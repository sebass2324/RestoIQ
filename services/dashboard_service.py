"""
KPIs de negocio calculados desde la tabla `ventas` de un usuario.

Se usa tanto en el Dashboard como en el reporte post-upload, para que
ambas pantallas muestren siempre la misma información — una sola fuente
de verdad — en vez de que cada una calcule sus propios números.
"""

import os
from datetime import datetime
import pandas as pd
from sqlalchemy import func
from models import db
from models.venta import Venta
from models.dataset_usuario import DatasetUsuario
from models.modelo_ml import ModeloML
from models.configuracion_analisis import ConfiguracionAnalisis
from services.sales_model import SalesModel
from services.dataset_hash import combinar_hash_config

DIAS_SEMANA = ["Lunes", "Martes", "Miércoles", "Jueves", "Viernes", "Sábado", "Domingo"]

# Misma carpeta que usa prediccion.py — un .pkl por usuario. Se
# duplica la ruta aquí a propósito (ver decisión de arquitectura:
# Dashboard y Predicción no comparten un service de orquestación
# para no acoplarlos entre sí).
MODELOS_DIR = "ml_models"


def obtener_kpis_usuario(user_id: int):
    """
    Retorna un dict con KPIs de negocio del usuario, o None si todavía
    no tiene ninguna venta cargada (aún no subió ningún archivo).
    """
    total_filas = Venta.query.filter_by(user_id=user_id).count()
    if total_filas == 0:
        return None

    ingreso_expr = func.coalesce(Venta.total, Venta.cantidad * Venta.precio)

    tiene_ingresos = (
        Venta.query
        .filter(Venta.user_id == user_id)
        .filter(db.or_(Venta.total.isnot(None), Venta.precio.isnot(None)))
        .first() is not None
    )

    ingreso_total = (
        db.session.query(func.sum(ingreso_expr))
        .filter(Venta.user_id == user_id)
        .scalar() or 0
    )

    top_producto = (
        db.session.query(Venta.producto, func.sum(Venta.cantidad).label("cantidad"))
        .filter(Venta.user_id == user_id)
        .group_by(Venta.producto)
        .order_by(func.sum(Venta.cantidad).desc())
        .first()
    )

    mejor_dia = (
        db.session.query(Venta.dia_semana, func.avg(Venta.cantidad).label("promedio"))
        .filter(Venta.user_id == user_id)
        .group_by(Venta.dia_semana)
        .order_by(func.avg(Venta.cantidad).desc())
        .first()
    )

    productos_distintos = (
        db.session.query(func.count(func.distinct(Venta.producto)))
        .filter(Venta.user_id == user_id)
        .scalar() or 0
    )

    fecha_min, fecha_max = (
        db.session.query(func.min(Venta.fecha), func.max(Venta.fecha))
        .filter(Venta.user_id == user_id)
        .first()
    )

    # Serie diaria: últimos 30 DÍAS CON VENTAS (no 30 días de calendario,
    # porque el historial puede tener huecos o ser de un periodo pasado).
    serie_rows = (
        db.session.query(
            Venta.fecha,
            func.sum(ingreso_expr).label("ingreso"),
            func.sum(Venta.cantidad).label("cantidad"),
        )
        .filter(Venta.user_id == user_id)
        .group_by(Venta.fecha)
        .order_by(Venta.fecha.desc())
        .limit(30)
        .all()
    )
    serie_diaria = [
        {
            "fecha":    row.fecha.strftime("%Y-%m-%d"),
            "ingreso":  round(float(row.ingreso or 0), 2),
            "cantidad": int(row.cantidad or 0),
        }
        for row in reversed(serie_rows)
    ]

    top_rows = (
        db.session.query(
            Venta.producto,
            func.sum(ingreso_expr).label("ingreso"),
            func.sum(Venta.cantidad).label("cantidad"),
        )
        .filter(Venta.user_id == user_id)
        .group_by(Venta.producto)
        .order_by(func.sum(Venta.cantidad).desc())
        .limit(8)
        .all()
    )
    top_productos = [
        {
            "producto": row.producto,
            "ingreso":  round(float(row.ingreso or 0), 2),
            "cantidad": int(row.cantidad or 0),
        }
        for row in top_rows
    ]

    dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()

    return {
        "tiene_ingresos":        tiene_ingresos,
        "ingreso_total":         round(float(ingreso_total), 2),
        "top_producto":          top_producto.producto if top_producto else "—",
        "top_producto_cantidad": int(top_producto.cantidad) if top_producto else 0,
        "mejor_dia":             DIAS_SEMANA[mejor_dia.dia_semana] if mejor_dia else "—",
        "productos_distintos":   int(productos_distintos),
        "filas_totales":         total_filas,
        "fecha_min":             fecha_min.strftime("%d/%m/%Y") if fecha_min else None,
        "fecha_max":             fecha_max.strftime("%d/%m/%Y") if fecha_max else None,
        "nombre_archivo":        dataset.nombre_archivo if dataset else None,
        "fecha_subida":          dataset.fecha_subida if dataset else None,
        "serie_diaria":          serie_diaria,
        "top_productos":         top_productos,
    }


# ════════════════════════════════════════════════════════════════
# Datos de demanda para el Dashboard (Predicción de Demanda y
# Planificación de Abastecimiento) — no gestión de inventario.
#
# NOTA DE ARQUITECTURA: _cargar_dataframe_usuario y
# _obtener_o_entrenar_modelo son una copia intencional de la lógica
# equivalente en blueprints/prediccion.py. Se decidió (sesión de
# rediseño del Dashboard) NO crear un service de orquestación
# compartido entre Dashboard y Predicción, para que evolucionen sin
# acoplarse — a costa de mantener esta lógica en dos lugares. Si
# alguna vez se corrige un bug en la carga de datos o el
# reentrenamiento, hay que replicarlo aquí también.
# ════════════════════════════════════════════════════════════════

def _ruta_modelo(user_id: int) -> str:
    return os.path.join(MODELOS_DIR, f"user_{user_id}.pkl")


def _cargar_dataframe_usuario(user_id: int):
    """Lee TODAS las ventas limpias del usuario desde MySQL."""
    ventas = Venta.query.filter_by(user_id=user_id).all()
    if not ventas:
        return None

    data = [{
        "fecha":               v.fecha,
        "producto":            v.producto,
        "cantidad":            v.cantidad,
        "precio":              v.precio,
        "total":               v.total,
        "dia_semana":          v.dia_semana,
        "mes":                 v.mes,
        "semana_anio":         v.semana_anio,
        "es_finde":            v.es_finde,
        "es_feriado":          v.es_feriado,
        "es_puente":           v.es_puente,
        "es_quincena":         v.es_quincena,
        "promocion":           v.promocion,
        "descuento_pct":       v.descuento_pct,
        "es_evento_especial":  v.es_evento_especial,
    } for v in ventas]

    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _obtener_o_entrenar_modelo(user_id: int, df: pd.DataFrame, config: ConfiguracionAnalisis) -> SalesModel:
    """
    Devuelve un SalesModel listo para predecir (cargado desde caché o
    recién entrenado). A diferencia de la versión en prediccion.py, no
    hace falta devolver un dict de métricas aparte: todo lo que
    necesita el Módulo de Confianza (mae, mape, holdout, top_features)
    viaja dentro de model.metricas, ya sea que el modelo se acabe de
    entrenar o se haya cargado desde el .pkl cacheado.
    """
    dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()
    if dataset is None:
        raise ValueError("No hay datos limpios para este usuario. Sube un archivo primero.")

    hash_actual = combinar_hash_config(dataset.hash, config)
    registro_modelo = ModeloML.query.filter_by(user_id=user_id).first()
    ruta = _ruta_modelo(user_id)

    hash_desactualizado = (
        config.reentrenar_automatico
        and (registro_modelo is None or registro_modelo.dataset_hash != hash_actual)
    )
    necesita_reentrenar = (
        registro_modelo is None
        or not os.path.exists(ruta)
        or hash_desactualizado
    )

    if not necesita_reentrenar:
        return SalesModel.cargar(ruta)

    model = SalesModel()
    metricas = model.entrenar(df, config=config, verbose=False)

    os.makedirs(MODELOS_DIR, exist_ok=True)
    model.guardar(ruta)

    if registro_modelo is None:
        registro_modelo = ModeloML(user_id=user_id, ruta_pkl=ruta, dataset_hash=hash_actual)
        db.session.add(registro_modelo)

    registro_modelo.dataset_hash        = hash_actual
    registro_modelo.estrategia          = metricas.get("estrategia")
    registro_modelo.mae                 = metricas.get("mae")
    registro_modelo.mape                = metricas.get("mape")
    registro_modelo.ruta_pkl            = ruta
    registro_modelo.fecha_entrenamiento = datetime.utcnow()
    db.session.commit()

    return model


def obtener_datos_demanda_usuario(user_id: int) -> dict:
    """
    Arma los datos del Dashboard de Predicción de Demanda y
    Planificación de Abastecimiento. Retorna un dict con:

      - estado: "sin_datos" | "sin_configuracion" | "listo"
      - (si estado == "listo") todo lo que necesitan los 4 bloques:
        Resumen de Demanda, Proyección de Demanda, Plan de
        Preparación y Módulo de Confianza Científica.

    Todo lo que se devuelve es derivable de: historial de ventas,
    demanda pronosticada, métricas del modelo o configuración del
    negocio — nunca de conceptos de inventario/stock, que este
    sistema no gestiona.
    """
    dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()
    if dataset is None:
        return {"estado": "sin_datos"}

    config = ConfiguracionAnalisis.query.filter_by(user_id=user_id).first()
    if config is None:
        return {"estado": "sin_configuracion"}

    df = _cargar_dataframe_usuario(user_id)
    if df is None:
        return {"estado": "sin_datos"}

    model = _obtener_o_entrenar_modelo(user_id, df, config)
    registro_modelo = ModeloML.query.filter_by(user_id=user_id).first()
    fecha_entrenamiento = registro_modelo.fecha_entrenamiento if registro_modelo else None

    resultado = model.predecir(dias=config.horizonte_dias, dias_operacion=config.dias_operacion_set())

    diario       = resultado["diario"]
    por_producto = resultado["por_producto"]

    demanda_total_horizonte = int(diario["cantidad_total_pred"].sum())

    ranking_productos = (
        por_producto.groupby("producto")["cantidad_pred"]
        .sum()
        .sort_values(ascending=False)
    )
    plan_preparacion = [
        {"producto": prod, "cantidad_estimada": int(cant)}
        for prod, cant in ranking_productos.items()
    ]
    producto_top_nombre    = ranking_productos.index[0] if len(ranking_productos) else "—"
    producto_top_cantidad  = int(ranking_productos.iloc[0]) if len(ranking_productos) else 0

    fila_pico  = diario.loc[diario["cantidad_total_pred"].idxmax()]
    fila_valle = diario.loc[diario["cantidad_total_pred"].idxmin()]

    # Series por producto para el selector del gráfico de Proyección
    # de Demanda ("Todos los productos" vs. un producto puntual) — se
    # arma en el backend una sola vez, sin pedir datos nuevos cuando
    # el usuario cambia el filtro en el frontend.
    series_por_producto = {
        prod: grupo[["fecha", "cantidad_pred"]].to_dict(orient="records")
        for prod, grupo in por_producto.groupby("producto")
    }

    metricas = model.metricas or {}

    # Confiabilidad en etiqueta simple, para el Dashboard "resumen
    # ejecutivo" (el detalle numérico completo vive en /prediccion)
    precision_pct = round(100 - metricas["mape"], 1) if metricas.get("mape") is not None else None
    if precision_pct is None:
        confiabilidad_label = "No disponible"
    elif precision_pct >= 80:
        confiabilidad_label = "Alta"
    elif precision_pct >= 50:
        confiabilidad_label = "Media"
    else:
        confiabilidad_label = "Baja"

    return {
        "estado":           "listo",
        "horizonte_dias":   config.horizonte_dias,
        "estrategia":       metricas.get("estrategia"),

        # ── Franja superior: decisiones inmediatas ──
        "producto_top_nombre":     producto_top_nombre,
        "producto_top_cantidad":   producto_top_cantidad,
        "dia_pico":            {"fecha": fila_pico["fecha"], "cantidad": int(fila_pico["cantidad_total_pred"])},
        "confiabilidad_label":  confiabilidad_label,
        "precision_pct":        precision_pct,
        "fecha_ultimo_dato":    metricas.get("fecha_hasta"),

        # ── Franja central: qué hacer ──
        "demanda_total_horizonte": demanda_total_horizonte,
        "serie_diaria_total":  diario[["fecha", "cantidad_total_pred"]].to_dict(orient="records"),
        "plan_preparacion": plan_preparacion,  # completo — /prediccion lo usa entero; el Dashboard solo pinta el top 5

        # ── Franja inferior: ¿puedo confiar? ──
        "confianza": {
            "mae":             metricas.get("mae"),
            "mape":            metricas.get("mape"),
            "precision_pct":   precision_pct,
            "top_features":    metricas.get("top_features", []),
            "holdout":         metricas.get("holdout"),
        },
        "registros_analizados": len(df),

        # ── "El sistema aprende de tus datos" — transparencia sobre
        # cuánto historial respalda el modelo actual, para que se
        # entienda que no reemplaza conocimiento, lo acumula. ──
        "info_modelo": {
            "registros_usados":  len(df),
            "meses_historial":   (
                round(metricas["dias_historial"] / 30.44, 1)
                if metricas.get("dias_historial") else None
            ),
            "fecha_ultimo_entrenamiento": (
                fecha_entrenamiento.strftime("%d/%m/%Y %H:%M")
                if fecha_entrenamiento else None
            ),
        },

        # ── Datos completos para la vista de exploración (/prediccion) ──
        "dia_valle":           {"fecha": fila_valle["fecha"], "cantidad": int(fila_valle["cantidad_total_pred"])},
        "series_por_producto": series_por_producto,
        "productos":           sorted(por_producto["producto"].unique().tolist()),
    }