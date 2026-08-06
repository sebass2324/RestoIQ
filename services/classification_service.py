"""
services/classification_service.py

Lógica de negocio de la Prioridad de Abastecimiento, extraída de
blueprints/abastecimiento.py para que sea reutilizable sin pasar por
Flask/HTTP — mismo motivo que prediction_service.py: Insights
(services/decision_engine.py) necesita este resultado como un dict
de Python plano.

No conoce Flask, no conoce HTML. Los errores de negocio se señalan
con ValueError.
"""

import os
import importlib
from datetime import datetime, timedelta
import pandas as pd
from services.classification_model import ModeloAbastecimiento
from services.matriz_confusion_img import generar_matriz_confusion_png
from services.dataset_hash import calcular_hash
from models import db
from models.venta import Venta
from models.dataset_usuario import DatasetUsuario
from models.modelo_clasificacion import ModeloClasificacion
from models.configuracion_analisis import ConfiguracionAnalisis

MODELOS_DIR = "ml_models"
HORIZONTE_DEFECTO = 7

# Mismo criterio que services/prediction_service.py (duplicado a
# propósito, mismo patrón de independencia entre servicios que ya
# usa este proyecto): reentrenar solo si el historial creció al
# menos UMBRAL_FILAS_NUEVAS filas O un UMBRAL_CRECIMIENTO_PCT%.
UMBRAL_FILAS_NUEVAS    = 20
UMBRAL_CRECIMIENTO_PCT = 0.05


def _supera_umbral_reentrenamiento(filas_actuales: int, filas_anteriores: int) -> bool:
    if not filas_anteriores or filas_anteriores <= 0:
        return True
    crecimiento_absoluto = filas_actuales - filas_anteriores
    if crecimiento_absoluto <= 0:
        return False
    crecimiento_relativo = crecimiento_absoluto / filas_anteriores
    return crecimiento_absoluto >= UMBRAL_FILAS_NUEVAS or crecimiento_relativo >= UMBRAL_CRECIMIENTO_PCT


def _ruta_modelo(user_id: int) -> str:
    return os.path.join(MODELOS_DIR, f"user_{user_id}_clasificacion.pkl")


def _ruta_matriz(user_id: int) -> str:
    return os.path.join(MODELOS_DIR, f"user_{user_id}_matriz_confusion.png")


def _cargar_feriados():
    for mod in ["services.data_generator", "data_generator"]:
        try:
            m = importlib.import_module(mod)
            return m.FERIADOS_ECUADOR
        except Exception:
            continue
    return set()


FERIADOS = _cargar_feriados()


def _features_fecha(fecha: pd.Timestamp) -> dict:
    fecha_str = fecha.strftime("%Y-%m-%d")
    dia = fecha.weekday()
    ayer = (fecha - timedelta(days=1)).strftime("%Y-%m-%d")
    maniana = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
    dia_mes = fecha.day
    return {
        "dia_semana":  dia,
        "mes":         fecha.month,
        "semana_anio": int(fecha.isocalendar()[1]),
        "es_finde":    int(dia in [5, 6]),
        "es_feriado":  int(fecha_str in FERIADOS),
        "es_puente":   int(ayer in FERIADOS or maniana in FERIADOS),
        "es_quincena": int(1 <= dia_mes <= 7 or 15 <= dia_mes <= 21),
    }


def _fechas_futuras(dias: int, dias_operacion=None) -> list:
    hoy = pd.Timestamp(datetime.now().date())
    fechas = []
    cursor = hoy
    intentos, limite = 0, dias * 4 + 14
    while len(fechas) < dias and intentos < limite:
        cursor = cursor + timedelta(days=1)
        intentos += 1
        if dias_operacion is None or cursor.weekday() in dias_operacion:
            fechas.append(cursor)
    return fechas


def _cargar_dataframe_usuario(user_id: int):
    """Lee ventas del usuario, incluida categoria (propia de este módulo)."""
    ventas = Venta.query.filter_by(user_id=user_id).all()
    if not ventas:
        return None
    data = [{
        "fecha":               v.fecha,
        "producto":            v.producto,
        "categoria":           v.categoria,
        "cantidad":            v.cantidad,
        "precio":              v.precio,
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


def _obtener_modelo(user_id: int, df: pd.DataFrame, forzar: bool = False):
    """Retorna (modelo, deltas)."""
    dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()
    if dataset is None:
        raise ValueError("No hay datos limpios para este usuario. Sube un archivo primero.")

    hash_actual = calcular_hash(df)
    registro = ModeloClasificacion.query.filter_by(user_id=user_id).first()
    ruta = _ruta_modelo(user_id)

    hash_cambio = registro is None or registro.dataset_hash != hash_actual
    supera_umbral = (
        registro is None
        or _supera_umbral_reentrenamiento(len(df), registro.filas_entrenamiento or 0)
    )

    necesita_reentrenar = (
        forzar
        or registro is None
        or not os.path.exists(ruta)
        or (hash_cambio and supera_umbral)
    )

    if not necesita_reentrenar:
        return ModeloAbastecimiento.cargar(ruta), None

    valores_previos = (
        {"f1_macro": registro.f1_macro, "accuracy": registro.accuracy}
        if registro is not None else None
    )

    modelo = ModeloAbastecimiento()
    metricas = modelo.entrenar(df)

    os.makedirs(MODELOS_DIR, exist_ok=True)
    modelo.guardar(ruta)

    if metricas.get("matriz_confusion") and metricas.get("clases"):
        ruta_matriz = _ruta_matriz(user_id)
        generar_matriz_confusion_png(
            metricas["matriz_confusion"], metricas["clases"], ruta_matriz
        )

    if registro is None:
        registro = ModeloClasificacion(user_id=user_id, ruta_pkl=ruta, dataset_hash=hash_actual)
        db.session.add(registro)

    registro.dataset_hash        = hash_actual
    registro.filas_entrenamiento = len(df)
    registro.f1_macro            = metricas.get("f1_macro")
    registro.accuracy            = metricas.get("accuracy")
    registro.ruta_pkl            = ruta
    registro.fecha_entrenamiento = datetime.utcnow()
    db.session.commit()

    deltas = None
    if valores_previos is not None and valores_previos["f1_macro"] is not None:
        deltas = {
            "f1_macro": round((metricas.get("f1_macro") or 0) - valores_previos["f1_macro"], 4),
            "accuracy": round((metricas.get("accuracy") or 0) - valores_previos["accuracy"], 4),
        }

    return modelo, deltas


def _factores_dia(feats_dia: dict) -> list:
    """Etiquetas legibles de los factores de contexto activos ese
    día. NO incluye promoción/evento: para fechas futuras esos
    valores siempre se asumen en 0."""
    etiquetas = []
    if feats_dia.get("es_finde"):    etiquetas.append("Fin de semana")
    if feats_dia.get("es_feriado"):  etiquetas.append("Feriado")
    if feats_dia.get("es_puente"):   etiquetas.append("Puente")
    if feats_dia.get("es_quincena"): etiquetas.append("Quincena")
    return etiquetas or ["Día laboral"]


def _predecir_horizonte(modelo: ModeloAbastecimiento, df: pd.DataFrame, dias: int, dias_operacion=None) -> list:
    cat_por_producto = (
        df.dropna(subset=["categoria"]).groupby("producto")["categoria"].first().to_dict()
        if "categoria" in df.columns else {}
    )
    precio_por_producto = (
        df.groupby("producto")["precio"].mean().to_dict()
        if "precio" in df.columns else {}
    )
    productos = sorted(df["producto"].unique().tolist())
    fechas = _fechas_futuras(dias, dias_operacion)

    # Armar TODOS los contextos primero (día × producto), sin predecir
    # todavía — la predicción se hace una sola vez, en lote, más abajo.
    # Predecir uno por uno acá adentro dispara joblib.Parallel una vez
    # por cada llamada, que en un horizonte con muchos días×productos
    # es mucho overhead de memoria innecesario (causó un Out of Memory
    # en producción con recursos limitados).
    contextos = []
    metadatos = []  # guarda fecha/producto/categoria/factores en el mismo orden que contextos
    for fecha in fechas:
        feats_dia = _features_fecha(fecha)
        factores = _factores_dia(feats_dia)
        for prod in productos:
            contextos.append({
                **feats_dia,
                "fecha":               fecha,
                "producto":            prod,
                "categoria":           cat_por_producto.get(prod),
                "precio":              precio_por_producto.get(prod),
                "promocion":           0,
                "descuento_pct":       0,
                "es_evento_especial":  0,
            })
            metadatos.append({
                "fecha": fecha.strftime("%Y-%m-%d"),
                "producto": prod,
                "categoria": cat_por_producto.get(prod) or "—",
                "factores": factores,
            })

    predicciones = modelo.predecir_lote(contextos)

    # Reagrupar por fecha, en el mismo orden que ya tenías
    por_fecha = {}
    for meta, r in zip(metadatos, predicciones):
        fila = {
            "producto":       meta["producto"],
            "categoria":      meta["categoria"],
            "prioridad":      r["prioridad"],
            "confianza":      r["confianza"],
            "probabilidades": r["probabilidades"],
            "factores":       meta["factores"],
        }
        por_fecha.setdefault(meta["fecha"], []).append(fila)

    orden_prioridad = {"Alta": 0, "Media": 1, "Baja": 2}
    resultado = []
    for fecha in fechas:
        fecha_str = fecha.strftime("%Y-%m-%d")
        productos_dia = por_fecha.get(fecha_str, [])
        productos_dia.sort(key=lambda p: orden_prioridad.get(p["prioridad"], 3))
        resultado.append({"fecha": fecha_str, "productos": productos_dia})
    return resultado


def obtener_config(user_id: int):
    return ConfiguracionAnalisis.query.filter_by(user_id=user_id).first()


# ════════════════════════════════════════════════════════════
# CASOS DE USO — punto de entrada para blueprints/abastecimiento.py
# Y para services/decision_engine.py (vía Insights)
# ════════════════════════════════════════════════════════════

def ejecutar_clasificacion(user_id: int, dias: int = None, forzar: bool = False) -> dict:
    """
    Caso de uso completo: obtiene/reentrena el clasificador y arma el
    resultado como un dict de Python plano (sin 'ok', sin nada HTTP).

    Lanza ValueError si no hay datos o falta la columna categoria.
    """
    dias = int(dias) if dias is not None else HORIZONTE_DEFECTO

    df = _cargar_dataframe_usuario(user_id)
    if df is None:
        raise ValueError("No se encontraron datos. Sube un archivo primero.")

    if "categoria" not in df.columns or df["categoria"].isna().all():
        raise ValueError(
            "Tu archivo no incluye una columna de categoría de producto. "
            "Es necesaria para que este modelo pueda evaluar productos nuevos "
            "sin historial. Vuelve a subir tu archivo incluyéndola."
        )

    config = obtener_config(user_id)
    dias_operacion = config.dias_operacion_set() if config else None

    modelo, deltas = _obtener_modelo(user_id, df, forzar=forzar)
    dias_pred = _predecir_horizonte(modelo, df, dias, dias_operacion)

    conteo_hoy = {"Alta": 0, "Media": 0, "Baja": 0}
    if dias_pred:
        for p in dias_pred[0]["productos"]:
            conteo_hoy[p["prioridad"]] += 1

    m = modelo.metricas or {}
    categorias = sorted({
        p["categoria"] for d in dias_pred for p in d["productos"] if p["categoria"] != "—"
    })

    return {
        "dias":       dias_pred,
        "resumen_proximo_dia": conteo_hoy,
        "categorias": categorias,
        "metricas": {
            "f1_macro":                 m.get("f1_macro"),
            "accuracy":                 m.get("accuracy"),
            "precision_macro":          m.get("precision_macro"),
            "recall_macro":             m.get("recall_macro"),
            "baseline_f1_mayoritaria":  m.get("baseline_f1_mayoritaria"),
            "mejora_vs_baseline_pct":   m.get("mejora_vs_baseline_pct"),
            "por_clase":                m.get("por_clase"),
            "top_features":             modelo.importancias[:6],
            "confianza_promedio":       m.get("confianza_promedio"),
            "matriz_confusion":         m.get("matriz_confusion"),
            "clases":                   m.get("clases"),
            "deltas":                   deltas,
        },
    }


def reentrenar_clasificacion(user_id: int) -> dict:
    """Fuerza reentrenamiento. Retorna {metricas, deltas}."""
    df = _cargar_dataframe_usuario(user_id)
    if df is None:
        raise ValueError("No hay datos disponibles.")

    modelo, deltas = _obtener_modelo(user_id, df, forzar=True)
    return {"metricas": modelo.metricas, "deltas": deltas}
