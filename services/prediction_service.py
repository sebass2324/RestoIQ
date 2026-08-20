"""
services/prediction_service.py

Lógica de negocio de la Predicción de Demanda, extraída de
blueprints/prediccion.py para que sea reutilizable sin pasar por
Flask/HTTP — el caso de uso concreto es services/decision_engine.py
(Insights), que necesita este mismo resultado como un dict de Python,
no como una respuesta JSON de un endpoint.

Este módulo NO conoce Flask, NO conoce HTML, NO arma respuestas HTTP.
Solo recibe un user_id y devuelve diccionarios de Python. Los errores
de negocio (falta configuración, no hay datos) se señalan con
excepciones (ValueError) — el blueprint que llame a esto decide cómo
convertir eso en una respuesta HTTP; este módulo no lo sabe ni le
importa.
"""

import os
from datetime import datetime
import pandas as pd
from services.sales_model import SalesModel
from services.dataset_hash import combinar_hash_config
from services.regresion_lineal_img import generar_grafico_regresion_png
from models import db
from models.venta import Venta
from models.dataset_usuario import DatasetUsuario
from models.modelo_ml import ModeloML
from models.configuracion_analisis import ConfiguracionAnalisis
from models.evento_futuro import EventoFuturo, EventoEspecialFuturo

MODELOS_DIR = "ml_models"

# Umbral híbrido: reentrenar solo si el historial creció al menos
# UMBRAL_FILAS_NUEVAS filas O un UMBRAL_CRECIMIENTO_PCT%, lo que ocurra
# primero. Sin esto, cualquier cambio (aunque sean 2 filas nuevas)
# dispara un reentrenamiento completo — y ese costo (memoria, tiempo)
# crece con el tamaño del historial, no es fijo. Con esto, subir un
# archivo chico actualiza los datos en MySQL igual, pero no reentrena
# hasta que el crecimiento realmente lo justifique.
UMBRAL_FILAS_NUEVAS   = 20
UMBRAL_CRECIMIENTO_PCT = 0.05


def _supera_umbral_reentrenamiento(filas_actuales: int, filas_anteriores: int) -> bool:
    if not filas_anteriores or filas_anteriores <= 0:
        return True  # sin base de comparación (primer entrenamiento o dato viejo sin este campo) → entrenar
    crecimiento_absoluto = filas_actuales - filas_anteriores
    if crecimiento_absoluto <= 0:
        return False  # el historial no creció (o encogió), no hay nada nuevo que justifique reentrenar
    crecimiento_relativo = crecimiento_absoluto / filas_anteriores
    return crecimiento_absoluto >= UMBRAL_FILAS_NUEVAS or crecimiento_relativo >= UMBRAL_CRECIMIENTO_PCT


def _ruta_modelo(user_id: int) -> str:
    return os.path.join(MODELOS_DIR, f"user_{user_id}.pkl")


def _ruta_grafico_regresion(user_id: int) -> str:
    return os.path.join(MODELOS_DIR, f"user_{user_id}_regresion_lineal.png")


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
        "dias_distancia_cobro": v.dias_distancia_cobro,
        "pico_comida_rapida":   int(v.pico_comida_rapida) if v.pico_comida_rapida is not None else 0,
        "fase_liquidez":         v.fase_liquidez,
        "promocion":           int(v.promocion) if v.promocion is not None else 0,
        "descuento_pct":       v.descuento_pct,
        "es_evento_especial":  int(v.es_evento_especial) if v.es_evento_especial is not None else 0,
        # Clima nocturno únicamente (horario real de operación,
        # 4pm-11pm). "lluvia_manana_mm"/"temp_manana_promed" se
        # probaron y, mediante prueba de ablación, se confirmó que no
        # aportan señal real (diferencia de WAPE: 0.04 puntos, dentro
        # del ruido) -- su alta importancia aparente era colinealidad
        # con la lluvia nocturna (correlación 0.38), no una relación
        # causal genuina. Se excluyen para simplificar el modelo sin
        # costo de precisión.
        "lluvia_nocturna_mm":    v.lluvia_nocturna_mm,
        "temp_nocturna_promed":  v.temp_nocturna_promed,
    } for v in ventas]

    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def _obtener_modelo(user_id: int, df: pd.DataFrame, config: ConfiguracionAnalisis, forzar: bool = False):
    """
    Devuelve (SalesModel listo, dict de métricas).

    Reentrena solo si: se fuerza explícitamente, no existe un modelo
    previo, el .pkl desapareció del disco, o el hash combinado
    (dataset + flags de configuración) cambió Y el usuario tiene
    activado "reentrenar automático".
    """
    dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()
    if dataset is None:
        raise ValueError("No hay datos limpios para este usuario. Sube un archivo primero.")

    hash_actual = combinar_hash_config(dataset.hash, config)
    registro_modelo = ModeloML.query.filter_by(user_id=user_id).first()
    ruta = _ruta_modelo(user_id)

    hash_cambio = registro_modelo is None or registro_modelo.dataset_hash != hash_actual
    supera_umbral = (
        registro_modelo is None
        or _supera_umbral_reentrenamiento(len(df), registro_modelo.filas_entrenamiento or 0)
    )
    hash_desactualizado = config.reentrenar_automatico and hash_cambio and supera_umbral

    necesita_reentrenar = (
        forzar
        or registro_modelo is None
        or not os.path.exists(ruta)
        or hash_desactualizado
    )

    if not necesita_reentrenar:
        metricas_cache = {
            "estrategia": registro_modelo.estrategia,
            "mae":        registro_modelo.mae,
            "mape":       registro_modelo.mape,
        }
        return SalesModel.cargar(ruta), metricas_cache

    model = SalesModel()
    metricas = model.entrenar(df, config=config, verbose=False)

    os.makedirs(MODELOS_DIR, exist_ok=True)
    model.guardar(ruta)

    # Gráfico explicativo del modelo GANADOR (matplotlib/seaborn) —
    # se genera UNA VEZ acá, nunca en el hot path de una vista.
    holdout = metricas.get("holdout")
    if holdout and holdout.get("scatter"):
        scatter = holdout["scatter"]
        ruta_grafico = _ruta_grafico_regresion(user_id)
        generar_grafico_regresion_png(
            real=[p["real"] for p in scatter],
            predicho=[p["predicho"] for p in scatter],
            r2=holdout.get("r2_ganador", 0),
            nombre_modelo=holdout.get("modelo_ganador", "el modelo"),
            ruta_salida=ruta_grafico,
        )

    if registro_modelo is None:
        registro_modelo = ModeloML(user_id=user_id, ruta_pkl=ruta, dataset_hash=hash_actual)
        db.session.add(registro_modelo)

    registro_modelo.dataset_hash        = hash_actual
    registro_modelo.filas_entrenamiento = len(df)
    registro_modelo.estrategia          = metricas.get("estrategia")
    registro_modelo.mae                 = metricas.get("mae")
    registro_modelo.mape                = metricas.get("mape")
    registro_modelo.ruta_pkl            = ruta
    registro_modelo.fecha_entrenamiento = datetime.utcnow()
    db.session.commit()

    return model, metricas


def _calcular_contexto_historico(df: pd.DataFrame):
    """Promedios derivados 100% del historial real — ¿un día/producto
    está por encima o por debajo de lo que normalmente pasa?"""
    totales_por_fecha = df.groupby("fecha")["cantidad"].sum().reset_index()
    totales_por_fecha["dia_semana"] = pd.to_datetime(totales_por_fecha["fecha"]).dt.weekday
    promedio_por_dia_semana = (
        totales_por_fecha.groupby("dia_semana")["cantidad"].mean().round(2).to_dict()
    )
    dias_historial = df["fecha"].nunique()
    promedio_diario_por_producto = (
        df.groupby("producto")["cantidad"].sum() / max(dias_historial, 1)
    ).round(2).to_dict()
    return promedio_por_dia_semana, promedio_diario_por_producto


def _clasificar_nivel(cantidad_predicha_diaria: float, promedio_historico_diario: float) -> str:
    """ALTO/MEDIO/BAJO comparando la predicción de un producto contra
    su PROPIO promedio histórico diario."""
    if not promedio_historico_diario:
        return "MEDIO"
    ratio = cantidad_predicha_diaria / promedio_historico_diario
    if ratio > 1.2:
        return "ALTO"
    if ratio < 0.8:
        return "BAJO"
    return "MEDIO"


def obtener_config(user_id: int):
    """Devuelve la ConfiguracionAnalisis del usuario, o None si no configuró su negocio."""
    return ConfiguracionAnalisis.query.filter_by(user_id=user_id).first()


def obtener_productos_usuario(user_id: int) -> list:
    """Lista de productos únicos del usuario, para selectores en la UI
    (ej. declarar un evento futuro para un producto específico)."""
    filas = Venta.query.with_entities(Venta.producto).filter_by(user_id=user_id).distinct().all()
    return sorted({f[0] for f in filas})


# ════════════════════════════════════════════════════════════
# EVENTOS FUTUROS — promociones (por producto) y eventos especiales
# (por día completo) que el dueño YA SABE que van a pasar en una
# fecha futura -- ver models/evento_futuro.py para la explicación
# completa de por qué son 2 mecanismos separados.
# ════════════════════════════════════════════════════════════

def obtener_eventos_futuros(user_id: int) -> dict:
    """
    {"YYYY-MM-DD": {producto_o_"__TODOS__": {"promocion":0/1,
    "es_evento_especial":0/1, "descuento_pct":float}}}
    """
    promos = EventoFuturo.query.filter_by(user_id=user_id).all()
    especiales = EventoEspecialFuturo.query.filter_by(user_id=user_id).all()

    resultado = {}
    for e in promos:
        fecha_str = e.fecha.strftime("%Y-%m-%d")
        clave = e.producto if e.producto else "__TODOS__"
        entrada = resultado.setdefault(fecha_str, {}).setdefault(clave, {})
        entrada["promocion"] = int(e.es_promocion)
        entrada["descuento_pct"] = e.descuento_pct or 0.0

    for ev in especiales:
        fecha_str = ev.fecha.strftime("%Y-%m-%d")
        entrada = resultado.setdefault(fecha_str, {}).setdefault("__TODOS__", {})
        entrada["es_evento_especial"] = 1

    return resultado


def guardar_evento_futuro(user_id: int, fecha, producto=None, es_promocion=False,
                          es_evento_especial=False, descuento_pct=None, nota=None) -> dict:
    resultado = {"promocion_guardada": False, "evento_especial_guardado": False}

    if es_promocion or descuento_pct:
        evento = EventoFuturo.query.filter_by(user_id=user_id, fecha=fecha, producto=producto).first()
        if evento is None:
            evento = EventoFuturo(user_id=user_id, fecha=fecha, producto=producto)
            db.session.add(evento)
        evento.es_promocion = bool(es_promocion)
        evento.descuento_pct = float(descuento_pct) if descuento_pct not in (None, "") else None
        evento.nota = nota
        resultado["promocion_guardada"] = True

    if es_evento_especial:
        especial = EventoEspecialFuturo.query.filter_by(user_id=user_id, fecha=fecha).first()
        if especial is None:
            especial = EventoEspecialFuturo(user_id=user_id, fecha=fecha)
            db.session.add(especial)
        especial.nota = nota
        resultado["evento_especial_guardado"] = True

    db.session.commit()
    return resultado


def eliminar_evento_futuro(user_id: int, fecha, producto=None) -> bool:
    evento = EventoFuturo.query.filter_by(user_id=user_id, fecha=fecha, producto=producto).first()
    if evento is None:
        return False
    db.session.delete(evento)
    db.session.commit()
    return True


def eliminar_evento_especial_futuro(user_id: int, fecha) -> bool:
    especial = EventoEspecialFuturo.query.filter_by(user_id=user_id, fecha=fecha).first()
    if especial is None:
        return False
    db.session.delete(especial)
    db.session.commit()
    return True


# ════════════════════════════════════════════════════════════
# CASOS DE USO — punto de entrada para blueprints/prediccion.py
# Y para services/decision_engine.py (vía Insights)
# ════════════════════════════════════════════════════════════

def ejecutar_prediccion(user_id: int, dias: int = None, forzar: bool = False) -> dict:
    """
    Caso de uso completo: obtiene/reentrena el modelo y arma el
    resultado de predicción de demanda como un dict de Python plano
    (sin 'ok', sin nada HTTP — eso lo decide quien llame a esto).

    Lanza ValueError si el usuario no configuró su negocio o no tiene
    datos — el llamador decide cómo mostrarlo (JSON 400, flash, etc.).
    """
    config = obtener_config(user_id)
    if config is None:
        raise ValueError("Configura tu negocio antes de predecir.")

    dias = int(dias) if dias is not None else config.horizonte_dias

    df = _cargar_dataframe_usuario(user_id)
    if df is None:
        raise ValueError("No se encontraron datos. Sube un archivo primero.")

    model, _ = _obtener_modelo(user_id, df, config, forzar=forzar)
    eventos_futuros = obtener_eventos_futuros(user_id)
    resultado = model.predecir(dias=dias, dias_operacion=config.dias_operacion_set(),
                               eventos_futuros=eventos_futuros)

    promedio_dia_semana, promedio_diario_producto = _calcular_contexto_historico(df)

    diario_df = resultado["diario"]
    diario_records = diario_df.to_dict(orient="records")
    for fila in diario_records:
        dia_semana = pd.Timestamp(fila["fecha"]).weekday()
        fila["promedio_historico"] = promedio_dia_semana.get(dia_semana)

    fila_pico = max(diario_records, key=lambda f: f["cantidad_total_pred"])
    pct_dia_critico = None
    if fila_pico.get("promedio_historico"):
        pct_dia_critico = round(
            (fila_pico["cantidad_total_pred"] / fila_pico["promedio_historico"] - 1) * 100, 1
        )

    por_producto_df = resultado["por_producto"]
    total_por_producto = por_producto_df.groupby("producto")["cantidad_pred"].sum()
    niveles_producto = {}
    for producto, total in total_por_producto.items():
        promedio_diario_predicho = total / max(dias, 1)
        promedio_hist = promedio_diario_producto.get(producto, 0)
        niveles_producto[producto] = {
            "nivel": _clasificar_nivel(promedio_diario_predicho, promedio_hist),
            "promedio_historico_diario": round(promedio_hist, 2),
        }

    metricas = model.metricas or {}

    return {
        "estrategia":        resultado["resumen"]["estrategia"],
        "resumen":           resultado["resumen"],
        "diario":            diario_records,
        "por_producto":      resultado["por_producto"].to_dict(orient="records"),
        "pivote":            resultado["pivote"].to_dict(orient="records"),
        "dia_critico": {
            "fecha":              fila_pico["fecha"],
            "cantidad":           fila_pico["cantidad_total_pred"],
            "pct_sobre_promedio": pct_dia_critico,
        },
        "niveles_producto": niveles_producto,
        "calidad_modelo": {
            "mae":              metricas.get("mae"),
            "mape":             metricas.get("mape"),
            "wape":             metricas.get("wape"),
            "rmse":             metricas.get("rmse"),
            "nrmse":            metricas.get("nrmse"),
            "nmae":             metricas.get("nmae"),
            "mae_provisional":  metricas.get("mae_provisional"),
            "dias_historial":   metricas.get("dias_historial"),
            "umbral_dias_ml":   metricas.get("umbral_dias_ml"),
            "holdout":          metricas.get("holdout"),
            "importancias":     metricas.get("importancias_top10"),
            "grafico_regresion_disponible": bool(
                metricas.get("holdout") and metricas["holdout"].get("scatter")
            ),
        },
    }


def reentrenar_prediccion(user_id: int) -> dict:
    """Fuerza reentrenamiento. Retorna solo las métricas (dict plano)."""
    config = obtener_config(user_id)
    if config is None:
        raise ValueError("Configura tu negocio antes de predecir.")

    df = _cargar_dataframe_usuario(user_id)
    if df is None:
        raise ValueError("No hay datos disponibles.")

    _, metricas = _obtener_modelo(user_id, df, config, forzar=True)
    return metricas