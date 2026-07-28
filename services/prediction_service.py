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

MODELOS_DIR = "ml_models"


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
        "es_quincena":         v.es_quincena,
        "promocion":           v.promocion,
        "descuento_pct":       v.descuento_pct,
        "es_evento_especial":  v.es_evento_especial,
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

    # En un arranque en frío (redeploy/reinicio), el .pkl no está en
    # disco local aunque el registro en MySQL diga que existe — antes
    # de asumir "hay que reentrenar", intentar descargarlo de Supabase
    # Storage. Si Supabase no está configurado (desarrollo local), esto
    # no hace nada y se comporta exactamente como antes.
    if registro_modelo is not None:
        from services.storage_service import asegurar_local
        asegurar_local(os.path.basename(ruta), ruta)

    hash_desactualizado = (
        config.reentrenar_automatico
        and (registro_modelo is None or registro_modelo.dataset_hash != hash_actual)
    )
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
        from services.storage_service import subir
        subir(ruta_grafico, os.path.basename(ruta_grafico))

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
    resultado = model.predecir(dias=dias, dias_operacion=config.dias_operacion_set())

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
        "objetivo_analisis": config.objetivo_analisis,
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