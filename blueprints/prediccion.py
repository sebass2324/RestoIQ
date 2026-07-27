import os
from datetime import datetime
import pandas as pd
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from services.sales_model import SalesModel
from services.dataset_hash import combinar_hash_config
from models import db
from models.venta import Venta
from models.dataset_usuario import DatasetUsuario
from models.modelo_ml import ModeloML
from models.configuracion_analisis import ConfiguracionAnalisis

prediccion_bp = Blueprint("prediccion", __name__)

# Carpeta separada de models/ (paquete Python de SQLAlchemy) para no
# mezclar código importable con artefactos binarios de ML.
MODELOS_DIR = "ml_models"


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


def _obtener_modelo(user_id: int, df: pd.DataFrame, config: ConfiguracionAnalisis, forzar: bool = False):
    """
    Devuelve (SalesModel listo, dict de métricas).

    Reentrena solo si:
      - se fuerza explícitamente (botón "Reentrenar"), o
      - no existe un modelo previo para este usuario, o
      - el .pkl desapareció del disco, o
      - el hash combinado (dataset + flags de configuración) cambió
        Y el usuario tiene activado "reentrenar automático".

    Si el usuario desactivó "reentrenar automático", el chequeo de
    hash se ignora por completo — solo el botón manual (forzar=True)
    dispara un nuevo entrenamiento, tal como se definió en el wizard.
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

    # Reentrenar, usando la configuración de negocio del usuario
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

    return model, metricas


def _obtener_config_o_none(user_id: int):
    return ConfiguracionAnalisis.query.filter_by(user_id=user_id).first()


def _calcular_contexto_historico(df: pd.DataFrame):
    """
    Promedios derivados 100% del historial real (sin inventar datos
    nuevos como "capacidad"), usados para contextualizar la
    predicción: ¿un día/producto está por encima o por debajo de lo
    que normalmente pasa?

    Retorna:
      - promedio_por_dia_semana: {0..6: promedio histórico de unidades
        totales vendidas ese día de la semana} — para la línea de
        referencia del gráfico y el % del "día crítico".
      - promedio_diario_por_producto: {producto: promedio histórico de
        unidades vendidas por día} — para clasificar ALTO/MEDIO/BAJO.
    """
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
    su PROPIO promedio histórico diario (no contra otros productos)."""
    if not promedio_historico_diario:
        return "MEDIO"
    ratio = cantidad_predicha_diaria / promedio_historico_diario
    if ratio > 1.2:
        return "ALTO"
    if ratio < 0.8:
        return "BAJO"
    return "MEDIO"


@prediccion_bp.route("/prediccion")
@login_required
def index():
    config = _obtener_config_o_none(current_user.id)
    if config is None:
        flash("Antes de predecir, cuéntanos cómo opera tu negocio.", "info")
        return redirect(url_for("configuracion.index"))
    return render_template("prediccion/index.html", config=config)


@prediccion_bp.route("/prediccion/ejecutar", methods=["POST"])
@login_required
def ejecutar():
    """
    Recibe { dias: N } desde el frontend (opcional — si no viene, usa
    el horizonte configurado por el usuario), obtiene o reentrena el
    modelo, y devuelve cantidad demandada + ingreso estimado.
    """
    try:
        config = _obtener_config_o_none(current_user.id)
        if config is None:
            return jsonify({"ok": False, "error": "Configura tu negocio antes de predecir."}), 400

        body = request.get_json(silent=True) or {}
        dias = int(body.get("dias", config.horizonte_dias))

        df = _cargar_dataframe_usuario(current_user.id)
        if df is None:
            return jsonify({
                "ok": False,
                "error": "No se encontraron datos. Sube un archivo primero."
            }), 400

        model, _ = _obtener_modelo(current_user.id, df, config)
        resultado = model.predecir(dias=dias, dias_operacion=config.dias_operacion_set())

        promedio_dia_semana, promedio_diario_producto = _calcular_contexto_historico(df)

        diario_df = resultado["diario"]
        # Línea de referencia: promedio histórico real del mismo día de
        # semana, alineado fecha a fecha con la predicción — no una
        # "capacidad" inventada, sino "qué es normal ese día" según tus
        # propios datos.
        diario_records = diario_df.to_dict(orient="records")
        for fila in diario_records:
            dia_semana = pd.Timestamp(fila["fecha"]).weekday()
            fila["promedio_historico"] = promedio_dia_semana.get(dia_semana)

        # Día crítico: el de mayor demanda pronosticada, con el % que
        # representa sobre el promedio histórico de ese mismo día de
        # semana (no sobre el promedio del propio período pronosticado).
        fila_pico = max(diario_records, key=lambda f: f["cantidad_total_pred"])
        pct_dia_critico = None
        if fila_pico.get("promedio_historico"):
            pct_dia_critico = round(
                (fila_pico["cantidad_total_pred"] / fila_pico["promedio_historico"] - 1) * 100, 1
            )

        # Nivel por producto: ALTO/MEDIO/BAJO contra el propio
        # historial de cada producto (no ranking entre productos).
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

        return jsonify({
            "ok":           True,
            "estrategia":   resultado["resumen"]["estrategia"],
            "objetivo_analisis": config.objetivo_analisis,
            "resumen":      resultado["resumen"],
            "diario":       diario_records,
            "por_producto": resultado["por_producto"].to_dict(orient="records"),
            "pivote":       resultado["pivote"].to_dict(orient="records"),
            "dia_critico": {
                "fecha":               fila_pico["fecha"],
                "cantidad":            fila_pico["cantidad_total_pred"],
                "pct_sobre_promedio":  pct_dia_critico,
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
                "holdout":          metricas.get("holdout"),  # incluye baseline_nombre, mejora_pct, n_folds, etc.
                # Solo existe si la estrategia es LightGBM (Promedio
                # Móvil no tiene concepto de "importancia de features").
                # Ya viaja calculado desde el entrenamiento — se dibuja
                # con Chart.js en el navegador, no con matplotlib en
                # el servidor (ver services/analisis_modelo.py para el
                # análisis offline con matplotlib/seaborn).
                "importancias":     metricas.get("importancias_top10"),
            },
        })

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


@prediccion_bp.route("/prediccion/reentrenar", methods=["POST"])
@login_required
def reentrenar():
    """Fuerza reentrenamiento con los datos y configuración actuales del usuario."""
    try:
        config = _obtener_config_o_none(current_user.id)
        if config is None:
            return jsonify({"ok": False, "error": "Configura tu negocio antes de predecir."}), 400

        df = _cargar_dataframe_usuario(current_user.id)
        if df is None:
            return jsonify({"ok": False, "error": "No hay datos disponibles."}), 400

        _, metricas = _obtener_modelo(current_user.id, df, config, forzar=True)
        return jsonify({"ok": True, "metricas": metricas})

    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500