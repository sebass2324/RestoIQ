import os
from flask import Blueprint, render_template
from flask_login import login_required, current_user
from services.dashboard_service import obtener_datos_demanda_usuario

dashboard_bp = Blueprint("dashboard", __name__)


def _resumen_modelo_prediccion(user_id: int):
    """Lee el MAE normalizado y el R² del modelo de predicción ya
    entrenado, cargando el .pkl en caché — NUNCA reentrena aquí (eso
    es responsabilidad exclusiva de /prediccion). Devuelve None si el
    usuario todavía no tiene un modelo entrenado."""
    from services.sales_model import SalesModel
    from models.modelo_ml import ModeloML

    registro = ModeloML.query.filter_by(user_id=user_id).first()
    if registro is None or not registro.ruta_pkl or not os.path.exists(registro.ruta_pkl):
        return None

    modelo = SalesModel.cargar(registro.ruta_pkl)
    m = modelo.metricas or {}
    holdout = m.get("holdout") or {}
    return {
        "nmae": m.get("nmae"),
        "r2":   holdout.get("r2_restoiq"),
    }


def _resumen_modelo_clasificacion(user_id: int):
    """Igual que arriba, pero para el clasificador de abastecimiento
    (Accuracy + F1 macro). Tampoco reentrena — solo lee el .pkl."""
    from services.classification_model import ModeloAbastecimiento
    from models.modelo_clasificacion import ModeloClasificacion

    registro = ModeloClasificacion.query.filter_by(user_id=user_id).first()
    if registro is None or not registro.ruta_pkl or not os.path.exists(registro.ruta_pkl):
        return None

    modelo = ModeloAbastecimiento.cargar(registro.ruta_pkl)
    m = modelo.metricas or {}
    return {
        "accuracy": m.get("accuracy"),
        "f1_macro": m.get("f1_macro"),
    }


@dashboard_bp.route("/dashboard")
@login_required
def index():
    """
    Centro de decisiones de RestoIQ. Responde 3 preguntas: cómo se
    comportará la demanda, qué preparar/abastecer, y qué tan
    confiables son las predicciones — nunca gestión de inventario.

    datos["estado"] puede ser:
      - "sin_datos":          el usuario aún no subió ningún archivo
      - "sin_configuracion":  subió datos pero no configuró su negocio
      - "listo":               hay demanda pronosticada para mostrar
    """
    datos = obtener_datos_demanda_usuario(current_user.id)
    return render_template(
        "dashboard/index.html",
        usuario=current_user,
        datos=datos,
        resumen_prediccion=_resumen_modelo_prediccion(current_user.id),
        resumen_clasificacion=_resumen_modelo_clasificacion(current_user.id),
    )