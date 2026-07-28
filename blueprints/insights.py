"""
blueprints/insights.py

Insights (Decision Engine). NO persiste nada en base de datos — cada
vez que se genera, corre predicción + clasificación en vivo y las
cruza. Simple a propósito: sin historial de ejecuciones guardadas.
"""

from flask import Blueprint, render_template, jsonify, redirect, url_for, flash
from flask_login import login_required, current_user
from services.prediction_service import ejecutar_prediccion, obtener_config as obtener_config_prediccion
from services.classification_service import ejecutar_clasificacion
from services.decision_engine import generar_insights, calcular_resumen
from models import db

insights_bp = Blueprint("insights", __name__)


@insights_bp.route("/insights")
@login_required
def index():
    config = obtener_config_prediccion(current_user.id)
    if config is None:
        flash("Antes de ver insights, cuéntanos cómo opera tu negocio.", "info")
        return redirect(url_for("configuracion.index"))
    return render_template("insights/index.html", config=config)


@insights_bp.route("/insights/generar", methods=["POST"])
@login_required
def generar():
    try:
        config = obtener_config_prediccion(current_user.id)
        if config is None:
            return jsonify({"ok": False, "error": "Configura tu negocio antes de ver insights."}), 400

        dias = config.horizonte_dias

        try:
            datos_prediccion = ejecutar_prediccion(current_user.id, dias=dias)
        except ValueError as e:
            return jsonify({"ok": False, "error": f"Predicción de demanda: {e}"}), 400

        try:
            datos_clasificacion = ejecutar_clasificacion(current_user.id, dias=dias)
        except ValueError as e:
            return jsonify({"ok": False, "error": f"Clasificación de prioridad: {e}"}), 400

        insights = generar_insights(datos_prediccion, datos_clasificacion)

        return jsonify({
            "ok": True,
            "insights": insights,
            "total": len(insights),
            "resumen": calcular_resumen(insights),
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500