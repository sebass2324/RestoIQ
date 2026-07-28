"""
blueprints/prediccion.py

Wrapper HTTP fino sobre services/prediction_service.py — toda la
lógica de negocio vive en el service (reutilizable, sin Flask). Este
archivo solo traduce: request → llamada al service → jsonify/render.
"""

import os
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import login_required, current_user
from services.prediction_service import (
    ejecutar_prediccion, reentrenar_prediccion, obtener_config, _ruta_grafico_regresion,
)
from models import db

prediccion_bp = Blueprint("prediccion", __name__)


@prediccion_bp.route("/prediccion")
@login_required
def index():
    config = obtener_config(current_user.id)
    if config is None:
        flash("Antes de predecir, cuéntanos cómo opera tu negocio.", "info")
        return redirect(url_for("configuracion.index"))
    return render_template("prediccion/index.html", config=config)


@prediccion_bp.route("/prediccion/ejecutar", methods=["POST"])
@login_required
def ejecutar():
    try:
        body = request.get_json(silent=True) or {}
        dias = body.get("dias")
        resultado = ejecutar_prediccion(current_user.id, dias=dias)
        return jsonify({"ok": True, **resultado})

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


@prediccion_bp.route("/prediccion/grafico-regresion")
@login_required
def grafico_regresion():
    ruta = _ruta_grafico_regresion(current_user.id)
    from services.storage_service import asegurar_local
    asegurar_local(os.path.basename(ruta), ruta)
    if not os.path.exists(ruta):
        return jsonify({"ok": False, "error": "Todavía no hay un gráfico disponible."}), 404
    return send_file(ruta, mimetype="image/png")


@prediccion_bp.route("/prediccion/reentrenar", methods=["POST"])
@login_required
def reentrenar():
    try:
        metricas = reentrenar_prediccion(current_user.id)
        return jsonify({"ok": True, "metricas": metricas})

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500