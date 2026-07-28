"""
blueprints/abastecimiento.py

Wrapper HTTP fino sobre services/classification_service.py — toda la
lógica de negocio vive en el service (reutilizable, sin Flask).
"""

import os
from flask import Blueprint, render_template, request, jsonify, redirect, url_for, flash, send_file
from flask_login import login_required, current_user
from services.classification_service import (
    ejecutar_clasificacion, reentrenar_clasificacion, obtener_config, _ruta_matriz,
)
from models import db

abastecimiento_bp = Blueprint("abastecimiento", __name__)


@abastecimiento_bp.route("/abastecimiento")
@login_required
def index():
    config = obtener_config(current_user.id)
    if config is None:
        flash("Antes de continuar, cuéntanos cómo opera tu negocio.", "info")
        return redirect(url_for("configuracion.index"))
    return render_template("abastecimiento/index.html", config=config)


@abastecimiento_bp.route("/abastecimiento/ejecutar", methods=["POST"])
@login_required
def ejecutar():
    try:
        body = request.get_json(silent=True) or {}
        dias = body.get("dias")
        resultado = ejecutar_clasificacion(current_user.id, dias=dias)
        return jsonify({"ok": True, **resultado})

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


@abastecimiento_bp.route("/abastecimiento/matriz-confusion")
@login_required
def matriz_confusion_imagen():
    ruta = _ruta_matriz(current_user.id)
    if not os.path.exists(ruta):
        return jsonify({"ok": False, "error": "Todavía no hay un modelo entrenado."}), 404
    return send_file(ruta, mimetype="image/png")


@abastecimiento_bp.route("/abastecimiento/reentrenar", methods=["POST"])
@login_required
def reentrenar():
    try:
        resultado = reentrenar_clasificacion(current_user.id)
        return jsonify({"ok": True, **resultado})

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500