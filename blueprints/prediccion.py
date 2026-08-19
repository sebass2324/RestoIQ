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
    obtener_eventos_futuros, guardar_evento_futuro, eliminar_evento_futuro,
    eliminar_evento_especial_futuro,
)
from services.insumos_service import calcular_prevision_insumos, tiene_recetas_cargadas
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


# ════════════════════════════════════════════════════════════
# EVENTOS FUTUROS — el dueño declara promociones/eventos que ya
# sabe que van a pasar en una fecha específica del horizonte.
# ════════════════════════════════════════════════════════════

@prediccion_bp.route("/prediccion/eventos-futuros", methods=["GET"])
@login_required
def listar_eventos_futuros():
    return jsonify({"ok": True, "eventos": obtener_eventos_futuros(current_user.id)})


@prediccion_bp.route("/prediccion/eventos-futuros", methods=["POST"])
@login_required
def guardar_evento():
    try:
        body = request.get_json(silent=True) or {}
        fecha_str = body.get("fecha")
        if not fecha_str:
            return jsonify({"ok": False, "error": "Falta la fecha."}), 400

        from datetime import datetime as dt
        fecha = dt.strptime(fecha_str, "%Y-%m-%d").date()
        producto = body.get("producto") or None  # None/"" = aplica a todos los productos

        evento = guardar_evento_futuro(
            current_user.id, fecha, producto=producto,
            es_promocion=body.get("es_promocion", False),
            es_evento_especial=body.get("es_evento_especial", False),
            descuento_pct=body.get("descuento_pct"),
            nota=body.get("nota"),
        )
        return jsonify({"ok": True, "fecha": fecha_str, "producto": producto})

    except ValueError:
        return jsonify({"ok": False, "error": "Fecha inválida, usa formato YYYY-MM-DD."}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


@prediccion_bp.route("/prediccion/eventos-futuros/<fecha_str>", methods=["DELETE"])
@login_required
def borrar_evento(fecha_str):
    try:
        from datetime import datetime as dt
        fecha = dt.strptime(fecha_str, "%Y-%m-%d").date()
        producto = request.args.get("producto") or None  # ?producto=X en la URL, ausente = "todos"

        # El evento especial SIEMPRE vive bajo "todos" (nunca por
        # producto) -- se borra junto con la fila "todos" para que la
        # UI (que muestra promoción + evento especial combinados en
        # una sola fila cuando ambos aplican a "todos") pueda borrar
        # los dos con un solo clic.
        promocion_eliminada = eliminar_evento_futuro(current_user.id, fecha, producto=producto)
        evento_especial_eliminado = False
        if producto is None:
            evento_especial_eliminado = eliminar_evento_especial_futuro(current_user.id, fecha)

        return jsonify({
            "ok": True,
            "promocion_eliminada": promocion_eliminada,
            "evento_especial_eliminado": evento_especial_eliminado,
        })
    except ValueError:
        return jsonify({"ok": False, "error": "Fecha inválida, usa formato YYYY-MM-DD."}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500


# ════════════════════════════════════════════════════════════
# PREVISIÓN DE INSUMOS — convierte la demanda de platos ya predicha
# en necesidad de ingredientes, usando las recetas cargadas (ver
# services/insumos_service.py). SIEMPRE marcado como estimación --
# ver "es_estimado" en la respuesta, la UI debe mostrarlo así.
# ════════════════════════════════════════════════════════════

@prediccion_bp.route("/prediccion/insumos", methods=["POST"])
@login_required
def insumos():
    try:
        if not tiene_recetas_cargadas(current_user.id):
            return jsonify({"ok": False, "error": "Todavía no hay recetas cargadas para este negocio."}), 400

        body = request.get_json(silent=True) or {}
        dias = body.get("dias")
        resultado_prediccion = ejecutar_prediccion(current_user.id, dias=dias)
        prevision = calcular_prevision_insumos(current_user.id, resultado_prediccion["por_producto"])
        return jsonify({"ok": True, **prevision})

    except ValueError as e:
        return jsonify({"ok": False, "error": str(e)}), 400
    except Exception as e:
        db.session.rollback()
        return jsonify({"ok": False, "error": str(e)}), 500