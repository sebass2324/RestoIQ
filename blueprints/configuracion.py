from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_required, current_user
from models import db
from models.venta import Venta
from models.dataset_usuario import DatasetUsuario
from models.configuracion_analisis import ConfiguracionAnalisis

configuracion_bp = Blueprint("configuracion", __name__)

DIAS_SEMANA_LABELS = [
    ("0", "L"), ("1", "M"), ("2", "M"), ("3", "J"),
    ("4", "V"), ("5", "S"), ("6", "D"),
]


def _columnas_disponibles(user_id: int) -> dict:
    """
    Determina qué columnas opcionales de negocio están realmente
    disponibles en los datos del usuario, para deshabilitar los
    toggles que no tienen soporte en su dataset.
    """
    tiene_promocion = (
        Venta.query.filter(Venta.user_id == user_id, Venta.promocion.isnot(None)).first()
        is not None
    )
    tiene_descuento = (
        Venta.query.filter(Venta.user_id == user_id, Venta.descuento_pct.isnot(None)).first()
        is not None
    )
    tiene_evento = (
        Venta.query.filter(Venta.user_id == user_id, Venta.es_evento_especial.isnot(None)).first()
        is not None
    )
    return {
        "promocion": tiene_promocion,
        "descuento": tiene_descuento,
        "evento":    tiene_evento,
    }


@configuracion_bp.route("/configuracion", methods=["GET", "POST"])
@login_required
def index():
    if not DatasetUsuario.query.filter_by(user_id=current_user.id).first():
        flash("Primero sube un archivo de ventas.", "error")
        return redirect(url_for("upload.index"))

    config = ConfiguracionAnalisis.query.filter_by(user_id=current_user.id).first()
    es_primera_vez = config is None
    columnas = _columnas_disponibles(current_user.id)

    if request.method == "POST":
        if config is None:
            config = ConfiguracionAnalisis(user_id=current_user.id)
            db.session.add(config)

        config.horizonte_dias = int(request.form.get("horizonte_dias", 7))

        dias_marcados = request.form.getlist("dias_operacion")
        config.dias_operacion = ",".join(dias_marcados) if dias_marcados else "0,1,2,3,4,5,6"

        # Los checkboxes solo tienen efecto si la columna existe —
        # doble candado, igual que en SalesModel._construir_features.
        config.considerar_promociones = bool(request.form.get("considerar_promociones")) and columnas["promocion"]
        config.considerar_descuentos  = bool(request.form.get("considerar_descuentos"))  and columnas["descuento"]
        config.considerar_eventos     = bool(request.form.get("considerar_eventos"))     and columnas["evento"]
        config.considerar_feriados    = bool(request.form.get("considerar_feriados"))

        config.reentrenar_automatico  = bool(request.form.get("reentrenar_automatico"))
        config.objetivo_analisis      = request.form.get("objetivo_analisis", "ingresos")

        db.session.commit()

        flash("Configuración guardada correctamente.", "success")
        return redirect(url_for("prediccion.index"))

    return render_template(
        "configuracion/index.html",
        config=config,
        es_primera_vez=es_primera_vez,
        columnas=columnas,
        dias_semana=DIAS_SEMANA_LABELS,
    )
