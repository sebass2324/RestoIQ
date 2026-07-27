import os
from datetime import datetime
import pandas as pd
from flask import Blueprint, render_template, request, redirect, url_for, flash, session
from flask_login import login_required, current_user
from werkzeug.utils import secure_filename
from services.data_cleaner import DataCleaner
from services.dataset_hash import calcular_hash
from services.dataset_merge import cargar_historial_df, validar_lote, fusionar
from services.dashboard_service import obtener_kpis_usuario
from models import db
from models.venta import Venta
from models.dataset_usuario import DatasetUsuario
from models.configuracion_analisis import ConfiguracionAnalisis

upload_bp = Blueprint("upload", __name__)

UPLOAD_FOLDER = "uploads_temp"
EXTENSIONES_PERMITIDAS = {"csv", "xlsx", "xls"}


def extension_permitida(nombre):
    return "." in nombre and nombre.rsplit(".", 1)[1].lower() in EXTENSIONES_PERMITIDAS


def _guardar_ventas_usuario(df: pd.DataFrame, user_id: int, nombre_archivo: str) -> str:
    """
    Persiste el HISTORIAL COMPLETO ya fusionado (historial previo + lote
    nuevo, deduplicado) como fuente de verdad. Se sigue implementando
    como "borrar todo y reinsertar" por simplicidad transaccional —
    la diferencia con el diseño anterior es que `df` acá ya es el
    acumulado, no solo el archivo recién subido.
    """
    dataset_hash = calcular_hash(df)
    registros = df.to_dict(orient="records")

    try:
        Venta.query.filter_by(user_id=user_id).delete(synchronize_session=False)

        nuevas_ventas = []
        for r in registros:
            fecha = r["fecha"]
            fecha = fecha.date() if hasattr(fecha, "date") else fecha
            precio = r.get("precio")
            total = r.get("total")

            nuevas_ventas.append(Venta(
                user_id=user_id,
                fecha=fecha,
                producto=str(r["producto"]),
                categoria=str(r["categoria"]) if pd.notna(r.get("categoria")) else None,
                cantidad=float(r["cantidad"]),
                precio=float(precio) if pd.notna(precio) else None,
                total=float(total) if pd.notna(total) else None,
                dia_semana=int(r["dia_semana"]),
                mes=int(r["mes"]),
                semana_anio=int(r["semana_anio"]),
                es_finde=bool(r["es_finde"]),
                es_feriado=bool(r["es_feriado"]),
                es_puente=bool(r["es_puente"]),
                es_quincena=bool(r["es_quincena"]),
                promocion=bool(r["promocion"]) if pd.notna(r.get("promocion")) else None,
                descuento_pct=float(r["descuento_pct"]) if pd.notna(r.get("descuento_pct")) else None,
                es_evento_especial=bool(r["es_evento_especial"]) if pd.notna(r.get("es_evento_especial")) else None,
            ))

        db.session.bulk_save_objects(nuevas_ventas)

        dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()
        if dataset is None:
            dataset = DatasetUsuario(user_id=user_id)
            db.session.add(dataset)

        dataset.hash = dataset_hash
        dataset.filas = len(df)
        dataset.nombre_archivo = nombre_archivo   # último archivo agregado
        dataset.fecha_subida = datetime.utcnow()

        db.session.commit()

    except Exception:
        db.session.rollback()
        raise

    return dataset_hash


def _renderizar_reporte(reporte, df_lote, df_final, n_duplicados):
    """
    Arma la respuesta común a los 3 caminos de éxito (fusión automática,
    fusión confirmada, reemplazo confirmado) — evita repetir esta lógica
    tres veces.
    """
    reporte["productos_detectados"] = int(df_lote["producto"].nunique())
    reporte["periodo_desde"] = df_lote["fecha"].min().strftime("%d/%m/%Y")
    reporte["periodo_hasta"] = df_lote["fecha"].max().strftime("%d/%m/%Y")
    reporte["duplicados_con_historial"] = n_duplicados
    reporte["filas_historial_total"] = len(df_final)

    preview = df_lote.head(10).to_dict(orient="records")
    kpis = obtener_kpis_usuario(current_user.id)
    tiene_configuracion = ConfiguracionAnalisis.query.filter_by(user_id=current_user.id).first() is not None

    # Ver comentario extenso en la versión anterior de este archivo sobre
    # por qué se renderiza directo en vez de session + redirect (límite
    # de 4KB de la cookie de sesión con reportes grandes).
    return render_template(
        "upload/reporte.html",
        reporte=reporte,
        preview=preview,
        kpis=kpis,
        tiene_configuracion=tiene_configuracion,
    )


@upload_bp.route("/upload", methods=["GET", "POST"])
@login_required
def index():
    if request.method == "POST":
        if "archivo" not in request.files or request.files["archivo"].filename == "":
            flash("No se seleccionó ningún archivo.", "error")
            return redirect(url_for("upload.index"))

        archivo = request.files["archivo"]
        if not extension_permitida(archivo.filename):
            flash("Formato no permitido. Usa CSV o Excel (.xlsx).", "error")
            return redirect(url_for("upload.index"))

        nombre_seguro = secure_filename(archivo.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        ruta = os.path.join(UPLOAD_FOLDER, nombre_seguro)
        archivo.save(ruta)

        cleaner = DataCleaner()
        df_nuevo, reporte = cleaner.procesar(ruta, nombre_seguro)

        if df_nuevo is None:
            try:
                os.remove(ruta)
            except Exception:
                pass
            flash(reporte["errores"][0], "error")
            return redirect(url_for("upload.index"))

        df_historial = cargar_historial_df(current_user.id)
        validacion = validar_lote(df_nuevo, df_historial)

        if not validacion["ok"]:
            # NO se borra el archivo temporal: se necesita para la
            # confirmación. Solo guardamos la ruta (string chico) en
            # session — no el DataFrame completo.
            session["upload_pendiente"] = {"ruta": ruta, "nombre": nombre_seguro}
            return render_template("upload/confirmar_negocio.html", validacion=validacion)

        try:
            os.remove(ruta)
        except Exception as e:
            print(f"Error eliminando archivo temporal: {e}")

        df_final, n_duplicados = fusionar(df_historial, df_nuevo)

        try:
            _guardar_ventas_usuario(df_final, current_user.id, nombre_seguro)
        except Exception as e:
            flash(f"El archivo se limpió pero no se pudo guardar en la base de datos: {e}", "error")
            return redirect(url_for("upload.index"))

        flash("Archivo procesado y agregado a tu historial correctamente.", "success")
        return _renderizar_reporte(reporte, df_nuevo, df_final, n_duplicados)

    return render_template("upload/index.html")


@upload_bp.route("/upload/confirmar", methods=["POST"])
@login_required
def confirmar():
    """
    Resuelve la advertencia de "esto parece otro negocio" mostrada por
    /upload. accion viene del formulario de confirmar_negocio.html:
      - "fusionar":   integrar igual al historial existente
      - "reemplazar": empezar un historial nuevo (borra el anterior)
      - "cancelar":   descartar el archivo, no tocar nada
    """
    accion = request.form.get("accion")
    pendiente = session.pop("upload_pendiente", None)

    if not pendiente:
        flash("No hay una subida pendiente de confirmar.", "error")
        return redirect(url_for("upload.index"))

    ruta, nombre = pendiente["ruta"], pendiente["nombre"]

    if accion == "cancelar":
        try:
            os.remove(ruta)
        except Exception:
            pass
        flash("Subida cancelada.", "info")
        return redirect(url_for("upload.index"))

    cleaner = DataCleaner()
    df_nuevo, reporte = cleaner.procesar(ruta, nombre)
    try:
        os.remove(ruta)
    except Exception:
        pass

    if df_nuevo is None:
        flash(reporte["errores"][0], "error")
        return redirect(url_for("upload.index"))

    df_historial = cargar_historial_df(current_user.id)

    if accion == "reemplazar":
        df_final, n_duplicados = df_nuevo, 0
    else:  # "fusionar" a pesar de la advertencia
        df_final, n_duplicados = fusionar(df_historial, df_nuevo)

    try:
        _guardar_ventas_usuario(df_final, current_user.id, nombre)
    except Exception as e:
        flash(f"El archivo se limpió pero no se pudo guardar en la base de datos: {e}", "error")
        return redirect(url_for("upload.index"))

    flash("Archivo procesado correctamente.", "success")
    return _renderizar_reporte(reporte, df_nuevo, df_final, n_duplicados)


@upload_bp.route("/upload/reporte")
@login_required
def reporte():
    flash("Sube un archivo para ver su resumen.", "info")
    return redirect(url_for("upload.index"))