from flask import Blueprint, render_template, request, redirect, url_for, flash
from flask_login import login_user, logout_user, login_required
from models import db
from models.users import Usuario
from services.supabase_client import get_supabase

auth_bp = Blueprint("auth", __name__)


def _obtener_o_crear_usuario_local(supabase_user, nombre_fallback=None):
    """Busca el perfil local por supabase_id (o por email, por si ya
    existía de antes) y lo crea si es la primera vez que este usuario
    de Supabase inicia sesión aquí."""
    usuario = Usuario.query.filter_by(supabase_id=supabase_user.id).first()
    if usuario:
        return usuario

    usuario = Usuario.query.filter_by(email=supabase_user.email).first()
    if usuario:
        usuario.supabase_id = supabase_user.id
        db.session.commit()
        return usuario

    nombre = (
        nombre_fallback
        or (supabase_user.user_metadata or {}).get("nombre")
        or supabase_user.email.split("@")[0]
    )
    usuario = Usuario(nombre=nombre, email=supabase_user.email, supabase_id=supabase_user.id)
    db.session.add(usuario)
    db.session.commit()
    return usuario


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        nombre    = request.form.get("nombre", "").strip()
        email     = request.form.get("email", "").strip()
        password  = request.form.get("password", "")
        confirmar = request.form.get("confirmar", "")
        terminos  = request.form.get("terminos")

        if not nombre or not email or not password or not confirmar:
            flash("Todos los campos son obligatorios.", "error")
            return render_template("auth/register.html")

        if not terminos:
            flash("Debes aceptar los términos y condiciones para continuar.", "error")
            return render_template("auth/register.html")

        if password != confirmar:
            flash("Las contraseñas no coinciden.", "error")
            return render_template("auth/register.html")

        if len(password) < 8:
            flash("La contraseña debe tener al menos 8 caracteres.", "error")
            return render_template("auth/register.html")

        supabase = get_supabase()
        try:
            resultado = supabase.auth.sign_up({
                "email": email,
                "password": password,
                "options": {"data": {"nombre": nombre}},
            })
        except Exception as e:
            # Supabase devuelve un error tipo "User already registered"
            flash("No se pudo crear la cuenta. El correo puede que ya esté registrado.", "error")
            return render_template("auth/register.html")

        if resultado.user is None:
            flash("No se pudo crear la cuenta. Intenta de nuevo.", "error")
            return render_template("auth/register.html")

        _obtener_o_crear_usuario_local(resultado.user, nombre_fallback=nombre)

        if resultado.session is None:
            # Supabase tiene activada la confirmación por correo — el
            # usuario debe hacer clic en el link que le llega antes de
            # poder iniciar sesión.
            flash("Cuenta creada. Revisa tu correo para confirmarla antes de iniciar sesión.", "success")
            return redirect(url_for("auth.login"))

        # Confirmación de correo desactivada — puede entrar directo.
        flash("Cuenta creada correctamente. Inicia sesión.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Completa todos los campos.", "error")
            return render_template("auth/login.html")

        supabase = get_supabase()
        try:
            resultado = supabase.auth.sign_in_with_password({"email": email, "password": password})
        except Exception:
            flash("Correo o contraseña incorrectos.", "error")
            return render_template("auth/login.html")

        if resultado.user is None:
            flash("Correo o contraseña incorrectos.", "error")
            return render_template("auth/login.html")

        usuario = _obtener_o_crear_usuario_local(resultado.user)
        login_user(usuario)
        return redirect(url_for("dashboard.index"))

    return render_template("auth/login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))


# ══ Login con Google (vía Supabase) ══

@auth_bp.route("/google")
def google_login():
    supabase = get_supabase()
    resultado = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {"redirect_to": url_for("auth.callback", _external=True)},
    })
    return redirect(resultado.url)


@auth_bp.route("/callback")
def callback():
    """Vuelta desde Supabase después del login con Google (flujo PKCE:
    llega un ?code=... que hay que intercambiar por una sesión)."""
    code = request.args.get("code")
    if not code:
        flash("No se pudo completar el inicio de sesión con Google.", "error")
        return redirect(url_for("auth.login"))

    supabase = get_supabase()
    try:
        resultado = supabase.auth.exchange_code_for_session({"auth_code": code})
    except Exception:
        flash("El link de inicio de sesión expiró o no es válido. Intenta de nuevo.", "error")
        return redirect(url_for("auth.login"))

    usuario = _obtener_o_crear_usuario_local(resultado.user)
    login_user(usuario)
    return redirect(url_for("dashboard.index"))


# ══ Olvidé mi contraseña (Supabase envía el correo) ══

@auth_bp.route("/forgot-password", methods=["GET", "POST"])
def forgot_password():
    if request.method == "POST":
        email = request.form.get("email", "").strip()
        supabase = get_supabase()
        try:
            supabase.auth.reset_password_for_email(email, {
                "redirect_to": url_for("auth.reset_password", _external=True),
            })
        except Exception:
            pass  # no revelamos si el correo existe o no

        flash("Si el correo está registrado, te enviamos un link para restablecer tu contraseña.", "success")
        return redirect(url_for("auth.login"))

    return render_template("auth/forgot_password.html")


@auth_bp.route("/reset-password", methods=["GET", "POST"])
def reset_password():
    code = request.args.get("code") or request.form.get("code")

    if request.method == "POST":
        password = request.form.get("password", "")
        if len(password) < 8:
            flash("La contraseña debe tener al menos 8 caracteres.", "error")
            return render_template("auth/reset_password.html", code=code)

        supabase = get_supabase()
        try:
            sesion = supabase.auth.exchange_code_for_session({"auth_code": code})
            supabase.auth.set_session(sesion.session.access_token, sesion.session.refresh_token)
            supabase.auth.update_user({"password": password})
        except Exception:
            flash("El link expiró o no es válido. Solicita uno nuevo.", "error")
            return redirect(url_for("auth.forgot_password"))

        flash("Contraseña actualizada. Ya puedes iniciar sesión.", "success")
        return redirect(url_for("auth.login"))

    if not code:
        flash("El link no es válido. Solicita uno nuevo.", "error")
        return redirect(url_for("auth.forgot_password"))

    return render_template("auth/reset_password.html", code=code)