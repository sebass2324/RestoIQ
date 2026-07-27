from flask import Blueprint, render_template, request, redirect, url_for, flash
from werkzeug.security import generate_password_hash, check_password_hash
from flask_login import login_user, logout_user, login_required
from models import db
from models.users import Usuario
from sqlalchemy.exc import IntegrityError

auth_bp = Blueprint("auth", __name__)


@auth_bp.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        nombre   = request.form.get("nombre", "").strip()
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not nombre or not email or not password:
            flash("Todos los campos son obligatorios.", "error")
            return render_template("auth/register.html")

        if len(password) < 8:
            flash("La contraseña debe tener al menos 8 caracteres.", "error")
            return render_template("auth/register.html")

        nuevo_usuario = Usuario(
            nombre=nombre,
            email=email,
            password_hash=generate_password_hash(password)
        )

        try:
            db.session.add(nuevo_usuario)
            db.session.commit()
            flash("Cuenta creada correctamente. Inicia sesión.", "success")
            return redirect(url_for("auth.login"))

        except IntegrityError:
            db.session.rollback()
            flash("Este correo ya está registrado.", "error")
            return render_template("auth/register.html")

    return render_template("auth/register.html")


@auth_bp.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email    = request.form.get("email", "").strip()
        password = request.form.get("password", "")

        if not email or not password:
            flash("Completa todos los campos.", "error")
            return render_template("auth/login.html")

        # Buscar usuario por email
        usuario = Usuario.query.filter_by(email=email).first()

        # Verificar que existe y que la contraseña es correcta
        if not usuario or not check_password_hash(usuario.password_hash, password):
            flash("Correo o contraseña incorrectos.", "error")
            return render_template("auth/login.html")

        # Iniciar sesión
        login_user(usuario)
        return redirect(url_for("dashboard.index"))

    return render_template("auth/login.html")


@auth_bp.route("/logout")
@login_required
def logout():
    logout_user()
    return redirect(url_for("auth.login"))