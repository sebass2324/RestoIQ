from models import db
from flask_login import UserMixin


class Usuario(UserMixin, db.Model):

    __tablename__ = "usuarios"

    id = db.Column(db.Integer, primary_key=True)

    nombre = db.Column(
        db.String(100),
        nullable=False
    )

    email = db.Column(
        db.String(120),
        unique=True,
        nullable=False
    )

    # ID del usuario en Supabase Auth (UUID). Ahí vive la contraseña
    # real, el estado de verificación de email, etc. — esta tabla solo
    # guarda el perfil local para las relaciones con Venta, ModeloML, etc.
    supabase_id = db.Column(
        db.String(64),
        unique=True,
        nullable=False
    )

    activo = db.Column(
        db.Boolean,
        default=True
    )