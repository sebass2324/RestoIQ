from datetime import datetime
from models import db


class DatasetUsuario(db.Model):
    """
    Una fila por usuario. Guarda el hash del último dataset limpio subido,
    para poder comparar contra el hash con el que se entrenó el modelo
    (ver ModeloML) y decidir si hace falta reentrenar.
    """

    __tablename__ = "dataset_usuario"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    hash = db.Column(db.String(64), nullable=False)

    filas = db.Column(db.Integer, nullable=False, default=0)

    nombre_archivo = db.Column(db.String(255), nullable=True)

    fecha_subida = db.Column(db.DateTime, default=datetime.utcnow)
