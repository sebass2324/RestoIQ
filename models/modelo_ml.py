from datetime import datetime
from models import db


class ModeloML(db.Model):
    """
    Una fila por usuario. Registra con qué hash de dataset se entrenó
    el modelo guardado en `ruta_pkl`. Si DatasetUsuario.hash != dataset_hash
    aquí guardado, el modelo está desactualizado y hay que reentrenar.
    """

    __tablename__ = "modelo_ml"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    dataset_hash = db.Column(db.String(64), nullable=False)

    estrategia = db.Column(db.String(50), nullable=True)

    mae = db.Column(db.Float, nullable=True)

    mape = db.Column(db.Float, nullable=True)

    ruta_pkl = db.Column(db.String(255), nullable=False)

    fecha_entrenamiento = db.Column(db.DateTime, default=datetime.utcnow)
