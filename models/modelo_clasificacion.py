from datetime import datetime
from models import db


class ModeloClasificacion(db.Model):
    """
    Una fila por usuario. Registro de caché del clasificador de prioridad
    de abastecimiento — mismo patrón que ModeloML, pero separado porque
    es un modelo independiente (RandomForest, no LGBM) con sus propias
    métricas. Si dataset_hash difiere del hash actual, hay que reentrenar.
    """

    __tablename__ = "modelo_clasificacion"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    dataset_hash = db.Column(db.String(64), nullable=False)

    f1_macro = db.Column(db.Float, nullable=True)
    accuracy = db.Column(db.Float, nullable=True)

    ruta_pkl = db.Column(db.String(255), nullable=False)

    fecha_entrenamiento = db.Column(db.DateTime, default=datetime.utcnow)
