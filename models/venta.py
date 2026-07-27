from models import db


class Venta(db.Model):
    """
    Cada fila representa una venta ya limpia (fecha, producto, cantidad)
    perteneciente a un usuario. Reemplaza por completo el enfoque anterior
    de leer un CSV suelto desde uploads_temp/: ahora la fuente de verdad
    para entrenar y predecir es SIEMPRE esta tabla, filtrada por user_id.
    """

    __tablename__ = "ventas"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id", ondelete="CASCADE"),
        nullable=False,
    )

    fecha = db.Column(db.Date, nullable=False)

    producto = db.Column(db.String(200), nullable=False)

    # Opcional: solo si el archivo del usuario la trae (DataCleaner ya la
    # detecta como columna opcional). La usa el modelo de clasificación
    # de prioridad de abastecimiento — sin ella, ese modelo no puede
    # generalizar a productos nuevos.
    categoria = db.Column(db.String(100), nullable=True)

    cantidad = db.Column(db.Float, nullable=False)

    precio = db.Column(db.Float, nullable=True)

    total = db.Column(db.Float, nullable=True)

    # ── Columnas de negocio OPCIONALES (nullable) ──
    # Se llenan solo si el archivo del usuario las trae. Su ausencia
    # nunca rompe el pipeline: SalesModel simplemente no las usa como
    # feature si están vacías o si el usuario no las activó en su
    # configuración (ver ConfiguracionAnalisis).
    promocion          = db.Column(db.Boolean, nullable=True)
    descuento_pct       = db.Column(db.Float, nullable=True)
    es_evento_especial = db.Column(db.Boolean, nullable=True)

    # ── Features temporales (ya calculadas por DataCleaner, se guardan
    #    tal cual para no recalcularlas en cada predicción) ──
    dia_semana = db.Column(db.Integer, nullable=False)
    mes = db.Column(db.Integer, nullable=False)
    semana_anio = db.Column(db.Integer, nullable=False)
    es_finde = db.Column(db.Boolean, default=False)
    es_feriado = db.Column(db.Boolean, default=False)
    es_puente = db.Column(db.Boolean, default=False)
    es_quincena = db.Column(db.Boolean, default=False)

    __table_args__ = (
        db.Index("ix_ventas_user_fecha_producto", "user_id", "fecha", "producto"),
    )