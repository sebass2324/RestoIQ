"""
models/evento_futuro.py

Dos mecanismos SEPARADOS, porque son conceptualmente distintos en los
propios datos de entrenamiento (ver services/data_generator.py):

- EventoFuturo (promoción): se decide de forma INDEPENDIENTE por
  producto y por día (12% de probabilidad cada uno) -- nunca "todo el
  local en oferta". Por eso se declara por producto específico, con
  "todos" (producto=None) como atajo cuando de verdad aplica a todo
  el menú.

- EventoEspecialFuturo (evento especial: partido, grupo grande, etc.):
  se decide UNA VEZ por día (5% de probabilidad), y afecta a TODOS
  los productos ese día por igual -- nunca a uno solo. Por eso NO
  tiene campo "producto": su sola existencia para una fecha ya
  significa "aplica a todo ese día".
"""

from models import db


class EventoFuturo(db.Model):
    __tablename__ = "eventos_futuros"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False, index=True)
    fecha = db.Column(db.Date, nullable=False, index=True)
    producto = db.Column(db.String(255), nullable=True)  # NULL = aplica a TODOS los productos ese día

    es_promocion  = db.Column(db.Boolean, default=False)
    descuento_pct = db.Column(db.Float, nullable=True)
    nota = db.Column(db.String(255), nullable=True)

    __table_args__ = (
        db.UniqueConstraint("user_id", "fecha", "producto", name="uq_evento_futuro_fecha_producto"),
    )


class EventoEspecialFuturo(db.Model):
    __tablename__ = "eventos_especiales_futuros"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False, index=True)
    fecha = db.Column(db.Date, nullable=False, index=True)
    nota = db.Column(db.String(255), nullable=True)  # ej. "partido Barcelona vs Emelec"

    __table_args__ = (
        db.UniqueConstraint("user_id", "fecha", name="uq_evento_especial_futuro_fecha"),
    )