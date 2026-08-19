"""
models/receta_insumo.py

Bill of Materials: cuánto de cada ingrediente lleva una unidad de cada
producto. Permite convertir la predicción de demanda de PLATOS
(cantidad_pred por producto) en previsión de INSUMOS (cuánta carne,
pan, queso, etc. hace falta comprar).

IMPORTANTE -- honestidad de los datos: las cantidades por defecto que
carga el seed (ver scripts/cargar_recetas.py) son ESTIMACIONES basadas
en porciones estándar de comida rápida/venezolana-ecuatoriana, no
mediciones reales de El Chamo Burger (documentado explícitamente en el
archivo de origen, hoja "Notas_y_Supuestos"). La UI debe mostrar esto
como estimación calibrable, nunca como dato medido -- ver
services/insumos_service.py.
"""

from models import db


class RecetaInsumo(db.Model):
    __tablename__ = "recetas_insumos"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), nullable=False, index=True)
    producto = db.Column(db.String(255), nullable=False, index=True)
    ingrediente = db.Column(db.String(255), nullable=False)
    cantidad_por_unidad = db.Column(db.Float, nullable=False)
    unidad = db.Column(db.String(20), nullable=False)  # "g", "unidad", "ml", etc.

    __table_args__ = (
        db.UniqueConstraint("user_id", "producto", "ingrediente", name="uq_receta_producto_ingrediente"),
    )
