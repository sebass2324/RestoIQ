from datetime import datetime
from models import db


class ConfiguracionAnalisis(db.Model):
    """
    Configuración de negocio de un usuario (una fila por usuario).

    A diferencia de un modelo de settings genérico, estos campos SÍ
    cambian el comportamiento real del pipeline de ML:
      - horizonte_dias      → cuántos días predice SalesModel.predecir()
      - dias_operacion      → qué días de la semana se incluyen en la predicción
      - considerar_*        → qué features opcionales entran al modelo
      - reentrenar_automatico → si el hash del dataset dispara reentreno solo,
                                 o si el usuario prefiere controlarlo a mano
      - objetivo_analisis   → NO cambia el modelo, solo qué vista se
                                 muestra por defecto en /prediccion
    """

    __tablename__ = "configuracion_analisis"

    id = db.Column(db.Integer, primary_key=True)

    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id", ondelete="CASCADE"),
        nullable=False,
        unique=True,
    )

    horizonte_dias = db.Column(db.Integer, nullable=False, default=7)

    # Guardado como "0,1,2,3,4,5" (ISO weekday: 0=Lunes ... 6=Domingo)
    dias_operacion = db.Column(db.String(20), nullable=False, default="0,1,2,3,4,5,6")

    considerar_promociones = db.Column(db.Boolean, default=False)
    considerar_descuentos  = db.Column(db.Boolean, default=False)
    considerar_eventos     = db.Column(db.Boolean, default=False)
    considerar_feriados    = db.Column(db.Boolean, default=True)

    reentrenar_automatico  = db.Column(db.Boolean, default=True)

    objetivo_analisis = db.Column(db.String(20), nullable=False, default="ingresos")
    # valores válidos: 'inventario' | 'produccion' | 'ingresos'

    fecha_actualizacion = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # ── Helpers ──────────────────────────────────────────────────────────

    def dias_operacion_set(self) -> set:
        """Retorna los días de operación como set de ints (weekday())."""
        if not self.dias_operacion:
            return {0, 1, 2, 3, 4, 5, 6}
        return {int(d) for d in self.dias_operacion.split(",") if d.strip() != ""}
