"""
Sincroniza el esquema de MySQL con los modelos de SQLAlchemy.

A diferencia de la versión anterior (que solo hacía db.create_all()),
esta versión también detecta columnas NUEVAS en tablas que YA EXISTEN
y las agrega con ALTER TABLE. Esto evita el error clásico:

    Unknown column 'ventas.promocion' in 'field list'

...que pasa porque db.create_all() NUNCA modifica una tabla existente,
solo crea las que faltan por completo.

No reemplaza a una herramienta de migraciones real (Flask-Migrate/Alembic)
— no versiona los cambios ni soporta rollback — pero cubre el caso de
uso de este proyecto: agregar columnas nuevas sin perder datos.

Uso:
    python crear_tablas.py
"""

from sqlalchemy import inspect, text
from app import create_app
from models import db

# Columnas que deben existir en `ventas` más allá de las originales.
# Formato: nombre -> tipo SQL para ALTER TABLE.
COLUMNAS_VENTAS_NUEVAS = {
    "promocion":          "BOOLEAN NULL",
    "descuento_pct":       "FLOAT NULL",
    "es_evento_especial": "BOOLEAN NULL",
    "categoria":          "VARCHAR(100) NULL",
}


def sincronizar_esquema():
    app = create_app()
    with app.app_context():
        inspector = inspect(db.engine)

        # 1) Agregar columnas faltantes a tablas que YA existen
        if inspector.has_table("ventas"):
            columnas_actuales = {c["name"] for c in inspector.get_columns("ventas")}
            for nombre, tipo in COLUMNAS_VENTAS_NUEVAS.items():
                if nombre not in columnas_actuales:
                    print(f"  → Agregando columna ventas.{nombre} ...")
                    db.session.execute(text(f"ALTER TABLE ventas ADD COLUMN {nombre} {tipo}"))
            db.session.commit()

        # 2) Crear tablas que no existan en absoluto
        db.create_all()

        print(" Esquema sincronizado: usuarios, ventas, dataset_usuario, "
              "modelo_ml, modelo_clasificacion, configuracion_analisis")


if __name__ == "__main__":
    sincronizar_esquema()