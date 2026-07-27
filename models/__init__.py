from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

# Importar los modelos AL FINAL (después de crear `db`) para que
# SQLAlchemy registre sus tablas en db.metadata. Sin estos imports,
# db.create_all() no crearía las tablas nuevas aunque los archivos existan.
from models.users import Usuario            # noqa: E402,F401
from models.venta import Venta              # noqa: E402,F401
from models.dataset_usuario import DatasetUsuario  # noqa: E402,F401
from models.modelo_ml import ModeloML       # noqa: E402,F401
from models.configuracion_analisis import ConfiguracionAnalisis  # noqa: E402,F401