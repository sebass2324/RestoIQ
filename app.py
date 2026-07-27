from flask import Flask
from config import Config
from services.login_manager import login_manager
from models import db

# Blueprints activos
from blueprints.auth import auth_bp
from blueprints.dashboard import dashboard_bp
from blueprints.upload import upload_bp
from blueprints.prediccion import prediccion_bp
from blueprints.abastecimiento import abastecimiento_bp
from blueprints.configuracion import configuracion_bp




def create_app():
    app = Flask(__name__)

    # Cargar configuración
    app.config.from_object(Config)

    # Inicializar SQLAlchemy
    db.init_app(app)

    # Inicializar Login Manager
    login_manager.init_app(app)
    login_manager.login_view = "auth.login"

    # Registrar blueprints activos
    app.register_blueprint(auth_bp)
    app.register_blueprint(dashboard_bp)
    app.register_blueprint(upload_bp)
    app.register_blueprint(prediccion_bp)
    app.register_blueprint(abastecimiento_bp)
    app.register_blueprint(configuracion_bp)

    # TOarDO: registr cuando estén listos
    # app.register_blueprint(products_bp)
    # app.register_blueprint(sales_bp)

    return app


app = create_app()

if __name__ == "__main__":
    app.run(debug=True)