import os

class Config:
    # Seguridad
    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "restoiq_desarrollo_2026"
    )

    # Conexión MySQL
    SQLALCHEMY_DATABASE_URI = (
        "mysql+mysqlconnector://root:admin123@localhost/RestoIQ"
    )

    # Desactiva seguimiento interno innecesario
    SQLALCHEMY_TRACK_MODIFICATIONS = False