import os

class Config:
    # Seguridad
    SECRET_KEY = os.getenv(
        "SECRET_KEY",
        "restoiq_desarrollo_2026"
    )

    # Conexión MySQL — en producción (Render) se lee de la variable de
    # entorno DATABASE_URL; en desarrollo local, si esa variable no
    # existe, cae al MySQL local de siempre. El mismo código sirve
    # para los dos entornos sin cambiar una línea.
    SQLALCHEMY_DATABASE_URI = os.getenv(
        "DATABASE_URL",
        "mysql+mysqlconnector://root:admin123@localhost/RestoIQ"
    )

    # Desactiva seguimiento interno innecesario
    SQLALCHEMY_TRACK_MODIFICATIONS = False

    # Supabase Auth (si ya la tenías configurada) — distinto de
    # Supabase STORAGE, que usa services/storage_service.py y lee sus
    # propias variables (SUPABASE_URL / SUPABASE_SERVICE_KEY) por
    # separado, con la service_role key, no esta.
    SUPABASE_URL = os.getenv("SUPABASE_URL")
    SUPABASE_KEY = os.getenv("SUPABASE_KEY")