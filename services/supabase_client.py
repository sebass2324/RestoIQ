"""
services/supabase_client.py — Cliente de Supabase Auth.

Se cachea en flask.g (dura solo el request actual — flask.g se
resetea automáticamente en cada request nuevo, así que esto NO es un
cliente global compartido entre usuarios). El estado del flujo OAuth
(PKCE) se guarda en la sesión de Flask vía FlaskSessionStorage — ver
ese archivo para el porqué.
"""

from flask import g, current_app
from supabase.client import Client, ClientOptions
from services.flask_storage import FlaskSessionStorage


def get_supabase() -> Client:
    if "supabase" not in g:
        g.supabase = Client(
            current_app.config["SUPABASE_URL"],
            current_app.config["SUPABASE_KEY"],
            options=ClientOptions(
                storage=FlaskSessionStorage(),
                flow_type="pkce",
            ),
        )
    return g.supabase