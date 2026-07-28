"""
services/flask_storage.py — Le dice al cliente de Supabase Auth cómo
guardar el estado del flujo OAuth (PKCE) entre requests.

Sin esto, el "code_verifier" que Supabase genera en /google se pierde
antes de llegar a /callback (porque cada request crea un cliente
nuevo sin memoria compartida) — y por eso salía el error
"El link de inicio de sesión expiró o no es válido".

Usa flask.session (cookie firmada del navegador) en vez de memoria del
proceso: persiste entre requests del MISMO navegador, y cada usuario
tiene su propia cookie — no hay forma de que se mezcle con la de otro.
"""

from gotrue import SyncSupportedStorage
from flask import session


class FlaskSessionStorage(SyncSupportedStorage):
    def __init__(self):
        self.storage = session

    def get_item(self, key: str):
        return self.storage.get(key)

    def set_item(self, key: str, value: str) -> None:
        self.storage[key] = value

    def remove_item(self, key: str) -> None:
        self.storage.pop(key, None)
