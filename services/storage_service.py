"""
services/storage_service.py

Storage persistente en Supabase (API REST de Supabase Storage, sin
agregar el SDK completo — solo requests) para los artefactos de ML
(.pkl de modelos, .png de gráficos) que antes vivían solo en disco
local.

Por qué: en Render (y la mayoría de hosting gratuito/barato), el
disco del servidor es EFÍMERO — se borra en cada redeploy o reinicio.
El servidor se trata como desechable; el storage persistente es la
única fuente de verdad para estos archivos.

Uso recomendado (lo que llaman los modelos al cargar):
    asegurar_local(nombre_remoto, ruta_local)
        → si ya está en disco local, no hace nada (evita una llamada
          de red innecesaria en el caso normal — mismo proceso, sin
          reinicio de por medio).
        → si no está, lo descarga de Supabase antes de devolver el
          control (arranque en frío tras un redeploy).

Diseño importante: si las variables de entorno de Supabase NO están
configuradas (ej. corriendo en tu máquina local), todas las funciones
no hacen nada y devuelven False/None — el sistema sigue funcionando
exactamente como antes, solo con disco local. El mismo código sirve
para desarrollo local y producción sin cambiar una línea.

Variables de entorno necesarias (configurar en Render, NUNCA
hardcodear ni commitear):
    SUPABASE_URL           ej. https://xxxx.supabase.co
    SUPABASE_SERVICE_KEY   la "service_role key" (NO la anon key —
                            necesita permiso de escritura)
    SUPABASE_BUCKET        nombre del bucket, ej. "modelos-ml"
"""

import os
import time
import logging
import requests

logger = logging.getLogger(__name__)

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
BUCKET       = os.environ.get("SUPABASE_BUCKET", "modelos-ml")

HABILITADO = bool(SUPABASE_URL and SUPABASE_KEY)

TIMEOUT = 30
CODIGOS_REINTENTABLES = {502, 503, 504}  # errores transitorios del lado de Supabase, no del archivo en sí


def _url(nombre_remoto: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{nombre_remoto}"


def _con_reintento(fn, intentos=2):
    """Corre fn() hasta 'intentos' veces si responde con un código
    transitorio (502/503/504). No reintenta ante 404 (no existe) ni
    401/403 (credenciales mal) — esos no se arreglan solos."""
    ultima_resp = None
    for intento in range(intentos):
        resp = fn()
        ultima_resp = resp
        if resp is None or resp.status_code not in CODIGOS_REINTENTABLES:
            return resp
        logger.warning(f"[storage_service] Respuesta {resp.status_code} de Supabase, "
                       f"reintentando ({intento + 1}/{intentos})...")
        time.sleep(1)
    return ultima_resp


def subir(ruta_local: str, nombre_remoto: str) -> bool:
    """Sube un archivo local a Supabase Storage (sobreescribe si ya existe)."""
    if not HABILITADO:
        return False
    try:
        with open(ruta_local, "rb") as f:
            contenido = f.read()
        resp = _con_reintento(lambda: requests.post(
            _url(nombre_remoto),
            headers={"Authorization": f"Bearer {SUPABASE_KEY}", "x-upsert": "true"},
            data=contenido,
            timeout=TIMEOUT,
        ))
        if resp is None or resp.status_code not in (200, 201):
            logger.warning(f"[storage_service] Falló la subida de {nombre_remoto}: "
                           f"{getattr(resp, 'status_code', 'sin respuesta')}")
            return False
        return True
    except Exception:
        logger.exception(f"[storage_service] Error subiendo {nombre_remoto}")
        return False


def descargar(nombre_remoto: str, ruta_local: str) -> bool:
    """Descarga un archivo de Supabase Storage a disco local, SIEMPRE
    (sin chequear si ya existe local — para eso está asegurar_local).
    Retorna True si se descargó, False si no existe o Supabase no
    está configurado."""
    if not HABILITADO:
        return False
    try:
        resp = _con_reintento(lambda: requests.get(
            _url(nombre_remoto),
            headers={"Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=TIMEOUT,
        ))
        if resp is None or resp.status_code != 200:
            return False
        carpeta = os.path.dirname(ruta_local)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        with open(ruta_local, "wb") as f:
            f.write(resp.content)
        return True
    except Exception:
        logger.exception(f"[storage_service] Error descargando {nombre_remoto}")
        return False


def asegurar_local(nombre_remoto: str, ruta_local: str) -> bool:
    """Lo que deberían llamar los modelos al cargar: garantiza que
    ruta_local exista en disco si es posible, evitando una descarga
    innecesaria cuando el archivo ya está ahí (caso normal, mismo
    proceso sin reinicio de por medio).

    Retorna True si el archivo está disponible en disco al terminar
    (ya estaba, o se descargó ahora), False si no se pudo conseguir
    de ninguna forma — el llamador debe interpretar eso como 'no
    existe, hay que regenerarlo'.
    """
    if os.path.exists(ruta_local):
        return True
    return descargar(nombre_remoto, ruta_local)