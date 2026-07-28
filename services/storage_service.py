"""
services/storage_service.py

Storage persistente en Supabase (API REST de Supabase Storage, sin
agregar el SDK completo — solo requests, que ya es una dependencia
liviana) para los artefactos de ML (.pkl de modelos, .png de
gráficos) que antes vivían solo en disco local.

Por qué: en Render (y la mayoría de hosting gratuito/barato), el
disco del servidor es EFÍMERO — se borra en cada redeploy o reinicio.
El servidor se trata como desechable; el storage persistente es la
única fuente de verdad para estos archivos.

Patrón de uso (en SalesModel, ModeloAbastecimiento, etc.):
    guardar():  escribe en disco local (rápido) Y sube a Supabase.
    cargar():   si el archivo YA está en disco local (mismo proceso,
                sin reinicio de por medio), lo usa directo. Si no
                está (arranque en frío tras un redeploy), lo
                descarga de Supabase antes de leerlo.

Diseño importante: si las variables de entorno de Supabase NO están
configuradas (ej. corriendo en tu máquina local), subir()/descargar()
no hacen nada y devuelven False — el sistema sigue funcionando
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
import requests

SUPABASE_URL = os.environ.get("SUPABASE_URL", "").rstrip("/")
SUPABASE_KEY = os.environ.get("SUPABASE_SERVICE_KEY", "")
BUCKET       = os.environ.get("SUPABASE_BUCKET", "modelos-ml")

HABILITADO = bool(SUPABASE_URL and SUPABASE_KEY)


def _url(nombre_remoto: str) -> str:
    return f"{SUPABASE_URL}/storage/v1/object/{BUCKET}/{nombre_remoto}"


def subir(ruta_local: str, nombre_remoto: str) -> bool:
    """Sube un archivo local a Supabase Storage (sobreescribe si ya
    existe). Si Supabase no está configurado, no hace nada."""
    if not HABILITADO:
        return False
    try:
        with open(ruta_local, "rb") as f:
            contenido = f.read()
        resp = requests.post(
            _url(nombre_remoto),
            headers={"Authorization": f"Bearer {SUPABASE_KEY}", "x-upsert": "true"},
            data=contenido,
            timeout=30,
        )
        if resp.status_code not in (200, 201):
            print(f"[storage_service] Falló la subida de {nombre_remoto}: {resp.status_code} {resp.text}")
            return False
        return True
    except Exception as e:
        print(f"[storage_service] Error subiendo {nombre_remoto}: {e}")
        return False


def descargar(nombre_remoto: str, ruta_local: str) -> bool:
    """Descarga un archivo de Supabase Storage a disco local. Retorna
    True si se descargó, False si no existe o Supabase no está
    configurado — en ese caso el llamador debe asumir 'no existe' y
    reentrenar/regenerar el archivo desde cero."""
    if not HABILITADO:
        return False
    try:
        resp = requests.get(
            _url(nombre_remoto),
            headers={"Authorization": f"Bearer {SUPABASE_KEY}"},
            timeout=30,
        )
        if resp.status_code != 200:
            return False
        carpeta = os.path.dirname(ruta_local)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        with open(ruta_local, "wb") as f:
            f.write(resp.content)
        return True
    except Exception as e:
        print(f"[storage_service] Error descargando {nombre_remoto}: {e}")
        return False
