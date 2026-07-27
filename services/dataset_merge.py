"""
services/dataset_merge.py

Fusión del historial acumulado de un usuario con un lote nuevo subido.

Reemplaza el comportamiento anterior (cada upload BORRABA el historial
previo) por el estándar de cualquier ERP: cada archivo es un LOTE que
se integra al historial existente, nunca lo sustituye.

Flujo:
    cargar_historial_df(user_id)      → historial actual completo
    validar_lote(nuevo, historial)    → ¿es seguro fusionar automático?
    fusionar(historial, nuevo)        → historial + lote, sin duplicados
"""

import pandas as pd
from models.venta import Venta
from services.dataset_hash import COLUMNAS_HASH

# % de productos del lote nuevo NUNCA vistos en el historial a partir
# del cual se sospecha que el archivo pertenece a otro negocio, y se
# pide confirmación explícita en vez de fusionar en silencio.
UMBRAL_OTRO_NEGOCIO = 0.85

COLUMNAS_VENTA = [
    "fecha", "producto", "categoria", "cantidad", "precio", "total",
    "dia_semana", "mes", "semana_anio", "es_finde", "es_feriado",
    "es_puente", "es_quincena", "promocion", "descuento_pct", "es_evento_especial",
]


def cargar_historial_df(user_id: int) -> pd.DataFrame:
    """Historial COMPLETO actual del usuario, tal como está en MySQL."""
    ventas = Venta.query.filter_by(user_id=user_id).all()
    if not ventas:
        return pd.DataFrame(columns=COLUMNAS_VENTA)
    data = [{col: getattr(v, col) for col in COLUMNAS_VENTA} for v in ventas]
    df = pd.DataFrame(data)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def validar_lote(df_nuevo: pd.DataFrame, df_historial: pd.DataFrame) -> dict:
    """
    Compara el lote nuevo contra el historial ANTES de fusionar.

    Retorna {"ok": True} si es seguro fusionar automáticamente, o
    {"ok": False, "motivo": "otro_negocio", ...} si la mayoría de los
    productos del lote nunca aparecieron en el historial — señal de que
    el archivo pertenece a otro negocio, no a un período nuevo del mismo.

    Con historial vacío (primera subida del usuario) siempre es seguro:
    no hay contra qué comparar.
    """
    if df_historial.empty:
        return {"ok": True}

    productos_hist  = set(df_historial["producto"].unique())
    productos_nuevo = set(df_nuevo["producto"].unique())
    desconocidos = productos_nuevo - productos_hist
    pct = len(desconocidos) / len(productos_nuevo) if productos_nuevo else 0.0

    if pct >= UMBRAL_OTRO_NEGOCIO:
        return {
            "ok": False,
            "motivo": "otro_negocio",
            "pct_desconocidos": round(pct * 100, 1),
            "ejemplos_desconocidos": sorted(desconocidos)[:8],
            "ejemplos_conocidos": sorted(productos_hist)[:8],
        }
    return {"ok": True}


def fusionar(df_historial: pd.DataFrame, df_nuevo: pd.DataFrame):
    """
    Combina historial + lote nuevo, eliminando filas EXACTAMENTE
    duplicadas (misma clave que usa dataset_hash: fecha+producto+
    cantidad+precio+total). No intenta resolver conflictos donde la
    misma fecha+producto trae cantidades DISTINTAS entre archivos —
    esas filas se conservan ambas (podrían ser turnos o sucursales
    distintas reportadas por separado); es una decisión conservadora:
    nunca se descartan ventas reales sin que coincidan exactamente.

    Retorna (df_fusionado_ordenado_por_fecha, n_duplicados_omitidos).
    """
    columnas = [c for c in COLUMNAS_VENTA if c in df_historial.columns or c in df_nuevo.columns]
    combinado = pd.concat([df_historial, df_nuevo], ignore_index=True, sort=False)
    for c in columnas:
        if c not in combinado.columns:
            combinado[c] = None

    claves = [c for c in COLUMNAS_HASH if c in combinado.columns]
    antes = len(combinado)
    combinado = combinado.drop_duplicates(subset=claves, keep="first")
    n_duplicados = antes - len(combinado)

    combinado = combinado.sort_values("fecha").reset_index(drop=True)
    return combinado, n_duplicados
