"""
Hash determinista de un DataFrame de ventas ya limpio.

Se usa para saber si los datos de un usuario cambiaron desde el último
entrenamiento, sin tener que comparar el DataFrame completo cada vez:
comparamos DatasetUsuario.hash contra ModeloML.dataset_hash.
"""

import hashlib
import pandas as pd

COLUMNAS_HASH = ["fecha", "producto", "cantidad", "precio", "total"]


def calcular_hash(df: pd.DataFrame) -> str:
    columnas = [c for c in COLUMNAS_HASH if c in df.columns]
    payload = (
        df[columnas]
        .sort_values(columnas)
        .astype(str)
        .to_csv(index=False)
        .encode("utf-8")
    )
    return hashlib.sha256(payload).hexdigest()


def combinar_hash_config(dataset_hash: str, config) -> str:
    """
    Combina el hash del dataset con los flags de configuración que
    afectan qué features usa el modelo. Si el usuario cambia, por
    ejemplo, "considerar promociones" sin subir datos nuevos, el
    dataset_hash NO cambia — pero el modelo entrenado ya no coincide
    con lo que el usuario pidió. Esta combinación resuelve eso.
    """
    partes = [
        dataset_hash,
        str(int(config.considerar_promociones)),
        str(int(config.considerar_descuentos)),
        str(int(config.considerar_eventos)),
        str(int(config.considerar_feriados)),
    ]
    return hashlib.sha256("|".join(partes).encode("utf-8")).hexdigest()