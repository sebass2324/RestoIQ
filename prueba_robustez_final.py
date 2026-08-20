"""
prueba_robustez_final.py

Cierre del módulo de predicción: genera el dataset OFICIAL (con toda
la configuración ya validada: quincena real, promociones, dispersión
70, eventos reales) 5 veces con semillas distintas, entrena el modelo
completo cada vez, y reporta el rango de WAPE/R² observado.

Esto responde la pregunta correcta para la tesis: no "¿cuál es EL
número exacto?", sino "¿el modelo es consistente entre distintas
realizaciones de un negocio con el mismo patrón de fondo?".

Ejecutar desde la raiz del proyecto:
    python prueba_robustez_final.py

Advertencia: genera y entrena 5 veces completas -- puede tardar
15-25 minutos. Déjalo corriendo de fondo.
"""

import random
import time
import numpy as np
import pandas as pd

from services.data_generator import DataGenerator
from services.sales_model import SalesModel

SEMILLAS = [42, 7, 123, 2024, 55]


def main():
    resultados = []

    for i, semilla in enumerate(SEMILLAS, 1):
        print(f"\n[{i}/{len(SEMILLAS)}] Generando dataset con semilla {semilla}...")
        random.seed(semilla)
        np.random.seed(semilla)

        gen = DataGenerator(
            tipo_negocio="elchamoburger", meses=48,
            incluir_promociones=True, incluir_descuentos=False,
        )
        df = gen.generar()

        print(f"    Filas generadas: {len(df)}")

        if "precio_unitario" in df.columns and "precio" not in df.columns:
            df = df.rename(columns={"precio_unitario": "precio"})
        df["fecha"] = pd.to_datetime(df["fecha"])

        t0 = time.time()
        modelo = SalesModel()
        metricas = modelo.entrenar(df, config=None, verbose=False)
        dt = time.time() - t0

        holdout = metricas.get("holdout") or {}
        fila = {
            "semilla": semilla,
            "wape": holdout.get("wape_restoiq"),
            "r2": holdout.get("r2_restoiq"),
            "mejora_pct": holdout.get("mejora_pct"),
            "segundos": round(dt, 1),
        }
        resultados.append(fila)
        print(f"    WAPE={fila['wape']}  R²={fila['r2']}  mejora={fila['mejora_pct']}%  ({dt:.0f}s)")

    tabla = pd.DataFrame(resultados)
    print("\n=== RESULTADOS POR SEMILLA ===")
    print(tabla.to_string(index=False))

    wapes = tabla["wape"].dropna()
    r2s = tabla["r2"].dropna()

    print("\n=== RESUMEN DE ESTABILIDAD ===")
    print(f"  WAPE: media={wapes.mean():.2f}  desv.est={wapes.std():.2f}  rango=[{wapes.min():.2f}, {wapes.max():.2f}]")
    print(f"  R²:   media={r2s.mean():.3f}  desv.est={r2s.std():.3f}  rango=[{r2s.min():.3f}, {r2s.max():.3f}]")

    tabla.to_csv("resultados_robustez_final.csv", index=False)
    print("\nGuardado en resultados_robustez_final.csv")
    print("\nEste rango es el número que citas en tu tesis: no un punto exacto,")
    print("sino el intervalo observado en múltiples realizaciones del negocio.")


if __name__ == "__main__":
    main()
