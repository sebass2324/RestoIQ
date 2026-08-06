"""
probar_dataset_grande.py — corré esto desde la raíz de tu proyecto:

    python probar_dataset_grande.py

Genera un dataset sintético grande (4 años, como el real que vas a
usar), entrena el modelo, y muestra todo: tiempos, métricas, features
usadas, y un ciclo completo de guardar/cargar/predecir — para
confirmar que el sistema aguanta el volumen antes de meterle el
dataset real de El Chamo Burger.
"""

import time
import pandas as pd
from services.data_generator import DataGenerator
from services.sales_model import SalesModel

print("=" * 60)
print("Generando dataset sintético de 4 años...")
print("=" * 60)

t0 = time.time()
gen = DataGenerator(tipo_negocio="restaurante", meses=48, incluir_promociones=True, incluir_descuentos=True)
df = gen.generar(sucio=False).rename(columns={"precio_unitario": "precio"})
print(f"Generado en {time.time() - t0:.1f}s — {len(df):,} filas, {df['producto'].nunique()} productos")
print(f"Rango de fechas: {df['fecha'].min()} a {df['fecha'].max()}")
print()

print("=" * 60)
print("Entrenando el modelo...")
print("=" * 60)

t0 = time.time()
modelo = SalesModel()
resultado = modelo.entrenar(df, verbose=True)
tiempo_entrenamiento = time.time() - t0
print(f"\n⏱  Tiempo total de entrenamiento: {tiempo_entrenamiento:.1f}s")
print()

print("=" * 60)
print("MÉTRICAS")
print("=" * 60)
h = resultado.get("holdout")
if h:
    print(f"Modelo ganador:     {h['modelo_ganador']}")
    print(f"MAE (RestoIQ):      {h['mae_restoiq']}")
    print(f"WAPE (RestoIQ):     {h['wape_restoiq']}%")
    print(f"RMSE (RestoIQ):     {h['rmse_restoiq']}")
    print(f"R² (RestoIQ):       {h['r2_restoiq']}")
    print(f"Mejora vs baseline: {h['mejora_pct']}%")
    print()
    print(f"MAE (Regresión Lineal): {h['mae_baseline']}  |  WAPE: {h['wape_baseline']}%")
    print(f"MAE (Naive lag-7):      {h['mae_naive']}  |  WAPE: {h['wape_naive']}%")
else:
    print("Sin holdout (historial insuficiente para walk-forward)")

print()
print("=" * 60)
print("FEATURES USADAS")
print("=" * 60)
for f in resultado["features_usadas"]:
    print(f"  - {f}")

print()
print("Top 5 variables más importantes:", resultado.get("top_features"))

print()
print("=" * 60)
print("Probando ciclo completo: guardar → cargar → predecir")
print("=" * 60)

t0 = time.time()
modelo.guardar("/tmp/prueba_modelo_grande.pkl")
print(f"Guardado en {time.time() - t0:.2f}s")

t0 = time.time()
modelo_cargado = SalesModel.cargar("/tmp/prueba_modelo_grande.pkl")
print(f"Cargado en {time.time() - t0:.2f}s")

t0 = time.time()
prediccion = modelo_cargado.predecir(dias=7)
print(f"Predicción (7 días) generada en {time.time() - t0:.2f}s — {len(prediccion['por_producto'])} filas")
print()
print("Primeras 5 predicciones:")
print(prediccion["por_producto"].head(5).to_string(index=False))
print()
print("Resumen:", prediccion["resumen"])

print()
print("=" * 60)
print(f"✅ TODO OK — tiempo total del script: {time.time() - t0:.1f}s (última medición) / entrenamiento solo: {tiempo_entrenamiento:.1f}s")
print("=" * 60)
