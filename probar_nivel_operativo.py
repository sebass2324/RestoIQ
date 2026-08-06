"""
probar_nivel_operativo.py — corré esto desde la raíz de tu proyecto:

    python probar_nivel_operativo.py

Entrena el clasificador de Nivel Operativo con tu dataset de
El Chamo Burger (data/el_chamo_burger.csv si ya lo generaste, o lo
genera de nuevo si no existe) y muestra todas las métricas + una
predicción de ejemplo para una fecha futura.
"""

import os
import pandas as pd
from services.nivel_operativo import ClasificadorNivelOperativo

RUTA_CSV = "data/el_chamo_burger.csv"

if os.path.exists(RUTA_CSV):
    print(f"Usando dataset ya generado: {RUTA_CSV}")
    df = pd.read_csv(RUTA_CSV)
else:
    print(f"No encontré {RUTA_CSV} — genero uno nuevo...")
    from services.data_generator import DataGenerator
    gen = DataGenerator(tipo_negocio="elchamoburger", meses=48)
    df = gen.guardar_csv(RUTA_CSV)

print(f"{len(df):,} filas cargadas\n")

print("=" * 60)
print("Entrenando (walk-forward)...")
print("=" * 60)

modelo = ClasificadorNivelOperativo()
m = modelo.entrenar(df)

print(f"\nGanador: {m['algoritmo_ganador']}  ({m['n_folds']} pliegues)")
print(f"Accuracy: {m['accuracy']}  |  F1 Macro: {m['f1_macro']}")
print(f"Mejora vs. no hacer nada: {m['mejora_vs_baseline_pct']}%\n")

print("=" * 60)
print("COMPARACIÓN DE ALGORITMOS")
print("=" * 60)
for algo, met in m["comparacion_algoritmos"].items():
    print(f"{algo}: Accuracy={met['accuracy']}  F1 Macro={met['f1_macro']}")

print()
print("=" * 60)
print("POR CLASE")
print("=" * 60)
for clase, met in m["por_clase"].items():
    print(f"{clase}: precision={met['precision']}  recall={met['recall']}  "
          f"f1={met['f1']}  soporte={met['soporte']}")

print()
print("=" * 60)
print("MATRIZ DE CONFUSIÓN (filas=real, columnas=predicho)")
print("=" * 60)
print("           ", "  ".join(f"{c:>10}" for c in m["clases"]))
for i, fila in enumerate(m["matriz_confusion"]):
    print(f"{m['clases'][i]:>10} ", "  ".join(f"{v:>10}" for v in fila))

print()
print("=" * 60)
print("VARIABLES MÁS IMPORTANTES")
print("=" * 60)
for item in modelo.importancias[:8]:
    print(f"  {item['variable']}: {item['importancia']}")

print()
print("=" * 60)
print("PREDICCIÓN DE EJEMPLO — próximos 7 días")
print("=" * 60)
hoy = pd.Timestamp.now().normalize()
for i in range(1, 8):
    fecha = hoy + pd.Timedelta(days=i)
    resultado = modelo.predecir(fecha)
    print(f"  {fecha.strftime('%Y-%m-%d')} ({fecha.strftime('%A')}): "
          f"{resultado['nivel_operativo']}  (confianza {resultado['confianza']})")
