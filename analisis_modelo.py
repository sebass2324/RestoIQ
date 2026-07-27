"""
analisis_modelo.py — Script de análisis OFFLINE del modelo de RestoIQ.

IMPORTANTE: esto NO es parte de la aplicación web. No lo ejecuta
ningún usuario de RestoIQ, no corre en el servidor Flask, y no se
llama desde ninguna ruta HTTP. Es una herramienta que el equipo de
desarrollo corre manualmente, en su propia computadora, para generar
gráficos de análisis más profundos que los que muestra la interfaz
en producción.

Por qué separado de la app:
  matplotlib y seaborn generan cada gráfico completo del lado del
  servidor (renderizan una imagen, la codifican, la envían) — un
  costo real de CPU y memoria por cada petición si se usaran dentro
  de la app en vivo. RestoIQ, en producción, usa Chart.js para todo
  lo que ve un usuario real: el navegador de CADA usuario dibuja sus
  propios gráficos con sus propios recursos, no el servidor.
  matplotlib/seaborn se reservan para este script de análisis y
  experimentación offline, donde ese costo no importa — se corre una
  vez, en una máquina de desarrollo, no en cada visita a una página.

Qué necesita: la ruta a un .pkl de un modelo ya entrenado (ver
ml_models/user_<id>.pkl). No se conecta a MySQL ni a Flask — todo lo
que necesita ya está guardado dentro del propio .pkl, en
self.metricas (top_features, importancias_top10, holdout con su
serie de puntos scatter).

Uso:
    python analisis_modelo.py ml_models/user_1.pkl
    python analisis_modelo.py ml_models/user_1.pkl --salida analisis/
"""
import sys
import argparse
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, str(Path(__file__).resolve().parent))
from services.sales_model import SalesModel

sns.set_theme(style="whitegrid", palette="deep")

COLOR_PRIMARIO = "#534AB7"


def graficar_importancia(metricas: dict, carpeta_salida: Path):
    """Importancia de variables — barra horizontal, matplotlib puro."""
    importancias = metricas.get("importancias_top10")
    if not importancias:
        print("  [omitido] El modelo no tiene importancia de variables guardada "
              "(probablemente la estrategia activa es Promedio Móvil, no LightGBM).")
        return

    nombres = list(importancias.keys())[::-1]
    valores = list(importancias.values())[::-1]

    fig, ax = plt.subplots(figsize=(8, 5), dpi=140)
    ax.barh(nombres, valores, color=COLOR_PRIMARIO, height=0.6)
    ax.set_xlabel("Importancia (ganancia acumulada en el modelo)")
    ax.set_title("¿Qué variables pesan más en las predicciones de RestoIQ?", fontsize=13, weight="bold")
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()

    ruta = carpeta_salida / "analisis_importancia.png"
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  [ok] {ruta}")


def graficar_regresion(metricas: dict, carpeta_salida: Path):
    """
    Real vs. Predicho con seaborn.regplot — incluye una línea de
    regresión ajustada con banda de intervalo de confianza, más
    riguroso estadísticamente que la línea y=x fija que se usa en el
    scatter interactivo de la app (ese es para el usuario de negocio;
    este es para el análisis técnico del equipo).
    """
    holdout = metricas.get("holdout")
    if not holdout or not holdout.get("scatter"):
        print("  [omitido] No hay datos de holdout guardados en este modelo todavía "
              "(historial insuficiente para haber corrido la validación walk-forward).")
        return

    df = pd.DataFrame(holdout["scatter"])

    fig, ax = plt.subplots(figsize=(7, 6), dpi=140)
    sns.regplot(
        data=df, x="real", y="predicho", ax=ax,
        scatter_kws={"alpha": 0.4, "color": COLOR_PRIMARIO, "s": 25},
        line_kws={"color": "#DC2626", "linewidth": 2},
    )
    max_val = max(df["real"].max(), df["predicho"].max()) * 1.05
    ax.plot([0, max_val], [0, max_val], linestyle="--", color="#888780", linewidth=1, label="Predicción exacta (y = x)")
    ax.set_xlabel("Demanda real (unidades)")
    ax.set_ylabel("Demanda predicha (unidades)")
    ax.set_title(f"Real vs. Predicho — {holdout.get('modelo_ganador', 'RestoIQ')}", fontsize=13, weight="bold")
    ax.legend()
    fig.tight_layout()

    ruta = carpeta_salida / "analisis_regresion.png"
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  [ok] {ruta}")


def graficar_distribucion_errores(metricas: dict, carpeta_salida: Path):
    """Distribución del error (predicho - real) — para ver si el
    modelo tiene sesgo sistemático (se corre hacia un lado) o si el
    error es simétrico alrededor de cero."""
    holdout = metricas.get("holdout")
    if not holdout or not holdout.get("scatter"):
        print("  [omitido] No hay datos de holdout guardados en este modelo todavía.")
        return

    df = pd.DataFrame(holdout["scatter"])
    df["error"] = df["predicho"] - df["real"]

    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    sns.histplot(df["error"], kde=True, ax=ax, color=COLOR_PRIMARIO, bins=30)
    ax.axvline(0, color="#DC2626", linestyle="--", linewidth=1.5, label="Sin error (0)")
    ax.axvline(df["error"].mean(), color="#1D9E75", linestyle="-", linewidth=1.5,
               label=f"Error promedio ({df['error'].mean():.2f})")
    ax.set_xlabel("Error (predicho − real)")
    ax.set_title("Distribución del error — ¿el modelo sobreestima o subestima en promedio?", fontsize=12, weight="bold")
    ax.legend()
    fig.tight_layout()

    ruta = carpeta_salida / "analisis_distribucion_errores.png"
    fig.savefig(ruta)
    plt.close(fig)
    print(f"  [ok] {ruta}")


def main():
    parser = argparse.ArgumentParser(description="Análisis offline del modelo de RestoIQ (matplotlib + seaborn)")
    parser.add_argument("ruta_pkl", help="Ruta al modelo entrenado, ej. ml_models/user_1.pkl")
    parser.add_argument("--salida", default="analisis", help="Carpeta donde guardar las imágenes (default: ./analisis)")
    args = parser.parse_args()

    carpeta_salida = Path(args.salida)
    carpeta_salida.mkdir(exist_ok=True)

    print(f"Cargando modelo desde {args.ruta_pkl} ...")
    modelo = SalesModel.cargar(args.ruta_pkl)
    metricas = modelo.metricas or {}

    print(f"Estrategia: {metricas.get('estrategia')} | "
          f"MAE={metricas.get('mae')} | WAPE={metricas.get('wape')}%\n")

    print("Generando gráficos de análisis...")
    graficar_importancia(metricas, carpeta_salida)
    graficar_regresion(metricas, carpeta_salida)
    graficar_distribucion_errores(metricas, carpeta_salida)
    print(f"\nListo. Imágenes guardadas en: {carpeta_salida.resolve()}")


if __name__ == "__main__":
    main()
