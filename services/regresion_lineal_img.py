"""
services/regresion_lineal_img.py

Genera el gráfico Real vs. Predicho del modelo GANADOR (el que
realmente se usa en producción — RestoIQ, Regresión Lineal o Naive,
lo que haya ganado por WAPE ese entrenamiento) con la línea de ajuste
OLS + banda de confianza al 95% (lo que dibuja sns.regplot), la
diagonal de predicción exacta (y=x) como referencia, y el R² como
resumen numérico del ajuste.

Consolidado a UN SOLO gráfico (antes convivía con un scatter
interactivo de Chart.js que mostraba lo mismo cuando el ganador era
Regresión Lineal — redundante). Este reemplaza a ambos.

Se ejecuta UNA SOLA VEZ por entrenamiento (llamado desde el blueprint,
justo después de SalesModel.entrenar()) — NUNCA en el hot path de una
request de usuario. Mismo principio de arquitectura que
matriz_confusion_img.py.

Buenas prácticas de recursos aplicadas:
  - Backend 'Agg' (sin display, no intenta abrir una ventana).
  - Los puntos ya vienen sub-muestreados a ≤500 (ver sales_model.py,
    'scatter') — el gráfico no reprocesa miles de filas.
  - plt.close(fig) explícito: sin esto, matplotlib acumula figuras en
    memoria durante la vida del proceso de Flask (que es de larga
    duración) — una fuga de memoria lenta pero real en un servidor
    que reentrena modelos repetidamente.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns


def generar_grafico_regresion_png(real: list, predicho: list, r2: float,
                                   nombre_modelo: str, ruta_salida: str):
    sns.set_style("whitegrid")
    fig, ax = plt.subplots(figsize=(6.4, 5))

    sns.regplot(
        x=real, y=predicho, ax=ax, ci=95,
        scatter_kws={"alpha": .45, "s": 28, "color": "#534AB7"},
        line_kws={"color": "#1D9E75", "linewidth": 2.5},
    )

    lim_superior = max(max(real), max(predicho)) * 1.05 if real and predicho else 1
    ax.plot([0, lim_superior], [0, lim_superior], linestyle="--",
             color="#888780", linewidth=1.3, label="Predicción exacta (y = x)")

    ax.text(0.04, 0.94, f"R² = {r2:.2f}", transform=ax.transAxes,
            fontsize=13, fontweight="bold", color="#2C2C2A", va="top")

    ax.set_xlabel("Demanda real (unidades)", fontsize=11)
    ax.set_ylabel(f"Demanda predicha — {nombre_modelo} (unidades)", fontsize=11)
    ax.set_title(f"Cómo predice {nombre_modelo} frente a la realidad",
                 fontsize=13, fontweight="bold", pad=14)
    ax.legend(loc="lower right", fontsize=9, frameon=True)
    ax.set_xlim(0, lim_superior)
    ax.set_ylim(0, lim_superior)

    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=140)
    plt.close(fig)  # libera la memoria de la figura 