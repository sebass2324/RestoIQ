"""
services/matriz_confusion_img.py

Genera la imagen de la matriz de confusión con matplotlib/seaborn.

Se ejecuta UNA SOLA VEZ por entrenamiento (dentro de abastecimiento.py,
justo después de modelo.entrenar()) — NUNCA en el hot path de una
request de usuario. Mismo principio de arquitectura que
services/analisis_modelo.py: matplotlib no corre en vivo por cada vista,
solo cuando se genera/actualiza un artefacto (acá: un PNG en vez de un
.pkl, pero el mismo costo puntual).
"""

import matplotlib
matplotlib.use("Agg")  # sin display — servidor
import matplotlib.pyplot as plt
import seaborn as sns


def generar_matriz_confusion_png(matriz, clases, ruta_salida: str, titulo="Matriz de confusión"):
    fig, ax = plt.subplots(figsize=(4.2, 4.2))
    sns.heatmap(
        matriz, annot=True, fmt="d", cmap="Blues",
        xticklabels=clases, yticklabels=clases,
        cbar=False, linewidths=1, linecolor="white",
        annot_kws={"size": 13, "weight": "bold"}, ax=ax,
    )
    ax.set_xlabel("predicción")
    ax.set_ylabel("actual")
    ax.set_title(titulo, fontsize=12, weight="bold")
    plt.tight_layout()
    fig.savefig(ruta_salida, dpi=140)
    plt.close(fig)
