import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import io, base64

# Traduce cada feature técnica a una categoría que un dueño de negocio entiende
CATEGORIAS = {
    "lag_7":            "Ventas recientes",
    "lag_14":           "Ventas recientes",
    "lag_28":           "Ventas recientes",
    "rolling_7_mean":   "Tendencia reciente",
    "rolling_14_mean":  "Tendencia reciente",
    "rolling_7_std":    "Tendencia reciente",
    "dia_semana":       "Día de la semana",
    "es_finde":         "Día de la semana",
    "es_feriado":       "Feriados",
    "es_puente":        "Feriados",
    "es_quincena":      "Día de pago (quincena)",
    "mes":              "Temporada del año",
    "semana_anio":      "Temporada del año",
    "producto_encoded": "Tipo de producto",
}

COLOR_PRINCIPAL = "#534AB7"
COLOR_TEXTO     = "#2C2C2A"
COLOR_GRID      = "#E2E4EE"


def grafico_feature_importance(modelo_lgbm, feature_names, top_n=8) -> str:
    """
    Agrupa la importancia técnica de LightGBM en categorías de negocio
    y genera un gráfico horizontal simple, legible para un usuario no técnico.
    """
    serie = pd.Series(modelo_lgbm.feature_importances_, index=feature_names)

    # Agrupar por categoría de negocio
    agrupado = (
        serie.rename(index=lambda f: CATEGORIAS.get(f, f))
        .groupby(level=0)
        .sum()
        .sort_values(ascending=True)  # ascending porque barh dibuja de abajo hacia arriba
    )

    # Convertir a porcentaje del total (más intuitivo que "importancia relativa")
    porcentajes = (agrupado / agrupado.sum() * 100).round(1)

    fig, ax = plt.subplots(figsize=(8, 4.5))

    # Degradado de color: la barra más importante más oscura
    n = len(porcentajes)
    colores = [COLOR_PRINCIPAL if i == n - 1 else "#A6A0DE" for i in range(n)]
    # resalta solo la barra top; el resto en tono más claro
    colores[-1] = COLOR_PRINCIPAL

    barras = ax.barh(porcentajes.index, porcentajes.values, color=colores, height=0.6)

    # Etiquetas de porcentaje al final de cada barra
    for barra, valor in zip(barras, porcentajes.values):
        ax.text(
            barra.get_width() + 0.5, barra.get_y() + barra.get_height() / 2,
            f"{valor:.0f}%", va="center", fontsize=11, color=COLOR_TEXTO, fontweight="bold"
        )

    # Limpieza visual: sin bordes, sin ticks innecesarios
    for spine in ["top", "right", "left", "bottom"]:
        ax.spines[spine].set_visible(False)
    ax.tick_params(left=False, bottom=False, labelsize=11)
    ax.set_xticks([])
    ax.set_xlim(0, porcentajes.max() * 1.2)

    ax.set_title("¿Qué influye más en tus ventas?", fontsize=15, fontweight="bold",
                 color=COLOR_TEXTO, loc="left", pad=14)
    fig.text(0.125, 0.90, "Factores que el modelo usó para predecir tu demanda",
              fontsize=10, color="#8A8A88")

    fig.tight_layout(rect=[0, 0, 1, 0.90])

    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=130, bbox_inches="tight", facecolor="white")
    buf.seek(0)
    img_b64 = base64.b64encode(buf.read()).decode("utf-8")
    plt.close(fig)
    return img_b64