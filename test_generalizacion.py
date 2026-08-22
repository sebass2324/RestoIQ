"""
Prueba de generalización a productos NUNCA vistos en entrenamiento.

Repite, a escala y de forma medible, el mismo experimento que detectó
el problema original de memorización (~99% en productos conocidos vs.
~26% en productos nuevos con producto-ID como feature): separa una
porción de PRODUCTOS COMPLETOS (no solo fechas) antes de entrenar, y
compara qué tan bien el modelo predice esos productos que jamás vio.

Si el gap entre "conocidos" y "nunca vistos" es chico, el modelo está
generalizando por categoría/comportamiento, no memorizando identidad.
Si el gap es grande (como el 99% vs 26% original), hay memorización.
"""

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from classification_model import ModeloAbastecimiento, CLASES


def evaluar_generalizacion(df: pd.DataFrame, frac_productos_nuevos=0.2, seed=42):
    rng = np.random.RandomState(seed)
    productos = df["producto"].unique()
    n_nuevos = max(1, int(len(productos) * frac_productos_nuevos))
    productos_nuevos = set(rng.choice(productos, size=n_nuevos, replace=False))

    df_train_completo = df[~df["producto"].isin(productos_nuevos)].copy()
    df_nuevos          = df[df["producto"].isin(productos_nuevos)].copy()

    modelo = ModeloAbastecimiento()
    metricas_normales = modelo.entrenar(df_train_completo)

    # Etiqueta "verdadera" para los productos nunca vistos: la misma
    # fórmula de score que usa el modelo, con el fallback por categoría
    # que ya trae _raw_de (el producto no está en stats_producto porque
    # nunca se entrenó con él).
    pd_nuevos = modelo._agregar_producto_dia(df_nuevos)
    pd_nuevos = modelo._asegurar_contexto_temporal(pd_nuevos)
    y_true_nuevos = modelo._a_clase(modelo._construir_scores(pd_nuevos))

    contextos = pd_nuevos.to_dict("records")
    predicciones = modelo.predecir_lote(contextos)
    y_pred_nuevos = [p["prioridad"] for p in predicciones]

    acc_nuevos = accuracy_score(y_true_nuevos, y_pred_nuevos)
    f1_nuevos = f1_score(y_true_nuevos, y_pred_nuevos, labels=CLASES, average="macro", zero_division=0)

    return {
        "productos_entrenamiento": len(productos) - n_nuevos,
        "productos_nunca_vistos": n_nuevos,
        "f1_macro_conocidos (holdout temporal normal)": metricas_normales["f1_macro"],
        "accuracy_conocidos": metricas_normales["accuracy"],
        "f1_macro_productos_nuevos": round(float(f1_nuevos), 4),
        "accuracy_productos_nuevos": round(float(acc_nuevos), 4),
        "gap_f1": round(metricas_normales["f1_macro"] - float(f1_nuevos), 4),
    }


if __name__ == "__main__":
    np.random.seed(7)
    fechas = pd.date_range("2023-06-01", periods=500, freq="D")
    categorias_productos = {
        "Hamburguesas": [f"Hamb_{i}" for i in range(6)],
        "Pepito": [f"Pepito_{i}" for i in range(4)],
        "Arepas": [f"Arepa_{i}" for i in range(8)],
        "Cachapas": [f"Cachapa_{i}" for i in range(5)],
        "Bebidas": [f"Bebida_{i}" for i in range(6)],
    }
    filas = []
    for f in fechas:
        for cat, prods in categorias_productos.items():
            for p in prods:
                base = 3 + (hash((cat, p)) % 12)
                cantidad = max(0, int(np.random.poisson(base * (1.3 if f.weekday() in (5, 6) else 1.0))))
                filas.append({
                    "fecha": f, "producto": p, "categoria": cat, "cantidad": cantidad,
                    "precio": 1 + (hash(p) % 10), "promocion": int(np.random.rand() < 0.08),
                    "descuento_pct": 0, "es_evento_especial": 0,
                    "lluvia_nocturna_mm": np.random.rand() * 8,
                    "temp_nocturna_promed": 23 + np.random.rand() * 4,
                })
    df = pd.DataFrame(filas)

    resultado = evaluar_generalizacion(df, frac_productos_nuevos=0.2)
    for k, v in resultado.items():
        print(f"{k}: {v}")