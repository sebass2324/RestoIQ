"""
prueba_coldstart_lgbm.py

Misma metodologia que prueba_coldstart_real.py, adaptada a la nueva
version del modelo (LightGBM, umbrales globales sobre cantidad cruda).

Saca N productos COMPLETOS del entrenamiento y mide el desempeño
SOLO sobre esos productos nunca vistos -- para confirmar si la
identidad de producto (one-hot) causa memorizacion, igual que se
midio antes con RandomForest.

Ejecutar desde la raiz del proyecto:
    python prueba_coldstart_lgbm.py
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app import create_app
from models.venta import Venta
from services.classification_model import ModeloAbastecimiento

USER_ID = 3              # <-- cambia esto por tu user_id real
N_PRODUCTOS_FUERA = 10
SEMILLA = 7


def val(v, campo, default):
    x = getattr(v, campo, default)
    return default if x is None else x


def cargar_ventas(user_id):
    ventas = Venta.query.filter_by(user_id=user_id).all()
    filas = [{
        "producto": v.producto, "fecha": v.fecha, "cantidad": v.cantidad,
        "categoria": val(v, "categoria", "Sin categoria"),
        "precio": val(v, "precio", 0),
        "promocion": val(v, "promocion", 0),
        "descuento_pct": val(v, "descuento_pct", 0),
        "es_evento_especial": val(v, "es_evento_especial", 0),
    } for v in ventas]
    return pd.DataFrame(filas)


def evaluar_coldstart(modelo, df_coldstart):
    """Reconstruye el mismo pipeline interno (agregar + historial +
    features) para los productos nunca vistos, y compara contra la
    clase real calculada con los MISMOS umbrales que fijo el modelo
    en entrenamiento (no se recalculan -- serian umbrales "del futuro"
    si se calcularan con datos que el modelo no uso)."""
    diario = modelo._agregar_producto_dia(df_coldstart)
    datos = modelo._con_historial(diario)
    if datos.empty:
        print("  (sin filas suficientes para cold-start -- historial muy corto)")
        return None
    y_real = modelo._a_clase(datos["cantidad"])
    X = modelo._construir_features(datos, entrenando=False)
    y_pred = modelo.modelo.predict(X)
    return {
        "accuracy": round(float(accuracy_score(y_real, y_pred)), 4),
        "f1_macro": round(float(f1_score(y_real, y_pred, labels=modelo.clases,
                                         average="macro", zero_division=0)), 4),
        "n_filas": len(y_real),
    }


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        productos = sorted(df["producto"].unique())
        print(f"Productos totales: {len(productos)}")

        random.seed(SEMILLA)
        productos_fuera = set(random.sample(productos, min(N_PRODUCTOS_FUERA, len(productos))))
        print(f"Productos sacados del entrenamiento (cold-start real): {productos_fuera}")

        df_train_pool = df[~df["producto"].isin(productos_fuera)].reset_index(drop=True)
        df_coldstart  = df[df["producto"].isin(productos_fuera)].reset_index(drop=True)

        print("\n=== Entrenando con productos conocidos (sin los 10 de cold-start) ===")
        modelo = ModeloAbastecimiento(random_state=SEMILLA)
        metricas = modelo.entrenar(df_train_pool)
        print(f"  Metricas normales (walk-forward, MISMOS productos): "
             f"acc={metricas['accuracy']}  f1_macro={metricas['f1_macro']}")

        print("\n=== Evaluando en productos NUNCA vistos ===")
        cold = evaluar_coldstart(modelo, df_coldstart)
        if cold:
            print(f"  acc={cold['accuracy']}  f1_macro={cold['f1_macro']}  (n={cold['n_filas']})")

        print("\n=== Top 10 features mas importantes ===")
        for f in modelo.importancias:
            print(f"  {f['variable']:<25} {f['importancia']}")


if __name__ == "__main__":
    main()
