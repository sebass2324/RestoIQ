"""
prueba_hiperparametros_lgbm.py

Busca hiperparametros de LightGBM para el modelo de clasificacion,
usando la misma validacion walk-forward que ya usa el proyecto (no
introduce una metodologia de evaluacion nueva).

Ejecutar desde la raiz del proyecto:
    python prueba_hiperparametros_lgbm.py
"""

import itertools
import time
import pandas as pd

from app import create_app
import services.classification_model as cm
from services.classification_model import ModeloAbastecimiento
from models.venta import Venta

USER_ID = 3

GRID = {
    "n_estimators":     [200, 350, 500],
    "learning_rate":    [0.02, 0.04, 0.08],
    "num_leaves":       [15, 31, 63],
}


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


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        print(f"Filas: {len(df)}  |  Productos: {df['producto'].nunique()}\n")

        combinaciones = list(itertools.product(
            GRID["n_estimators"], GRID["learning_rate"], GRID["num_leaves"]))
        print(f"Probando {len(combinaciones)} combinaciones...\n")

        resultados = []
        for i, (n_est, lr, leaves) in enumerate(combinaciones, start=1):
            cm.LGBM_PARAMS = {
                "n_estimators": n_est, "learning_rate": lr, "num_leaves": leaves,
                "min_child_samples": 12, "subsample": 0.85,
                "colsample_bytree": 0.85, "reg_lambda": 0.5,
            }
            t0 = time.time()
            modelo = ModeloAbastecimiento(random_state=42)
            m = modelo.entrenar(df)
            dt = time.time() - t0

            fila = {
                "n_estimators": n_est, "learning_rate": lr, "num_leaves": leaves,
                "accuracy": m["accuracy"], "f1_macro": m["f1_macro"],
                "segundos": round(dt, 1),
            }
            resultados.append(fila)
            print(f"[{i}/{len(combinaciones)}] n_est={n_est:<4} lr={lr:<5} "
                 f"leaves={leaves:<3} -> acc={fila['accuracy']}  "
                 f"f1_macro={fila['f1_macro']}  ({dt:.1f}s)")

        tabla = pd.DataFrame(resultados).sort_values("f1_macro", ascending=False)
        print("\n=== TOP 5 por f1_macro ===")
        print(tabla.head(5).to_string(index=False))

        actual = tabla[(tabla.n_estimators == 350) & (tabla.learning_rate == 0.04)
                       & (tabla.num_leaves == 31)]
        print(f"\nConfiguracion actual (350, 0.04, 31): "
             f"f1_macro={actual['f1_macro'].values[0] if len(actual) else 'no probada'}")

        tabla.to_csv("resultados_hiperparametros_lgbm.csv", index=False)
        print("\nGuardado en resultados_hiperparametros_lgbm.csv")


if __name__ == "__main__":
    main()
