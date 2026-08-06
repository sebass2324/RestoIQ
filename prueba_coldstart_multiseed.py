"""
prueba_coldstart_multiseed.py

Corre la prueba de cold-start real (sacar productos completos del
entrenamiento) con VARIAS semillas automaticamente, y da un resumen
estadistico (promedio y desviacion) -- mucho mas convincente para una
sustentacion que 1-2 corridas sueltas.

Ejecutar desde la raiz del proyecto:
    python prueba_coldstart_multiseed.py
"""

import random
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

from app import create_app
from models.venta import Venta
from services.classification_model import ModeloAbastecimiento

USER_ID = 3
N_PRODUCTOS_FUERA = 10
SEMILLAS = [42, 7, 123, 2024, 8]  # 5 corridas independientes


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
    diario = modelo._agregar_producto_dia(df_coldstart)
    datos = modelo._con_historial(diario)
    if datos.empty:
        return None
    y_real = modelo._a_clase(datos)  # ahora recibe el DataFrame completo
    X = modelo._construir_features(datos, entrenando=False)
    y_pred = modelo.modelo.predict(X)
    return {
        "accuracy": float(accuracy_score(y_real, y_pred)),
        "f1_macro": float(f1_score(y_real, y_pred, labels=modelo.clases,
                                   average="macro", zero_division=0)),
        "n_filas": len(y_real),
    }


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        productos = sorted(df["producto"].unique())
        print(f"Productos totales: {len(productos)}  |  Semillas a probar: {SEMILLAS}\n")

        resultados = []
        for semilla in SEMILLAS:
            random.seed(semilla)
            productos_fuera = set(random.sample(productos,
                                                 min(N_PRODUCTOS_FUERA, len(productos))))
            df_train_pool = df[~df["producto"].isin(productos_fuera)].reset_index(drop=True)
            df_coldstart  = df[df["producto"].isin(productos_fuera)].reset_index(drop=True)

            modelo = ModeloAbastecimiento(random_state=semilla)
            m = modelo.entrenar(df_train_pool)
            cold = evaluar_coldstart(modelo, df_coldstart)

            fila = {
                "semilla": semilla,
                "acc_normal": m["accuracy"], "f1_normal": m["f1_macro"],
                "acc_cold": cold["accuracy"] if cold else None,
                "f1_cold": cold["f1_macro"] if cold else None,
            }
            resultados.append(fila)
            print(f"Semilla {semilla:>5}: normal(acc={fila['acc_normal']:.4f} "
                 f"f1={fila['f1_normal']:.4f})  |  "
                 f"cold-start(acc={fila['acc_cold']:.4f} f1={fila['f1_cold']:.4f})")

        tabla = pd.DataFrame(resultados)
        print("\n=== RESUMEN (promedio ± desviacion estandar, sobre "
             f"{len(SEMILLAS)} semillas) ===")
        for col, nombre in [("acc_normal", "Accuracy normal"),
                            ("f1_normal", "F1 macro normal"),
                            ("acc_cold", "Accuracy cold-start"),
                            ("f1_cold", "F1 macro cold-start")]:
            print(f"  {nombre:<22} {tabla[col].mean():.4f} ± {tabla[col].std():.4f}")

        caida_acc = tabla["acc_normal"].mean() - tabla["acc_cold"].mean()
        print(f"\nCaida promedio de accuracy (normal -> cold-start): {caida_acc:+.4f}")
        if caida_acc > 0.10:
            print("⚠️  Caida mayor a 10 puntos -- señal de posible memorización.")
        else:
            print("OK -- sin caida relevante, evidencia consistente de generalización.")

        tabla.to_csv("resultados_coldstart_multiseed.csv", index=False)
        print("\nGuardado en resultados_coldstart_multiseed.csv")


if __name__ == "__main__":
    main()
