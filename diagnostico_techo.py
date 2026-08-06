"""
diagnostico_techo.py

Antes de cambiar nada al azar, este script mide DONDE esta el techo
del modelo, respondiendo 4 preguntas concretas:

  1. Cobertura de umbrales: cuantas filas usan umbral propio vs. respaldo
     (si muchas caen al respaldo, la etiqueta es menos precisa).
  2. Rendimiento por origen de umbral: el F1 es peor en las filas que
     cayeron al respaldo? (nos dice si el problema son esos productos).
  3. Casos frontera: cuantas filas estan muy cerca de un umbral (donde
     un cambio de 1 unidad salta de clase) -- ruido intrinseco irreducible.
  4. Techo teorico: si un modelo PERFECTO clasificara, cuanto acertaria
     dado el ruido de frontera? (nos dice si 75% ya esta cerca del maximo).

Ejecutar desde la raiz del proyecto:
    python diagnostico_techo.py
"""

import numpy as np
import pandas as pd

from app import create_app
from models.venta import Venta
from services.classification_model import ModeloAbastecimiento, CLASES


USER_ID = 3


def val(v, campo, default):
    x = getattr(v, campo, default)
    return default if x is None else x


def cargar_ventas(user_id):
    ventas = Venta.query.filter_by(user_id=user_id).all()
    print(f"[debug] Ventas encontradas: {len(ventas)}")
    filas = [{
        "producto": v.producto, "fecha": v.fecha, "cantidad": v.cantidad,
        "categoria": val(v, "categoria", "Sin categoria"),
        "precio": val(v, "precio", 0), "promocion": val(v, "promocion", 0),
        "descuento_pct": val(v, "descuento_pct", 0),
        "es_evento_especial": val(v, "es_evento_especial", 0),
    } for v in ventas]
    return pd.DataFrame(filas)


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        modelo = ModeloAbastecimiento(random_state=42)
        diario = modelo._agregar_producto_dia(df)
        datos = modelo._con_historial(diario)
        modelo._fijar_umbrales(datos)

        # --- 1. Cobertura de umbrales ---
        origenes = [modelo._origen_umbral(p, c) for p, c in
                    zip(datos["producto"],
                        datos.get("categoria", pd.Series([None]*len(datos))))]
        cob = pd.Series(origenes).value_counts(normalize=True)
        print("\n=== 1. COBERTURA DE UMBRALES ===")
        for origen in ("producto", "categoria", "global"):
            print(f"  {origen:<12}: {cob.get(origen, 0)*100:5.1f}% de las filas")
        print(f"  Productos con umbral propio: {len(modelo.umbrales_producto)} "
             f"de {datos['producto'].nunique()}")

        # --- 3. Casos frontera (ruido irreducible) ---
        y = modelo._a_clase(datos)
        cat_col = datos.get("categoria", pd.Series([None]*len(datos)))
        margenes = []
        for cantidad, prod, cat in zip(datos["cantidad"], datos["producto"], cat_col):
            u1, u2 = modelo._umbrales_para(prod, cat)
            # distancia relativa al umbral mas cercano
            d1, d2 = abs(cantidad - u1), abs(cantidad - u2)
            margen = min(d1, d2) / (u2 - u1) if (u2 - u1) > 0 else 1.0
            margenes.append(margen)
        margenes = np.array(margenes)
        frontera_10 = (margenes < 0.10).mean()
        frontera_05 = (margenes < 0.05).mean()
        print("\n=== 3. CASOS FRONTERA (ruido intrinseco) ===")
        print(f"  Filas a <5% de distancia de un umbral:  {frontera_05*100:5.1f}%")
        print(f"  Filas a <10% de distancia de un umbral: {frontera_10*100:5.1f}%")
        print("  (estas filas son casi imposibles de clasificar bien: un")
        print("   cambio minimo de ventas las salta de clase)")

        # --- 4. Distribucion de clases ---
        print("\n=== 4. DISTRIBUCION DE CLASES (deberia ser ~33/33/33) ===")
        dist = y.value_counts(normalize=True)
        for clase in CLASES:
            print(f"  {clase:<8}: {dist.get(clase, 0)*100:5.1f}%")

        # --- Estimacion de techo ---
        # Si ~X% de las filas estan en zona de frontera (irreducibles),
        # el techo teorico ronda 100% - (fraccion de esas que se pierden).
        techo_est = 100 - frontera_10 * 50  # asumiendo que la mitad de las de frontera se pierden
        print("\n=== RESUMEN ===")
        print(f"  Techo teorico estimado (grueso): ~{techo_est:.0f}%")
        print("  Si tu accuracy actual (~75%) esta cerca de este techo,")
        print("  el margen de mejora honesto es pequeño y el numero es defendible.")


if __name__ == "__main__":
    main()
