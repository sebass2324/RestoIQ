"""
prueba_mediana_justa.py

Corrige el defecto del diagnostico anterior: compara el WAPE del
modelo real contra una mediana por producto calculada SOLO con el
train de cada pliegue (nunca con el futuro), evaluada exactamente en
las mismas fechas de prueba -- comparacion justa, out-of-sample en
ambos casos.

Ejecutar desde la raiz del proyecto:
    python prueba_mediana_justa.py
"""

import numpy as np
import pandas as pd

from app import create_app
from models.venta import Venta
from models.configuracion_analisis import ConfiguracionAnalisis
from services.sales_model import SalesModel

USER_ID = 6  # <-- ajusta al user_id correcto


def cargar_ventas(user_id):
    ventas = Venta.query.filter_by(user_id=user_id).all()
    print(f"[debug] Ventas encontradas: {len(ventas)}")
    filas = []
    for v in ventas:
        filas.append({
            "fecha": v.fecha, "producto": v.producto, "cantidad": v.cantidad,
            "precio": v.precio, "total": v.total,
            "dia_semana": v.dia_semana, "mes": v.mes, "semana_anio": v.semana_anio,
            "es_finde": v.es_finde, "es_feriado": v.es_feriado,
            "es_puente": v.es_puente, "es_quincena": v.es_quincena,
            "promocion": v.promocion, "descuento_pct": v.descuento_pct,
            "es_evento_especial": v.es_evento_especial,
            "lluvia_nocturna_mm": v.lluvia_nocturna_mm,
            "temp_nocturna_promed": v.temp_nocturna_promed,
        })
    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def wape(y_true, y_pred):
    total_real = np.sum(np.abs(y_true))
    return round(float(np.sum(np.abs(y_true - y_pred)) / total_real) * 100, 2) if total_real else None


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis -- configura el negocio primero.")
            return

        print("=== Entrenando el modelo real ===")
        modelo = SalesModel()
        metricas = modelo.entrenar(df, config=config, verbose=False)
        wape_modelo = metricas.get("wape")
        dias_historial = metricas.get("dias_historial")
        print(f"WAPE del modelo real (ya reportado): {wape_modelo}")

        # Reutilizamos el mismo dataframe ya agregado (fecha, producto,
        # cantidad) que el propio modelo construyó internamente -- no
        # reimplementamos la agregación, para no arriesgar otro bug.
        df_feat = modelo.df_historial
        n_folds = modelo._calcular_n_folds(dias_historial)
        pliegues = modelo._pliegues_por_fecha(df_feat["fecha"].unique(), n_folds)

        y_reales_todos, pred_mediana_todos = [], []
        for fecha_corte, fecha_fin in pliegues:
            train = df_feat[df_feat["fecha"] < fecha_corte]
            test = df_feat[(df_feat["fecha"] >= fecha_corte) & (df_feat["fecha"] <= fecha_fin)]
            if train.empty or test.empty:
                continue
            medianas_train = train.groupby("producto")["cantidad"].median()
            fallback = train["cantidad"].median()
            pred = test["producto"].map(medianas_train).fillna(fallback)
            y_reales_todos.append(test["cantidad"].values)
            pred_mediana_todos.append(pred.values)

        y_reales = np.concatenate(y_reales_todos)
        pred_mediana = np.concatenate(pred_mediana_todos)
        wape_mediana_justa = wape(y_reales, pred_mediana)

        print(f"\n=== COMPARACIÓN JUSTA (ambos out-of-sample, mismos pliegues) ===")
        print(f"WAPE del modelo real:                        {wape_modelo}")
        print(f"WAPE de mediana por producto (solo train):    {wape_mediana_justa}")
        diferencia = round(wape_modelo - wape_mediana_justa, 2)
        print(f"\nDiferencia (modelo - mediana): {diferencia:+.2f} puntos")
        if diferencia < -1:
            print("-> El modelo le GANA a la mediana simple -- sí está aportando valor real.")
        elif diferencia > 1:
            print("-> El modelo pierde contra algo tan simple como una mediana -- ahí SÍ hay margen real de mejora en el modelo.")
        else:
            print("-> Prácticamente empatados -- el modelo no aporta mucho sobre algo trivial, revisar hiperparámetros/features.")


if __name__ == "__main__":
    main()