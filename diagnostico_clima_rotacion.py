"""
diagnostico_clima_rotacion.py

Diagnostico completo para entender por que bajaron las metricas tras
el nuevo generador de datos (Binomial Negativa + clima real):

  1. Importancia de features del modelo final -- especificamente,
     donde caen las 4 columnas de clima en el ranking.
  2. WAPE separado por nivel de rotacion (alta vs. baja) -- confirma
     si el error nuevo se concentra en productos de baja rotacion
     (demanda_base=1), como se sospecha.

Ejecutar desde la raiz del proyecto:
    python diagnostico_clima_rotacion.py
"""

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import TimeSeriesSplit

from app import create_app
from models.venta import Venta
from services.sales_model import SalesModel

USER_ID = 6  # <-- ajusta a tu user_id real


def val(v, campo, default):
    x = getattr(v, campo, default)
    return default if x is None else x


def cargar_ventas(user_id):
    ventas = Venta.query.filter_by(user_id=user_id).all()
    print(f"[debug] Ventas encontradas: {len(ventas)}")
    cols_clima = ("lluvia_manana_mm", "temp_manana_promed",
                  "lluvia_nocturna_mm", "temp_nocturna_promed")
    filas = []
    for v in ventas:
        fila = {
            "producto": v.producto, "fecha": v.fecha, "cantidad": v.cantidad,
            "categoria": val(v, "categoria", "Sin categoria"),
            "precio": val(v, "precio", 0),
            "promocion": val(v, "promocion", 0),
            "descuento_pct": val(v, "descuento_pct", 0),
            "es_evento_especial": val(v, "es_evento_especial", 0),
        }
        for c in cols_clima:
            valor = getattr(v, c, None)
            if valor is not None:
                fila[c] = valor
        filas.append(fila)
    return pd.DataFrame(filas)


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)

        print("\n=== 1. Entrenando modelo completo (para importancias) ===")
        modelo = SalesModel()
        metricas = modelo.entrenar(df, verbose=False)
        print(f"Estrategia: {metricas['estrategia']}  |  Features usadas: {modelo.features}")

        # Importancias completas, no solo el top10 que guarda self.metricas
        importancias = pd.Series(
            modelo.modelo.feature_importance(importance_type="gain"),
            index=modelo.modelo.feature_name(),
        ).sort_values(ascending=False)
        total = importancias.sum() or 1

        print("\n=== Ranking COMPLETO de importancias (% del total) ===")
        for feat, val_imp in importancias.items():
            marca = "  <-- CLIMA" if "lluvia" in feat or "temp_" in feat else ""
            print(f"  {feat:<28} {val_imp/total*100:5.1f}%{marca}")

        peso_clima = sum(v for f, v in importancias.items()
                         if "lluvia" in f or "temp_" in f) / total * 100
        print(f"\nPeso ACUMULADO del clima: {peso_clima:.1f}% del total de importancia")
        if peso_clima < 2:
            print("  -> El clima aporta muy poco. El modelo casi no lo está usando.")
        else:
            print("  -> El clima SÍ tiene peso real en las decisiones del modelo.")

        # --- 2. WAPE por rotación (alta vs. baja) ---
        print("\n=== 2. WAPE separado por rotación de producto ===")
        df2 = df.copy()
        df2["fecha"] = pd.to_datetime(df2["fecha"])
        df2["cantidad"] = pd.to_numeric(df2["cantidad"], errors="coerce")
        df2 = df2.dropna(subset=["fecha", "producto", "cantidad"])
        df2 = df2[df2["cantidad"] > 0]

        rotacion = df2.groupby("producto")["cantidad"].mean()
        mediana = rotacion.median()
        productos_alta = set(rotacion[rotacion >= mediana].index)
        productos_baja = set(rotacion[rotacion < mediana].index)
        print(f"Productos alta rotación (>= mediana de {mediana:.2f}/día): {len(productos_alta)}")
        print(f"Productos baja rotación (< mediana): {len(productos_baja)}")

        agg_spec = {"cantidad": ("cantidad", "sum")}
        for col, fn in (("promocion", "max"), ("descuento_pct", "mean"), ("es_evento_especial", "max"),
                        ("lluvia_manana_mm", "mean"), ("temp_manana_promed", "mean"),
                        ("lluvia_nocturna_mm", "mean"), ("temp_nocturna_promed", "mean")):
            if col in df2.columns:
                agg_spec[col] = (col, fn)
        df_agg = df2.groupby(["fecha", "producto"]).agg(**agg_spec).reset_index()

        df_feat = modelo._preparar_features(df_agg)
        df_feat_temporal = df_feat.sort_values("fecha").reset_index(drop=True)
        categoricas = modelo._categoricas_activas()

        import lightgbm as lgb
        parametros = {
            "objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 63, "max_bin": 255,
            "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "seed": 42,
        }

        n_folds = 3
        tscv = TimeSeriesSplit(n_splits=n_folds)
        reales_todo, preds_todo, productos_todo = [], [], []

        for idx_train, idx_test in tscv.split(df_feat_temporal):
            train_feat = df_feat_temporal.iloc[idx_train]
            test_feat = df_feat_temporal.iloc[idx_test]
            X_train, y_train = train_feat[modelo.features], train_feat["cantidad"]
            X_test, y_test = test_feat[modelo.features], test_feat["cantidad"]

            train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categoricas)
            m = lgb.train(parametros, train_set, num_boost_round=700)
            pred = np.maximum(0, m.predict(X_test))

            reales_todo.append(y_test.values)
            preds_todo.append(pred)
            productos_todo.append(test_feat["producto"].astype(str).values)

        y_reales = np.concatenate(reales_todo)
        y_preds = np.concatenate(preds_todo)
        productos_arr = np.concatenate(productos_todo)

        def wape(y_true, y_pred):
            total_real = np.sum(np.abs(y_true))
            return round(float(np.sum(np.abs(y_true - y_pred)) / total_real) * 100, 2) if total_real else None

        mask_alta = np.isin(productos_arr, list(productos_alta))
        mask_baja = ~mask_alta

        print(f"\nWAPE GLOBAL:              {wape(y_reales, y_preds)}%")
        print(f"WAPE productos ALTA rotación: {wape(y_reales[mask_alta], y_preds[mask_alta])}%  (n={mask_alta.sum()})")
        print(f"WAPE productos BAJA rotación: {wape(y_reales[mask_baja], y_preds[mask_baja])}%  (n={mask_baja.sum()})")


if __name__ == "__main__":
    main()
