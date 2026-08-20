"""
prueba_promocion_ciega.py

Tu WAPE reportado (holdout) usa la promoción REAL conocida de cada
fecha histórica -- pero en producción, si no declaras "Eventos
futuros", la predicción siempre asume promocion=0. Este script mide
qué tan grande es esa brecha: compara el WAPE actual (promoción
conocida) contra un WAPE "ciego" (promoción forzada a 0 en las
features objetivo del holdout, igual que en producción sin eventos
declarados) -- mismo dataset exacto en ambos casos.

Ejecutar desde la raiz del proyecto:
    python prueba_promocion_ciega.py
"""

import pandas as pd
import numpy as np
import lightgbm as lgb

from app import create_app
from models.venta import Venta
from models.configuracion_analisis import ConfiguracionAnalisis
import services.sales_model as sm
from services.sales_model import TARGET, LGBM_PARAMS

USER_ID = 6 # <-- ajusta al user_id correcto


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
            "es_puente": v.es_puente,
            "dias_distancia_cobro": v.dias_distancia_cobro,
            "pico_comida_rapida":   int(v.pico_comida_rapida) if v.pico_comida_rapida is not None else 0,
            "fase_liquidez":        v.fase_liquidez,
            "promocion":           int(v.promocion) if v.promocion is not None else 0,
            "descuento_pct":       v.descuento_pct,
            "es_evento_especial":  int(v.es_evento_especial) if v.es_evento_especial is not None else 0,
            "lluvia_nocturna_mm":    v.lluvia_nocturna_mm,
            "temp_nocturna_promed":  v.temp_nocturna_promed,
        })
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def wape(y_true, y_pred):
    total = np.sum(np.abs(y_true))
    return round(float(np.sum(np.abs(y_true - y_pred)) / total) * 100, 2) if total else None


def evaluar(modelo, df_feat, pliegues, forzar_promocion_cero, k_valores):
    categoricas = modelo._categoricas_activas()
    y_reales_todos, preds_todos = [], []
    for fecha_corte, fecha_fin in pliegues:
        train = df_feat[df_feat["fecha"] < fecha_corte]
        if len(train) < 30:
            continue
        for k in k_valores:
            ds_train = modelo._construir_dataset_direct(train, k)
            if len(ds_train) < 30:
                continue
            ds_completo = modelo._construir_dataset_direct(df_feat, k)
            mask_test = (ds_completo["fecha"] >= fecha_corte - pd.Timedelta(days=k)) & \
                       (ds_completo["fecha"] <= fecha_fin - pd.Timedelta(days=k))
            ds_test = ds_completo[mask_test].copy()
            if ds_test.empty:
                continue

            # La diferencia clave: en el escenario "ciego", forzamos
            # promocion=0 en el TEST -- igual que en producción real
            # cuando no se declaran eventos futuros. El TRAIN se deja
            # intacto (el modelo sí aprendió la relación promo->demanda
            # del histórico real, solo cambia qué sabe al PREDECIR).
            if forzar_promocion_cero and "promocion" in ds_test.columns:
                ds_test["promocion"] = 0

            X_train, y_train = ds_train[modelo.features], ds_train[TARGET]
            X_test, y_test = ds_test[modelo.features], ds_test[TARGET]
            train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categoricas)
            m = lgb.train(dict(LGBM_PARAMS), train_set, num_boost_round=modelo.n_arboles_optimo or 700)
            preds = np.maximum(0, m.predict(X_test))
            y_reales_todos.append(y_test.values)
            preds_todos.append(preds)

    return wape(np.concatenate(y_reales_todos), np.concatenate(preds_todos))


def main():
    app = create_app()
    with app.app_context():
        df_raw = cargar_ventas(USER_ID)
        if df_raw.empty:
            print(f"[!] No hay ventas para USER_ID={USER_ID}.")
            return
        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis -- configura el negocio primero.")
            return

        modelo = sm.SalesModel()
        metricas = modelo.entrenar(df_raw, config=config, verbose=False)
        print(f"WAPE oficial (reportado por la app, promoción conocida): {metricas['holdout']['wape_restoiq']}")

        agg_spec = {"cantidad": ("cantidad", "sum")}
        for col, fn in (("promocion", "max"), ("es_evento_especial", "max"),
                        ("lluvia_nocturna_mm", "mean"), ("temp_nocturna_promed", "mean")):
            if col in df_raw.columns:
                agg_spec[col] = (col, fn)
        df_agg = df_raw.groupby(["fecha", "producto"]).agg(**agg_spec).reset_index()
        df_feat = modelo._preparar_features(df_agg)

        pliegues = modelo._pliegues_por_fecha(df_feat["fecha"].unique())
        k_valores = list(range(1, modelo.HORIZONTE_VALIDACION + 1))

        print("\nEvaluando escenario A (promoción conocida, igual al oficial) -- puede tardar unos minutos...")
        wape_conocida = evaluar(modelo, df_feat, pliegues, forzar_promocion_cero=False, k_valores=k_valores)

        print("Evaluando escenario B (promoción SIEMPRE en 0, como en producción sin eventos declarados)...")
        wape_ciega = evaluar(modelo, df_feat, pliegues, forzar_promocion_cero=True, k_valores=k_valores)

        print("\n=== RESUMEN ===")
        print(f"  WAPE con promoción conocida (como el holdout oficial): {wape_conocida}")
        print(f"  WAPE ciego (promoción=0, como producción sin declarar): {wape_ciega}")
        print(f"  Brecha real: {round(wape_ciega - wape_conocida, 2):+.2f} puntos")
        print()
        print("  Esa brecha es el costo REAL de no declarar promociones futuras --")
        print("  si es grande, vale la pena usar 'Eventos futuros' de forma activa;")
        print("  si es chica, el WAPE oficial ya es representativo de tu uso normal.")


if __name__ == "__main__":
    main()
