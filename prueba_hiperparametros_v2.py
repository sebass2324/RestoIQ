"""
prueba_hiperparametros_v2.py

Búsqueda de hiperparámetros sobre el modelo COMPLETO actual (Direct
Forecasting + quincena rediseñada + lag_14 + promociones) -- nunca se
probó esta combinación completa junta antes.

Ejecutar desde la raiz del proyecto:
    python prueba_hiperparametros_v2.py

Advertencia: cada combinación entrena 14 modelos (Direct Forecasting)
+ la validación honesta -- son ~2 minutos por combinación. Con 9
combinaciones, esto tarda ~20-25 minutos. Déjalo corriendo en segundo
plano si hace falta.
"""

import time
import pandas as pd

from app import create_app
from models.venta import Venta
from models.configuracion_analisis import ConfiguracionAnalisis
import services.sales_model as sm

USER_ID = 6  # <-- ajusta al user_id correcto

BASE = {
    "objective": "regression_l1", "max_bin": 255, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 0.1,
    "lambda_l2": 0.1, "verbose": -1, "seed": 42,
}

GRID = [
    {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 20},  # baseline actual
    {"learning_rate": 0.03, "num_leaves": 63, "min_data_in_leaf": 20},
    {"learning_rate": 0.08, "num_leaves": 63, "min_data_in_leaf": 20},
    {"learning_rate": 0.05, "num_leaves": 31, "min_data_in_leaf": 20},
    {"learning_rate": 0.05, "num_leaves": 95, "min_data_in_leaf": 20},
    {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 10},
    {"learning_rate": 0.05, "num_leaves": 63, "min_data_in_leaf": 40},
    {"learning_rate": 0.03, "num_leaves": 31, "min_data_in_leaf": 40},
    {"learning_rate": 0.08, "num_leaves": 95, "min_data_in_leaf": 10},
]


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


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        if df.empty:
            print(f"[!] No hay ventas para USER_ID={USER_ID}. Sube tu CSV en /upload primero.")
            return
        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis -- configura el negocio primero.")
            return

        resultados = []
        for i, combo in enumerate(GRID, 1):
            sm.LGBM_PARAMS = {**BASE, **combo}
            t0 = time.time()
            modelo = sm.SalesModel()
            metricas = modelo.entrenar(df, config=config, verbose=False)
            dt = time.time() - t0
            holdout = metricas.get("holdout") or {}
            fila = {
                **combo,
                "wape": holdout.get("wape_restoiq"),
                "r2": holdout.get("r2_restoiq"),
                "mejora_pct": holdout.get("mejora_pct"),
                "segundos": round(dt, 1),
            }
            resultados.append(fila)
            print(f"[{i}/{len(GRID)}] {combo} -> WAPE={fila['wape']} R²={fila['r2']} "
                 f"mejora={fila['mejora_pct']}% ({dt:.0f}s)", flush=True)

        tabla = pd.DataFrame(resultados).sort_values("wape")
        print("\n=== TOP resultados (ordenado por WAPE) ===")
        print(tabla.to_string(index=False))
        tabla.to_csv("resultados_hiperparametros_v2.csv", index=False)
        print("\nGuardado en resultados_hiperparametros_v2.csv")


if __name__ == "__main__":
    main()
