"""
prueba_sales_model.py

Entrena el modelo de prediccion de demanda con tus datos reales y
muestra todas las metricas relevantes.

Ejecutar desde la raiz del proyecto:
    python prueba_sales_model.py
"""

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
            "es_puente": v.es_puente,
            # Reemplazan a "es_quincena" (bandera binaria, poca señal
            # real) -- ver services/data_cleaner.py y data_generator.py.
            "dias_distancia_cobro": v.dias_distancia_cobro,
            "pico_comida_rapida":   int(v.pico_comida_rapida) if v.pico_comida_rapida is not None else 0,
            "fase_liquidez":        v.fase_liquidez,
            # Forzar a 0/1 numérico, nunca None -- si se deja tal cual,
            # pandas infiere dtype "object" en cuanto hay UNA sola fila
            # con NULL, y LightGBM rechaza esa columna.
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
            print(f"\n[!] No hay ventas guardadas para USER_ID={USER_ID} todavía.")
            print("    Sube tu CSV en /upload antes de correr este script.")
            return

        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis para este usuario -- configura el negocio primero.")
            return

        print("\n=== Entrenando SalesModel ===")
        modelo = SalesModel()
        metricas = modelo.entrenar(df, config=config, verbose=True)

        print("\n=== MÉTRICAS PRINCIPALES ===")
        print(f"  Estrategia:        {metricas.get('estrategia')}")
        print(f"  Días de historial: {metricas.get('dias_historial')}")

        holdout = metricas.get("holdout")
        if holdout:
            print("\n=== HOLDOUT ===")
            print(f"  WAPE:                 {holdout.get('wape_restoiq')}")
            print(f"  R²:                   {holdout.get('r2_restoiq')}")
            print(f"  Modelo ganador:       {holdout.get('modelo_ganador')}")
            print(f"  Mejora vs. baseline:  {holdout.get('mejora_pct')}%")
            print(f"  WAPE día+1:           {holdout.get('wape_dia_1')}")
            print(f"  WAPE por día:         {holdout.get('wape_por_dia_horizonte')}")
        else:
            print("\n[!] No se generó holdout.")

        importancias = metricas.get("importancias_top10")
        if importancias:
            print("\n=== TOP 10 IMPORTANCIAS ===")
            for k, v in importancias.items():
                print(f"  {k}: {v}")

        # Ranking COMPLETO (no solo el top 10), para ver dónde caen
        # exactamente dias_distancia_cobro / pico_comida_rapida /
        # fase_liquidez, aunque no entren en el top 10.
        print("\n=== RANKING COMPLETO (todas las features) ===")
        import pandas as pd_local
        gain_completo = pd_local.Series(
            modelo.modelos[1].feature_importance(importance_type="gain"),
            index=modelo.features,
        ).sort_values(ascending=False)
        total = gain_completo.sum()
        for i, (feat, val) in enumerate(gain_completo.items(), start=1):
            marca = "  <-- NUEVA" if feat in ("dias_distancia_cobro", "pico_comida_rapida", "fase_liquidez") else ""
            print(f"  {i:2d}. {feat:<25} {val:>12.1f}  ({val/total*100:5.2f}%){marca}")


if __name__ == "__main__":
    main()