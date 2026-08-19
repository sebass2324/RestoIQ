"""
prueba_ablacion_clima.py (v2 - robusta)

Compara 2 versiones, usando SIEMPRE tu propio SalesModel.entrenar()
(nunca reimplementa la logica interna, para no arriesgarse a adivinar
mal un nombre de metodo):

  A) Con las 4 variables de clima (como esta ahora)
  B) Sin lluvia_manana_mm ni temp_manana_promed en el DataFrame de
     entrada -- entrenar() nunca las ve, así que _preparar_features()
     no las incluye como feature, sin tocar nada del código real.

Ejecutar desde la raiz del proyecto:
    python prueba_ablacion_clima.py
"""

import pandas as pd

from app import create_app
from models.venta import Venta
from models.configuracion_analisis import ConfiguracionAnalisis
from services.sales_model import SalesModel

USER_ID = 14  # <-- ajusta al user_id CORRECTO, confirmado con el script de listado


def cargar_ventas(user_id):
    ventas = Venta.query.filter_by(user_id=user_id).all()
    print(f"[debug] Ventas encontradas: {len(ventas)}")
    filas = []
    for v in ventas:
        fila = {
            "fecha": v.fecha, "producto": v.producto, "cantidad": v.cantidad,
            "precio": v.precio, "total": v.total,
            "dia_semana": v.dia_semana, "mes": v.mes, "semana_anio": v.semana_anio,
            "es_finde": v.es_finde, "es_feriado": v.es_feriado,
            "es_puente": v.es_puente, "es_quincena": v.es_quincena,
            "promocion": v.promocion, "descuento_pct": v.descuento_pct,
            "es_evento_especial": v.es_evento_especial,
            "lluvia_manana_mm": v.lluvia_manana_mm,
            "temp_manana_promed": v.temp_manana_promed,
            "lluvia_nocturna_mm": v.lluvia_nocturna_mm,
            "temp_nocturna_promed": v.temp_nocturna_promed,
        }
        filas.append(fila)
    df = pd.DataFrame(filas)
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis para este usuario -- configura el negocio primero.")
            return

        print("\n=== A) Con las 4 variables de clima ===")
        modelo_a = SalesModel()
        metricas_a = modelo_a.entrenar(df, config=config, verbose=False)
        print(f"WAPE: {metricas_a.get('wape')}   |   Features: {modelo_a.features}")

        print("\n=== B) Sin lluvia_manana_mm ni temp_manana_promed ===")
        df_sin_manana = df.drop(columns=["lluvia_manana_mm", "temp_manana_promed"])
        modelo_b = SalesModel()
        metricas_b = modelo_b.entrenar(df_sin_manana, config=config, verbose=False)
        print(f"WAPE: {metricas_b.get('wape')}   |   Features: {modelo_b.features}")

        wape_a, wape_b = metricas_a.get("wape"), metricas_b.get("wape")
        if wape_a is not None and wape_b is not None:
            diferencia = round(wape_b - wape_a, 2)
            print(f"\nDiferencia (B - A): {diferencia} puntos")
            if abs(diferencia) < 1.0:
                print("-> Practicamente igual: la mañana no aportaba señal real,")
                print("   su importancia alta era colinealidad con la noche.")
            else:
                print("-> Diferencia notable: la mañana sí aporta algo real,")
                print("   vale la pena investigar más.")


if __name__ == "__main__":
    main()
