"""
prueba_ablacion_historico.py

Complementa prueba_importancia_permutacion.py: lag_1/lag_7/lag_28 y
rolling_* no se pueden "permutar" desde afuera (se calculan DENTRO de
entrenar(), no son columnas subibles). En su lugar, se prueba
quitando la FAMILIA completa de features via monkeypatch del modulo
(mismo truco ya usado para la busqueda de hiperparametros) y se
compara el WAPE resultante contra el modelo completo.

Ejecutar desde la raiz del proyecto:
    python prueba_ablacion_historico.py
"""

import pandas as pd

from app import create_app
from models.venta import Venta
from models.configuracion_analisis import ConfiguracionAnalisis
import services.sales_model as sm

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


def entrenar_y_medir(df, config, features_historicas, features_producto, etiqueta):
    sm.FEATURES_HISTORICAS = features_historicas
    sm.FEATURES_PRODUCTO   = features_producto
    modelo = sm.SalesModel()
    metricas = modelo.entrenar(df, config=config, verbose=False)
    wape = metricas.get("wape")
    print(f"{etiqueta:<45} WAPE={wape}   features={modelo.features}")
    return wape


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis -- configura el negocio primero.")
            return

        # Guardamos los originales para restaurarlos al final, por si
        # el proceso sigue vivo después (no debería afectar, pero es
        # una buena práctica no dejar el módulo "parchado").
        historicas_original = list(sm.FEATURES_HISTORICAS)
        producto_original   = list(sm.FEATURES_PRODUCTO)

        try:
            print("=== A) Modelo completo (como está ahora) ===")
            wape_completo = entrenar_y_medir(df, config, historicas_original, producto_original,
                                             "Completo")

            print("\n=== B) SIN lag/rolling (FEATURES_HISTORICAS vacío) ===")
            wape_sin_historico = entrenar_y_medir(df, config, [], producto_original,
                                                  "Sin lag/rolling")

            print("\n=== C) SIN producto/categoria (FEATURES_PRODUCTO vacío) ===")
            wape_sin_producto = entrenar_y_medir(df, config, historicas_original, [],
                                                 "Sin producto/categoria")

            print("\n=== D) SIN lag/rolling NI producto/categoria ===")
            wape_sin_ambos = entrenar_y_medir(df, config, [], [],
                                              "Sin historico ni producto")

            print("\n=== RESUMEN ===")
            print(f"  Completo:                  {wape_completo}")
            print(f"  Sin lag/rolling:           {wape_sin_historico}  (+{round(wape_sin_historico - wape_completo, 2)} puntos)")
            print(f"  Sin producto/categoria:    {wape_sin_producto}  (+{round(wape_sin_producto - wape_completo, 2)} puntos)")
            print(f"  Sin ninguno de los dos:    {wape_sin_ambos}  (+{round(wape_sin_ambos - wape_completo, 2)} puntos)")

        finally:
            sm.FEATURES_HISTORICAS = historicas_original
            sm.FEATURES_PRODUCTO   = producto_original


if __name__ == "__main__":
    main()
