"""
obtener_clima_historico.py — corré esto desde la raíz de tu proyecto:

    python obtener_clima_historico.py

Descarga temperatura y lluvia REALES de Durán, Ecuador, HORA POR HORA,
para el rango de fechas de tu dataset — vía la API gratuita de
Open-Meteo (sin clave). Arma 4 variables por día, separando la franja
de la mañana (7am-3pm, negocio cerrado) de la noche (3pm-11pm, horario
real de El Chamo Burger):

    lluvia_manana_mm      — mm de lluvia acumulados entre 7am y 3pm
    temp_manana_promed    — temperatura promedio entre 7am y 3pm
    lluvia_nocturna_mm    — mm de lluvia acumulados entre 3pm y 11pm
    temp_nocturna_promed  — temperatura promedio entre 3pm y 11pm

Guarda el resultado en data/clima_duran.csv. Después se cruza con tu
dataset de ventas por la columna 'fecha' (ejemplo al final).
"""

import sys
sys.path.insert(0, ".")

import pandas as pd
from services.clima_service import obtener_clima_historico

# Ajustá este rango a las fechas reales de tu dataset
FECHA_DESDE = "2022-08-11"
FECHA_HASTA = "2026-08-10"


if __name__ == "__main__":
    print(f"Pidiendo clima de Durán, hora por hora, {FECHA_DESDE} a {FECHA_HASTA}...")
    print("(puede tardar más que el resumen diario — son muchos más datos)")

    df_clima = obtener_clima_historico(FECHA_DESDE, FECHA_HASTA)

    if df_clima is None:
        print("❌ No se pudo descargar el clima — revisá tu conexión a internet e intentá de nuevo.")
        sys.exit(1)

    df_clima.to_csv("data/clima_duran.csv", index=False)
    print(f"\n✅ Guardado: data/clima_duran.csv ({len(df_clima)} días)")
    print(df_clima.head(10).to_string(index=False))
    print()
    print(f"Días con lluvia en la mañana: {(df_clima['lluvia_manana_mm'] > 0).sum()} de {len(df_clima)}")
    print(f"Días con lluvia en la noche:  {(df_clima['lluvia_nocturna_mm'] > 0).sum()} de {len(df_clima)}")
    print(f"Temperatura promedio mañana: {df_clima['temp_manana_promed'].mean():.1f}°C")
    print(f"Temperatura promedio noche:  {df_clima['temp_nocturna_promed'].mean():.1f}°C")

    # ── Cómo cruzarlo con tu dataset de ventas ──
    # df_ventas = pd.read_csv("data/el_chamo_burger.csv")
    # df_ventas["fecha"] = pd.to_datetime(df_ventas["fecha"])
    # df_ventas = df_ventas.merge(df_clima, on="fecha", how="left")
    # df_ventas.to_csv("data/el_chamo_burger_con_clima.csv", index=False)
