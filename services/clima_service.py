"""
services/clima_service.py

Clima real de Durán vía Open-Meteo . Se pide por
HORA (no el resumen diario) porque El Chamo Burger opera de 4pm a
11pm — el clima de la mañana (negocio cerrado) y el de la noche
(negocio operando) importan de forma distinta, así que se separan en
2 franjas en vez de promediar el día completo.

Mismo criterio de confiabilidad que storage_service.py: si la API no
responde, se degrada de forma segura — nunca rompe una predicción por
un problema de red externo.
"""

import time
import requests
import pandas as pd

LATITUD, LONGITUD = -2.17, -79.83  # Durán, Guayas, Ecuador
URL_HISTORICO  = "https://archive-api.open-meteo.com/v1/archive"
URL_PRONOSTICO = "https://api.open-meteo.com/v1/forecast"
TIMEOUT = 10  # segundos — nunca dejar una predicción colgada esperando al clima
REINTENTOS = 3
ESPERA_ENTRE_REINTENTOS = 3  # segundos, se duplica en cada intento (backoff simple)

HORA_INICIO_MANIANA, HORA_FIN_MANIANA = 7, 15   # 7:00 a 14:59
HORA_INICIO_NOCHE,   HORA_FIN_NOCHE   = 15, 23  # 15:00 a 22:59 (cierre del local)

_CACHE_HISTORICO = {}  # {(fecha_desde, fecha_hasta): DataFrame o None}
# El clima real NO depende de la semilla aleatoria del generador --
# scripts que regeneran el dataset varias veces (ej. pruebas de
# robustez con distintas semillas) pedirían el MISMO rango histórico
# repetidas veces sin este caché, arriesgando un límite de tasa de la
# API que se ve igual que una falla de red, pero en realidad es "ya
# pediste esto hace 10 segundos".


def _pedir_con_reintentos(url, params):
    """GET con reintentos y espera creciente -- las fallas de DNS/red
    intermitentes (comunes en redes hogareñas/corporativas) suelen
    resolverse solas en unos segundos; sin esto, un corte momentáneo
    tira todo el clima del dataset a valores por defecto sin avisar."""
    ultimo_error = None
    espera = ESPERA_ENTRE_REINTENTOS
    for intento in range(1, REINTENTOS + 1):
        try:
            resp = requests.get(url, params=params, timeout=TIMEOUT)
            resp.raise_for_status()
            return resp
        except Exception as e:
            ultimo_error = e
            if intento < REINTENTOS:
                print(f"  Intento {intento}/{REINTENTOS} falló ({e.__class__.__name__}), reintentando en {espera}s...")
                time.sleep(espera)
                espera *= 2
    raise ultimo_error


def _dividir_manana_noche(horas, temperaturas, precipitaciones) -> pd.DataFrame:
    """A partir de listas horarias, arma UNA fila por día con las 4
    variables: lluvia/temperatura de mañana y de noche, por separado."""
    df = pd.DataFrame({
        "fecha_hora": pd.to_datetime(horas),
        "temperatura": temperaturas,
        "precipitacion": precipitaciones,
    })
    df["fecha"] = df["fecha_hora"].dt.normalize()
    df["hora"]  = df["fecha_hora"].dt.hour

    manana = df[(df["hora"] >= HORA_INICIO_MANIANA) & (df["hora"] < HORA_FIN_MANIANA)]
    noche  = df[(df["hora"] >= HORA_INICIO_NOCHE)   & (df["hora"] < HORA_FIN_NOCHE)]

    agg_manana = manana.groupby("fecha").agg(
        lluvia_manana_mm=("precipitacion", "sum"),
        temp_manana_promed=("temperatura", "mean"),
    ).round(2)
    agg_noche = noche.groupby("fecha").agg(
        lluvia_nocturna_mm=("precipitacion", "sum"),
        temp_nocturna_promed=("temperatura", "mean"),
    ).round(2)

    resultado = agg_manana.join(agg_noche, how="outer").reset_index()
    return resultado


def obtener_clima_historico(fecha_desde, fecha_hasta):
    """Clima real ya ocurrido, hora por hora, para hacer más realista
    el dataset de entrenamiento. None si la API no responde."""
    clave = (str(fecha_desde), str(fecha_hasta))
    if clave in _CACHE_HISTORICO:
        return _CACHE_HISTORICO[clave]

    try:
        params = {
            "latitude": LATITUD, "longitude": LONGITUD,
            "start_date": str(fecha_desde), "end_date": str(fecha_hasta),
            "hourly": "temperature_2m,precipitation", "timezone": "auto",
        }
        resp = _pedir_con_reintentos(URL_HISTORICO, params)
        h = resp.json()["hourly"]
        resultado = _dividir_manana_noche(h["time"], h["temperature_2m"], h["precipitation"])
        _CACHE_HISTORICO[clave] = resultado
        return resultado
    except Exception as e:
        print(f"  No se pudo obtener clima histórico: {e}")
        _CACHE_HISTORICO[clave] = None  # no reintentar de red en cada llamada repetida dentro de la misma corrida
        return None


def obtener_pronostico(dias=16):
    """Pronóstico real hora por hora, para los próximos días (Open-Meteo
    da hasta 16 adelante). None si la API no responde."""
    try:
        params = {
            "latitude": LATITUD, "longitude": LONGITUD,
            "hourly": "temperature_2m,precipitation",
            "forecast_days": dias, "timezone": "auto",
        }
        resp = _pedir_con_reintentos(URL_PRONOSTICO, params)
        h = resp.json()["hourly"]
        return _dividir_manana_noche(h["time"], h["temperature_2m"], h["precipitation"])
    except Exception as e:
        print(f"  No se pudo obtener el pronóstico: {e}")
        return None