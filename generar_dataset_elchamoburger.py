"""
generar_dataset_elchamoburger.py — corré esto desde la raíz de tu proyecto:

    python generar_dataset_elchamoburger.py

Genera el dataset sintético de 4 años de El Chamo Burger (con el menú,
precios y patrones reales que ya cargamos en data_generator.py) y lo
guarda como CSV en data/el_chamo_burger.csv — listo para subir a la
app por /upload, o para usar directo en pruebas.
"""

from services.data_generator import DataGenerator

gen = DataGenerator(tipo_negocio="elchamoburger", meses=48)
df = gen.guardar_csv("data/el_chamo_burger.csv")
gen.resumen(df)
