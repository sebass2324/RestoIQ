"""
generar_dataset_elchamoburger.py — corré esto desde la raíz de tu proyecto:

    python generar_dataset_elchamoburger.py

Genera el dataset sintético de 4 años de El Chamo Burger (con el menú,
precios y patrones reales que ya cargamos en data_generator.py) y lo
guarda como CSV en data/el_chamo_burger.csv — listo para subir a la
app por /upload, o para usar directo en pruebas.

incluir_promociones=True: el negocio SÍ hace promociones puntuales por
producto (ver el 12% de probabilidad en data_generator.py), así que el
modelo puede aprender su efecto real.

incluir_descuentos=False (dejado explícito, aunque sea el valor por
defecto): el negocio no maneja descuentos porcentuales todavía -- no
tiene sentido generar una columna con datos que no reflejan cómo
opera el negocio hoy.
"""

from services.data_generator import DataGenerator

gen = DataGenerator(
    tipo_negocio="elchamoburger", meses=48,
    incluir_promociones=True,
    incluir_descuentos=False,
)
df = gen.guardar_csv("data/el_chamo_burger.csv")
gen.resumen(df)