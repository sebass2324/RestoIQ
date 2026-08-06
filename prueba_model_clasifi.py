from services.classification_model import ModeloAbastecimiento

# Cambia el 5 por el user_id real que quieras probar
modelo = ModeloAbastecimiento.cargar("ml_models/user_3_clasificacion.pkl")

resultado_normal = modelo.diagnostico_producto_nuevo("Bebidas", {
    "dia_semana": 4, "mes": 8, "semana_anio": 32,
    "es_finde": 0, "es_feriado": 0, "es_puente": 0, "es_quincena": 0
})
resultado_feriado = modelo.diagnostico_producto_nuevo("Bebidas", {
    "dia_semana": 5, "mes": 12, "semana_anio": 52,
    "es_finde": 1, "es_feriado": 1, "es_puente": 0, "es_quincena": 0
})
print("Normal:", resultado_normal)
print("Feriado+finde:", resultado_feriado)
