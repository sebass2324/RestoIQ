import pandas as pd
import numpy as np
from faker import Faker
from datetime import datetime, timedelta
import random

fake = Faker("es_ES")
np.random.seed(42)
random.seed(42)


# ════════════════════════════════════════════════════════════
# FERIADOS ECUADOR (fijos + móviles aproximados 2022-2026)
# ════════════════════════════════════════════════════════════

FERIADOS_ECUADOR = set([
    # Fijos (se repiten cada año)
    *[f"{y}-01-01" for y in range(2022, 2027)],  # Año Nuevo
    *[f"{y}-05-01" for y in range(2022, 2027)],  # Día del Trabajo
    *[f"{y}-08-10" for y in range(2022, 2027)],  # Primer Grito Independencia
    *[f"{y}-10-09" for y in range(2022, 2027)],  # Independencia de Guayaquil
    *[f"{y}-11-02" for y in range(2022, 2027)],  # Día de Difuntos
    *[f"{y}-11-03" for y in range(2022, 2027)],  # Independencia de Cuenca
    *[f"{y}-12-25" for y in range(2022, 2027)],  # Navidad
    # Móviles aproximados (Carnaval, Semana Santa, Batalla de Pichincha)
    "2022-02-28", "2022-03-01",   # Carnaval 2022
    "2022-04-15", "2022-04-16",   # Semana Santa 2022
    "2022-05-24",                  # Batalla de Pichincha 2022
    "2023-02-20", "2023-02-21",   # Carnaval 2023
    "2023-04-07", "2023-04-08",   # Semana Santa 2023
    "2023-05-26",                  # Batalla de Pichincha 2023
    "2024-02-12", "2024-02-13",   # Carnaval 2024
    "2024-03-29", "2024-03-30",   # Semana Santa 2024
    "2024-05-24",                  # Batalla de Pichincha 2024
    "2025-03-03", "2025-03-04",   # Carnaval 2025
    "2025-04-18", "2025-04-19",   # Semana Santa 2025
    "2025-05-26",                  # Batalla de Pichincha 2025
    "2026-02-16", "2026-02-17",   # Carnaval 2026
    "2026-04-03", "2026-04-04",   # Semana Santa 2026
    "2026-05-25",                  # Batalla de Pichincha 2026
])


# ════════════════════════════════════════════════════════════
# PERFILES DE NEGOCIO
# ════════════════════════════════════════════════════════════

PERFILES = {

    "restaurante": {
        "nombre": "Restaurante / Comida rápida",
        "productos": [
            ("Hamburguesa Clásica",  4.50, 20, None),
            ("Hamburguesa Especial", 6.50, 12, None),
            ("Pizza Familiar",      12.00,  9, "finde"),
            ("Pizza Personal",       7.00, 11, "finde"),
            ("Papas Fritas",         2.50, 25, None),
            ("Papas con Queso",      3.50, 14, None),
            ("Pollo Broaster",       6.50, 13, None),
            ("Alitas BBQ",           8.00,  8, "finde"),
            ("Gaseosa",              1.50, 30, None),
            ("Jugo Natural",         2.00, 18, None),
            ("Ensalada César",       5.00,  7, "semana"),
            ("Hot Dog",              3.00, 10, None),
            ("Brownie",              2.50,  9, None),
        ],
        "factor_dia": {0:0.75, 1:0.80, 2:0.85, 3:0.90, 4:1.10, 5:1.55, 6:1.40},
        "factor_mes": {1:0.85, 2:0.90, 3:1.00, 4:1.05, 5:1.10, 6:1.20,
                       7:1.25, 8:1.15, 9:0.95, 10:1.00, 11:1.10, 12:1.30},
    },

    "panaderia": {
        "nombre": "Panadería / Pastelería",
        "productos": [
            ("Pan de Sal",           0.20, 80, "semana"),
            ("Pan de Dulce",         0.20, 80, "semana"),
            ("Pan Mixto",            0.20, 80, "semana"),
            ("Pan Integral",         0.30, 40, "semana"),
            ("Croissant",            1.20, 25, None),
            ("Empanada de Queso",    0.80, 30, None),
            ("Empanada de Pollo",    1.00, 20, None),
            ("Torta de Chocolate",   3.50,  8, "finde"),
            ("Torta de Vainilla",    3.00,  7, "finde"),
            ("Muffin",               1.50, 15, None),
            ("Galletas x6",          2.00, 12, None),
            ("Donut",                1.00, 18, None),
            ("Palito de Queso",      0.50, 35, None),
            ("Café",                 1.50, 22, "semana"),
            ("Chocolate Caliente",   1.80, 15, None),
            ("Cheesecake",           4.00,  6, "finde"),
            ("Rol de Canela",        1.20, 10, None),
        ],
        "factor_dia": {0:1.10, 1:0.90, 2:0.85, 3:0.90, 4:1.00, 5:1.20, 6:1.40},
        "factor_mes": {1:0.90, 2:0.95, 3:1.00, 4:1.00, 5:1.05, 6:1.00,
                       7:0.95, 8:0.95, 9:1.00, 10:1.05, 11:1.15, 12:1.50},
    },

    "cafeteria": {
        "nombre": "Cafetería",
        "productos": [
            ("Café Americano",       1.50, 35, "semana"),
            ("Cappuccino",           2.50, 20, "semana"),
            ("Latte",                2.80, 18, "semana"),
            ("Espresso",             1.20, 15, "semana"),
            ("Té",                   1.00, 12, None),
            ("Chocolate Caliente",   2.00, 10, None),
            ("Sándwich de Jamón",    3.00, 14, "semana"),
            ("Sándwich Vegetal",     3.50,  8, "semana"),
            ("Tostada con Queso",    2.00, 12, "semana"),
            ("Muffin de Arándanos",  2.00, 10, None),
            ("Croissant",            1.80, 15, None),
            ("Ensalada del Día",     4.50,  7, "semana"),
            ("Jugo Natural",         2.50,  9, None),
            ("Agua con Gas",         1.00, 10, None),
        ],
        "factor_dia": {0:1.20, 1:1.10, 2:1.00, 3:1.05, 4:1.10, 5:0.70, 6:0.50},
        "factor_mes": {1:1.00, 2:1.00, 3:1.05, 4:1.05, 5:1.10, 6:0.80,
                       7:0.75, 8:0.80, 9:1.10, 10:1.10, 11:1.05, 12:0.90},
    },

    "heladeria": {
        "nombre": "Heladería",
        "productos": [
            ("Helado Simple",        1.50, 25, "verano"),
            ("Helado Doble",         2.50, 18, "verano"),
            ("Sundae",               3.50, 12, "verano"),
            ("Malteada",             4.00, 10, "verano"),
            ("Banana Split",         5.00,  7, "verano"),
            ("Copa de Frutos",       4.50,  6, "verano"),
            ("Paleta",               1.00, 20, "verano"),
            ("Yogurt Helado",        2.00, 15, None),
            ("Crepe",                3.00,  8, None),
            ("Waffle con Helado",    4.00,  9, "verano"),
            ("Granizado",            1.50, 22, "verano"),
            ("Agua",                 1.00, 10, None),
        ],
        "factor_dia": {0:0.70, 1:0.75, 2:0.80, 3:0.85, 4:1.00, 5:1.60, 6:1.50},
        "factor_mes": {1:0.40, 2:0.45, 3:0.60, 4:0.80, 5:1.10, 6:1.40,
                       7:1.50, 8:1.40, 9:1.00, 10:0.70, 11:0.50, 12:0.45},
    },

    "pizzeria": {
        "nombre": "Pizzería",
        "productos": [
            ("Pizza Margarita",      8.00, 12, "finde"),
            ("Pizza Pepperoni",     10.00, 15, "finde"),
            ("Pizza Hawaiana",       9.50, 10, "finde"),
            ("Pizza Vegetariana",    9.00,  7, "finde"),
            ("Pizza BBQ Pollo",     11.00,  9, "finde"),
            ("Pizza 4 Quesos",      10.50,  8, "finde"),
            ("Calzone",              8.50,  6, None),
            ("Pasta Boloñesa",       7.00,  8, None),
            ("Pasta Carbonara",      7.50,  7, None),
            ("Ensalada Caesar",      5.00,  5, "semana"),
            ("Pan de Ajo",           2.50, 14, None),
            ("Gaseosa",              1.50, 20, None),
            ("Cerveza",              2.50, 12, "finde"),
            ("Tiramisú",             3.50,  6, None),
        ],
        "factor_dia": {0:0.60, 1:0.65, 2:0.70, 3:0.80, 4:1.20, 5:1.70, 6:1.50},
        "factor_mes": {1:0.80, 2:0.85, 3:0.95, 4:1.00, 5:1.05, 6:1.15,
                       7:1.20, 8:1.10, 9:1.00, 10:1.00, 11:1.10, 12:1.25},
    },

    "generico": {
        "nombre": "Negocio genérico de comida",
        "productos": [
            ("Producto A", 5.00, 20, None),
            ("Producto B", 3.50, 15, None),
            ("Producto C", 8.00, 10, "finde"),
            ("Producto D", 2.00, 25, None),
            ("Producto E", 6.00, 12, "semana"),
            ("Producto F", 1.50, 30, None),
            ("Producto G", 4.00, 18, None),
            ("Producto H", 7.00,  8, "finde"),
        ],
        "factor_dia": {0:0.80, 1:0.85, 2:0.90, 3:0.95, 4:1.05, 5:1.40, 6:1.30},
        "factor_mes": {1:0.90, 2:0.92, 3:0.95, 4:1.00, 5:1.05, 6:1.10,
                       7:1.15, 8:1.10, 9:1.00, 10:1.00, 11:1.05, 12:1.20},
    },

    # ── El Chamo Burger — caso de estudio real de la tesis ──
    # Durán, Guayas. Menú, precios y patrones de venta salen de la
    # entrevista al propietario (Anexo — Instrumento de Recolección
    # de Datos), no son inventados. demanda_base por producto refleja
    # lo que dijo el dueño: hamburguesas y pepitos "tienen más salida",
    # cachapas/arepas especiales rotan menos.
    "elchamoburger": {
        "nombre": "El Chamo Burger (Durán)",
        "productos": [
            # Hamburguesas — Clásicas
            ("La Sencilla", 2.00, 10, None),
            ("La Especial", 2.25, 6, None),
            ("La Sub-Especial", 2.75, 3, None),
            ("La Completa", 3.00, 8, None),
            # Hamburguesas — Otras Protagonistas
            ("Pollo Sencilla", 2.75, 3, None),
            ("Pollo Completa", 3.25, 2, None),
            ("Lomo Sencilla", 3.25, 6, None),
            ("Lomo Completa", 3.75, 2, None),
            ("Chancho Sencilla", 2.75, 2, None),
            ("Chancho Completa", 3.25, 2, None),
            # Hamburguesas — Las Gigantes de la Casa
            ("Doble Carne", 3.75, 2, None),
            ("La Monster", 4.25, 1, None),
            ("La Trifásica", 6.00, 1, None),
            # Pepito
            ("Pepito Sencillo", 7.00, 6, None),
            ("Pepito Completo", 8.50, 5, None),
            ("Pepito Pollo", 8.00, 2, None),
            ("Pepito Carne", 9.00, 2, None),
            ("Pepito Monster", 11.00, 1, None),
            ("Monstrico", 3.50, 2, None),
            # Sabor Ecuatoriano — Papas
            ("Salchipapa", 2.00, 3, None),
            ("Papipollo", 2.50, 2, None),
            ("Papichancho", 3.50, 2, None),
            ("Papaparillera", 8.00, 1, None),
            ("Porción de Papa", 1.50, 2, None),
            # Sabor Ecuatoriano — Bandejitas
            ("Bandejita Con Todo", 3.50, 1, None),
            ("Bandejita La Bestia", 5.00, 1, None),
            # Arepas Venecas
            ("Arepa Dominó", 2.50, 1, None),
            ("Arepa Catira", 3.50, 1, None),
            ("Arepa Reina Pepiada", 3.50, 1, None),
            ("Arepa Rumbera", 3.50, 5, None),
            ("Arepa Pelúa", 3.00, 1, None),
            ("Arepa Llanera", 3.50, 1, None),
            ("Arepa Sifrina", 3.00, 1, None),
            ("Arepa Perico", 2.50, 1, None),
            # Patacones Burguer
            ("Patacón Carne Mechada", 4.00, 1, None),
            ("Patacón Pollo Mechado", 4.00, 1, None),
            ("Patacón Carne Hamburguesa", 3.50, 1, None),
            ("Patacón Chancho", 4.50, 1, None),
            ("Patacón Parrillero", 4.50, 1, None),
            # Cachapas — rotación baja, confirmado por el dueño
            ("Cachapa Viuda", 1.50, 1, None),
            ("Cachapa Sencilla", 4.50, 1, None),
            ("Cachapa Chorizo", 6.50, 1, None),
            ("Cachapa Pollo Mechado", 6.50, 1, None),
            ("Cachapa Tocineta", 7.00, 1, None),
            ("Cachapa Chancho", 7.50, 1, None),
            ("Cachapa Carne Mechada", 6.50, 1, None),
            ("Cachapa Parrillera", 11.50, 1, None),
            ("Cachapa Llanera", 7.50, 1, None),
            ("Cachapa Chuleta Ahumada", 8.00, 1, None),
            # Hotdog Jumbo
            ("Jumbo Sencillo", 1.50, 8, None),
            ("Jumbo Especial", 2.00, 6, None),
            ("Jumbo Mega", 2.50, 2, None),
            ("Choriperro", 2.00, 2, None),
            # Bebidas — precios REALES del menú (MENU.pdf). Demanda base
            # calibrada como complemento de la comida (sin combos, según
            # la entrevista, pregunta 8): suman ~32% del volumen total de
            # comida, y ninguna bebida individual supera al producto más
            # vendido del local (La Sencilla, demanda_base=10) -- antes
            # el agua empataba con la hamburguesa top, lo cual no tenía
            # sentido de negocio para un local que no es una tienda de
            # bebidas.
            ("Agua 300ml",             0.50, 6, None),
            ("Jugo Natural Maracuyá",  0.75, 3, None),
            ("Jugo Natural Mora",      0.75, 3, None),
            ("Nutrimalta 350ml",       0.50, 4, None),
            ("Nutrimalta 550ml",       1.00, 2, None),
            ("Cola Coca-Cola",         0.50, 8, None),
            ("Cola Sprite",            0.50, 4, None),
            ("Cola Fanta",             0.50, 4, None),
            ("Cola Inca",              0.50, 3, None),
            ("Cola Tropical",          0.50, 2, None),
        ],
        # Ranking real del dueño: sábado > viernes > domingo > martes >
        # miércoles > jueves > lunes. "El doble" entre mejor y peor día
        # → sábado (1.40) es exactamente el doble de lunes (0.70).
        "factor_dia": {0:0.70, 1:0.80, 2:0.77, 3:0.75, 4:1.15, 5:1.40, 6:1.10},
        # Picos reales que mencionó el dueño: día de la madre (mayo),
        # fiestas de Durán (octubre), diciembre. El resto del año, parejo.
        "factor_mes": {1:0.90, 2:0.90, 3:0.92, 4:0.95, 5:1.20, 6:0.95,
                       7:0.95, 8:0.95, 9:0.95, 10:1.20, 11:0.95, 12:1.30},
        # Calibración de volumen: la suma de demanda_base de todos los
        # productos, ya ponderada por factor_dia/factor_mes, generaba
        # ~182 u/día en promedio. La entrevista dice explícitamente
        # "entre 300 y 350 unidades aproximadamente" en un día normal
        # (pregunta de Volumen aproximado). 1.8x lleva el promedio a
        # ~327 u/día, dentro del rango declarado por el dueño, sin
        # tener que re-escribir a mano los 63 demanda_base individuales
        # (las proporciones relativas entre productos se conservan).
        "escala_volumen": 1.8,
    },
}


# ════════════════════════════════════════════════════════════
# CATEGORÍAS POR PRODUCTO (para el modelo de clasificación)
# Agrupan productos afines → permiten generalizar a productos nuevos.
# ════════════════════════════════════════════════════════════

CATEGORIAS = {
    "restaurante": {
        "Hamburguesa Clásica": "Platos", "Hamburguesa Especial": "Platos",
        "Pizza Familiar": "Platos", "Pizza Personal": "Platos",
        "Pollo Broaster": "Platos", "Hot Dog": "Platos",
        "Papas Fritas": "Acompañamientos", "Papas con Queso": "Acompañamientos",
        "Alitas BBQ": "Acompañamientos", "Ensalada César": "Acompañamientos",
        "Gaseosa": "Bebidas", "Jugo Natural": "Bebidas",
        "Brownie": "Postres",
    },
    "panaderia": {
        "Pan de Sal": "Panes", "Pan de Dulce": "Panes",
        "Pan Mixto": "Panes", "Pan Integral": "Panes",
        "Croissant": "Bollería", "Empanada de Queso": "Bollería",
        "Empanada de Pollo": "Bollería", "Muffin": "Bollería",
        "Donut": "Bollería", "Palito de Queso": "Bollería",
        "Rol de Canela": "Bollería",
        "Torta de Chocolate": "Pastelería", "Torta de Vainilla": "Pastelería",
        "Galletas x6": "Pastelería", "Cheesecake": "Pastelería",
        "Café": "Bebidas", "Chocolate Caliente": "Bebidas",
    },
    "cafeteria": {
        "Café Americano": "Bebidas", "Cappuccino": "Bebidas",
        "Latte": "Bebidas", "Espresso": "Bebidas", "Té": "Bebidas",
        "Chocolate Caliente": "Bebidas", "Jugo Natural": "Bebidas",
        "Agua con Gas": "Bebidas",
        "Sándwich de Jamón": "Comida", "Sándwich Vegetal": "Comida",
        "Tostada con Queso": "Comida", "Ensalada del Día": "Comida",
        "Muffin de Arándanos": "Panadería", "Croissant": "Panadería",
    },
    "heladeria": {
        "Helado Simple": "Helados", "Helado Doble": "Helados",
        "Paleta": "Helados", "Yogurt Helado": "Helados", "Granizado": "Helados",
        "Sundae": "Especiales", "Malteada": "Especiales",
        "Banana Split": "Especiales", "Copa de Frutos": "Especiales",
        "Crepe": "Especiales", "Waffle con Helado": "Especiales",
        "Agua": "Bebidas",
    },
    "pizzeria": {
        "Pizza Margarita": "Pizzas", "Pizza Pepperoni": "Pizzas",
        "Pizza Hawaiana": "Pizzas", "Pizza Vegetariana": "Pizzas",
        "Pizza BBQ Pollo": "Pizzas", "Pizza 4 Quesos": "Pizzas", "Calzone": "Pizzas",
        "Pasta Boloñesa": "Pastas", "Pasta Carbonara": "Pastas",
        "Ensalada Caesar": "Entradas", "Pan de Ajo": "Entradas",
        "Gaseosa": "Bebidas", "Cerveza": "Bebidas",
        "Tiramisú": "Postres",
    },
    "generico": {
        "Producto A": "Regulares", "Producto B": "Regulares",
        "Producto D": "Regulares", "Producto F": "Regulares", "Producto G": "Regulares",
        "Producto C": "Especiales", "Producto E": "Especiales", "Producto H": "Especiales",
    },

    "elchamoburger": {
        "La Sencilla": "Hamburguesas", "La Especial": "Hamburguesas",
        "La Sub-Especial": "Hamburguesas", "La Completa": "Hamburguesas",
        "Pollo Sencilla": "Hamburguesas", "Pollo Completa": "Hamburguesas",
        "Lomo Sencilla": "Hamburguesas", "Lomo Completa": "Hamburguesas",
        "Chancho Sencilla": "Hamburguesas", "Chancho Completa": "Hamburguesas",
        "Doble Carne": "Hamburguesas", "La Monster": "Hamburguesas", "La Trifásica": "Hamburguesas",
        "Pepito Sencillo": "Pepito", "Pepito Completo": "Pepito", "Pepito Pollo": "Pepito",
        "Pepito Carne": "Pepito", "Pepito Monster": "Pepito", "Monstrico": "Pepito",
        "Salchipapa": "Papas", "Papipollo": "Papas", "Papichancho": "Papas",
        "Papaparillera": "Papas", "Porción de Papa": "Papas",
        "Bandejita Con Todo": "Bandejitas", "Bandejita La Bestia": "Bandejitas",
        "Arepa Dominó": "Arepas", "Arepa Catira": "Arepas", "Arepa Reina Pepiada": "Arepas",
        "Arepa Rumbera": "Arepas", "Arepa Pelúa": "Arepas", "Arepa Llanera": "Arepas",
        "Arepa Sifrina": "Arepas", "Arepa Perico": "Arepas",
        "Patacón Carne Mechada": "Patacones", "Patacón Pollo Mechado": "Patacones",
        "Patacón Carne Hamburguesa": "Patacones", "Patacón Chancho": "Patacones",
        "Patacón Parrillero": "Patacones",
        "Cachapa Viuda": "Cachapas", "Cachapa Sencilla": "Cachapas", "Cachapa Chorizo": "Cachapas",
        "Cachapa Pollo Mechado": "Cachapas", "Cachapa Tocineta": "Cachapas",
        "Cachapa Chancho": "Cachapas", "Cachapa Carne Mechada": "Cachapas",
        "Cachapa Parrillera": "Cachapas", "Cachapa Llanera": "Cachapas",
        "Cachapa Chuleta Ahumada": "Cachapas",
        "Jumbo Sencillo": "Hotdogs", "Jumbo Especial": "Hotdogs",
        "Jumbo Mega": "Hotdogs", "Choriperro": "Hotdogs",
        "Agua 300ml": "Bebidas", "Jugo Natural Maracuyá": "Bebidas",
        "Jugo Natural Mora": "Bebidas", "Nutrimalta 350ml": "Bebidas",
        "Nutrimalta 550ml": "Bebidas", "Cola Coca-Cola": "Bebidas",
        "Cola Sprite": "Bebidas", "Cola Fanta": "Bebidas",
        "Cola Inca": "Bebidas", "Cola Tropical": "Bebidas",
    },
}


# ════════════════════════════════════════════════════════════
# CLÁSICO DEL ASTILLERO (Barcelona SC vs. Emelec) — solo El Chamo Burger
# ════════════════════════════════════════════════════════════
CLASICOS_ASTILLERO = {
    "2024-10-20": "tarde",   # 17:00, Barcelona vs. Emelec (primicias.ec)
    "2025-09-14": "tarde",   # 17:30, Emelec vs. Barcelona (primicias.ec)
    "2026-03-08": "noche",   # 18:00, Barcelona vs. Emelec (espn)
    "2026-07-12": "noche",   # 18:10, Emelec vs. Barcelona (ecuavisa)
}
F_CLASICO_TARDE = 1.30
F_CLASICO_NOCHE = 0.80

DISPERSION_NB_BASE = 25


# ════════════════════════════════════════════════════════════
# HELPERS DE FEATURES TEMPORALES
# ════════════════════════════════════════════════════════════

def _features_temporales(fecha_dt: pd.Timestamp) -> dict:
    fecha_str = fecha_dt.strftime("%Y-%m-%d")
    dia_semana = fecha_dt.weekday()
    es_finde   = int(dia_semana in [5, 6])
    es_feriado = int(fecha_str in FERIADOS_ECUADOR)
    ayer    = (fecha_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    maniana = (fecha_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    es_puente = int(ayer in FERIADOS_ECUADOR or maniana in FERIADOS_ECUADOR)
    dia_mes = fecha_dt.day
    dias_en_mes = fecha_dt.days_in_month

    # Distancia al día de cobro más cercano (15 o fin/inicio de mes) --
    # variable CONTINUA, no binaria. Le da a LightGBM puntos de corte
    # reales para encontrar dónde empieza a importar la cercanía al
    # pago, en vez de forzar una ventana fija de antemano. Maneja
    # meses de 28 a 31 días correctamente.
    dist_15 = abs(dia_mes - 15)
    dist_fin = min(dia_mes - 1, dias_en_mes - dia_mes)
    dias_distancia_cobro = min(dist_15, dist_fin)

    # Pico: cobro que coincide con fin de semana (viernes/sábado) --
    # incluye el caso de pago adelantado cuando el 15/30 cae en fin de
    # semana (13/14/28/29 + viernes).
    es_ventana_pago = (
        dia_mes in (15, 16, dias_en_mes, 1)
        or (dia_mes in (13, 14, dias_en_mes - 2, dias_en_mes - 1) and dia_semana == 4)
    )
    es_finde_alto = dia_semana in (4, 5)
    pico_comida_rapida = int(es_ventana_pago and es_finde_alto)

    # Fase de liquidez: 3=cobro activo, 2=post-pago inmediato,
    # 1=neutral, 0=escasez real (días antes del pago, sin adelanto).
    if pico_comida_rapida == 1 or dia_mes in (15, dias_en_mes, 1):
        fase_liquidez = 3
    elif dia_mes in (2, 16, 17):
        fase_liquidez = 2
    elif dia_mes in (13, 14, dias_en_mes - 2, dias_en_mes - 1) and pico_comida_rapida == 0:
        fase_liquidez = 0
    else:
        fase_liquidez = 1

    return {
        "dia_semana":   dia_semana,
        "mes":          fecha_dt.month,
        "semana_anio":  fecha_dt.isocalendar().week,
        "es_finde":     es_finde,
        "es_feriado":   es_feriado,
        "es_puente":    es_puente,
        "dias_distancia_cobro": dias_distancia_cobro,
        "pico_comida_rapida":   pico_comida_rapida,
        "fase_liquidez":        fase_liquidez,
    }


PROMO_PROBABILIDAD = 0.12
PROMO_BOOST_RANGO  = (1.25, 1.70)

DESCUENTO_PROBABILIDAD    = 0.15
DESCUENTO_PCT_OPCIONES    = [10, 15, 20, 25, 30]
DESCUENTO_BOOST_POR_PUNTO = 0.008


class DataGenerator:
    def __init__(self, tipo_negocio="restaurante", años=2, meses=None,
                 incluir_promociones=False, incluir_descuentos=False):
        if tipo_negocio not in PERFILES:
            raise ValueError(
                f"Tipo '{tipo_negocio}' no reconocido. "
                f"Opciones: {list(PERFILES.keys())}"
            )
        self.perfil = PERFILES[tipo_negocio]
        self.tipo   = tipo_negocio
        self.meses = meses if meses is not None else round(años * 12)
        self.años  = self.meses / 12
        self.incluir_promociones = incluir_promociones
        self.incluir_descuentos  = incluir_descuentos

    def generar(self, sucio=False) -> pd.DataFrame:
        fecha_inicio = datetime.today() - timedelta(days=round(self.meses * 30.44))
        fecha_fin    = datetime.today() - timedelta(days=1)
        fechas       = pd.date_range(fecha_inicio, fecha_fin, freq="D")

        productos  = self.perfil["productos"]
        factor_dia = self.perfil["factor_dia"]
        factor_mes = self.perfil["factor_mes"]

        clima_por_fecha = {}
        if self.tipo == "elchamoburger":
            try:
                from services.clima_service import obtener_clima_historico
                df_clima = obtener_clima_historico(
                    fecha_inicio.strftime("%Y-%m-%d"), fecha_fin.strftime("%Y-%m-%d")
                )
                if df_clima is not None:
                    # Forzar a datetime real, sin importar si el servicio
                    # devuelve la columna "fecha" como texto o ya como
                    # Timestamp -- antes, si venía como texto, row["fecha"]
                    # no tenía .strftime() y el except de abajo lo tragaba
                    # en silencio, dejando clima_por_fecha vacío sin avisar.
                    df_clima = df_clima.copy()
                    df_clima["fecha"] = pd.to_datetime(df_clima["fecha"])
                    clima_por_fecha = {
                        row["fecha"].strftime("%Y-%m-%d"): row.to_dict()
                        for _, row in df_clima.iterrows()
                    }
            except Exception as e:
                # Antes: "except Exception: pass" -- tragaba CUALQUIER
                # error sin avisar (fue lo que ocultó este bug). Ahora se
                # imprime para poder diagnosticar, pero NUNCA rompe la
                # generación del dataset (clima_por_fecha simplemente
                # queda vacío y el resto sigue funcionando sin clima).
                print(f"[data_generator] No se pudo obtener clima: {e}")

        choque_anterior = {nombre: 0.0 for nombre, _, _, _ in self.perfil["productos"]}
        PESO_MOMENTUM  = 0.15
        DECAIMIENTO    = 0.35

        demanda_base_promedio = np.mean([d for _, _, d, _ in self.perfil["productos"]])
        dispersion_por_producto = {
            # La dispersión se calcula sobre la PROPORCIÓN entre
            # productos (demanda_base / promedio), que no cambia al
            # escalar el volumen total -- por eso no se multiplica acá
            # por escala_volumen, solo abajo en demanda_esperada.
            nombre: max(2.0, DISPERSION_NB_BASE * (demanda_base / demanda_base_promedio) ** 0.5)
            for nombre, _, demanda_base, _ in self.perfil["productos"]
        }
        escala_volumen = self.perfil.get("escala_volumen", 1.0)

        filas = []
        for i, fecha in enumerate(fechas):
            # Tendencia de crecimiento orgánico ~0.5% mensual, compuesta
            # (no lineal) -- sobre 3 años equivale a ~19.7% acumulado,
            # bastante más modesto que un 20%/año lineal, y más realista
            # para un negocio pequeño ya establecido (4 años operando,
            # según la entrevista, no en fase de expansión acelerada).
            meses_transcurridos = i / 30.44
            tendencia = 1.005 ** meses_transcurridos
            f_dia           = factor_dia[fecha.weekday()]
            f_mes           = factor_mes[fecha.month]
            evento_especial = random.uniform(1.3, 2.0) if random.random() < 0.05 else 1.0

            feats = _features_temporales(fecha)
            clima_dia = clima_por_fecha.get(fecha.strftime("%Y-%m-%d"), {})

            f_clima = 1.0
            if self.tipo == "elchamoburger" and clima_dia:
                # Solo clima NOCTURNO (horario real de operación,
                # 4pm-11pm), calibrado con el propietario. Antes también
                # se incluía lluvia/temperatura de MAÑANA, pero una
                # prueba de ablación sobre el modelo real (comparar WAPE
                # con y sin esas 2 variables) mostró una diferencia de
                # solo 0.04 puntos -- no aportaban señal causal propia,
                # su alta importancia aparente era colinealidad con la
                # noche (correlación 0.38). Se simplifica para no
                # simular un efecto ya comprobado irrelevante.
                # Calibrado según la entrevista: "en lluvia suele
                # afectar negativamente pero no tanto, pero sí afecta"
                # -- un efecto real y medible, pero moderado, no un
                # colapso de la demanda. Los valores anteriores (hasta
                # -42.5% en lluvia fuerte) generaban una caída de ~52%
                # en los días más lluviosos del histórico real de
                # Durán, muy por encima de "no tanto".
                lluvia_noche = clima_dia.get("lluvia_nocturna_mm") or 0
                if lluvia_noche > 25:
                    f_clima *= 0.80   # lluvia extrema: ~-20%
                elif lluvia_noche > 20:
                    f_clima *= 0.85   # lluvia fuerte: ~-15%
                elif lluvia_noche > 15:
                    f_clima *= 0.90   # lluvia moderada: ~-10%
                elif lluvia_noche > 5:
                    f_clima *= 0.95   # lluvia ligera continua: ~-5%
                elif lluvia_noche > 2:
                    f_clima *= 0.995  # llovizna leve: ~-0.5%


                temp_noche = clima_dia.get("temp_nocturna_promed")
                if temp_noche is not None:
    # Calor sofocante nocturno: reduce el apetito presencial en locales sin AC
                    if temp_noche >= 28.0:
                         f_clima *= 0.93  
        
    # Noche fresca ideal en la costa: incentiva el consumo y la salida de clientes
                    elif 22.0 <= temp_noche <= 24.5:
                        f_clima *= 1.04  # Bono del +4% por clima agradable
        
    # "Frío" extremo para la costa (Madrugadas excepcionales o frentes fríos)
                    elif temp_noche < 21.0:
                        f_clima *= 0.97  # Ligero castigo del 3% porque la gente prefiere quedarse dentro



            f_feriado = 1.0
            if self.tipo == "elchamoburger":
                f_feriado = 1.0
            elif feats["es_feriado"] or feats["es_finde"]:
                f_feriado = 1.25 if self.tipo in ("restaurante", "pizzeria", "heladeria") else 0.80

            # Confirmado con el propietario: 25-30%, motor real de gasto.
            # Ahora impulsado por fase_liquidez (3=cobro activo/pico,
            # 2=post-pago inmediato, 0=escasez), no por una bandera
            # binaria -- mismo efecto de negocio, mejor estructurado
            # para que el árbol encuentre el patrón.
            mapa_f_quincena = {3: 1.28, 2: 1.15, 1: 1.0, 0: 0.90}
            f_quincena = mapa_f_quincena[feats["fase_liquidez"]]

            f_clasico = 1.0
            if self.tipo == "elchamoburger":
                tipo_partido = CLASICOS_ASTILLERO.get(fecha.strftime("%Y-%m-%d"))
                if tipo_partido == "tarde":
                    f_clasico = F_CLASICO_TARDE
                elif tipo_partido == "noche":
                    f_clasico = F_CLASICO_NOCHE

            for nombre, precio, demanda_base, temporada in productos:
                demanda_base = demanda_base * escala_volumen

                f_temp = 1.0
                if temporada == "finde"  and fecha.weekday() in [4, 5, 6]:
                    f_temp = 1.5
                elif temporada == "semana" and fecha.weekday() in [0, 1, 2, 3]:
                    f_temp = 1.3
                elif temporada == "verano" and fecha.month in [6, 7, 8]:
                    f_temp = 1.6

                promocion_val = 0
                f_promo = 1.0
                if self.incluir_promociones and random.random() < PROMO_PROBABILIDAD:
                    promocion_val = 1
                    f_promo = random.uniform(*PROMO_BOOST_RANGO)

                descuento_val = 0
                f_descuento = 1.0
                if self.incluir_descuentos and random.random() < DESCUENTO_PROBABILIDAD:
                    descuento_val = random.choice(DESCUENTO_PCT_OPCIONES)
                    f_descuento = 1 + descuento_val * DESCUENTO_BOOST_POR_PUNTO

                impulso = 1 + choque_anterior[nombre] * PESO_MOMENTUM

                demanda_esperada = (
                    demanda_base
                    * f_dia * f_mes * f_temp
                    * f_feriado * f_quincena * f_clasico * f_clima
                    * tendencia * evento_especial
                    * f_promo * f_descuento
                    * impulso
                )

                dispersion = dispersion_por_producto[nombre]
                mu = max(demanda_esperada, 0.01)
                p  = dispersion / (dispersion + mu)
                cantidad = int(np.random.negative_binomial(dispersion, p))

                choque_hoy = (cantidad - demanda_esperada) / mu
                choque_anterior[nombre] = choque_hoy * DECAIMIENTO

                if cantidad == 0:
                    continue

                fila = {
                    "fecha":           fecha.strftime("%Y-%m-%d"),
                    "producto":        nombre,
                    "categoria":       CATEGORIAS.get(self.tipo, {}).get(nombre, "General"),
                    "cantidad":        int(cantidad),
                    "precio_unitario": precio,
                    "total":           round(cantidad * precio, 2),
                    "dia_semana":      feats["dia_semana"],
                    "mes":             feats["mes"],
                    "semana_anio":     feats["semana_anio"],
                    "es_finde":        feats["es_finde"],
                    "es_feriado":      feats["es_feriado"],
                    "es_puente":       feats["es_puente"],
                    "dias_distancia_cobro": feats["dias_distancia_cobro"],
                    "pico_comida_rapida":   feats["pico_comida_rapida"],
                    "fase_liquidez":        feats["fase_liquidez"],
                }
                if self.incluir_promociones:
                    fila["promocion"] = promocion_val
                if self.incluir_descuentos:
                    fila["descuento_pct"] = descuento_val
                fila["es_evento_especial"] = int(evento_especial > 1.0)

                if clima_dia:
                    for col in ("lluvia_nocturna_mm", "temp_nocturna_promed"):
                        fila[col] = clima_dia.get(col)

                filas.append(fila)

        df = pd.DataFrame(filas)

        if sucio:
            df = self._agregar_errores(df)

        return df

    def _agregar_errores(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        idx_nombres = df.sample(frac=0.03).index
        for idx in idx_nombres:
            prod = df.loc[idx, "producto"]
            errores = [
                prod.lower(),
                prod.upper(),
                prod[: len(prod) // 2] + ".",
                prod.replace("a", "").replace("e", ""),
            ]
            df.loc[idx, "producto"] = random.choice(errores)

        formatos = ["%d/%m/%Y", "%m-%d-%Y", "%d.%m.%Y", "%d %b %Y"]
        idx_fechas = df.sample(frac=0.05).index
        for idx in idx_fechas:
            fecha_dt = pd.to_datetime(df.loc[idx, "fecha"])
            df.loc[idx, "fecha"] = fecha_dt.strftime(random.choice(formatos))

        for col in ["cantidad", "precio_unitario"]:
            df.loc[df.sample(frac=0.02).index, col] = np.nan

        df = pd.concat([df, df.sample(frac=0.01)], ignore_index=True)

        idx_precio = df.sample(frac=0.10).index
        df["precio_unitario"] = df["precio_unitario"].astype(object)
        df.loc[idx_precio, "precio_unitario"] = (
            df.loc[idx_precio, "precio_unitario"]
            .apply(lambda x: f"${x}" if pd.notna(x) else x)
        )

        idx_neg = df.sample(frac=0.005).index
        df.loc[idx_neg, "cantidad"] = (
            df.loc[idx_neg, "cantidad"]
            .apply(lambda x: -abs(x) if pd.notna(x) else x)
        )

        return df.sample(frac=1).reset_index(drop=True)

    def guardar_csv(self, ruta: str, sucio=False) -> pd.DataFrame:
        import os
        carpeta = os.path.dirname(ruta)
        if carpeta:
            os.makedirs(carpeta, exist_ok=True)
        df = self.generar(sucio=sucio)
        df.to_csv(ruta, index=False, encoding="utf-8")
        print(f"✅ Guardado: {ruta} ({len(df):,} filas)")
        return df

    def resumen(self, df: pd.DataFrame):
        print(f"\n{'='*55}")
        print(f"DATASET: {self.perfil['nombre']} ({self.años} años)")
        print(f"{'='*55}")
        print(f"Filas:             {len(df):,}")
        print(f"Productos únicos:  {df['producto'].nunique()}")
        print(f"Período:           {df['fecha'].min()} → {df['fecha'].max()}")
        if "total" in df.columns:
            total = pd.to_numeric(df["total"], errors="coerce").sum()
            print(f"Ingreso total:     ${total:,.2f}")
        print(f"\nFeatures incluidas: {[c for c in df.columns if c not in ('fecha','producto','cantidad','precio_unitario','total')]}")
        print(f"\nTop 5 productos por cantidad vendida:")
        cant = pd.to_numeric(df["cantidad"], errors="coerce")
        top  = df.assign(cantidad=cant).groupby("producto")["cantidad"].sum().sort_values(ascending=False).head(5)
        for prod, c in top.items():
            print(f"  {prod:<28} {int(c):>8,} unidades")

        if "promocion" in df.columns:
            pct_promo = 100 * df["promocion"].mean()
            print(f"\nFilas con promoción activa:  {pct_promo:.1f}%")
        if "descuento_pct" in df.columns:
            con_descuento = df[df["descuento_pct"] > 0]
            pct_desc = 100 * len(con_descuento) / len(df) if len(df) else 0
            desc_prom = con_descuento["descuento_pct"].mean() if len(con_descuento) else 0
            print(f"Filas con descuento activo:  {pct_desc:.1f}%  (promedio {desc_prom:.1f}%)")
        print("=" * 55)


def generar_todos(carpeta="data", años=2):
    import os
    os.makedirs(carpeta, exist_ok=True)
    for tipo in PERFILES:
        gen = DataGenerator(tipo_negocio=tipo, años=años)
        gen.guardar_csv(f"{carpeta}/train_{tipo}.csv",      sucio=False)
        gen.guardar_csv(f"{carpeta}/test_sucio_{tipo}.csv", sucio=True)


if __name__ == "__main__":
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    tipo = sys.argv[1] if len(sys.argv) > 1 else "restaurante"
    if tipo not in PERFILES:
        print(f"Tipo '{tipo}' no reconocido. Opciones: {list(PERFILES.keys())}")
        sys.exit(1)

    print(f"\nGenerando dataset para: {tipo}\n")

    while True:
        entrada = input("Meses de historial a generar (ej. 24): ").strip()
        try:
            meses = int(entrada)
            if meses <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Ingresa un número entero mayor a 0.")

    print("\nVariables de negocio a incluir:")
    print("  1. Ninguna")
    print("  2. Promociones")
    print("  3. Promociones y descuentos")
    while True:
        opcion = input("Elige una opción [1-3]: ").strip()
        if opcion in ("1", "2", "3"):
            break
        print("  Ingresa 1, 2 o 3.")

    incluir_promociones = opcion in ("2", "3")
    incluir_descuentos  = opcion == "3"

    gen = DataGenerator(
        tipo_negocio=tipo,
        meses=meses,
        incluir_promociones=incluir_promociones,
        incluir_descuentos=incluir_descuentos,
    )
    df_limpio = gen.guardar_csv(f"data/train_{tipo}.csv",      sucio=False)
    df_sucio  = gen.guardar_csv(f"data/test_sucio_{tipo}.csv", sucio=True)

    gen.resumen(df_limpio)