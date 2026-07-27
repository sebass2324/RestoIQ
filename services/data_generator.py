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
}


# ════════════════════════════════════════════════════════════
# HELPERS DE FEATURES TEMPORALES
# ════════════════════════════════════════════════════════════

def _features_temporales(fecha_dt: pd.Timestamp) -> dict:
    """
    Dado un Timestamp, devuelve todas las features derivadas de la fecha.
    Estas son las mismas features que se calcularán en producción cuando
    el usuario suba su CSV real, así el modelo puede generalizar.
    """
    fecha_str = fecha_dt.strftime("%Y-%m-%d")
    dia_semana = fecha_dt.weekday()          # 0=Lun … 6=Dom
    es_finde   = int(dia_semana in [5, 6])
    es_feriado = int(fecha_str in FERIADOS_ECUADOR)

    # Feriado o puente (día antes/después de feriado)
    ayer    = (fecha_dt - timedelta(days=1)).strftime("%Y-%m-%d")
    maniana = (fecha_dt + timedelta(days=1)).strftime("%Y-%m-%d")
    es_puente = int(ayer in FERIADOS_ECUADOR or maniana in FERIADOS_ECUADOR)

    # Quincena: semana de cobro (1-7 y 15-21 de cada mes)
    dia_mes = fecha_dt.day
    es_quincena = int(1 <= dia_mes <= 7 or 15 <= dia_mes <= 21)

    return {
        "dia_semana":   dia_semana,          # 0-6
        "mes":          fecha_dt.month,       # 1-12
        "semana_anio":  fecha_dt.isocalendar().week,  # 1-53
        "es_finde":     es_finde,             # 0/1
        "es_feriado":   es_feriado,           # 0/1
        "es_puente":    es_puente,            # 0/1
        "es_quincena":  es_quincena,          # 0/1  ← patrón de cobro Ecuador
    }


# ════════════════════════════════════════════════════════════
# PROMOCIONES Y DESCUENTOS (sintéticos, independientes entre sí)
# ════════════════════════════════════════════════════════════

# Probabilidad de que una fila (fecha, producto) tenga promoción activa,
# y el rango de aumento de demanda que produce cuando ocurre.
PROMO_PROBABILIDAD = 0.12
PROMO_BOOST_RANGO  = (1.25, 1.70)

# Probabilidad de que una fila tenga descuento (independiente de la
# promoción — puede haber descuento sin promoción y viceversa, según
# se decidió explícitamente para este generador). Los valores posibles
# de descuento y cuánto empuja la demanda por cada punto porcentual.
DESCUENTO_PROBABILIDAD    = 0.15
DESCUENTO_PCT_OPCIONES    = [10, 15, 20, 25, 30]   # entero tipo porcentaje (15 = 15%)
DESCUENTO_BOOST_POR_PUNTO = 0.008                   # 20% de descuento → +16% de demanda


# ════════════════════════════════════════════════════════════
# CLASE GENERADORA
# ════════════════════════════════════════════════════════════

class DataGenerator:
    """
    Genera datasets sintéticos de ventas DIARIAS por producto para RestoIQ.

    Columnas de salida (CSV limpio):
        fecha, producto, cantidad, precio_unitario, total,
        dia_semana, mes, semana_anio,
        es_finde, es_feriado, es_puente, es_quincena
        [+ promocion]        si incluir_promociones=True
        [+ descuento_pct]    si incluir_descuentos=True

    El CSV sucio mantiene las mismas columnas pero introduce errores en
    fecha, producto, cantidad y precio_unitario para probar el DataCleaner.
    Las columnas de negocio opcionales (promocion, descuento_pct) NO se
    ensucian, igual que el resto de features de contexto.
    """

    def __init__(self, tipo_negocio="restaurante", años=2, meses=None,
                 incluir_promociones=False, incluir_descuentos=False):
        if tipo_negocio not in PERFILES:
            raise ValueError(
                f"Tipo '{tipo_negocio}' no reconocido. "
                f"Opciones: {list(PERFILES.keys())}"
            )
        self.perfil = PERFILES[tipo_negocio]
        self.tipo   = tipo_negocio

        # `meses`, si viene, manda sobre `años` — pero `años` se
        # mantiene como parámetro válido para no romper a quien ya
        # llama DataGenerator(tipo_negocio=..., años=...) (ej.
        # generar_todos() y el bloque de prueba de sales_model.py).
        self.meses = meses if meses is not None else round(años * 12)
        self.años  = self.meses / 12  # usado en el cálculo de tendencia

        self.incluir_promociones = incluir_promociones
        # Descuentos requiere que promociones esté activo a nivel de
        # menú (opción 3 = "promociones y descuentos"), pero a nivel de
        # fila son independientes entre sí (puede haber descuento sin
        # promoción activa en esa fila específica, y viceversa).
        self.incluir_descuentos  = incluir_descuentos

    # ── Generación principal ──────────────────────────────────────────────

    def generar(self, sucio=False) -> pd.DataFrame:
        """
        Genera el dataset completo.

        Parámetros:
            sucio: si True agrega errores realistas para probar el DataCleaner.

        Retorna:
            DataFrame con features de contexto ya incluidas.
        """
        fecha_inicio = datetime.today() - timedelta(days=round(self.meses * 30.44))
        fecha_fin    = datetime.today() - timedelta(days=1)
        fechas       = pd.date_range(fecha_inicio, fecha_fin, freq="D")

        productos  = self.perfil["productos"]
        factor_dia = self.perfil["factor_dia"]
        factor_mes = self.perfil["factor_mes"]

        filas = []
        for i, fecha in enumerate(fechas):
            tendencia       = 1 + (i / len(fechas)) * (0.20 * self.años)
            f_dia           = factor_dia[fecha.weekday()]
            f_mes           = factor_mes[fecha.month]
            evento_especial = random.uniform(1.3, 2.0) if random.random() < 0.05 else 1.0

            # Features temporales de contexto (las mismas que en producción)
            feats = _features_temporales(fecha)

            # Factor extra por feriado: más ventas en heladería/restaurante,
            # menos en cafetería (que depende del tráfico laboral)
            f_feriado = 1.0
            if feats["es_feriado"] or feats["es_finde"]:
                f_feriado = 1.25 if self.tipo in ("restaurante", "pizzeria", "heladeria") else 0.80

            # Factor de quincena: leve aumento de demanda
            f_quincena = 1.10 if feats["es_quincena"] else 1.0

            for nombre, precio, demanda_base, temporada in productos:

                f_temp = 1.0
                if temporada == "finde"  and fecha.weekday() in [4, 5, 6]:
                    f_temp = 1.5
                elif temporada == "semana" and fecha.weekday() in [0, 1, 2, 3]:
                    f_temp = 1.3
                elif temporada == "verano" and fecha.month in [6, 7, 8]:
                    f_temp = 1.6

                # ── Promoción y descuento (independientes entre sí) ──
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

                demanda_esperada = (
                    demanda_base
                    * f_dia * f_mes * f_temp
                    * f_feriado * f_quincena
                    * tendencia * evento_especial
                    * f_promo * f_descuento
                )
                ruido    = np.random.normal(0, demanda_esperada * 0.15)
                cantidad = max(0, round(demanda_esperada + ruido))

                if cantidad == 0:
                    continue

                fila = {
                    "fecha":           fecha.strftime("%Y-%m-%d"),
                    "producto":        nombre,
                    "categoria":       CATEGORIAS.get(self.tipo, {}).get(nombre, "General"),
                    "cantidad":        int(cantidad),
                    "precio_unitario": precio,
                    "total":           round(cantidad * precio, 2),
                    # ── Features de contexto ──────────────────────────────
                    "dia_semana":      feats["dia_semana"],
                    "mes":             feats["mes"],
                    "semana_anio":     feats["semana_anio"],
                    "es_finde":        feats["es_finde"],
                    "es_feriado":      feats["es_feriado"],
                    "es_puente":       feats["es_puente"],
                    "es_quincena":     feats["es_quincena"],
                }
                if self.incluir_promociones:
                    fila["promocion"] = promocion_val
                if self.incluir_descuentos:
                    fila["descuento_pct"] = descuento_val

                filas.append(fila)

        df = pd.DataFrame(filas)

        if sucio:
            df = self._agregar_errores(df)

        return df

    # ── Errores sintéticos ────────────────────────────────────────────────

    def _agregar_errores(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Introduce errores realistas en las columnas crudas (fecha, producto,
        cantidad, precio_unitario).  Las columnas de features NO se tocan:
        el DataCleaner las regenerará a partir de la fecha corregida.
        """
        df = df.copy()

        # Nombres mal escritos ~3%
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

        # Fechas en distintos formatos ~5%
        formatos = ["%d/%m/%Y", "%m-%d-%Y", "%d.%m.%Y", "%d %b %Y"]
        idx_fechas = df.sample(frac=0.05).index
        for idx in idx_fechas:
            fecha_dt = pd.to_datetime(df.loc[idx, "fecha"])
            df.loc[idx, "fecha"] = fecha_dt.strftime(random.choice(formatos))

        # Valores nulos ~2%
        for col in ["cantidad", "precio_unitario"]:
            df.loc[df.sample(frac=0.02).index, col] = np.nan

        # Duplicados ~1%
        df = pd.concat([df, df.sample(frac=0.01)], ignore_index=True)

        # Precios con símbolo $ ~10%  ← FIX: convertir a object antes de asignar strings
        idx_precio = df.sample(frac=0.10).index
        df["precio_unitario"] = df["precio_unitario"].astype(object)
        df.loc[idx_precio, "precio_unitario"] = (
            df.loc[idx_precio, "precio_unitario"]
            .apply(lambda x: f"${x}" if pd.notna(x) else x)
        )

        # Cantidades negativas ~0.5%
        idx_neg = df.sample(frac=0.005).index
        df.loc[idx_neg, "cantidad"] = (
            df.loc[idx_neg, "cantidad"]
            .apply(lambda x: -abs(x) if pd.notna(x) else x)
        )

        return df.sample(frac=1).reset_index(drop=True)

    # ── I/O ──────────────────────────────────────────────────────────────

    def guardar_csv(self, ruta: str, sucio=False) -> pd.DataFrame:
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


# ════════════════════════════════════════════════════════════
# FUNCIÓN DE CONVENIENCIA — genera todos los perfiles
# ════════════════════════════════════════════════════════════

def generar_todos(carpeta="data", años=2):
    """Genera un dataset limpio y uno sucio para cada tipo de negocio."""
    import os
    os.makedirs(carpeta, exist_ok=True)
    for tipo in PERFILES:
        gen = DataGenerator(tipo_negocio=tipo, años=años)
        gen.guardar_csv(f"{carpeta}/train_{tipo}.csv",      sucio=False)
        gen.guardar_csv(f"{carpeta}/test_sucio_{tipo}.csv", sucio=True)


# ════════════════════════════════════════════════════════════
# EJECUCIÓN DIRECTA
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys, os
    # Permite correr desde cualquier directorio
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    tipo = sys.argv[1] if len(sys.argv) > 1 else "restaurante"
    if tipo not in PERFILES:
        print(f"Tipo '{tipo}' no reconocido. Opciones: {list(PERFILES.keys())}")
        sys.exit(1)

    print(f"\nGenerando dataset para: {tipo}\n")

    # ── Meses de historial (parámetro principal — la cantidad de
    #    registros es una consecuencia y se muestra al final, en el
    #    resumen, no se pide como input) ──
    while True:
        entrada = input("Meses de historial a generar (ej. 24): ").strip()
        try:
            meses = int(entrada)
            if meses <= 0:
                raise ValueError
            break
        except ValueError:
            print("  Ingresa un número entero mayor a 0.")

    # ── Variables de negocio ──
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