import re
import pandas as pd
import numpy as np
from rapidfuzz import fuzz, process
from scipy import stats
from datetime import datetime
from services.data_generator import FERIADOS_ECUADOR
from datetime import timedelta

# DICCIONARIO DE COLUMNAS — alias por idioma y variante

COLUMNAS_OBJETIVO = {
    "fecha": [
        "fecha", "date", "dia", "día", "day", "f_venta", "fecha_venta",
        "fecha_pedido", "fecha_compra", "fecha_transaccion", "fecha_transacción",
        "transaction_date", "order_date", "sale_date", "periodo", "periodo_venta",
        "fechaventa", "fecha_registro", "created_at", "created", "timestamp",
    ],
    "producto": [
        "producto", "product", "item", "articulo", "artículo",
        "descripcion", "descripción", "nombre", "name", "detalle",
        "detail", "concepto", "mercancia", "mercancía", "bien",
        "nombre_producto", "product_name", "item_name", "sku", "referencia",
    ],
    "cantidad": [
        "cantidad", "qty", "quantity", "cant", "unidades", "units",
        "num", "numero", "número", "piezas", "pieces", "amount",
        "volumen", "volume", "cant_vendida", "cantidad_vendida",
        "units_sold", "qty_sold",
    ],
    "precio": [
        "precio", "price", "valor", "value", "monto", "precio_unitario",
        "unit_price", "valor_unitario", "costo_unitario", "p_unitario",
        "precio_unit", "unit_cost", "tarifa", "rate", "precio_venta",
        "sale_price", "selling_price", "precio_base",
    ],
    "total": [
        "total", "subtotal", "importe", "venta", "ingreso", "revenue",
        "total_venta", "venta_total", "monto_total", "total_amount",
        "precio_total", "valor_total", "sales", "sale_amount",
        "total_ingreso", "importe_total", "gross", "gross_sales",
    ],
    "cliente": [
        "cliente", "client", "customer", "comprador", "buyer",
        "nombre_cliente", "customer_name", "client_name", "consumidor",
    ],
    "vendedor": [
        "vendedor", "seller", "employee", "empleado", "agente",
        "agent", "staff", "personal", "cajero",
    ],
    "categoria": [
        "categoria", "categoría", "category", "grupo", "group",
        "tipo", "type", "linea", "línea", "familia",
    ],
    "promocion": [
        "promocion", "promoción", "promo", "en_promocion", "en_promoción",
        "en_oferta", "oferta", "on_promotion", "promotion", "is_promo",
    ],
    "descuento_pct": [
        "descuento", "descuento_pct", "discount", "discount_pct", "rebaja",
        "porcentaje_descuento", "desc_pct", "descuento_porcentaje", "pct_descuento",
    ],
    "es_evento_especial": [
        "evento_especial", "evento", "festividad", "special_event",
        "es_evento", "dia_especial", "día_especial", "fecha_especial", "event",
    ],
}

# Columnas que DEBEN existir para que el análisis funcione
COLUMNAS_REQUERIDAS = ["fecha", "producto", "cantidad"]

# Columnas que enriquecen el análisis pero no son obligatorias
COLUMNAS_OPCIONALES = [
    "precio", "total", "cliente", "vendedor", "categoria",
    "promocion", "descuento_pct", "es_evento_especial",
]

# Score mínimo de similitud para aceptar una columna (0-100)
SCORE_MINIMO = 70


# ════════════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ════════════════════════════════════════════════════════════

class DataCleaner:
    """
    Motor de limpieza inteligente para archivos de ventas de restaurantes.

    Funciona con cualquier CSV o Excel independientemente de:
    - El idioma de las columnas (español, inglés)
    - El nombre exacto de las columnas (detecta por similitud)
    - Las columnas que tenga o no tenga (solo requiere fecha, producto, cantidad)
    - El formato de las fechas (detecta automáticamente)
    - La codificación del archivo (UTF-8, Latin-1, etc.)
    """

    def __init__(self):
        self.reporte = {
            "archivo":            None,
            "filas_originales":   0,
            "filas_finales":      0,
            "duplicados":         0,
            "nulos_eliminados":   0,
            "nulos_rellenados":   0,
            "fechas_corregidas":  0,
            "nombres_corregidos": [],
            "outliers":           [],
            "columnas_detectadas":{},
            "columnas_ignoradas": [],
            "columnas_faltantes": [],
            "errores":            [],
            "advertencias":       [],
            "tiene_precio":       False,
            "tiene_total":        False,
            "tiene_promocion":         False,
            "tiene_descuento":         False,
            "tiene_evento_especial":   False,
        }

    # ════════════════════════════════════════
    # PASO 1 — LEER ARCHIVO
    # ════════════════════════════════════════

    def leer_archivo(self, ruta, nombre_archivo):
        """
        Lee CSV o Excel con detección automática de:
        - Separador (coma, punto y coma, tab)
        - Codificación (UTF-8, Latin-1, CP1252)
        - Hoja activa en Excel
        """
        self.reporte["archivo"] = nombre_archivo
        ext = nombre_archivo.rsplit(".", 1)[-1].lower()

        try:
            if ext == "csv":
                df = self._leer_csv(ruta)
            elif ext in ("xlsx", "xls"):
                df = self._leer_excel(ruta)
            else:
                self.reporte["errores"].append(
                    f"Formato '.{ext}' no soportado. Usa CSV o Excel (.xlsx)."
                )
                return None

            # Limpiar nombres de columnas (espacios, saltos de línea)
            df.columns = [str(c).strip().replace("\n", " ") for c in df.columns]

            # Eliminar filas completamente vacías
            df = df.dropna(how="all")

            self.reporte["filas_originales"] = len(df)
            return df

        except Exception as e:
            self.reporte["errores"].append(f"Error al leer el archivo: {str(e)}")
            return None

    def _leer_csv(self, ruta):
        """Intenta múltiples separadores y codificaciones."""
        separadores  = [",", ";", "\t", "|"]
        codificaciones = ["utf-8", "latin-1", "cp1252", "utf-8-sig"]

        for codif in codificaciones:
            for sep in separadores:
                try:
                    df = pd.read_csv(
                        ruta, sep=sep, encoding=codif,
                        on_bad_lines="skip", low_memory=False
                    )
                    # Verificar que realmente separó en columnas
                    if len(df.columns) > 1:
                        return df
                except Exception:
                    continue

        # Último intento: dejar que pandas detecte
        return pd.read_csv(ruta, sep=None, engine="python", on_bad_lines="skip")

    def _leer_excel(self, ruta):
        with pd.ExcelFile(ruta) as xl:
            for hoja in xl.sheet_names:
                df = xl.parse(hoja)

                if len(df) > 0 and len(df.columns) > 1:
                    return df

            return xl.parse(xl.sheet_names[0])

    # ════════════════════════════════════════
    # PASO 2 — DETECTAR COLUMNAS
    # ════════════════════════════════════════

    def detectar_columnas(self, df):
        """
        Compara cada columna del archivo contra todos los alias conocidos
        usando fuzzy matching. Tolera errores de tipeo, mayúsculas,
        abreviaciones y columnas en inglés o español.
        """
        # Normalizar nombres de columnas para comparar
        cols_normalizadas = {
            str(c).lower().strip().replace(" ", "_").replace("-", "_"): c
            for c in df.columns
        }

        mapeo = {}      # { "fecha": "Fecha Venta" }
        usadas  = set() # columnas del archivo ya asignadas

        for col_objetivo, variantes in COLUMNAS_OBJETIVO.items():
            mejor_col    = None
            mejor_score  = 0

            for col_norm, col_original in cols_normalizadas.items():
                if col_original in usadas:
                    continue
                for variante in variantes:
                    score = fuzz.ratio(col_norm, variante)
                    if score > mejor_score:
                        mejor_score = score
                        mejor_col   = col_original

            if mejor_score >= SCORE_MINIMO:
                mapeo[col_objetivo] = mejor_col
                usadas.add(mejor_col)

        self.reporte["columnas_detectadas"] = {
            k: str(v) for k, v in mapeo.items()
        }

        # Detectar columnas ignoradas (no reconocidas)
        self.reporte["columnas_ignoradas"] = [
            str(c) for c in df.columns if c not in usadas
        ]
        if self.reporte["columnas_ignoradas"]:
            self.reporte["advertencias"].append(
                f"Columnas no reconocidas (ignoradas): "
                f"{', '.join(self.reporte['columnas_ignoradas'])}"
            )

        # Verificar columnas requeridas
        faltantes = [c for c in COLUMNAS_REQUERIDAS if c not in mapeo]
        self.reporte["columnas_faltantes"] = faltantes

        if faltantes:
            self.reporte["errores"].append(
                f"No se detectaron columnas requeridas: {', '.join(faltantes)}. "
                f"El archivo debe tener columnas de fecha, producto y cantidad "
                f"(pueden llamarse diferente, el sistema las detecta automáticamente)."
            )
            return None

        # Verificar columnas opcionales disponibles
        self.reporte["tiene_precio"] = "precio" in mapeo
        self.reporte["tiene_total"]  = "total"  in mapeo
        self.reporte["tiene_promocion"]       = "promocion" in mapeo
        self.reporte["tiene_descuento"]       = "descuento_pct" in mapeo
        self.reporte["tiene_evento_especial"] = "es_evento_especial" in mapeo

        if not self.reporte["tiene_precio"] and not self.reporte["tiene_total"]:
            self.reporte["advertencias"].append(
                "No se encontró columna de precio ni total. "
                "El análisis se basará en cantidades vendidas, no en ingresos."
            )

        # Renombrar columnas al estándar interno y conservar solo las detectadas
        df = df.rename(columns={v: k for k, v in mapeo.items()})
        columnas_a_conservar = list(mapeo.keys())
        df = df[columnas_a_conservar]

        return df

    # ════════════════════════════════════════
    # PASO 3 — LIMPIEZA
    # ════════════════════════════════════════

    def limpiar(self, df):
        """Ejecuta todos los pasos de limpieza en orden."""
        df = self._eliminar_duplicados(df)
        df = self._limpiar_fechas(df)
        df = self._limpiar_texto(df)
        df = self._limpiar_numericos(df)
        df = self._limpiar_columnas_negocio(df)
        df = self._estandarizar_productos(df)
        df = self._detectar_outliers(df)
        df = self._calcular_total(df)
        df = self._agregar_features_temporales(df)
        df = df.reset_index(drop=True)
        self.reporte["filas_finales"] = len(df)
        return df

    def _eliminar_duplicados(self, df):
        antes = len(df)
        df = df.drop_duplicates()
        self.reporte["duplicados"] = antes - len(df)
        return df

    def _limpiar_fechas(self, df):
        """
        Convierte fechas en cualquier formato a datetime.

        Las fechas en formato ISO (YYYY-MM-DD) son INEQUÍVOCAS por
        definición (año siempre primero) y se parsean con formato
        explícito, sin pasar por dayfirst=True.

        Por qué: pd.to_datetime(valor, dayfirst=True) intercambia
        día y mes incluso en fechas ISO cuando ambos componentes son
        <= 12 (ej. "2020-01-08" se interpretaba como 2020-08-01).
        Esto es un comportamiento real y confirmado de pandas con
        parseo por-valor (no vectorizado) — no una fecha ambigua mal
        escrita por el usuario. Bug encontrado en sesión de depuración
        del Dashboard: dataset con 6+ años de historial perdía 334
        fechas de calendario, siempre en los años parciales de los
        bordes del rango, con el patrón "solo sobreviven los días
        08-12 de cada mes" — exactamente lo que produce este swap.

        Para formatos NO-ISO (donde sí puede haber ambigüedad real,
        ej. "08/01/2020"), se mantiene dayfirst=True como fallback,
        y después los formatos manuales explícitos.
        """
        formatos_manuales = [
            "%Y-%m-%d", "%d/%m/%Y", "%m/%d/%Y", "%d-%m-%Y",
            "%Y/%m/%d", "%d.%m.%Y", "%Y.%m.%d", "%d %b %Y",
            "%d %B %Y", "%B %d, %Y", "%b %d, %Y",
        ]
        patron_iso = re.compile(r"^\d{4}-\d{2}-\d{2}")
        convertidas  = 0
        fechas_limpias = []

        for valor in df["fecha"]:
            fecha_ok = None
            texto = str(valor).strip()

            # ISO primero, sin ambigüedad, sin dayfirst
            if patron_iso.match(texto):
                try:
                    fecha_ok = pd.to_datetime(texto[:10], format="%Y-%m-%d")
                    convertidas += 1
                except Exception:
                    fecha_ok = None

            # No era ISO (o falló) → fallback al parseo con dayfirst
            if fecha_ok is None:
                try:
                    fecha_ok = pd.to_datetime(valor, dayfirst=True)
                    convertidas += 1
                except Exception:
                    for fmt in formatos_manuales:
                        try:
                            fecha_ok = datetime.strptime(texto, fmt)
                            convertidas += 1
                            break
                        except Exception:
                            continue

            fechas_limpias.append(fecha_ok)

        df["fecha"] = fechas_limpias
        antes = len(df)
        df = df.dropna(subset=["fecha"])
        self.reporte["nulos_eliminados"] += antes - len(df)
        self.reporte["fechas_corregidas"] = convertidas
        return df

    def _limpiar_texto(self, df):
        """Limpia columnas de texto: elimina espacios extra y caracteres raros."""
        cols_texto = ["producto"]
        if "cliente"   in df.columns: cols_texto.append("cliente")
        if "vendedor"  in df.columns: cols_texto.append("vendedor")
        if "categoria" in df.columns: cols_texto.append("categoria")

        for col in cols_texto:
            if col in df.columns:
                df[col] = (
                    df[col]
                    .astype(str)
                    .str.strip()
                    .str.replace(r"\s+", " ", regex=True)
                    .str.replace(r"[^\w\s\-áéíóúüñÁÉÍÓÚÜÑ]", "", regex=True)
                )
                # Eliminar filas donde el texto quedó vacío o "nan"
                df = df[~df[col].isin(["", "nan", "NaN", "None"])]

        return df

    def _limpiar_numericos(self, df):
        """
        Convierte columnas numéricas, eliminando símbolos de moneda,
        separadores de miles y comas decimales europeas.
        """
        cols_numericas = []
        if "cantidad" in df.columns: cols_numericas.append("cantidad")
        if "precio"   in df.columns: cols_numericas.append("precio")
        if "total"    in df.columns: cols_numericas.append("total")

        for col in cols_numericas:
            df[col] = (
                df[col]
                .astype(str)
                .str.replace(r"[\$€£,\s]", "", regex=True)  # quitar $, €, ,
                .str.replace(r"(?<=\d)\.(?=\d{3})", "", regex=True)  # miles: 1.000
                .str.replace(",", ".", regex=False)  # decimal europeo: 1,5 → 1.5
            )
            df[col] = pd.to_numeric(df[col], errors="coerce")

            # Rellenar nulos con mediana por producto
            nulos = df[col].isna().sum()
            if nulos > 0:
                df[col] = df.groupby("producto")[col].transform(
                    lambda x: x.fillna(x.median())
                )
                df[col] = df[col].fillna(df[col].median())
                self.reporte["nulos_rellenados"] += int(nulos)

            # Eliminar valores negativos o cero en cantidad
            if col == "cantidad":
                antes = len(df)
                df = df[df["cantidad"] > 0]
                eliminados = antes - len(df)
                if eliminados > 0:
                    self.reporte["advertencias"].append(
                        f"Se eliminaron {eliminados} filas con cantidad cero o negativa."
                    )

        return df

    def _limpiar_columnas_negocio(self, df):
        """
        Normaliza promocion / descuento_pct / es_evento_especial —
        SOLO si el archivo las trae. Son 100% opcionales: si no existen
        en df.columns, este método no las toca ni las inventa, así que
        el resto del pipeline sigue funcionando igual que con un
        dataset que nunca tuvo estas columnas.
        """
        for col in ("promocion", "es_evento_especial"):
            if col in df.columns:
                df[col] = df[col].apply(self._normalizar_booleano)

        if "descuento_pct" in df.columns:
            df["descuento_pct"] = (
                df["descuento_pct"]
                .astype(str)
                .str.replace("%", "", regex=False)
                .str.replace(",", ".", regex=False)
            )
            df["descuento_pct"] = pd.to_numeric(df["descuento_pct"], errors="coerce").fillna(0.0)

        return df

    @staticmethod
    def _normalizar_booleano(valor):
        """Interpreta Sí/No, True/False, 1/0, X/'' como booleano."""
        if pd.isna(valor):
            return False
        texto = str(valor).strip().lower()
        return texto in ("1", "true", "si", "sí", "yes", "x", "verdadero", "t", "y")

    def _estandarizar_productos(self, df):
        """
        Unifica nombres de productos similares.
        'Pizza Fam.', 'pizza familiar', 'PIZZA FAMILIAR' → 'Pizza Familiar'
        Usa fuzzy matching con score > 85%.
        """
        df = df[df["producto"].notna() & (df["producto"].astype(str).str.strip() != "") & (df["producto"].astype(str).str.lower() != "nan") & (df["producto"].astype(str).str.lower() != "none")]
        df["producto"] = df["producto"].str.title()
        productos_unicos = df["producto"].unique().tolist()

        if len(productos_unicos) <= 1:
            return df

        correcciones    = {}
        productos_vistos = set()

        for producto in sorted([str(p) for p in productos_unicos if pd.notna(p)],key=len,reverse=True):
            if producto in productos_vistos:
                continue

            similares = process.extract(
                producto, productos_unicos,
                scorer=fuzz.ratio, limit=10
            )

            for similar, score, _ in similares:
                if similar == producto or similar in productos_vistos:
                    continue
                if score >= 85:
                    # Preferir el nombre más largo (más descriptivo)
                    nombre_final = producto if len(producto) >= len(similar) else similar
                    correcciones[similar] = nombre_final
                    self.reporte["nombres_corregidos"].append(
                        f"'{similar}' → '{nombre_final}' (similitud: {score}%)"
                    )
                    productos_vistos.add(similar)

            productos_vistos.add(producto)

        if correcciones:
            df["producto"] = df["producto"].replace(correcciones)

        return df

    def _detectar_outliers(self, df):
        outliers_encontrados = []
        filas_globales = df.loc[df["producto"] == df["producto"]].copy()  

        for producto in df["producto"].unique():
            mask    = df["producto"] == producto
            subset  = df.loc[mask].reset_index(drop=True)
            valores = subset["cantidad"].astype(float)

            if len(valores) < 5:
                continue

            z_scores = np.abs(stats.zscore(valores))

            for pos in range(len(valores)):
                z = float(z_scores[pos])
                if z > 3:
                    fila = subset.iloc[pos]
                    cantidad = int(fila["cantidad"])
                    fecha    = str(pd.to_datetime(fila["fecha"]).date())

                    outliers_encontrados.append({
                    "producto": producto,
                    "fecha":    fecha,
                    "cantidad": cantidad,
                    "z_score":  round(z, 2),
                })
                    self.reporte["advertencias"].append(
                    f"Valor anómalo detectado: {producto} registró "
                    f"{cantidad} unidades el {fecha} "
                    f"(Z-score: {z:.1f}). Verifica si es correcto."
                )

        self.reporte["outliers"] = outliers_encontrados
        return df

    def _calcular_total(self, df):
        """
        Estrategia de 3 niveles para obtener la columna total:
        1. Ya existe → usarla
        2. Hay cantidad y precio → calcularla
        3. No hay nada → dejar en None y avisar
        """
        if "total" in df.columns:
            pass  # ya existe, no hacer nada

        elif "cantidad" in df.columns and "precio" in df.columns:
            df["total"] = (df["cantidad"] * df["precio"]).round(2)
            self.reporte["advertencias"].append(
                "Columna 'total' calculada automáticamente (cantidad × precio)."
            )

        else:
            df["total"] = None

        return df
    
    def _agregar_features_temporales(self, df):   
        fecha = pd.to_datetime(df["fecha"])
        df["dia_semana"]  = fecha.dt.dayofweek
        df["mes"]         = fecha.dt.month
        df["semana_anio"] = fecha.dt.isocalendar().week.astype(int)
        df["es_finde"]    = fecha.dt.dayofweek.isin([5, 6]).astype(int)
        fechas_str        = fecha.dt.strftime("%Y-%m-%d")
        df["es_feriado"]  = fechas_str.isin(FERIADOS_ECUADOR).astype(int)
        ayer    = (fecha - timedelta(days=1)).dt.strftime("%Y-%m-%d")
        maniana = (fecha + timedelta(days=1)).dt.strftime("%Y-%m-%d")
        df["es_puente"]   = (ayer.isin(FERIADOS_ECUADOR) | maniana.isin(FERIADOS_ECUADOR)).astype(int)
        dia_mes           = fecha.dt.day
        df["es_quincena"] = ((dia_mes.between(1, 7)) | (dia_mes.between(15, 21))).astype(int)
        return df

    

    # ════════════════════════════════════════
    # MÉTODO PRINCIPAL
    # ════════════════════════════════════════

    def procesar(self, ruta, nombre_archivo):
        """
        Método principal. Ejecuta todo el pipeline de limpieza.
        Retorna (df_limpio, reporte) donde df_limpio es None si hubo errores críticos.

        Uso:
            cleaner = DataCleaner()
            df, reporte = cleaner.procesar("/ruta/archivo.csv", "ventas.csv")
            if df is not None:
                # usar df limpio
        """
        # Paso 1: leer archivo
        df = self.leer_archivo(ruta, nombre_archivo)
        if df is None:
            return None, self.reporte

        # Paso 2: detectar columnas
        df = self.detectar_columnas(df)
        if df is None:
            return None, self.reporte

        # Paso 3: limpiar
        df = self.limpiar(df)

        # Resumen final en el reporte
        self.reporte["exito"] = len(self.reporte["errores"]) == 0

        return df, self.reporte
    
    

    def resumen_texto(self):
        """
        Devuelve un resumen legible del reporte para mostrar al usuario.
        """
        r = self.reporte
        lineas = [
            f"✅ Archivo procesado: {r['archivo']}",
            f"📊 Filas originales: {r['filas_originales']} → Filas limpias: {r['filas_finales']}",
        ]
        if r["duplicados"]:
            lineas.append(f"🗑️  Duplicados eliminados: {r['duplicados']}")
        if r["nulos_eliminados"]:
            lineas.append(f"❌ Filas con datos inválidos eliminadas: {r['nulos_eliminados']}")
        if r["nulos_rellenados"]:
            lineas.append(f"🔧 Valores nulos rellenados con mediana: {r['nulos_rellenados']}")
        if r["nombres_corregidos"]:
            lineas.append(f"📝 Nombres de productos unificados: {len(r['nombres_corregidos'])}")
            for c in r["nombres_corregidos"]:
                lineas.append(f"   • {c}")
        if r["outliers"]:
            lineas.append(f"⚠️  Valores anómalos detectados: {len(r['outliers'])} (no eliminados)")
        if r["columnas_ignoradas"]:
            lineas.append(f"ℹ️  Columnas ignoradas: {', '.join(r['columnas_ignoradas'])}")
        if r["advertencias"]:
            lineas.append("⚠️  Advertencias:")
            for a in r["advertencias"]:
                lineas.append(f"   • {a}")
        if r["errores"]:
            lineas.append("🚨 Errores:")
            for e in r["errores"]:
                lineas.append(f"   • {e}")
        return "\n".join(lineas)