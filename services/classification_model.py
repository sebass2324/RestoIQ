"""
services/classification_model.py

Modelo de CLASIFICACIÓN — Prioridad de Abastecimiento (LightGBM).
Segundo modelo de ML de RestoIQ, independiente del regresor (LGBM
también, ver sales_model.py): no predice demanda ni usa su salida.
Clasifica cada (producto, día) en un nivel de prioridad operativa
para planificar compras/preparación.

    Clases:  Baja · Media · Alta

────────────────────────────────────────────────────────────────────
ETIQUETA (ground-truth), calculada, no observada directamente:

  Por producto (percentil [0,1] entre productos, del train):
      rotacion_pct      = percentil de la mediana de cantidad diaria
      volatilidad_pct   = percentil del coef. de variación (std/media)
      impacto_promo_pct = percentil de cuánto sube la demanda en promo
  Por día:
      presion_dia = promedio(es_finde, es_feriado, promocion,
                              es_puente, fase_liquidez/3)
      — fase_liquidez (0=escasez, 1=neutral, 2=post-pago, 3=cobro
        activo) viene de calcular_features_liquidez() en
        data_generator.py. Reemplaza al es_quincena binario, que
        classification_model.py y classification_service.py
        mantenían cada uno por su cuenta y terminaron divergiendo
        (esa duplicación fue la causa del bug original). Ahora hay
        una sola fuente de verdad.
  Por categoría:
      tendencia = ratio de demanda de los últimos 7 días de la
                  categoría vs. su promedio histórico (causal,
                  shift(1) antes de rolling — ver _tendencia_pct).

      criticidad = 0.35·rotacion_pct + 0.25·presion_dia +
                   0.15·tendencia_norm + 0.15·volatilidad_pct +
                   0.10·impacto_promo_pct
      (PESOS_SCORE; antes era un promedio simple de 4 señales sin
      tendencia. Rotación pesa más porque es la señal con más datos
      por producto; impacto_promo pesa menos porque depende de que
      existan filas con y sin promoción, más ruidoso).

  Terciles (percentiles 33/66 del train) → 3 clases balanceadas.

  El clima (lluvia/temperatura nocturna) NO entra en la fórmula de
  la etiqueta: es un modulador de demanda de un solo tipo de negocio
  (El Chamo Burger), no una señal de criticidad de producto
  generalizable. Entra solo como feature (ver abajo), para que el
  modelo aprenda la correlación directo de los datos.

────────────────────────────────────────────────────────────────────
FEATURES:
  · categoría (one-hot)
  · features indirectas de categoría (percentil promedio de rotación/
    volatilidad/impacto_promo de esa categoría) — la única señal de
    producto "por identidad" que usa el modelo. Se probó identidad de
    producto directa (one-hot) y midió ~99% en productos conocidos
    vs. ~26% en productos nuevos: memorización, no aprendizaje del
    patrón de negocio. Se descartó por eso.
  · tendencia reciente por categoría (cat_tendencia_7).
  · features causales POR PRODUCTO (lag7/14/28_ratio, vol7_ratio,
    tendencia_prod, dias_sin_venta, participacion_cat) — todas
    expresadas como ratios/relativos al propio histórico del
    producto, nunca cantidad cruda, por la misma razón que se
    descartó la identidad directa: un ratio generaliza a un producto
    nuevo con historial corto, una cantidad absoluta delata volumen
    (=identidad) del producto. Rolling CAUSAL (shift(1) antes de
    rolling, igual patrón que sales_model.py) — nunca miran el propio
    día que predicen.
  · contexto temporal: dia_semana/mes/semana_anio/es_finde/es_feriado/
    es_puente + fase_liquidez/dias_distancia_cobro/pico_comida_rapida
    (calcular_features_liquidez, ver arriba).
  · clima nocturno si el archivo lo trae (lluvia_nocturna_mm,
    temp_nocturna_promed) — se rellena con 0 si el negocio no opera
    de noche o no tiene esos datos.
  · promoción/descuento/evento si el archivo los trae.

  NUNCA usa: precio crudo, cantidad/demanda cruda, ni identidad de
  producto.

────────────────────────────────────────────────────────────────────
CALIDAD DEL MODELO:
  · LightGBM (LGBMClassifier), mismo motor que el regresor de
    demanda — consistencia de stack, mejor manejo de features
    ralas (dummies de categoría) que RandomForest.
  · Probabilidades calibradas (CalibratedClassifierCV): predict_proba()
    es un número confiable ("80% de confianza" ≈ 80% de aciertos reales).
  · Comparación contra 2 baselines (mayoritaria y aleatoria) en cada
    entrenamiento, para cuantificar si el modelo aporta valor real.
"""

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix,
)

try:
    from services.data_generator import calcular_features_liquidez, FERIADOS_ECUADOR
except ImportError:
    from data_generator import calcular_features_liquidez, FERIADOS_ECUADOR


CLASES = ["Baja", "Media", "Alta"]
BANDAS_PRECIO = ["Bajo", "Medio", "Alto"]

COMPONENTES_PRODUCTO = ["rotacion", "volatilidad", "impacto_promo"]
# es_quincena NO está acá: fase_liquidez (ver PESOS_SCORE / _presion_dia)
# lo reemplaza con la ventana real reportada por el negocio.
FLAGS_DIA = ["es_finde", "es_feriado", "promocion", "es_puente"]

# Pesos del índice de criticidad — antes promedio simple de 4 señales,
# sin tendencia. Rotación pesa más (más datos por producto, señal más
# estable); impacto_promo pesa menos (requiere filas con y sin promo,
# más ruidoso). Deben sumar 1.0.
PESOS_SCORE = {
    "rotacion": 0.35,
    "presion_dia": 0.25,
    "tendencia": 0.15,
    "volatilidad": 0.15,
    "impacto_promo": 0.10,
}

# Precio entra SOLO como banda (Bajo/Medio/Alto), nunca crudo: el valor
# exacto actúa como ID de producto y rompe la generalización.
FEATURES_PRODUCTO   = ["categoria", "promocion", "descuento_pct", "es_evento_especial"]
FEATURES_TEMPORALES = ["dia_semana", "mes", "semana_anio", "es_finde", "es_feriado", "es_puente",
                       "fase_liquidez", "dias_distancia_cobro", "pico_comida_rapida"]
# Solo existen para negocios que reportan clima nocturno (ver
# clima_service.py). Si no están en el archivo, quedan en 0 vía el
# reindex de _construir_features — no rompen el entrenamiento.
FEATURES_CLIMA = ["lluvia_nocturna_mm", "temp_nocturna_promed"]
FEATURES_CAUSALES_PRODUCTO = [
    "lag7_ratio", "lag14_ratio", "lag28_ratio",
    "vol7_ratio", "tendencia_prod", "dias_sin_venta", "participacion_cat",
]
_DEFAULT_CAUSAL_PRODUCTO = {c: (1.0 if c == "tendencia_prod" else 0.0) for c in FEATURES_CAUSALES_PRODUCTO}
_DEFAULT_CONTEXTO_TEMPORAL = {
    "dia_semana": 0, "mes": 1, "semana_anio": 1, "es_finde": 0, "es_feriado": 0,
    "es_puente": 0, "fase_liquidez": 1, "dias_distancia_cobro": 7, "pico_comida_rapida": 0,
}

FRACCION_TRAIN = 0.8

LGBM_PARAMS = {
    "n_estimators": 300, "learning_rate": 0.05, "num_leaves": 31,
    "min_child_samples": 20, "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_alpha": 0.1, "reg_lambda": 0.1, "importance_type": "gain",
}


class ModeloAbastecimiento:
    """Clasificador de prioridad de abastecimiento a nivel producto-día."""

    def __init__(self, random_state=42):
        self.modelo = None                 # CalibratedClassifierCV una vez entrenado
        self.mejores_hiperparametros = {}
        self.random_state = random_state

        self.feature_names  = []
        self.cols_categoria = []
        self.stats_producto  = {}     # {prod: {rotacion, volatilidad, impacto_promo}} (raw)
        self.stats_categoria = {}     # promedio raw por categoría (fallback + feature indirecta)
        self.stat_global     = {}     # fallback global (raw promedio)
        self._umbrales_precio = None  # terciles de precio promedio por producto (train)
        self._arrays = {}             # {componente: np.array ordenado del train} para percentiles
        self.umbrales = None          # (u1, u2) terciles del score en train
        self.clases = CLASES

        self.tendencia_categoria = {}  # {categoria: serie causal de ratio de demanda}
        self.hist_producto = {}        # {producto: DataFrame causal de features relativas}

        self.metricas = {}
        self.importancias = []

    # ────────────────────────────────────────
    # 1. AGREGACIÓN A PRODUCTO-DÍA
    # ────────────────────────────────────────

    def _agregar_producto_dia(self, df):
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])

        agg = {"cantidad": "sum"}
        if "precio" in df.columns:             agg["precio"] = "mean"
        if "descuento_pct" in df.columns:      agg["descuento_pct"] = "mean"
        if "promocion" in df.columns:          agg["promocion"] = "max"
        if "es_evento_especial" in df.columns: agg["es_evento_especial"] = "max"
        if "categoria" in df.columns:          agg["categoria"] = "first"
        for c in FEATURES_TEMPORALES + FEATURES_CLIMA:
            if c in df.columns:                agg[c] = "first"

        pd_df = (df.groupby(["producto", "fecha"], as_index=False)
                   .agg(agg).sort_values("fecha").reset_index(drop=True))

        for c in ["promocion", "es_evento_especial", "es_finde", "es_feriado",
                  "es_puente", "pico_comida_rapida"]:
            if c in pd_df.columns:
                pd_df[c] = pd_df[c].astype(float).fillna(0).astype(int)
        return pd_df

    # ────────────────────────────────────────
    # 2. CONTEXTO TEMPORAL — dia_semana/feriados/liquidez
    # ────────────────────────────────────────

    def _asegurar_contexto_temporal(self, df):
        """Autocompleta columnas de calendario/liquidez faltantes. En
        entrenamiento normalmente ya vienen del archivo; en predicción
        (contexto con solo 'fecha', o sin fecha) se calculan acá, con
        calcular_features_liquidez como única fuente de verdad (ver
        docstring del módulo)."""
        df = df.copy()
        if "fecha" not in df.columns:
            for c, v in _DEFAULT_CONTEXTO_TEMPORAL.items():
                if c not in df.columns:
                    df[c] = v
            return df

        fechas = pd.to_datetime(df["fecha"])
        if "dia_semana" not in df.columns:
            df["dia_semana"] = fechas.dt.weekday
        if "mes" not in df.columns:
            df["mes"] = fechas.dt.month
        if "semana_anio" not in df.columns:
            df["semana_anio"] = fechas.dt.isocalendar().week.astype(int)
        if "es_finde" not in df.columns:
            df["es_finde"] = fechas.dt.weekday.isin([5, 6]).astype(int)
        if "es_feriado" not in df.columns:
            df["es_feriado"] = fechas.dt.strftime("%Y-%m-%d").isin(FERIADOS_ECUADOR).astype(int)
        if "es_puente" not in df.columns:
            ayer    = (fechas - pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
            maniana = (fechas + pd.Timedelta(days=1)).dt.strftime("%Y-%m-%d")
            df["es_puente"] = (ayer.isin(FERIADOS_ECUADOR) | maniana.isin(FERIADOS_ECUADOR)).astype(int)

        faltan_liquidez = [c for c in ("fase_liquidez", "dias_distancia_cobro", "pico_comida_rapida")
                           if c not in df.columns]
        if faltan_liquidez:
            calculado = fechas.apply(calcular_features_liquidez)
            for c in faltan_liquidez:
                df[c] = calculado.apply(lambda d: d[c])
        return df

    # ────────────────────────────────────────
    # 3. ESTADÍSTICOS DE PRODUCTO (SOLO TRAIN)
    # ────────────────────────────────────────

    def _raw_producto(self, g):
        """Calcula (rotacion, volatilidad, impacto_promo) crudos de un grupo."""
        q = g["cantidad"].astype(float)
        media = q.mean()
        rot = float(q.median())
        vol = float(q.std(ddof=0) / media) if media > 0 else 0.0
        imp = 0.0
        if "promocion" in g.columns:
            con = q[g["promocion"] == 1]
            sin = q[g["promocion"] == 0]
            if len(con) > 0 and len(sin) > 0 and sin.mean() > 0:
                imp = max(0.0, float((con.mean() - sin.mean()) / sin.mean()))
        return {"rotacion": rot, "volatilidad": vol, "impacto_promo": imp}

    def _calcular_stats(self, df_train_pd):
        stats = {p: self._raw_producto(g)
                 for p, g in df_train_pd.groupby("producto")}
        self.stats_producto = stats

        for comp in COMPONENTES_PRODUCTO:
            self._arrays[comp] = np.sort([s[comp] for s in stats.values()])

        self.stats_categoria = {}
        if "categoria" in df_train_pd.columns:
            cat_de = (df_train_pd.groupby("producto")["categoria"].first().to_dict())
            acc = {}
            for prod, s in stats.items():
                cat = cat_de.get(prod)
                acc.setdefault(cat, []).append(s)
            for cat, lst in acc.items():
                self.stats_categoria[cat] = {
                    c: float(np.mean([s[c] for s in lst])) for c in COMPONENTES_PRODUCTO}

        self.stat_global = {
            c: float(np.mean([s[c] for s in stats.values()])) for c in COMPONENTES_PRODUCTO}

        # Umbrales de banda de precio — terciles del precio PROMEDIO
        # por producto, calculados solo con train.
        self._umbrales_precio = None
        if "precio" in df_train_pd.columns:
            precio_por_producto = df_train_pd.groupby("producto")["precio"].mean()
            if precio_por_producto.notna().any() and precio_por_producto.nunique() > 1:
                self._umbrales_precio = tuple(precio_por_producto.quantile([1/3, 2/3]).values)

    def _banda_precio(self, precio):
        if self._umbrales_precio is None or pd.isna(precio):
            return "Medio"
        p1, p2 = self._umbrales_precio
        if precio <= p1:
            return "Bajo"
        if precio <= p2:
            return "Medio"
        return "Alto"

    def _pct(self, valor, comp):
        """Percentil empírico [0,1]: fracción de productos del train ≤ valor."""
        arr = self._arrays.get(comp)
        if arr is None or len(arr) == 0:
            return 0.5
        return float(np.searchsorted(arr, valor, side="right") / len(arr))

    def _raw_de(self, prod, cat):
        if prod in self.stats_producto:
            return self.stats_producto[prod]
        if cat is not None and cat in self.stats_categoria:
            return self.stats_categoria[cat]
        return self.stat_global

    # ────────────────────────────────────────
    # 4. TENDENCIA CAUSAL POR CATEGORÍA (score + feature cat_tendencia_7)
    # ────────────────────────────────────────

    def _calcular_historial_categoria(self, df_pd):
        """
        Serie diaria continua de demanda por categoría, con rolling
        CAUSAL (shift(1) antes de rolling — mismo patrón anti-fuga que
        _preparar_features() en sales_model.py). Se construye con TODO
        el historial (no solo train): cada fila solo mira hacia atrás
        desde su propia fecha, no hay fuga hacia el futuro.
        """
        self.tendencia_categoria = {}
        if "categoria" not in df_pd.columns or "fecha" not in df_pd.columns:
            return
        diario = df_pd.groupby(["categoria", "fecha"], as_index=False)["cantidad"].sum()
        for cat, g in diario.groupby("categoria"):
            g = g.set_index("fecha").sort_index()
            rango = pd.date_range(g.index.min(), g.index.max(), freq="D")
            serie = g["cantidad"].reindex(rango, fill_value=0.0)
            media = float(serie.mean()) or 1.0
            rolling = serie.shift(1).rolling(7, min_periods=1).mean()
            self.tendencia_categoria[cat] = rolling / media

    def _tendencia_pct(self, categoria, fecha):
        """Ratio de demanda reciente (7 días) vs. el promedio histórico
        de la categoría. ~1.0 = ritmo normal, >1.0 = repuntando,
        <1.0 = enfriándose. Fallback neutral (1.0) si no hay historial."""
        serie = self.tendencia_categoria.get(categoria)
        if serie is None:
            return 1.0
        val = serie.get(pd.Timestamp(fecha))
        return 1.0 if val is None or pd.isna(val) else float(val)

    # ────────────────────────────────────────
    # 5. FEATURES CAUSALES POR PRODUCTO (relativas, nunca cantidad cruda)
    # ────────────────────────────────────────

    def _calcular_historial_producto(self, df_pd):
        """
        Por producto: lags/rolling de demanda expresados como RATIO al
        propio promedio histórico del producto (nunca la cantidad
        cruda — eso delataría volumen = identidad, igual que se
        descartó el one-hot de producto). Todo con shift(1) antes de
        cualquier rolling/lag, mismo patrón anti-fuga que
        sales_model.py y _calcular_historial_categoria.
        """
        self.hist_producto = {}
        if "fecha" not in df_pd.columns:
            return

        total_cat_dia = None
        if "categoria" in df_pd.columns:
            total_cat_dia = df_pd.groupby(["categoria", "fecha"])["cantidad"].sum()

        for prod, g in df_pd.groupby("producto"):
            g = g.set_index("fecha").sort_index()
            rango = pd.date_range(g.index.min(), g.index.max(), freq="D")
            cant = g["cantidad"].reindex(rango, fill_value=0.0)
            media_hist = float(cant.mean()) or 1.0
            cant_shift = cant.shift(1)

            r7  = cant_shift.rolling(7,  min_periods=1).mean()
            r14 = cant_shift.rolling(14, min_periods=1).mean()
            r28 = cant_shift.rolling(28, min_periods=1).mean()
            std7 = cant_shift.rolling(7, min_periods=1).std().fillna(0)

            sin_venta = (cant_shift == 0).astype(int)
            racha = (sin_venta != sin_venta.shift()).cumsum()
            dias_sin_venta = sin_venta.groupby(racha).cumsum().where(sin_venta == 1, 0)

            participacion = pd.Series(0.0, index=rango)
            if total_cat_dia is not None and "categoria" in g.columns:
                cat = g["categoria"].iloc[0]
                cat_total = total_cat_dia.loc[cat].reindex(rango, fill_value=0.0).shift(1)
                participacion = (cant_shift / cat_total.replace(0, np.nan)).fillna(0.0).clip(0, 1)

            self.hist_producto[prod] = pd.DataFrame({
                "lag7_ratio":       (r7 / media_hist).fillna(0.0),
                "lag14_ratio":      (r14 / media_hist).fillna(0.0),
                "lag28_ratio":      (r28 / media_hist).fillna(0.0),
                "vol7_ratio":       (std7 / media_hist).fillna(0.0),
                "tendencia_prod":   (r7 / r28.replace(0, np.nan)).fillna(1.0),
                "dias_sin_venta":   dias_sin_venta.astype(float),
                "participacion_cat": participacion,
            })

    def _causales_producto(self, producto, fecha):
        tabla = self.hist_producto.get(producto)
        if tabla is None:
            return dict(_DEFAULT_CAUSAL_PRODUCTO)
        fila = tabla.reindex([pd.Timestamp(fecha)]).iloc[0]
        if fila.isna().any():
            return dict(_DEFAULT_CAUSAL_PRODUCTO)
        return fila.to_dict()

    # ────────────────────────────────────────
    # 6. SCORE Y ETIQUETA
    # ────────────────────────────────────────

    def _presion_dia(self, row):
        flags = [float(row.get(f, 0)) for f in FLAGS_DIA if f in row.index]
        fase = float(row.get("fase_liquidez", 1)) / 3.0
        valores = flags + [fase]
        return float(np.mean(valores)) if valores else fase

    def _score_desde_raw(self, raw, row, tendencia=1.0):
        rot = self._pct(raw["rotacion"], "rotacion")
        vol = self._pct(raw["volatilidad"], "volatilidad")
        imp = self._pct(raw["impacto_promo"], "impacto_promo")
        presion = self._presion_dia(row)
        tend = tendencia / (tendencia + 1.0)  # ratio -> [0,1), 1.0 => 0.5
        return float(
            PESOS_SCORE["rotacion"] * rot +
            PESOS_SCORE["presion_dia"] * presion +
            PESOS_SCORE["tendencia"] * tend +
            PESOS_SCORE["volatilidad"] * vol +
            PESOS_SCORE["impacto_promo"] * imp
        )

    def _score_fila(self, row):
        cat = row["categoria"] if "categoria" in row.index else None
        raw = self._raw_de(row["producto"], cat)
        tendencia = self._tendencia_pct(cat, row["fecha"]) if cat is not None else 1.0
        return self._score_desde_raw(raw, row, tendencia)

    def _construir_scores(self, df_pd):
        return df_pd.apply(self._score_fila, axis=1)

    def _fijar_umbrales(self, scores_train):
        self.umbrales = tuple(np.quantile(scores_train, [1/3, 2/3]))

    def _a_clase(self, scores):
        u1, u2 = self.umbrales
        return pd.Series(np.where(scores <= u1, "Baja",
                         np.where(scores <= u2, "Media", "Alta")),
                         index=scores.index)

    # ────────────────────────────────────────
    # 7. FEATURES (X) — sin precio crudo, sin producto-ID
    # ────────────────────────────────────────

    def _construir_features(self, df_pd, entrenando):
        df_pd = self._asegurar_contexto_temporal(df_pd)

        cols_num = [c for c in FEATURES_PRODUCTO + FEATURES_TEMPORALES + FEATURES_CLIMA
                    if c in df_pd.columns and c != "categoria"]
        X = df_pd[cols_num].copy()
        for c in cols_num:
            X[c] = pd.to_numeric(X[c], errors="coerce").fillna(0)

        if "categoria" in df_pd.columns:
            dummies = pd.get_dummies(df_pd["categoria"].astype(str), prefix="cat")
            if entrenando:
                self.cols_categoria = list(dummies.columns)
            else:
                dummies = dummies.reindex(columns=self.cols_categoria, fill_value=0)
            X = pd.concat([X.reset_index(drop=True),
                           dummies.reset_index(drop=True)], axis=1)

            # Features indirectas de categoría (ver docstring del módulo).
            # Se mantienen como respaldo para productos nunca vistos en train.
            cats = df_pd["categoria"].astype(str)
            for comp in COMPONENTES_PRODUCTO:
                X[f"cat_{comp}_pct"] = cats.map(
                    lambda c: self._pct(self.stats_categoria.get(c, self.stat_global)[comp], comp)
                ).astype(float).values

        # NOTA: se probó incluir la identidad del producto (one-hot) como
        # feature directa. Midió ~99% accuracy en productos conocidos pero
        # solo ~26% en productos nuevos -- memorización de identidad, no
        # aprendizaje del patrón de negocio. Se descarta; solo quedan las
        # features indirectas de categoría y las causales relativas, que
        # sí generalizan.

        if "precio" in df_pd.columns and self._umbrales_precio is not None:
            bandas = df_pd["precio"].apply(self._banda_precio)
            dummies_precio = pd.get_dummies(bandas, prefix="precio")
            for banda in BANDAS_PRECIO:
                col = f"precio_{banda}"
                if col not in dummies_precio.columns:
                    dummies_precio[col] = 0
            X = pd.concat([X.reset_index(drop=True),
                           dummies_precio.reset_index(drop=True)], axis=1)

        if "categoria" in df_pd.columns:
            tiene_fecha = "fecha" in df_pd.columns
            X["cat_tendencia_7"] = [
                self._tendencia_pct(c, df_pd["fecha"].iloc[i]) if tiene_fecha else 1.0
                for i, c in enumerate(df_pd["categoria"].astype(str))
            ]

        if "producto" in df_pd.columns and "fecha" in df_pd.columns:
            causales = pd.DataFrame([
                self._causales_producto(p, f)
                for p, f in zip(df_pd["producto"], df_pd["fecha"])
            ])
            X = pd.concat([X.reset_index(drop=True), causales.reset_index(drop=True)], axis=1)
        else:
            for c, v in _DEFAULT_CAUSAL_PRODUCTO.items():
                X[c] = v

        if entrenando:
            self.feature_names = list(X.columns)
        else:
            X = X.reindex(columns=self.feature_names, fill_value=0)
        return X

    # ────────────────────────────────────────
    # 8. ENTRENAMIENTO + EVALUACIÓN
    # ────────────────────────────────────────

    def entrenar(self, df):
        pd_df = self._agregar_producto_dia(df)
        pd_df = self._asegurar_contexto_temporal(pd_df)
        if len(pd_df) < 30 or pd_df["producto"].nunique() < 2:
            raise ValueError("Datos insuficientes para clasificación.")

        corte = int(len(pd_df) * FRACCION_TRAIN)
        fecha_corte = pd_df["fecha"].iloc[corte]
        train = pd_df[pd_df["fecha"] < fecha_corte].reset_index(drop=True)
        test  = pd_df[pd_df["fecha"] >= fecha_corte].reset_index(drop=True)
        if len(train) < 20 or len(test) < 5:
            train, test = pd_df.iloc[:corte].copy(), pd_df.iloc[corte:].copy()

        self._calcular_stats(train)
        self._calcular_historial_categoria(pd_df)  # todo el historial, no solo train (ver docstring)
        self._calcular_historial_producto(pd_df)    # idem, causal por construcción

        s_train = self._construir_scores(train)
        self._fijar_umbrales(s_train)
        y_train = self._a_clase(s_train)
        y_test  = self._a_clase(self._construir_scores(test))

        X_train = self._construir_features(train, entrenando=True)
        X_test  = self._construir_features(test,  entrenando=False)

        self.mejores_hiperparametros = dict(LGBM_PARAMS)
        lgbm = LGBMClassifier(
            **LGBM_PARAMS, class_weight="balanced",
            random_state=self.random_state, verbosity=-1, n_jobs=1,
        )

        # Calibración: hace que predict_proba() (el campo "confianza") sea
        # un número real, no solo un ranking interno del árbol.
        n_folds_calib = min(3, max(2, y_train.value_counts().min()))
        self.modelo = CalibratedClassifierCV(lgbm, method="sigmoid", cv=n_folds_calib)
        self.modelo.fit(X_train, y_train)

        self._evaluar(X_test, y_test, y_train)

        # Importancias: promedio entre los estimadores internos de la
        # calibración (CalibratedClassifierCV no expone feature_importances_
        # directo, pero cada fold sí entrenó un LGBMClassifier real).
        imp_arrays = [cc.estimator.feature_importances_ for cc in self.modelo.calibrated_classifiers_]
        imp_media = np.mean(imp_arrays, axis=0)
        imp = sorted(zip(self.feature_names, imp_media), key=lambda t: t[1], reverse=True)
        self.importancias = [{"variable": v, "importancia": round(float(i), 4)}
                             for v, i in imp[:10]]
        return self.metricas

    def _evaluar(self, X_test, y_test, y_train):
        pred = self.modelo.predict(X_test)
        probs = self.modelo.predict_proba(X_test)
        confianza_promedio = float(np.mean(np.max(probs, axis=1)))

        acc = accuracy_score(y_test, pred)
        f1_macro = f1_score(y_test, pred, labels=CLASES, average="macro", zero_division=0)
        f1_weight = f1_score(y_test, pred, labels=CLASES, average="weighted", zero_division=0)
        prec, rec, f1c, sup = precision_recall_fscore_support(
            y_test, pred, labels=CLASES, zero_division=0)
        cm = confusion_matrix(y_test, pred, labels=CLASES)

        dummy = DummyClassifier(strategy="most_frequent").fit(X_test, y_test)
        d_strat = DummyClassifier(strategy="stratified",
                                  random_state=self.random_state).fit(X_test, y_test)
        f1_freq = f1_score(y_test, dummy.predict(X_test), labels=CLASES,
                           average="macro", zero_division=0)
        f1_strat = f1_score(y_test, d_strat.predict(X_test), labels=CLASES,
                            average="macro", zero_division=0)
        mejora = ((f1_macro - f1_freq) / f1_freq * 100) if f1_freq > 0 else float("inf")

        # Escala 0-1, nombres sin sufijo _pct: contrato que ya consumen
        # classification_service.py (guarda en BD) y templates/abastecimiento
        # (hace *100 al mostrar). No renombrar sin revisar ambos primero.
        self.metricas = {
            "accuracy": round(float(acc), 4),
            "f1_macro": round(float(f1_macro), 4),
            "f1_weighted": round(float(f1_weight), 4),
            "precision_macro": round(float(np.mean(prec)), 4),
            "recall_macro": round(float(np.mean(rec)), 4),
            "confianza_promedio": round(confianza_promedio, 4),
            "por_clase": {CLASES[i]: {
                "precision": round(float(prec[i]), 4),
                "recall": round(float(rec[i]), 4),
                "f1": round(float(f1c[i]), 4),
                "soporte": int(sup[i])} for i in range(len(CLASES))},
            "matriz_confusion": cm.tolist(),
            "clases": CLASES,
            "baseline_f1_mayoritaria": round(float(f1_freq), 4),
            "baseline_f1_aleatoria": round(float(f1_strat), 4),
            "mejora_vs_baseline_pct": round(float(mejora), 2),
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
            "umbrales_score": [round(float(u), 4) for u in self.umbrales],
            "hiperparametros": self.mejores_hiperparametros,
        }

    # ────────────────────────────────────────
    # 9. PREDICCIÓN
    # ────────────────────────────────────────

    def diagnostico_producto_nuevo(self, categoria: str, contexto_dia: dict) -> dict:
        """
        Chequeo de sanidad: predice para un producto que NO existe en
        stats_producto/hist_producto, para confirmar que cae al
        respaldo por categoría (o al neutral, para las features
        causales) en vez de fallar o devolver basura. Usar tras
        entrenar, antes de confiar en el modelo en producción.
        """
        contexto = {"producto": "__producto_inexistente__", "categoria": categoria,
                   **contexto_dia}
        return self.predecir(contexto)

    def predecir(self, contexto):
        if self.modelo is None:
            raise ValueError("El modelo no está entrenado.")
        X = self._construir_features(pd.DataFrame([contexto]), entrenando=False)
        clase = self.modelo.predict(X)[0]
        probs = self.modelo.predict_proba(X)[0]
        orden = list(self.modelo.classes_)
        return {"prioridad": clase,
                "confianza": round(float(probs[orden.index(clase)]), 4),
                "probabilidades": {c: round(float(probs[orden.index(c)]), 4)
                                   for c in orden}}

    def predecir_lote(self, contextos: list) -> list:
        """
        Igual que predecir(), pero para todos los contextos en una sola
        llamada — evita disparar el paralelismo interno de sklearn una
        vez por cada predicción (causaba OOM en Render con planes chicos).
        """
        if self.modelo is None:
            raise ValueError("El modelo no está entrenado.")
        if not contextos:
            return []
        X = self._construir_features(pd.DataFrame(contextos), entrenando=False)
        clases = self.modelo.predict(X)
        probs = self.modelo.predict_proba(X)
        orden = list(self.modelo.classes_)

        resultados = []
        for i, clase in enumerate(clases):
            fila_probs = probs[i]
            resultados.append({
                "prioridad": clase,
                "confianza": round(float(fila_probs[orden.index(clase)]), 4),
                "probabilidades": {c: round(float(fila_probs[orden.index(c)]), 4)
                                   for c in orden},
            })
        return resultados

    # ────────────────────────────────────────
    # 10. PERSISTENCIA
    # ────────────────────────────────────────

    def guardar(self, ruta):
        import joblib, os
        joblib.dump(self, ruta)
        from services.storage_service import subir
        subir(ruta, os.path.basename(ruta))

    @staticmethod
    def cargar(ruta):
        import joblib, os
        from services.storage_service import asegurar_local
        asegurar_local(os.path.basename(ruta), ruta)
        return joblib.load(ruta)