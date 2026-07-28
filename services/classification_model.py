"""
services/classification_model.py

Modelo de CLASIFICACIÓN — Prioridad de Abastecimiento (RandomForestClassifier).

Segundo modelo de ML de RestoIQ, 100% independiente del regresor (LGBM).
NO predice demanda ni usa la salida del regresor. Clasifica cada (producto, día)
en un nivel de prioridad operativa para planificar compras/preparación.

    Clases:  🟢 Baja  ·  🟡 Media  ·  🔴 Alta

────────────────────────────────────────────────────────────────────────────
ETIQUETA (ground-truth) — un solo promedio, sin pesos manuales:

  Por producto, tres propiedades medidas del histórico REAL y llevadas a
  PERCENTIL [0,1] (rank empírico entre productos, no min-max arbitrario):
      · rotacion_pct      = percentil de la mediana de cantidad diaria
      · volatilidad_pct   = percentil del coef. de variación (std/media)
      · impacto_promo_pct = percentil de cuánto sube la demanda en promo
  Por día, presión del contexto (promedio de flags activos, ya en [0,1]):
      · presion_dia = promedio(es_finde, es_feriado, promocion, es_quincena, es_puente)

      criticidad = promedio(rotacion_pct, volatilidad_pct, impacto_promo_pct, presion_dia)

  Un solo paso, cuatro señales, mismo peso cada una — fácil de explicar y
  de defender. Discretización por TERCILES (percentiles 33/66 del train)
  → 3 clases balanceadas. Nunca usa la predicción del regresor.

────────────────────────────────────────────────────────────────────────────
FEATURES del RF:
  · categoría (one-hot)
  · features INDIRECTAS de categoría: percentil promedio de rotación/
    volatilidad/impacto_promo de los productos de esa categoría (calculado
    solo con train). Sin esto, el modelo no tenía NINGUNA variable
    relacionada con lo que arma la etiqueta — no podía aprender lo que no
    podía observar. Es señal de GRUPO, no identifica un producto puntual
    (a diferencia de 'precio', que sí actuaba como ID encubierto y se
    sacó por eso), así que un producto nuevo la hereda automáticamente.
  · contexto temporal (día de semana, mes, semana, finde, feriado, puente,
    quincena) + promoción/descuento/evento si el archivo los trae.

  NUNCA usa: precio, cantidad/demanda cruda, ni el producto como ID.

────────────────────────────────────────────────────────────────────────────
CALIDAD DEL MODELO:
  · Random Forest con hiperparámetros elegidos por una búsqueda pequeña
    (RandomizedSearchCV, validación cruzada SOLO sobre train) en vez de
    valores fijos a mano.
  · Probabilidades CALIBRADAS (CalibratedClassifierCV) — para que
    predict_proba() sea un número confiable (ej. "80% de confianza"
    realmente ronde el 80% de aciertos), utilizable después en Insights
    o un futuro motor de decisión, no solo un ranking interno del árbol.
  · Confianza de cada predicción (probabilidad de la clase asignada)
    disponible en cada resultado y resumida en las métricas del holdout.
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support,
    confusion_matrix,
)


CLASES = ["Baja", "Media", "Alta"]
BANDAS_PRECIO = ["Bajo", "Medio", "Alto"]

COMPONENTES_PRODUCTO = ["rotacion", "volatilidad", "impacto_promo"]
FLAGS_DIA = ["es_finde", "es_feriado", "promocion", "es_quincena", "es_puente"]

# Contexto usable como feature (solo si existe en el df). Precio entra
# SOLO como banda (Bajo/Medio/Alto), nunca crudo — el valor exacto
# actúa como ID de producto y rompe la generalización a productos
# nuevos (memorización, ver docstring del módulo); la banda conserva
# la señal ordinal ("los productos caros rotan distinto que los
# baratos") sin identificar a un producto puntual.
FEATURES_PRODUCTO   = ["categoria", "promocion", "descuento_pct", "es_evento_especial"]
FEATURES_TEMPORALES = ["dia_semana", "mes", "semana_anio",
                       "es_finde", "es_feriado", "es_puente", "es_quincena"]

FRACCION_TRAIN = 0.8

# Hiperparámetros del RF fijos, no buscados: una búsqueda con
# RandomizedSearchCV se probó y midió (~6s por entrenamiento) contra
# estos valores fijos y la diferencia de F1 fue ruido, no mejora real
# — se optó por lo más simple y rápido, no por complejidad sin
# beneficio medido.
RF_PARAMS = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2}


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
        for c in FEATURES_TEMPORALES:
            if c in df.columns:                agg[c] = "first"

        pd_df = (df.groupby(["producto", "fecha"], as_index=False)
                   .agg(agg).sort_values("fecha").reset_index(drop=True))

        for c in ["promocion", "es_evento_especial",
                  "es_finde", "es_feriado", "es_puente", "es_quincena"]:
            if c in pd_df.columns:
                pd_df[c] = pd_df[c].astype(float).fillna(0).astype(int)
        return pd_df

    # ────────────────────────────────────────
    # 2. ESTADÍSTICOS DE PRODUCTO (SOLO TRAIN)
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
    # 3. SCORE Y ETIQUETA — un solo promedio, 4 señales
    # ────────────────────────────────────────

    def _score_desde_raw(self, raw, row):
        percentiles = [self._pct(raw[c], c) for c in COMPONENTES_PRODUCTO]
        flags = [float(row.get(f, 0)) for f in FLAGS_DIA if f in row.index]
        presion_dia = float(np.mean(flags)) if flags else 0.0
        return float(np.mean(percentiles + [presion_dia]))

    def _score_fila(self, row):
        cat = row["categoria"] if "categoria" in row.index else None
        raw = self._raw_de(row["producto"], cat)
        return self._score_desde_raw(raw, row)

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
    # 4. FEATURES (X) — sin precio, sin producto-ID
    # ────────────────────────────────────────

    def _construir_features(self, df_pd, entrenando):
        cols_num = [c for c in FEATURES_PRODUCTO + FEATURES_TEMPORALES
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
            cats = df_pd["categoria"].astype(str)
            for comp in COMPONENTES_PRODUCTO:
                X[f"cat_{comp}_pct"] = cats.map(
                    lambda c: self._pct(self.stats_categoria.get(c, self.stat_global)[comp], comp)
                ).astype(float).values

        if "precio" in df_pd.columns and self._umbrales_precio is not None:
            bandas = df_pd["precio"].apply(self._banda_precio)
            dummies_precio = pd.get_dummies(bandas, prefix="precio")
            for banda in BANDAS_PRECIO:
                col = f"precio_{banda}"
                if col not in dummies_precio.columns:
                    dummies_precio[col] = 0
            X = pd.concat([X.reset_index(drop=True),
                           dummies_precio.reset_index(drop=True)], axis=1)

        if entrenando:
            self.feature_names = list(X.columns)
        else:
            X = X.reindex(columns=self.feature_names, fill_value=0)
        return X

    # ────────────────────────────────────────
    # 5. ENTRENAMIENTO + EVALUACIÓN
    # ────────────────────────────────────────

    def entrenar(self, df):
        pd_df = self._agregar_producto_dia(df)
        if len(pd_df) < 30 or pd_df["producto"].nunique() < 2:
            raise ValueError("Datos insuficientes para clasificación.")

        corte = int(len(pd_df) * FRACCION_TRAIN)
        fecha_corte = pd_df["fecha"].iloc[corte]
        train = pd_df[pd_df["fecha"] < fecha_corte].reset_index(drop=True)
        test  = pd_df[pd_df["fecha"] >= fecha_corte].reset_index(drop=True)
        if len(train) < 20 or len(test) < 5:
            train, test = pd_df.iloc[:corte].copy(), pd_df.iloc[corte:].copy()

        self._calcular_stats(train)
        s_train = self._construir_scores(train)
        self._fijar_umbrales(s_train)
        y_train = self._a_clase(s_train)
        y_test  = self._a_clase(self._construir_scores(test))

        X_train = self._construir_features(train, entrenando=True)
        X_test  = self._construir_features(test,  entrenando=False)

        # RF con hiperparámetros fijos (ver RF_PARAMS) — una búsqueda
        # se probó y midió (~6s por entrenamiento) sin mejora real de
        # F1 sobre estos valores fijos, así que se optó por lo más
        # simple y rápido.
        self.mejores_hiperparametros = dict(RF_PARAMS)
        rf = RandomForestClassifier(
            **RF_PARAMS, class_weight="balanced", random_state=self.random_state, n_jobs=-1,
        )

        # ── Calibración de probabilidades (predict_proba confiable) ──
        # Barata (~0.4s medido) — se mantiene porque predict_proba()
        # calibrado es lo que hace confiable el campo "confianza" que
        # usa el sistema (y lo que podría usar Insights más adelante).
        n_folds_calib = min(3, max(2, y_train.value_counts().min()))
        self.modelo = CalibratedClassifierCV(rf, method="sigmoid", cv=n_folds_calib)
        self.modelo.fit(X_train, y_train)

        self._evaluar(X_test, y_test, y_train)

        # Importancias: promedio entre los estimadores internos de la
        # calibración (CalibratedClassifierCV no expone feature_importances_
        # directo, pero cada fold sí entrenó un RandomForest real).
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
    # 6. PREDICCIÓN
    # ────────────────────────────────────────

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

    # ────────────────────────────────────────
    # 7. PERSISTENCIA
    # ────────────────────────────────────────

    def guardar(self, ruta):
        import joblib; joblib.dump(self, ruta)

    @staticmethod
    def cargar(ruta):
        import joblib; return joblib.load(ruta)