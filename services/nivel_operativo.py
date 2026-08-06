"""
services/nivel_operativo.py — MÓDULO DE PRUEBA, tercer clasificador
en evaluación (no reemplaza a los anteriores todavía).

Objetivo de negocio: "¿cómo organizo el restaurante mañana?" — no
"¿cuánto voy a vender?" (eso ya lo responde el modelo de regresión).

    🔴 Alta   → nivel operativo alto: se prepara más, llega todo el personal.
    🟢 Normal → operación estándar.
    🟡 Baja   → se reduce la preparación anticipada de perecederos.

Por qué es un problema de ML genuino (y no una regla disfrazada):
las features son PURO CALENDARIO (día de semana, feriado, quincena,
evento deportivo...) — el calendario SUGIERE el nivel operativo, pero
no lo determina matemáticamente. Un viernes puede terminar siendo
"Alta" o "Normal" según cómo se dé en la realidad — ahí es donde el
modelo aprende un patrón real, no aplica una fórmula.

Independiente del modelo de regresión: NUNCA usa la demanda predicha
como entrada — solo el mismo calendario que cualquiera podría mirar
de antemano. Esto permite comparar los 2 modelos entre sí después
(¿coinciden la cantidad predicha y el nivel clasificado? si no
coinciden, es una señal de alerta para revisar la planificación) —
comparación que se implementa aparte, no acá.

Organización:
    1. Preparación de datos (agregación diaria)
    2. Construcción del target (terciles de demanda diaria total)
    3. Ingeniería de variables (calendario + Clásico del Astillero)
    4. Entrenamiento (LightGBM principal, RandomForest de referencia)
    5. Evaluación
    6. Predicción
"""

from datetime import timedelta
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, f1_score, precision_recall_fscore_support, confusion_matrix,
)
import lightgbm as lgb


CLASES = ["Baja", "Normal", "Alta"]
FRACCION_TRAIN = 0.8
TIPOS_CLASICO = ["ninguno", "tarde", "noche"]  # fijo, para que cada pliegue del walk-forward tenga las mismas columnas

LGBM_PARAMS = {"n_estimators": 200, "num_leaves": 31, "min_child_samples": 20, "learning_rate": 0.05}
RF_PARAMS   = {"n_estimators": 200, "max_depth": 10, "min_samples_leaf": 2}


class ClasificadorNivelOperativo:
    """Clasifica el nivel operativo del PRÓXIMO día (Baja/Normal/Alta)
    a partir únicamente de su calendario — nunca de la demanda
    predicha por el otro modelo."""

    def __init__(self, random_state=42):
        self.modelo = None
        self.algoritmo_ganador = None
        self.random_state = random_state
        self.umbral_bajo = None
        self.umbral_alto = None
        self.feature_names = []
        self.feriados = self._cargar_feriados()
        self.clasicos = self._cargar_clasicos()
        self.dia_historial = None
        self.metricas = {}
        self.importancias = []

    # ────────────────────────────────────────
    # Calendarios externos — mismo patrón que sales_model.py
    # ────────────────────────────────────────

    def _cargar_feriados(self):
        for mod in ["services.data_generator", "data_generator"]:
            try:
                import importlib
                return importlib.import_module(mod).FERIADOS_ECUADOR
            except Exception:
                continue
        return set()

    def _cargar_clasicos(self):
        for mod in ["services.data_generator", "data_generator"]:
            try:
                import importlib
                return importlib.import_module(mod).CLASICOS_ASTILLERO
            except Exception:
                continue
        return {}

    # ────────────────────────────────────────
    # 1. PREPARACIÓN DE DATOS — agregación diaria
    # ────────────────────────────────────────

    def _agregar_dia(self, df):
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])
        agg = {"cantidad": "sum"}
        if "promocion" in df.columns:
            agg["promocion"] = "max"
        dia_df = df.groupby("fecha", as_index=False).agg(agg).sort_values("fecha").reset_index(drop=True)

        # Calendario continuo (mismo fix que sales_model.py) — para
        # que lag_7/rolling_mean sean válidos día a día, sin huecos.
        rango = pd.date_range(dia_df["fecha"].min(), dia_df["fecha"].max(), freq="D")
        dia_df = (dia_df.set_index("fecha").reindex(rango, fill_value=0)
                  .rename_axis("fecha").reset_index())
        if "promocion" not in dia_df.columns:
            dia_df["promocion"] = 0

        # Historial REAL (lo que ya pasó) — no la salida de otro
        # modelo. "¿Cómo venía la semana pasada?" es información
        # legítima para predecir el nivel operativo de hoy, distinta
        # de "cuánto exactamente" que responde el regresor.
        dia_df["lag_7"] = dia_df["cantidad"].shift(7)
        dia_df["rolling_7_mean"]  = dia_df["cantidad"].shift(1).rolling(7,  min_periods=1).mean()
        dia_df["rolling_14_mean"] = dia_df["cantidad"].shift(1).rolling(14, min_periods=1).mean()

        return dia_df.dropna(subset=["lag_7"]).reset_index(drop=True)

    # ────────────────────────────────────────
    # 2. CONSTRUCCIÓN DEL TARGET — terciles de demanda diaria total
    # ────────────────────────────────────────
    # Una sola variable ordenada, cortada en 3 tercios — balanceado
    # por construcción, sin el riesgo de combinar 2 señales con
    # AND/OR que dejó una clase vacía en un intento anterior.

    def _fijar_umbrales(self, cantidades_train):
        self.umbral_bajo, self.umbral_alto = np.quantile(cantidades_train, [1/3, 2/3])

    def _clasificar_nivel(self, cantidad) -> str:
        if cantidad <= self.umbral_bajo:
            return "Baja"
        if cantidad >= self.umbral_alto:
            return "Alta"
        return "Normal"

    # ────────────────────────────────────────
    # 3. INGENIERÍA DE VARIABLES — puro calendario
    # ────────────────────────────────────────

    def _features_fecha(self, fecha: pd.Timestamp) -> dict:
        fecha_str = fecha.strftime("%Y-%m-%d")
        dia       = fecha.weekday()
        ayer      = (fecha - timedelta(days=1)).strftime("%Y-%m-%d")
        maniana   = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
        dia_mes   = fecha.day
        return {
            "dia_semana":      dia,
            "dia_mes":         dia_mes,
            "mes":             fecha.month,
            "semana_anio":     int(fecha.isocalendar()[1]),
            "es_finde":        int(dia in [5, 6]),
            "es_feriado":      int(fecha_str in self.feriados),
            "vispera_feriado": int(maniana in self.feriados),
            "es_puente":       int(ayer in self.feriados or maniana in self.feriados),
            "es_quincena":     int(1 <= dia_mes <= 7 or 15 <= dia_mes <= 21),
            "tipo_clasico":    self.clasicos.get(fecha_str, "ninguno"),  # "tarde" / "noche" / "ninguno"
        }

    def _construir_features(self, dia_df, entrenando):
        filas = [self._features_fecha(f) for f in dia_df["fecha"]]
        base = pd.DataFrame(filas)
        if "promocion" in dia_df.columns:
            base["promocion"] = dia_df["promocion"].fillna(0).astype(int).values
        base["lag_7"]           = dia_df["lag_7"].values
        base["rolling_7_mean"]  = dia_df["rolling_7_mean"].values
        base["rolling_14_mean"] = dia_df["rolling_14_mean"].values

        dummies = pd.get_dummies(
            pd.Categorical(base["tipo_clasico"], categories=TIPOS_CLASICO), prefix="clasico"
        )
        X = pd.concat([base.drop(columns=["tipo_clasico"]).reset_index(drop=True),
                       dummies.reset_index(drop=True)], axis=1)

        if entrenando:
            self.feature_names = list(X.columns)
        else:
            X = X.reindex(columns=self.feature_names, fill_value=0)
        return X

    # ────────────────────────────────────────
    # 4. ENTRENAMIENTO — LightGBM (principal) vs. RandomForest (referencia)
    # ────────────────────────────────────────

    def _entrenar_lightgbm(self, X_train, y_train):
        modelo = lgb.LGBMClassifier(
            **LGBM_PARAMS, class_weight="balanced", random_state=self.random_state,
            n_jobs=1, verbose=-1,
        )
        modelo.fit(X_train, y_train)
        return modelo

    def _entrenar_random_forest(self, X_train, y_train):
        modelo = RandomForestClassifier(
            **RF_PARAMS, class_weight="balanced", random_state=self.random_state, n_jobs=1,
        )
        modelo.fit(X_train, y_train)
        return modelo

    def entrenar(self, df):
        dia_df = self._agregar_dia(df)
        if len(dia_df) < 90:
            raise ValueError("Datos insuficientes para clasificar nivel operativo con walk-forward.")

        self.dia_historial = dia_df[["fecha", "cantidad"]].copy()  # para calcular lags al predecir fechas futuras

        from sklearn.model_selection import TimeSeriesSplit
        n_folds = 5 if len(dia_df) > 250 else 3
        tscv = TimeSeriesSplit(n_splits=n_folds)

        y_reales_todos, pred_lgbm_todos, pred_rf_todos = [], [], []

        for idx_train, idx_test in tscv.split(dia_df):
            train = dia_df.iloc[idx_train].reset_index(drop=True)
            test  = dia_df.iloc[idx_test].reset_index(drop=True)

            # Umbrales recalibrados EN CADA PLIEGUE, solo con el train
            # de ese pliegue — nunca con datos del futuro. Esto es lo
            # que corrige el desbalance que encontramos con un solo
            # corte fijo (la tendencia de crecimiento del negocio
            # dejaba desactualizado un umbral calculado una sola vez).
            self._fijar_umbrales(train["cantidad"])
            y_train = train["cantidad"].apply(self._clasificar_nivel)
            y_test  = test["cantidad"].apply(self._clasificar_nivel)

            X_train = self._construir_features(train, entrenando=True)
            X_test  = self._construir_features(test, entrenando=False)

            modelo_lgbm_fold = self._entrenar_lightgbm(X_train, y_train)
            modelo_rf_fold   = self._entrenar_random_forest(X_train, y_train)

            y_reales_todos.extend(y_test.tolist())
            pred_lgbm_todos.extend(modelo_lgbm_fold.predict(X_test).tolist())
            pred_rf_todos.extend(modelo_rf_fold.predict(X_test).tolist())

        metricas_lgbm = self._calcular_metricas(y_reales_todos, pred_lgbm_todos)
        metricas_rf   = self._calcular_metricas(y_reales_todos, pred_rf_todos)

        if metricas_lgbm["f1_macro"] >= metricas_rf["f1_macro"]:
            self.algoritmo_ganador, metricas_ganador = "LightGBM", metricas_lgbm
        else:
            self.algoritmo_ganador, metricas_ganador = "RandomForest", metricas_rf

        # Modelo FINAL: se reentrena con el 100% de los datos (mismo
        # criterio que sales_model.py) — el walk-forward de arriba es
        # solo para MEDIR honestamente, nunca es el modelo que queda
        # guardado para predecir de verdad.
        self._fijar_umbrales(dia_df["cantidad"])
        y_todo = dia_df["cantidad"].apply(self._clasificar_nivel)
        X_todo = self._construir_features(dia_df, entrenando=True)
        self.modelo = (self._entrenar_lightgbm(X_todo, y_todo) if self.algoritmo_ganador == "LightGBM"
                       else self._entrenar_random_forest(X_todo, y_todo))

        self.metricas = {
            **metricas_ganador,
            "algoritmo_ganador": self.algoritmo_ganador,
            "comparacion_algoritmos": {"LightGBM": metricas_lgbm, "RandomForest": metricas_rf},
            "n_folds": n_folds,
            "umbral_bajo": round(float(self.umbral_bajo), 1),
            "umbral_alto": round(float(self.umbral_alto), 1),
        }

        imp = sorted(zip(self.feature_names, self.modelo.feature_importances_),
                    key=lambda t: t[1], reverse=True)
        self.importancias = [{"variable": v, "importancia": round(float(i), 3)} for v, i in imp[:10]]

        return self.metricas

    # ────────────────────────────────────────
    # 5. EVALUACIÓN — métricas 0-1, 3 decimales
    # ────────────────────────────────────────

    def _calcular_metricas(self, y_true, y_pred):
        acc = accuracy_score(y_true, y_pred)
        f1_macro = f1_score(y_true, y_pred, labels=CLASES, average="macro", zero_division=0)
        f1_weight = f1_score(y_true, y_pred, labels=CLASES, average="weighted", zero_division=0)
        prec, rec, f1c, sup = precision_recall_fscore_support(y_true, y_pred, labels=CLASES, zero_division=0)
        cm = confusion_matrix(y_true, y_pred, labels=CLASES)

        # Baseline "ingenuo": predecir siempre la clase más frecuente
        # de verdad (sin usar ningún modelo) — sirve para saber si el
        # modelo realmente aporta algo sobre no hacer nada.
        clase_mayoritaria = pd.Series(y_true).mode()[0]
        pred_base = [clase_mayoritaria] * len(y_true)
        f1_base = f1_score(y_true, pred_base, labels=CLASES, average="macro", zero_division=0)
        mejora = ((f1_macro - f1_base) / f1_base * 100) if f1_base > 0 else float("inf")

        return {
            "accuracy":        round(float(acc), 3),
            "f1_macro":        round(float(f1_macro), 3),
            "f1_weighted":     round(float(f1_weight), 3),
            "precision_macro": round(float(np.mean(prec)), 3),
            "recall_macro":    round(float(np.mean(rec)), 3),
            "por_clase": {CLASES[i]: {
                "precision": round(float(prec[i]), 3),
                "recall":    round(float(rec[i]), 3),
                "f1":        round(float(f1c[i]), 3),
                "soporte":   int(sup[i]),
            } for i in range(len(CLASES))},
            "matriz_confusion": cm.tolist(),
            "clases": CLASES,
            "baseline_f1_mayoritaria": round(float(f1_base), 3),
            "mejora_vs_baseline_pct":  round(float(mejora), 1),
            "n_total": len(y_true),
        }

    # ────────────────────────────────────────
    # 6. PREDICCIÓN
    # ────────────────────────────────────────

    def predecir(self, fecha: pd.Timestamp) -> dict:
        if self.modelo is None:
            raise ValueError("El modelo no está entrenado.")

        hist = self.dia_historial.set_index("fecha")["cantidad"]
        fecha_lag7 = fecha - timedelta(days=7)
        lag_7 = float(hist.get(fecha_lag7, hist.mean() if len(hist) else 0))

        reciente = hist.sort_index()
        reciente = reciente[reciente.index < fecha].tail(14)
        rolling_7  = float(reciente.tail(7).mean())  if len(reciente) >= 1 else lag_7
        rolling_14 = float(reciente.mean())          if len(reciente) >= 1 else lag_7

        dia_df = pd.DataFrame({"fecha": [fecha], "lag_7": [lag_7],
                               "rolling_7_mean": [rolling_7], "rolling_14_mean": [rolling_14]})
        X = self._construir_features(dia_df, entrenando=False)
        clase = self.modelo.predict(X)[0]
        probs = self.modelo.predict_proba(X)[0]
        orden = list(self.modelo.classes_)
        return {"nivel_operativo": clase,
                "confianza": round(float(probs[orden.index(clase)]), 3)}

    # ────────────────────────────────────────
    # 7. PERSISTENCIA
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
