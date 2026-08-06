"""Clasificador de prioridad de abastecimiento.

La prioridad es una predicción de la demanda *del producto en ese día*:
``Baja``, ``Media`` o ``Alta``. La etiqueta sale de la cantidad que
realmente se vendió, comparada contra umbrales RELATIVOS al propio
producto (terciles calculados con SU historial, no con el de todos los
productos mezclados) -- así "Alta" significa "alto para este producto
específico", no "grande en términos absolutos". Un producto sin
suficiente historial cae al respaldo por categoría, y si tampoco hay
categoría con datos suficientes, al umbral global. Los umbrales se
calculan únicamente del periodo de entrenamiento de cada pliegue.

Las variables de historial se calculan siempre con ``shift(1)``. En
consecuencia, la fila de fecha *t* nunca ve su cantidad ni la del futuro.
"""

from __future__ import annotations

import os
from datetime import timedelta

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.metrics import (
    accuracy_score, confusion_matrix, f1_score,
    precision_recall_fscore_support,
)


CLASES = ["Baja", "Media", "Alta"]
FRACCION_TRAIN = 0.8
MINIMO_FILAS = 45
MIN_FILAS_UMBRAL_PRODUCTO = 10
MIN_FILAS_UMBRAL_CATEGORIA = 20
LGBM_PARAMS = {
    "n_estimators": 350, "learning_rate": 0.04, "num_leaves": 31,
    "min_child_samples": 12, "subsample": 0.85, "colsample_bytree": 0.85,
    "reg_lambda": 0.5,
}
FEATURES_CALENDARIO = [
    "dia_semana", "dia_mes", "mes", "semana_anio", "es_finde",
    "es_feriado", "es_puente", "es_quincena",
]
FEATURES_HISTORICAS = [
    "lag_1", "lag_7", "lag_14", "lag_28", "rolling_7_mean",
    "rolling_14_mean", "rolling_28_mean", "rolling_7_std",
    "media_historica", "volatilidad_historica",
]
FEATURES_COMERCIALES = ["promocion", "descuento_pct", "es_evento_especial"]


class ModeloAbastecimiento:
    """Prioridad producto-día con LightGBM y validación walk-forward."""

    def __init__(self, random_state=42):
        self.modelo = None
        self.random_state = random_state
        self.feature_names = []
        self.cols_categoria = []
        self.cols_producto = []
        self.umbrales_producto = {}
        self.umbrales_categoria = {}
        self.umbrales_global = None
        self.clases = CLASES
        self.metricas = {}
        self.importancias = []
        self.historial_producto = {}
        self.perfil_producto = {}
        self.perfil_categoria = {}
        self.media_global = 0.0
        self.mejores_hiperparametros = dict(LGBM_PARAMS)
        # Sesgo de decisión a favor de "Alta" (recall) sobre las otras 2
        # clases. 1.0 = neutral, igual que predict() por defecto. >1.0
        # favorece detectar más casos reales de Alta, a costa de algunas
        # falsas alarmas (más Media/Baja mal clasificadas como Alta).
        # No cambia las probabilidades que se le muestran al usuario,
        # solo la regla final de "cuál clase elijo".
        self.sesgo_alta = 1.0

    # Preparación ---------------------------------------------------------

    @staticmethod
    def _fecha_features(fecha):
        fecha = pd.Timestamp(fecha)
        return {
            "dia_semana": fecha.weekday(), "dia_mes": fecha.day,
            "mes": fecha.month, "semana_anio": int(fecha.isocalendar()[1]),
            "es_finde": int(fecha.weekday() >= 5),
            "es_quincena": int(1 <= fecha.day <= 7 or 15 <= fecha.day <= 21),
        }

    def _agregar_producto_dia(self, df):
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
        df = df.dropna(subset=["fecha", "producto", "cantidad"])
        agg = {"cantidad": "sum"}
        for c, fn in (("precio", "mean"), ("categoria", "first"),
                      ("descuento_pct", "mean"), ("promocion", "max"),
                      ("es_evento_especial", "max")):
            if c in df.columns:
                agg[c] = fn
        for c in FEATURES_CALENDARIO:
            if c in df.columns:
                agg[c] = "first"
        diario = df.groupby(["producto", "fecha"], as_index=False).agg(agg)

        # Un calendario continuo hace que lag_7 signifique realmente siete
        # días y no "siete registros anteriores" cuando hubo días sin venta.
        piezas = []
        for producto, g in diario.groupby("producto", sort=False):
            g = g.set_index("fecha").sort_index()
            fechas = pd.date_range(g.index.min(), g.index.max(), freq="D")
            p = g.reindex(fechas)
            p.index.name = "fecha"
            p["producto"] = producto
            p["cantidad"] = p["cantidad"].fillna(0.0)
            for c in ("categoria", "precio"):
                if c in p:
                    p[c] = p[c].ffill().bfill()
            for c in FEATURES_COMERCIALES:
                if c in p:
                    p[c] = pd.to_numeric(p[c], errors="coerce").fillna(0.0)
            piezas.append(p.reset_index())
        return pd.concat(piezas, ignore_index=True).sort_values(["fecha", "producto"]).reset_index(drop=True)

    def _con_historial(self, diario):
        piezas = []
        for _, g in diario.groupby("producto", sort=False):
            g = g.sort_values("fecha").copy()
            q = g["cantidad"].astype(float)
            for lag in (1, 7, 14, 28):
                g[f"lag_{lag}"] = q.shift(lag)
            previo = q.shift(1)
            g["rolling_7_mean"] = previo.rolling(7, min_periods=1).mean()
            g["rolling_14_mean"] = previo.rolling(14, min_periods=1).mean()
            g["rolling_28_mean"] = previo.rolling(28, min_periods=1).mean()
            g["rolling_7_std"] = previo.rolling(7, min_periods=2).std().fillna(0.0)
            g["media_historica"] = previo.expanding(min_periods=1).mean()
            g["volatilidad_historica"] = previo.expanding(min_periods=2).std().fillna(0.0)
            piezas.append(g)
        return pd.concat(piezas, ignore_index=True).dropna(subset=["lag_28"]).reset_index(drop=True)

    # Target --------------------------------------------------------------

    @staticmethod
    def _quantiles_seguros(cantidades):
        """u1, u2 (terciles) o None si no hay suficiente separación."""
        arr = np.asarray(cantidades, dtype=float)
        if len(arr) < 3:
            return None
        u1, u2 = np.quantile(arr, [1 / 3, 2 / 3])
        return None if u1 >= u2 else (float(u1), float(u2))

    def _fijar_umbrales(self, datos):
        """Umbrales RELATIVOS: por producto, con respaldo por categoría y
        global (mismo patrón producto->categoria->global que el resto del
        proyecto). 'Alta' significa 'alto PARA ESTE producto', no 'grande
        en términos absolutos' -- con umbrales globales, un producto de
        bajo volumen natural (ej. un postre especial) casi nunca llegaría
        a 'Alta' aunque tuviera su mejor día histórico, y uno de volumen
        alto natural (ej. papas) casi nunca bajaría de 'Alta' aunque
        tuviera un día flojo. Se calcula con TODO el 'datos' recibido
        (train de ese pliegue, nunca test) para no fugar información."""
        self.umbrales_global = self._quantiles_seguros(datos["cantidad"])
        if self.umbrales_global is None:
            raise ValueError(
                "No hay suficiente variación de ventas para formar tres prioridades útiles."
            )

        self.umbrales_categoria = {}
        if "categoria" in datos.columns:
            for cat, g in datos.groupby("categoria"):
                if len(g) >= MIN_FILAS_UMBRAL_CATEGORIA:
                    u = self._quantiles_seguros(g["cantidad"])
                    if u is not None:
                        self.umbrales_categoria[cat] = u

        self.umbrales_producto = {}
        for prod, g in datos.groupby("producto"):
            if len(g) >= MIN_FILAS_UMBRAL_PRODUCTO:
                u = self._quantiles_seguros(g["cantidad"])
                if u is not None:
                    self.umbrales_producto[prod] = u

    def _umbrales_para(self, producto, categoria):
        if producto in self.umbrales_producto:
            return self.umbrales_producto[producto]
        if categoria in self.umbrales_categoria:
            return self.umbrales_categoria[categoria]
        return self.umbrales_global

    def _origen_umbral(self, producto, categoria):
        """Indica qué nivel definió la etiqueta; útil para auditar el F1."""
        if producto in self.umbrales_producto:
            return "producto"
        if categoria in self.umbrales_categoria:
            return "categoria"
        return "global"

    def _a_clase(self, datos):
        """datos: DataFrame con columnas producto, categoria (opcional) y
        cantidad -- ya no una Series suelta, porque el umbral depende de
        QUÉ producto es cada fila."""
        cat_col = (datos["categoria"] if "categoria" in datos.columns
                  else pd.Series([None] * len(datos), index=datos.index))
        clases = []
        for cantidad, prod, cat in zip(datos["cantidad"], datos["producto"], cat_col):
            u1, u2 = self._umbrales_para(prod, cat)
            if cantidad <= u1:
                clases.append("Baja")
            elif cantidad <= u2:
                clases.append("Media")
            else:
                clases.append("Alta")
        return pd.Series(clases, index=datos.index)

    # Features ------------------------------------------------------------

    def _construir_features(self, df, entrenando):
        base = pd.DataFrame(index=df.index)
        for c in FEATURES_CALENDARIO:
            if c in df:
                base[c] = pd.to_numeric(df[c], errors="coerce").fillna(0)
            elif c in ("dia_semana", "dia_mes", "mes", "semana_anio", "es_finde", "es_quincena") and "fecha" in df:
                base[c] = [self._fecha_features(f)[c] for f in df["fecha"]]
            else:
                base[c] = 0
        for c in FEATURES_COMERCIALES + FEATURES_HISTORICAS:
            base[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0) if c in df else 0.0

        # LightGBM trata las variables categóricas nativamente: evita crear
        # decenas de dummies de producto y permite aprender particiones por
        # categoría sin imponer un orden artificial. Las categorías se fijan
        # con el train de cada pliegue; un producto nuevo queda como valor
        # desconocido (missing para LightGBM), no como una identidad vista.
        for columna in ("categoria", "producto"):
            valores = (
                df[columna].fillna("Sin categoria").astype(str)
                if columna in df else pd.Series("Sin categoria", index=df.index)
            )
            if entrenando:
                categorias = sorted(valores.unique().tolist())
                if columna == "categoria":
                    self.cols_categoria = categorias
                else:
                    self.cols_producto = categorias
            else:
                categorias = (
                    self.cols_categoria if columna == "categoria"
                    else self.cols_producto
                )
            base[columna] = pd.Categorical(valores, categories=categorias)
        if entrenando:
            self.feature_names = list(base.columns)
        return base.reindex(columns=self.feature_names, fill_value=0.0)

    @staticmethod
    def _entrenar_lgbm(X, y, random_state):
        modelo = lgb.LGBMClassifier(
            **LGBM_PARAMS, class_weight="balanced", random_state=random_state,
            n_jobs=1, verbose=-1,
        )
        categoricas = [c for c in ("categoria", "producto") if c in X.columns]
        modelo.fit(X, y, categorical_feature=categoricas)
        return modelo

    def _clasificar_desde_probs(self, fila_probs, orden):
        """Elige la clase final aplicando self.sesgo_alta -- NO cambia
        las probabilidades reales (esas se siguen reportando tal cual
        salen del modelo), solo la regla de decisión: en vez de argmax
        directo, la probabilidad de 'Alta' se multiplica por el sesgo
        antes de comparar. Con sesgo_alta=1.0 es idéntico a argmax."""
        idx_alta = orden.index("Alta")
        ajustadas = list(fila_probs)
        ajustadas[idx_alta] = ajustadas[idx_alta] * self.sesgo_alta
        return orden[int(np.argmax(ajustadas))]

    def _metricas(self, y_true, pred, probs=None):
        acc = accuracy_score(y_true, pred)
        prec, rec, f1c, sup = precision_recall_fscore_support(y_true, pred, labels=CLASES, zero_division=0)
        base = DummyClassifier(strategy="most_frequent").fit(np.zeros((len(y_true), 1)), y_true)
        f1_base = f1_score(y_true, base.predict(np.zeros((len(y_true), 1))), labels=CLASES, average="macro", zero_division=0)
        f1 = f1_score(y_true, pred, labels=CLASES, average="macro", zero_division=0)
        return {
            "accuracy": round(float(acc), 4), "f1_macro": round(float(f1), 4),
            "f1_weighted": round(float(f1_score(y_true, pred, labels=CLASES, average="weighted", zero_division=0)), 4),
            "precision_macro": round(float(np.mean(prec)), 4), "recall_macro": round(float(np.mean(rec)), 4),
            "confianza_promedio": round(float(np.mean(np.max(probs, axis=1))), 4) if probs is not None else None,
            "por_clase": {CLASES[i]: {"precision": round(float(prec[i]), 4), "recall": round(float(rec[i]), 4), "f1": round(float(f1c[i]), 4), "soporte": int(sup[i])} for i in range(3)},
            "matriz_confusion": confusion_matrix(y_true, pred, labels=CLASES).tolist(), "clases": CLASES,
            "baseline_f1_mayoritaria": round(float(f1_base), 4),
            "mejora_vs_baseline_pct": round(float((f1 - f1_base) / f1_base * 100), 2) if f1_base else None,
            "n_test": int(len(y_true)),
        }

    @staticmethod
    def _pliegues_por_fecha(datos, n_folds):
        """Genera validaciones walk-forward sin partir un mismo día.

        ``TimeSeriesSplit`` trabaja por fila. Como aquí cada fecha contiene
        varios productos, puede dejar productos del mismo día en train y test.
        Al separar primero las fechas únicas, el conjunto de prueba representa
        días completamente futuros para todos los productos.
        """
        fechas = np.sort(datos["fecha"].dropna().unique())
        if len(fechas) <= n_folds:
            return []

        # Conserva la proporción creciente de TimeSeriesSplit, pero sobre días
        # completos (no sobre pares producto-día).
        tam_test = len(fechas) // (n_folds + 1)
        if tam_test == 0:
            return []
        inicio_test = len(fechas) - n_folds * tam_test
        pliegues = []
        for i in range(n_folds):
            corte = inicio_test + i * tam_test
            fin = corte + tam_test if i < n_folds - 1 else len(fechas)
            fecha_corte, fecha_fin = fechas[corte], fechas[fin - 1]
            train = datos.loc[datos["fecha"] < fecha_corte].copy()
            test = datos.loc[
                (datos["fecha"] >= fecha_corte) & (datos["fecha"] <= fecha_fin)
            ].copy()
            if not train.empty and not test.empty:
                pliegues.append((train, test))
        return pliegues

    def entrenar(self, df):
        diario = self._agregar_producto_dia(df)
        datos = self._con_historial(diario)
        if len(datos) < MINIMO_FILAS or datos["producto"].nunique() < 2:
            raise ValueError("Datos insuficientes: se requieren al menos 45 días-producto con historial.")
        datos = datos.sort_values(["fecha", "producto"]).reset_index(drop=True)

        # Walk-forward: cada pliegue aprende umbrales y modelo solo del pasado.
        n_folds = 3 if len(datos) >= 120 else 2
        pliegues = self._pliegues_por_fecha(datos, n_folds)
        reales, predicciones, probabilidades, origenes = [], [], [], []
        for train, test in pliegues:
            try:
                self._fijar_umbrales(train)
            except ValueError:
                # En los primeros meses puede haber solo ceros o dos niveles
                # de venta; ese pliegue no permite medir tres clases aún.
                continue
            y_train, y_test = self._a_clase(train), self._a_clase(test)
            if y_train.nunique() < 3:
                continue
            X_train = self._construir_features(train, entrenando=True)
            X_test = self._construir_features(test, entrenando=False)
            modelo = self._entrenar_lgbm(X_train, y_train, self.random_state)
            orden = list(modelo.classes_)
            probs_test = modelo.predict_proba(X_test)
            pred_test = [self._clasificar_desde_probs(fila, orden) for fila in probs_test]
            reales.extend(y_test.tolist()); predicciones.extend(pred_test); probabilidades.extend(probs_test.tolist())
            categorias_test = test.get("categoria", pd.Series(None, index=test.index))
            origenes.extend(
                self._origen_umbral(producto, categoria)
                for producto, categoria in zip(test["producto"], categorias_test)
            )
        if not reales:
            raise ValueError("El historial no permite una validación temporal con las tres prioridades.")
        self.metricas = self._metricas(reales, predicciones, np.asarray(probabilidades))
        self.metricas["n_folds"] = len(pliegues)
        self.metricas["validacion"] = "walk-forward por fecha completa"
        conteo_origenes = pd.Series(origenes).value_counts()
        self.metricas["cobertura_umbral_validacion"] = {
            origen: round(float(conteo_origenes.get(origen, 0) / len(origenes)), 4)
            for origen in ("producto", "categoria", "global")
        }

        # Modelo que queda en producción: todo el historial disponible.
        self._fijar_umbrales(datos)
        y = self._a_clase(datos)
        if y.nunique() < 3:
            raise ValueError(
                "El historial no contiene ejemplos suficientes de Baja, Media y Alta."
            )
        X = self._construir_features(datos, entrenando=True)
        self.modelo = self._entrenar_lgbm(X, y, self.random_state)
        self.metricas.update({
            "n_train": int(len(y)),
            "umbrales_global": [round(x, 4) for x in self.umbrales_global],
            "productos_con_umbral_propio": len(self.umbrales_producto),
            "categorias_con_umbral_propio": len(self.umbrales_categoria),
            "hiperparametros": dict(LGBM_PARAMS), "algoritmo": "LightGBM",
        })
        imp = sorted(zip(self.feature_names, self.modelo.feature_importances_), key=lambda x: x[1], reverse=True)
        self.importancias = [{"variable": n, "importancia": round(float(v), 4)} for n, v in imp[:10]]

        self.historial_producto = {p: g.set_index("fecha")["cantidad"].astype(float).to_dict() for p, g in diario.groupby("producto")}
        self.perfil_producto = diario.groupby("producto")["cantidad"].mean().to_dict()
        self.perfil_categoria = diario.groupby("categoria")["cantidad"].mean().to_dict() if "categoria" in diario else {}
        self.media_global = float(diario["cantidad"].mean())
        return self.metricas

    # Predicción ----------------------------------------------------------

    def _historial_contexto(self, contexto):
        producto, fecha = contexto.get("producto"), pd.Timestamp(contexto.get("fecha", pd.Timestamp.today()))
        hist = self.historial_producto.get(producto, {})
        fallback = self.perfil_producto.get(producto, self.perfil_categoria.get(contexto.get("categoria"), self.media_global))
        serie = pd.Series(hist, dtype=float)
        def valor(dia): return float(serie.get(dia, fallback))
        previos = pd.Series([valor(fecha - timedelta(days=i)) for i in range(1, 29)])
        return {"lag_1": valor(fecha - timedelta(days=1)), "lag_7": valor(fecha - timedelta(days=7)), "lag_14": valor(fecha - timedelta(days=14)), "lag_28": valor(fecha - timedelta(days=28)), "rolling_7_mean": float(previos.iloc[:7].mean()), "rolling_14_mean": float(previos.iloc[:14].mean()), "rolling_28_mean": float(previos.mean()), "rolling_7_std": float(previos.iloc[:7].std(ddof=0)), "media_historica": float(serie[serie.index < fecha].mean()) if not serie.empty else fallback, "volatilidad_historica": float(serie[serie.index < fecha].std(ddof=0)) if len(serie) > 1 else 0.0}

    def _preparar_contextos(self, contextos):
        filas = []
        for contexto in contextos:
            fila = dict(contexto)
            fila.update({k: v for k, v in self._fecha_features(fila.get("fecha", pd.Timestamp.today())).items() if k not in fila})
            fila.update(self._historial_contexto(fila))
            filas.append(fila)
        return pd.DataFrame(filas)

    def diagnostico_producto_nuevo(self, categoria, contexto_dia):
        return self.predecir({"producto": "__producto_nuevo__", "categoria": categoria, **contexto_dia})

    def predecir(self, contexto):
        return self.predecir_lote([contexto])[0]

    def predecir_lote(self, contextos):
        if self.modelo is None: raise ValueError("El modelo no está entrenado.")
        if not contextos: return []
        X = self._construir_features(self._preparar_contextos(contextos), entrenando=False)
        probs, orden = self.modelo.predict_proba(X), list(self.modelo.classes_)
        clases = [self._clasificar_desde_probs(fila, orden) for fila in probs]
        return [{"prioridad": c, "confianza": round(float(probs[i, orden.index(c)]), 4), "probabilidades": {cl: round(float(probs[i, orden.index(cl)]), 4) for cl in orden}} for i, c in enumerate(clases)]

    def guardar(self, ruta):
        import joblib
        joblib.dump(self, ruta)
        from services.storage_service import subir
        subir(ruta, os.path.basename(ruta))

    @staticmethod
    def cargar(ruta):
        import joblib
        from services.storage_service import asegurar_local
        asegurar_local(os.path.basename(ruta), ruta)
        return joblib.load(ruta)
