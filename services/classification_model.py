"""Clasificador de prioridad de abastecimiento con LightGBM.

La etiqueta representa la demanda REAL diaria de cada producto: ``Baja``,
``Media`` o ``Alta``. Los límites son terciles del historial disponible del
producto (con respaldo por categoría o global) y se ajustan exclusivamente
con el periodo de entrenamiento de cada pliegue.

La validación es walk-forward por fechas completas. Las variables históricas
siempre se calculan con ``shift(1)``: ninguna fila usa su propia venta ni una
venta futura para construir sus variables de entrada.
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
MINIMO_FILAS = 45
MIN_FILAS_UMBRAL_PRODUCTO = 10
MIN_FILAS_UMBRAL_CATEGORIA = 20
LGBM_PARAMS = {
    "n_estimators": 350,
    "learning_rate": 0.04,
    "num_leaves": 31,
    "min_child_samples": 12,
    "subsample": 0.85,
    "colsample_bytree": 0.85,
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
    """Clasifica la prioridad producto-día con LightGBM multiclase."""

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

    # Preparación ---------------------------------------------------------

    @staticmethod
    def _fecha_features(fecha):
        fecha = pd.Timestamp(fecha)
        return {
            "dia_semana": fecha.weekday(),
            "dia_mes": fecha.day,
            "mes": fecha.month,
            "semana_anio": int(fecha.isocalendar()[1]),
            "es_finde": int(fecha.weekday() >= 5),
            "es_quincena": int(1 <= fecha.day <= 7 or 15 <= fecha.day <= 21),
        }

    def _agregar_producto_dia(self, df):
        df = df.copy()
        df["fecha"] = pd.to_datetime(df["fecha"])
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
        df = df.dropna(subset=["fecha", "producto", "cantidad"])

        agg = {"cantidad": "sum"}
        for columna, funcion in (
            ("precio", "mean"), ("categoria", "first"),
            ("descuento_pct", "mean"), ("promocion", "max"),
            ("es_evento_especial", "max"),
        ):
            if columna in df.columns:
                agg[columna] = funcion
        for columna in FEATURES_CALENDARIO:
            if columna in df.columns:
                agg[columna] = "first"

        diario = df.groupby(["producto", "fecha"], as_index=False).agg(agg)

        # Completa días sin ventas para que lag_7 sea exactamente una semana.
        piezas = []
        for producto, grupo in diario.groupby("producto", sort=False):
            grupo = grupo.set_index("fecha").sort_index()
            fechas = pd.date_range(grupo.index.min(), grupo.index.max(), freq="D")
            pieza = grupo.reindex(fechas)
            pieza.index.name = "fecha"
            pieza["producto"] = producto
            pieza["cantidad"] = pieza["cantidad"].fillna(0.0)
            for columna in ("categoria", "precio"):
                if columna in pieza:
                    pieza[columna] = pieza[columna].ffill().bfill()
            for columna in FEATURES_COMERCIALES:
                if columna in pieza:
                    pieza[columna] = pd.to_numeric(
                        pieza[columna], errors="coerce"
                    ).fillna(0.0)
            piezas.append(pieza.reset_index())

        return pd.concat(piezas, ignore_index=True).sort_values(
            ["fecha", "producto"]
        ).reset_index(drop=True)

    def _con_historial(self, diario):
        piezas = []
        for _, grupo in diario.groupby("producto", sort=False):
            grupo = grupo.sort_values("fecha").copy()
            cantidad = grupo["cantidad"].astype(float)
            for lag in (1, 7, 14, 28):
                grupo[f"lag_{lag}"] = cantidad.shift(lag)
            previo = cantidad.shift(1)
            grupo["rolling_7_mean"] = previo.rolling(7, min_periods=1).mean()
            grupo["rolling_14_mean"] = previo.rolling(14, min_periods=1).mean()
            grupo["rolling_28_mean"] = previo.rolling(28, min_periods=1).mean()
            grupo["rolling_7_std"] = previo.rolling(
                7, min_periods=2
            ).std().fillna(0.0)
            grupo["media_historica"] = previo.expanding(min_periods=1).mean()
            grupo["volatilidad_historica"] = previo.expanding(
                min_periods=2
            ).std().fillna(0.0)
            piezas.append(grupo)
        return pd.concat(piezas, ignore_index=True).dropna(
            subset=["lag_28"]
        ).reset_index(drop=True)

    # Etiqueta ------------------------------------------------------------

    @staticmethod
    def _quantiles_seguros(cantidades):
        """Retorna los terciles o ``None`` cuando no existen tres niveles."""
        valores = np.asarray(cantidades, dtype=float)
        if len(valores) < 3:
            return None
        bajo, alto = np.quantile(valores, [1 / 3, 2 / 3])
        return None if bajo >= alto else (float(bajo), float(alto))

    def _fijar_umbrales(self, datos):
        """Ajusta umbrales usando solo ``datos`` (el train del pliegue)."""
        self.umbrales_global = self._quantiles_seguros(datos["cantidad"])
        if self.umbrales_global is None:
            raise ValueError(
                "No hay suficiente variación de ventas para formar tres prioridades útiles."
            )

        self.umbrales_categoria = {}
        if "categoria" in datos.columns:
            for categoria, grupo in datos.groupby("categoria"):
                if len(grupo) >= MIN_FILAS_UMBRAL_CATEGORIA:
                    umbrales = self._quantiles_seguros(grupo["cantidad"])
                    if umbrales is not None:
                        self.umbrales_categoria[categoria] = umbrales

        self.umbrales_producto = {}
        for producto, grupo in datos.groupby("producto"):
            if len(grupo) >= MIN_FILAS_UMBRAL_PRODUCTO:
                umbrales = self._quantiles_seguros(grupo["cantidad"])
                if umbrales is not None:
                    self.umbrales_producto[producto] = umbrales

    def _umbrales_para(self, producto, categoria):
        if producto in self.umbrales_producto:
            return self.umbrales_producto[producto]
        if categoria in self.umbrales_categoria:
            return self.umbrales_categoria[categoria]
        return self.umbrales_global

    def _origen_umbral(self, producto, categoria):
        if producto in self.umbrales_producto:
            return "producto"
        if categoria in self.umbrales_categoria:
            return "categoria"
        return "global"

    def _a_clase(self, datos):
        categorias = (
            datos["categoria"] if "categoria" in datos.columns
            else pd.Series(None, index=datos.index)
        )
        clases = []
        for cantidad, producto, categoria in zip(
            datos["cantidad"], datos["producto"], categorias
        ):
            bajo, alto = self._umbrales_para(producto, categoria)
            if cantidad <= bajo:
                clases.append("Baja")
            elif cantidad <= alto:
                clases.append("Media")
            else:
                clases.append("Alta")
        return pd.Series(clases, index=datos.index)

    # Variables -----------------------------------------------------------

    def _construir_features(self, df, entrenando):
        base = pd.DataFrame(index=df.index)
        for columna in FEATURES_CALENDARIO:
            if columna in df:
                base[columna] = pd.to_numeric(
                    df[columna], errors="coerce"
                ).fillna(0)
            elif columna in (
                "dia_semana", "dia_mes", "mes", "semana_anio",
                "es_finde", "es_quincena",
            ) and "fecha" in df:
                base[columna] = [
                    self._fecha_features(fecha)[columna] for fecha in df["fecha"]
                ]
            else:
                base[columna] = 0

        for columna in FEATURES_COMERCIALES + FEATURES_HISTORICAS:
            base[columna] = (
                pd.to_numeric(df[columna], errors="coerce").fillna(0.0)
                if columna in df else 0.0
            )

        # Categorías nativas: se fijan con train. Un nivel no visto durante
        # el entrenamiento se convierte en missing y LightGBM lo procesa como
        # tal; nunca se crea una columna-identidad para memorizarlo.
        for columna in ("categoria", "producto"):
            valores = (
                df[columna].fillna("Sin categoria").astype(str)
                if columna in df else pd.Series("Sin categoria", index=df.index)
            )
            if entrenando:
                niveles = sorted(valores.unique().tolist())
                if columna == "categoria":
                    self.cols_categoria = niveles
                else:
                    self.cols_producto = niveles
            else:
                niveles = (
                    self.cols_categoria if columna == "categoria"
                    else self.cols_producto
                )
            base[columna] = pd.Categorical(valores, categories=niveles)

        if entrenando:
            self.feature_names = list(base.columns)
        return base.reindex(columns=self.feature_names, fill_value=0.0)

    @staticmethod
    def _entrenar_lgbm(X, y, random_state):
        modelo = lgb.LGBMClassifier(
            **LGBM_PARAMS,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=1,
            verbose=-1,
        )
        categoricas = [c for c in ("categoria", "producto") if c in X.columns]
        modelo.fit(X, y, categorical_feature=categoricas)
        return modelo

    # Evaluación ----------------------------------------------------------

    def _metricas(self, y_true, pred, probs=None):
        precision, recall, f1_por_clase, soporte = precision_recall_fscore_support(
            y_true, pred, labels=CLASES, zero_division=0
        )
        baseline = DummyClassifier(strategy="most_frequent").fit(
            np.zeros((len(y_true), 1)), y_true
        )
        f1_base = f1_score(
            y_true,
            baseline.predict(np.zeros((len(y_true), 1))),
            labels=CLASES,
            average="macro",
            zero_division=0,
        )
        f1 = f1_score(y_true, pred, labels=CLASES, average="macro", zero_division=0)
        return {
            "accuracy": round(float(accuracy_score(y_true, pred)), 4),
            "f1_macro": round(float(f1), 4),
            "f1_weighted": round(float(f1_score(
                y_true, pred, labels=CLASES, average="weighted", zero_division=0
            )), 4),
            "precision_macro": round(float(np.mean(precision)), 4),
            "recall_macro": round(float(np.mean(recall)), 4),
            "confianza_promedio": (
                round(float(np.mean(np.max(probs, axis=1))), 4)
                if probs is not None else None
            ),
            "por_clase": {
                CLASES[i]: {
                    "precision": round(float(precision[i]), 4),
                    "recall": round(float(recall[i]), 4),
                    "f1": round(float(f1_por_clase[i]), 4),
                    "soporte": int(soporte[i]),
                }
                for i in range(len(CLASES))
            },
            "matriz_confusion": confusion_matrix(y_true, pred, labels=CLASES).tolist(),
            "clases": CLASES,
            "baseline_f1_mayoritaria": round(float(f1_base), 4),
            "mejora_vs_baseline_pct": (
                round(float((f1 - f1_base) / f1_base * 100), 2)
                if f1_base else None
            ),
            "n_test": int(len(y_true)),
        }

    @staticmethod
    def _pliegues_por_fecha(datos, n_folds):
        """Walk-forward sin dividir los productos de una misma fecha."""
        fechas = np.sort(datos["fecha"].dropna().unique())
        if len(fechas) <= n_folds:
            return []
        tam_test = len(fechas) // (n_folds + 1)
        if tam_test == 0:
            return []

        inicio_test = len(fechas) - n_folds * tam_test
        pliegues = []
        for indice in range(n_folds):
            inicio = inicio_test + indice * tam_test
            fin = inicio + tam_test if indice < n_folds - 1 else len(fechas)
            fecha_inicio, fecha_fin = fechas[inicio], fechas[fin - 1]
            train = datos.loc[datos["fecha"] < fecha_inicio].copy()
            test = datos.loc[
                (datos["fecha"] >= fecha_inicio) & (datos["fecha"] <= fecha_fin)
            ].copy()
            if not train.empty and not test.empty:
                pliegues.append((train, test))
        return pliegues

    def entrenar(self, df):
        diario = self._agregar_producto_dia(df)
        datos = self._con_historial(diario)
        if len(datos) < MINIMO_FILAS or datos["producto"].nunique() < 2:
            raise ValueError(
                "Datos insuficientes: se requieren al menos 45 días-producto con historial."
            )
        datos = datos.sort_values(["fecha", "producto"]).reset_index(drop=True)

        n_folds = 3 if len(datos) >= 120 else 2
        pliegues = self._pliegues_por_fecha(datos, n_folds)
        reales, predicciones, probabilidades, origenes = [], [], [], []
        pliegues_evaluados = 0
        for train, test in pliegues:
            try:
                self._fijar_umbrales(train)
            except ValueError:
                continue
            y_train, y_test = self._a_clase(train), self._a_clase(test)
            if y_train.nunique() < len(CLASES):
                continue
            X_train = self._construir_features(train, entrenando=True)
            X_test = self._construir_features(test, entrenando=False)
            modelo = self._entrenar_lgbm(X_train, y_train, self.random_state)
            reales.extend(y_test.tolist())
            predicciones.extend(modelo.predict(X_test).tolist())
            probabilidades.extend(modelo.predict_proba(X_test).tolist())
            categorias_test = test.get("categoria", pd.Series(None, index=test.index))
            origenes.extend(
                self._origen_umbral(producto, categoria)
                for producto, categoria in zip(test["producto"], categorias_test)
            )
            pliegues_evaluados += 1

        if not reales:
            raise ValueError(
                "El historial no permite una validación temporal con las tres prioridades."
            )
        self.metricas = self._metricas(reales, predicciones, np.asarray(probabilidades))
        conteo_origenes = pd.Series(origenes).value_counts()
        self.metricas.update({
            "n_folds": pliegues_evaluados,
            "validacion": "walk-forward por fecha completa",
            "cobertura_umbral_validacion": {
                origen: round(float(conteo_origenes.get(origen, 0) / len(origenes)), 4)
                for origen in ("producto", "categoria", "global")
            },
        })

        # Artefacto final: usa todo el historial disponible, solo después de
        # haber medido de forma temporal los pliegues anteriores.
        self._fijar_umbrales(datos)
        y = self._a_clase(datos)
        if y.nunique() < len(CLASES):
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
            "hiperparametros": dict(LGBM_PARAMS),
            "algoritmo": "LightGBM multiclase",
        })
        importancia = sorted(
            zip(self.feature_names, self.modelo.feature_importances_),
            key=lambda item: item[1], reverse=True,
        )
        self.importancias = [
            {"variable": nombre, "importancia": round(float(valor), 4)}
            for nombre, valor in importancia[:10]
        ]

        self.historial_producto = {
            producto: grupo.set_index("fecha")["cantidad"].astype(float).to_dict()
            for producto, grupo in diario.groupby("producto")
        }
        self.perfil_producto = diario.groupby("producto")["cantidad"].mean().to_dict()
        self.perfil_categoria = (
            diario.groupby("categoria")["cantidad"].mean().to_dict()
            if "categoria" in diario else {}
        )
        self.media_global = float(diario["cantidad"].mean())
        return self.metricas

    # Predicción ----------------------------------------------------------

    def _historial_contexto(self, contexto):
        producto = contexto.get("producto")
        fecha = pd.Timestamp(contexto.get("fecha", pd.Timestamp.today()))
        historial = self.historial_producto.get(producto, {})
        respaldo = self.perfil_producto.get(
            producto,
            self.perfil_categoria.get(contexto.get("categoria"), self.media_global),
        )
        serie = pd.Series(historial, dtype=float)

        def valor(dia):
            return float(serie.get(dia, respaldo))

        previos = pd.Series([valor(fecha - timedelta(days=i)) for i in range(1, 29)])
        return {
            "lag_1": valor(fecha - timedelta(days=1)),
            "lag_7": valor(fecha - timedelta(days=7)),
            "lag_14": valor(fecha - timedelta(days=14)),
            "lag_28": valor(fecha - timedelta(days=28)),
            "rolling_7_mean": float(previos.iloc[:7].mean()),
            "rolling_14_mean": float(previos.iloc[:14].mean()),
            "rolling_28_mean": float(previos.mean()),
            "rolling_7_std": float(previos.iloc[:7].std(ddof=0)),
            "media_historica": (
                float(serie[serie.index < fecha].mean()) if not serie.empty else respaldo
            ),
            "volatilidad_historica": (
                float(serie[serie.index < fecha].std(ddof=0))
                if len(serie) > 1 else 0.0
            ),
        }

    def _preparar_contextos(self, contextos):
        filas = []
        for contexto in contextos:
            fila = dict(contexto)
            fila.update({
                clave: valor for clave, valor in self._fecha_features(
                    fila.get("fecha", pd.Timestamp.today())
                ).items() if clave not in fila
            })
            fila.update(self._historial_contexto(fila))
            filas.append(fila)
        return pd.DataFrame(filas)

    def diagnostico_producto_nuevo(self, categoria, contexto_dia):
        return self.predecir({
            "producto": "__producto_nuevo__", "categoria": categoria, **contexto_dia,
        })

    def predecir(self, contexto):
        return self.predecir_lote([contexto])[0]

    def predecir_lote(self, contextos):
        if self.modelo is None:
            raise ValueError("El modelo no está entrenado.")
        if not contextos:
            return []
        X = self._construir_features(
            self._preparar_contextos(contextos), entrenando=False
        )
        clases = self.modelo.predict(X)
        probabilidades = self.modelo.predict_proba(X)
        orden = list(self.modelo.classes_)
        return [
            {
                "prioridad": clase,
                "confianza": round(float(probabilidades[i, orden.index(clase)]), 4),
                "probabilidades": {
                    etiqueta: round(float(probabilidades[i, orden.index(etiqueta)]), 4)
                    for etiqueta in orden
                },
            }
            for i, clase in enumerate(clases)
        ]

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
