"""
RestoIQ — Modelo de predicción de ventas
=========================================
Predice para los próximos 7 días:
  • Cantidad vendida por producto por día  ← detallado
  • Ingreso total del negocio por día      ← agregado

Estrategia automática según historial disponible:
  +90 días → LightGBM (machine learning)
  -90 días → Promedio móvil ponderado (reglas simples)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_absolute_percentage_error, r2_score
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

UMBRAL_DIAS_ML = 90   # mínimo de días de calendario para intentar LightGBM
MINIMO_FILAS_LGBM = 30  # mínimo de filas YA con lags calculados (densidad real de ventas)

FEATURES_BASE = [
    "dia_semana", "mes", "semana_anio",
    "es_finde", "es_puente", "es_quincena",
    "producto_encoded",
    "lag_7", "lag_14", "lag_28",
    "rolling_7_mean", "rolling_14_mean", "rolling_7_std",
]

# Features opcionales: columna del dataset -> atributo de ConfiguracionAnalisis
# que decide si se usan. Solo se activan si AMBAS cosas son ciertas:
# el usuario lo pidió en su configuración Y la columna existe en sus datos.
FEATURES_OPCIONALES = {
    "es_feriado":         "considerar_feriados",
    "promocion":          "considerar_promociones",
    "descuento_pct":      "considerar_descuentos",
    "es_evento_especial": "considerar_eventos",
}

TARGET = "cantidad"


# ════════════════════════════════════════════════════════════
# CLASE PRINCIPAL
# ════════════════════════════════════════════════════════════

class SalesModel:
    """
    Entrena y sirve predicciones de ventas para RestoIQ.

    Uso:
        model = SalesModel()
        model.entrenar(df_limpio)
        resultado = model.predecir(dias=7)

        # Predicción detallada por producto:
        print(resultado["por_producto"])

        # Totales diarios:
        print(resultado["diario"])

        # Tabla pivote (fecha × producto):
        print(resultado["pivote"])
    """

    def __init__(self):
        self.modelo           = None
        self.estrategia       = None   # "lgbm" o "promedio_movil"
        self.label_encoder    = LabelEncoder()
        self.productos        = []
        self.precio_promedio  = {}
        self.df_historial     = None
        self.metricas         = {}
        self.fecha_ultimo_dato = None
        self.feriados         = self._cargar_feriados()
        # Features que este modelo entrenado realmente usa (se decide en
        # entrenar(), según la ConfiguracionAnalisis del usuario y qué
        # columnas trae su dataset). Por defecto: base + feriados.
        self.features = FEATURES_BASE + ["es_feriado"]

    # ── Feriados ──────────────────────────────────────────────────────────

    def _cargar_feriados(self):
        for mod in ["services.data_generator", "data_generator"]:
            try:
                import importlib
                m = importlib.import_module(mod)
                return m.FERIADOS_ECUADOR
            except Exception:
                continue
        return set()

    # ── Features temporales ───────────────────────────────────────────────

    def _features_fecha(self, fecha: pd.Timestamp) -> dict:
        fecha_str = fecha.strftime("%Y-%m-%d")
        dia       = fecha.weekday()
        ayer      = (fecha - timedelta(days=1)).strftime("%Y-%m-%d")
        maniana   = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
        dia_mes   = fecha.day
        return {
            "dia_semana":  dia,
            "mes":         fecha.month,
            "semana_anio": int(fecha.isocalendar()[1]),
            "es_finde":    int(dia in [5, 6]),
            "es_feriado":  int(fecha_str in self.feriados),
            "es_puente":   int(ayer in self.feriados or maniana in self.feriados),
            "es_quincena": int(1 <= dia_mes <= 7 or 15 <= dia_mes <= 21),
        }

    def _asegurar_features_contexto(self, df: pd.DataFrame) -> pd.DataFrame:
        if "dia_semana" not in df.columns:
            feats = df["fecha"].apply(self._features_fecha)
            for col in feats.iloc[0].keys():
                df[col] = feats.apply(lambda x: x[col])
        return df

    # ── Preparación de datos ─────────────────────────────────────────────

    def _preparar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Arma lag_7/14/28 y rolling_* por CALENDARIO, no por posición de
        fila. ANTES: `shift(N)` avanzaba N FILAS, que solo equivale a N
        días si el producto vendió TODOS los días sin huecos. Con
        ventas intermitentes (normal en un restaurante: un producto
        puede no venderse algunos días), un hueco desalineaba el lag
        silenciosamente — sin error, sin warning. Bug real detectado
        en revisión de arquitectura y corregido acá (ver test que lo
        reproduce y confirma el fix).

        FIX: por cada producto se reconstruye una serie DIARIA
        continua — un valor por cada día de calendario entre su
        primera y última venta, con cantidad=0 en los días sin venta
        registrada — y sobre esa serie continua sí es válido usar
        shift(N): ahí N filas literalmente equivale a N días.

        Efecto colateral esperado (correcto, no un bug nuevo): el
        modelo ahora también entrena con días de demanda CERO real
        para cada producto, que antes ni siquiera existían como fila
        (se descartaban en el filtro `cantidad > 0` de entrenar(),
        antes de llegar acá). Antes el modelo nunca veía un "0" como
        posible respuesta — esto es más correcto para forecasting de
        demanda, pero VA A CAMBIAR el MAE/WAPE reportado. Hay que
        re-correr la evaluación y actualizar las métricas de la tesis.
        """
        df = df.copy()
        df["producto_encoded"] = self.label_encoder.transform(df["producto"])

        columnas_negocio = [c for c in ("promocion", "descuento_pct", "es_evento_especial")
                            if c in df.columns]

        piezas = []
        for producto, grupo in df.groupby("producto"):
            grupo = grupo.set_index("fecha").sort_index()
            rango = pd.date_range(grupo.index.min(), grupo.index.max(), freq="D")

            pieza = pd.DataFrame({"fecha": rango})
            pieza["producto"] = producto
            pieza["producto_encoded"] = int(self.label_encoder.transform([producto])[0])
            pieza["cantidad"] = grupo["cantidad"].reindex(rango, fill_value=0).values

            for col in columnas_negocio:
                # Días de relleno (sin fila original) → 0 / sin
                # actividad, nunca NaN.
                pieza[col] = grupo[col].reindex(rango, fill_value=0).values

            for lag in (7, 14, 28):
                pieza[f"lag_{lag}"] = pieza["cantidad"].shift(lag)

            pieza["rolling_7_mean"] = (
                pieza["cantidad"].shift(1).rolling(7, min_periods=1).mean()
            )
            pieza["rolling_14_mean"] = (
                pieza["cantidad"].shift(1).rolling(14, min_periods=1).mean()
            )
            pieza["rolling_7_std"] = (
                pieza["cantidad"].shift(1).rolling(7, min_periods=1).std().fillna(0)
            )

            piezas.append(pieza)

        df_continuo = pd.concat(piezas, ignore_index=True)
        df_continuo = self._asegurar_features_contexto(df_continuo)
        df_continuo = df_continuo.dropna(subset=["lag_7", "lag_14", "lag_28"])
        return df_continuo

    # ── Entrenamiento ─────────────────────────────────────────────────────

    def entrenar(self, df: pd.DataFrame, config=None, verbose=True) -> dict:
        df = df.copy()
        df["fecha"]    = pd.to_datetime(df["fecha"])
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
        df = df.dropna(subset=["fecha", "producto", "cantidad"])
        df = df[df["cantidad"] > 0]

        self.productos        = sorted(df["producto"].unique().tolist())
        self.fecha_ultimo_dato = df["fecha"].max()
        self.precio_promedio  = (
            df.groupby("producto")["precio"].mean().to_dict()
            if "precio" in df.columns else {}
        )

        # Decidir qué features opcionales usa este modelo: requiere que
        # el usuario lo haya activado en su configuración Y que la
        # columna exista realmente en su dataset (doble candado).
        self.features = self._construir_features(df, config)

        # Agregar por fecha+producto — conservando las columnas de
        # negocio opcionales si están presentes y son necesarias.
        agg_spec = {"cantidad": ("cantidad", "sum")}
        if "promocion" in df.columns:
            agg_spec["promocion"] = ("promocion", "max")
        if "descuento_pct" in df.columns:
            agg_spec["descuento_pct"] = ("descuento_pct", "mean")
        if "es_evento_especial" in df.columns:
            agg_spec["es_evento_especial"] = ("es_evento_especial", "max")

        df_agg = (
            df.groupby(["fecha", "producto"])
            .agg(**agg_spec)
            .reset_index()
        )
        for col in ("promocion", "descuento_pct", "es_evento_especial"):
            if col in df_agg.columns:
                df_agg[col] = df_agg[col].fillna(0)

        # (El cálculo de features de contexto se hace más abajo, sobre
        # la serie diaria continua que arma _preparar_features — no
        # hace falta calcularlo acá también, sería redundante.)

        # Decidir estrategia: los días de calendario NO bastan — si las
        # ventas son intermitentes, un producto puede tener 90+ días de
        # historial pero nunca 28 días CON venta real, y entonces
        # lag_28 sale NaN para todas sus filas. Por eso se verifica la
        # densidad real (cuántas filas quedan con todos los lags
        # calculados) antes de decidir usar LightGBM.
        dias_historial = (df["fecha"].max() - df["fecha"].min()).days
        self.label_encoder.fit(self.productos)
        df_feat = self._preparar_features(df_agg)

        # df_historial ahora es la serie DIARIA CONTINUA (con huecos
        # rellenados en 0) que ya construyó _preparar_features — no el
        # df_agg original con huecos. Antes, _features_para_fecha_producto
        # necesitaba un fallback "a la media del producto" para fechas
        # dentro del rango histórico pero sin fila propia (huecos); con
        # la serie continua eso ya no hace falta para huecos internos,
        # el fallback solo se usa para fechas realmente fuera de rango.
        self.df_historial = df_feat[["fecha", "producto", "cantidad"]].copy()

        if verbose:
            print(f" Historial disponible: {dias_historial} días · {len(df_feat)} filas con lags completos")

        if dias_historial >= UMBRAL_DIAS_ML and len(df_feat) >= MINIMO_FILAS_LGBM:
            self.estrategia = "lgbm"
            if verbose:
                print(" Estrategia: LightGBM (machine learning)\n")
            self._entrenar_lgbm(df_feat, verbose)
        else:
            self.estrategia = "promedio_movil"
            if verbose:
                motivo = "datos insuficientes para ML" if dias_historial < UMBRAL_DIAS_ML \
                    else "ventas muy intermitentes por producto (pocos lags completos)"
                print(f" Estrategia: Promedio móvil ponderado ({motivo})\n")
            self._entrenar_promedio_movil(df_agg)

        # Evaluación honesta con datos reales nunca vistos en el
        # entrenamiento (holdout) + comparación contra un modelo de
        # referencia (Regresión Lineal). Reemplaza el MAE/MAPE de
        # producción por esta métrica única — ver ARQUITECTURA.md.
        self._evaluar_holdout(df_feat, dias_historial, verbose)

        # Siempre presentes, sin importar la estrategia ni si el
        # walk-forward llegó a correr — el frontend los usa para la
        # barra de progreso "Historial disponible: X / 90 días"
        # cuando no hay suficientes datos para validar todavía.
        self.metricas["dias_historial"] = dias_historial
        self.metricas["umbral_dias_ml"] = UMBRAL_DIAS_ML

        if verbose:
            self._imprimir_metricas()

        return self.metricas

    def _construir_features(self, df: pd.DataFrame, config) -> list:
        """
        Base siempre incluida. Cada feature opcional entra SOLO si:
          1. El usuario la activó en su ConfiguracionAnalisis (o no hay
             config todavía → se usa el default de esa columna), Y
          2. La columna existe de verdad en su dataset.
        Esto hace imposible que una config mal puesta ("sí, usar
        promociones") rompa el entrenamiento de alguien cuyo CSV no
        tiene esa columna: simplemente se ignora en silencio.
        """
        features = list(FEATURES_BASE)
        for columna, atributo_config in FEATURES_OPCIONALES.items():
            columna_existe = columna in df.columns and df[columna].notna().any()
            if config is None:
                quiere_usarla = (columna == "es_feriado")  # default histórico
            else:
                quiere_usarla = bool(getattr(config, atributo_config, False))
            if quiere_usarla and columna_existe:
                features.append(columna)
        return features

    def _entrenar_lgbm(self, df_feat, verbose):
        # Mismo fix que en _evaluar_holdout: df_feat viene ordenado por
        # (producto, fecha), no por fecha pura. TimeSeriesSplit asume
        # orden cronológico — sin reordenar aquí, los folds de esta
        # validación interna también cortarían mezclando productos en
        # vez de avanzar en el tiempo real.
        df_feat_temporal = df_feat.sort_values("fecha").reset_index(drop=True)
        X = df_feat_temporal[self.features]
        y = df_feat_temporal[TARGET]

        tscv = TimeSeriesSplit(n_splits=3)
        maes, mapes = [], []

        for train_idx, val_idx in tscv.split(X):
            m = lgb.LGBMRegressor(
                objective="regression_l1",  # optimiza MAE, no RMSE (L2 por defecto) — alineado con las métricas que reportamos (MAE/WAPE)
                n_estimators=500, learning_rate=0.05,
                num_leaves=63, min_child_samples=20,
                subsample=0.8, colsample_bytree=0.8,
                reg_alpha=0.1, reg_lambda=0.1,
                random_state=42, verbose=-1,
            )
            m.fit(
                X.iloc[train_idx], y.iloc[train_idx],
                eval_set=[(X.iloc[val_idx], y.iloc[val_idx])],
                callbacks=[lgb.early_stopping(50, verbose=False)],
            )
            preds = np.maximum(0, m.predict(X.iloc[val_idx]))
            maes.append(mean_absolute_error(y.iloc[val_idx], preds))
            mapes.append(mean_absolute_percentage_error(y.iloc[val_idx], preds) * 100)

        # ── Modelo final: early stopping en 2 fases ──────────────────
        # Antes: n_estimators=700 fijo, sin validación — el número de
        # árboles del modelo desplegado no dependía de los datos, era
        # una constante arbitraria. Eso es exactamente lo que un
        # jurado pregunta al "¿cómo evitas overfitting?".
        #
        # Fase 1: se reserva un tramo de validación cronológico real
        # (nunca aleatorio en series de tiempo) para que early stopping
        # encuentre cuántos árboles son los que de verdad mejoran el
        # desempeño, sin adivinar.
        # Fase 2: se reentrena con el 100% de los datos (incluido el
        # tramo reservado en la fase 1) usando ESE número de árboles ya
        # encontrado — así el modelo final no pierde datos de
        # entrenamiento, solo se benefició del tramo de validación para
        # decidir su complejidad.
        n_val = max(15, int(len(X) * 0.15))
        n_val = min(n_val, len(X) // 3)  # nunca más de un tercio de los datos
        corte_val = len(X) - n_val

        X_train_fs, X_val_fs = X.iloc[:corte_val], X.iloc[corte_val:]
        y_train_fs, y_val_fs = y.iloc[:corte_val], y.iloc[corte_val:]

        modelo_busqueda = lgb.LGBMRegressor(
            objective="regression_l1",  # mismo objetivo que el modelo final — la búsqueda de n_arboles_optimo debe optimizar lo mismo que se va a usar después
            n_estimators=2000,  # techo alto — early stopping decide el número real
            learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        modelo_busqueda.fit(
            X_train_fs, y_train_fs,
            eval_set=[(X_val_fs, y_val_fs)],
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        # best_iteration_ puede ser None si nunca dejó de mejorar antes
        # del techo — fallback de seguridad al valor fijo anterior.
        n_arboles_optimo = modelo_busqueda.best_iteration_ or 700

        self.modelo = lgb.LGBMRegressor(
            objective="regression_l1",
            n_estimators=n_arboles_optimo,
            learning_rate=0.05,
            num_leaves=63, min_child_samples=20,
            subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=0.1,
            random_state=42, verbose=-1,
        )
        self.modelo.fit(X, y)

        if verbose:
            print(f"  Early stopping: {n_val} filas de validación → {n_arboles_optimo} árboles óptimos "
                  f"(modelo final reentrenado con el 100% de los datos)")

        importancias = pd.Series(
            self.modelo.feature_importances_, index=self.features
        ).sort_values(ascending=False)

        self.metricas = {
            "estrategia":        "LightGBM",
            # mae/mape/wape/rmse quedan en None aquí a propósito — son
            # los que se muestran al usuario como "validados", y la
            # ÚNICA fuente de verdad para eso es _evaluar_holdout()
            # (walk-forward). Antes este bloque los rellenaba con la
            # validación cruzada interna de abajo, y si el walk-forward
            # no llegaba a correr (poco historial), esos valores
            # "fantasma" se quedaban como si fueran confiables —
            # causaba una contradicción real en el Dashboard: arriba
            # decía "historial insuficiente" y más abajo mostraba un
            # MAE numérico como si fuera válido.
            "mae":               None,
            "mape":              None,
            # Provisional: SOLO para mostrarse como "dato preliminar,
            # no validado" mientras no hay suficiente historial para
            # el walk-forward — nunca se presenta como si fuera la
            # métrica final.
            "mae_provisional":   round(float(np.mean(maes)), 2),
            "mape_provisional":  round(float(np.mean(mapes)), 2),
            "filas_entrenadas": len(df_feat),
            "n_arboles_optimo": int(n_arboles_optimo),
            "productos":        len(self.productos),
            "fecha_desde":      str(self.df_historial["fecha"].min().date()),
            "fecha_hasta":      str(self.fecha_ultimo_dato.date()),
            "top_features":     importancias.head(5).index.tolist(),
            "importancias_top10": importancias.head(10).round(1).to_dict(),
            "features_usadas":  self.features,
        }

    def _entrenar_promedio_movil(self, df_agg):
        """
        Calcula pesos por día de semana y promedio reciente por producto.
        No necesita librerías de ML — funciona con 2 semanas de datos.
        """
        # Factor por día de semana (cuánto vende cada día vs el promedio)
        self.factor_dia = (
            df_agg.groupby(df_agg["fecha"].dt.dayofweek)["cantidad"]
            .mean()
        )
        media_global = self.factor_dia.mean()
        self.factor_dia = (self.factor_dia / media_global).to_dict()

        # Promedio reciente por producto (últimos 14 días disponibles)
        fecha_corte = df_agg["fecha"].max() - timedelta(days=14)
        reciente    = df_agg[df_agg["fecha"] >= fecha_corte]
        self.promedio_producto = (
            reciente.groupby("producto")["cantidad"].mean().to_dict()
        )

        self.label_encoder.fit(self.productos)
        self.metricas = {
            "estrategia": "Promedio móvil ponderado",
            "mae":        None,
            "mape":       None,
            "mae_provisional": None,  # sin validación cruzada interna en este camino
            "productos":  len(self.productos),
            "fecha_desde": str(df_agg["fecha"].min().date()),
            "fecha_hasta": str(self.fecha_ultimo_dato.date()),
            "nota":       "Datos insuficientes para ML. Se usó promedio móvil.",
            "features_usadas": self.features,
        }

    # ── Evaluación honesta (holdout real + modelo de referencia) ──────────

    def _calcular_n_folds(self, dias_historial: int):
        """
        Cuántos pliegues (folds) de validación walk-forward usar, según
        la densidad de historial disponible. Cada pliegue entrena con
        una ventana expansiva (todo lo anterior) y evalúa contra un
        tramo posterior nunca visto — el mismo mecanismo de
        TimeSeriesSplit que ya usa _entrenar_lgbm() para su validación
        interna, aplicado aquí también a la comparación contra el
        modelo de referencia.

          < 60 días     → None (no hay margen para ni un solo pliegue
                           honesto sin dejar el entrenamiento casi vacío).
          60–120 días   → 2 pliegues.
          120–250 días  → 3 pliegues.
          > 250 días    → 5 pliegues.

        Si con el número de pliegues elegido algún pliegue quedara con
        menos de MINIMO_FILAS_LGBM filas de entrenamiento, se reduce el
        número de pliegues automáticamente antes de rendirse del todo.
        """
        if dias_historial < 60:
            return None
        if dias_historial <= 120:
            return 2
        if dias_historial <= 250:
            return 3
        return 5

    def _predecir_promedio_movil_holdout(self, train_feat: pd.DataFrame, test_feat: pd.DataFrame) -> np.ndarray:
        """
        Réplica de la lógica de _entrenar_promedio_movil/_predecir_promedio_movil,
        pero calculada SOLO con el tramo de entrenamiento de cada pliegue
        (nunca con self.factor_dia/self.promedio_producto de producción,
        para no pisar el modelo real ya entrenado con todo el historial).
        """
        factor_dia_temp = train_feat.groupby(train_feat["fecha"].dt.dayofweek)["cantidad"].mean()
        media_global = factor_dia_temp.mean()
        factor_dia_temp = (factor_dia_temp / media_global).to_dict() if media_global else {}

        fecha_corte_reciente = train_feat["fecha"].max() - timedelta(days=14)
        reciente = train_feat[train_feat["fecha"] >= fecha_corte_reciente]
        promedio_producto_temp = reciente.groupby("producto")["cantidad"].mean().to_dict()
        promedio_global = train_feat["cantidad"].mean() if len(train_feat) else 1

        preds = []
        for _, row in test_feat.iterrows():
            f_dia     = factor_dia_temp.get(row["fecha"].weekday(), 1.0)
            f_feriado = 1.25 if row["fecha"].strftime("%Y-%m-%d") in self.feriados else 1.0
            base      = promedio_producto_temp.get(row["producto"], promedio_global)
            preds.append(max(0, base * f_dia * f_feriado))
        return np.array(preds)

    @staticmethod
    def _wape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        """
        Weighted Absolute Percentage Error: Σ|real - predicho| / Σ real.
        A diferencia de MAPE (que divide CADA observación por su propio
        valor real y puede dispararse con productos de baja demanda),
        WAPE divide una sola vez por el total — un producto que vende
        3 unidades/día ya no puede inflar el error porcentual él solo.
        """
        total_real = np.sum(np.abs(y_true))
        if total_real == 0:
            return None
        return round(float(np.sum(np.abs(y_true - y_pred)) / total_real) * 100, 2)

    def _evaluar_holdout(self, df_feat: pd.DataFrame, dias_historial: int, verbose=False):
        """
        Validación walk-forward (rolling-origin) real, no un solo split
        fijo: usa TimeSeriesSplit para generar varios pliegues
        secuenciales, cada uno con ventana de entrenamiento expansiva
        (todo lo anterior) y un tramo de evaluación posterior nunca
        visto. En cada pliegue se entrena, SOLO con ese tramo de
        entrenamiento, una réplica de la misma estrategia de producción
        y un modelo de referencia (Regresión Lineal, mismas features).

        Las métricas se calculan sobre los residuos AGRUPADOS (pooled)
        de todos los pliegues juntos, no promediando ratios pliegue por
        pliegue — más robusto cuando los pliegues tienen tamaños
        distintos. Resultado en self.metricas["holdout"], serializado
        junto con el resto del modelo en el .pkl.

        Si no hay margen suficiente para al menos un pliegue honesto,
        se omite (self.metricas["holdout"] = None) en vez de forzar
        un número.
        """
        n_folds = self._calcular_n_folds(dias_historial)
        if n_folds is None:
            self.metricas["holdout"] = None
            if verbose:
                print(" Validación walk-forward: omitida (historial < 60 días)\n")
            return

        # Reduce n_folds si algún pliegue quedaría con muy poco
        # entrenamiento — nunca se fuerza un número que produciría
        # pliegues sin sentido estadístico.
        while n_folds >= 2:
            tamano_min_train = len(df_feat) // (n_folds + 1)
            if tamano_min_train >= MINIMO_FILAS_LGBM:
                break
            n_folds -= 1

        if n_folds < 2:
            self.metricas["holdout"] = None
            if verbose:
                print(" Validación walk-forward: omitida (datos insuficientes para ni 2 pliegues)\n")
            return

        # TimeSeriesSplit asume que las filas ya vienen en orden
        # cronológico puro — pero df_feat está ordenado por
        # (producto, fecha) para que los lags se calculen bien por
        # producto. Sin reordenar aquí, los "pliegues" cortarían por
        # posición de fila (mezclando productos), no por tiempo real,
        # y cada pliegue terminaría cubriendo el mismo rango de fechas
        # completo en vez de tramos crecientes — walk-forward inválido.
        # Los valores de lag ya están calculados y fijos por fila, así
        # que reordenar aquí no los afecta, solo corrige el corte.
        df_feat_temporal = df_feat.sort_values("fecha").reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_folds)

        y_reales_todos, pred_restoiq_todos, pred_baseline_todos, pred_naive_todos = [], [], [], []
        detalle_folds = []

        for fold_i, (idx_train, idx_test) in enumerate(tscv.split(df_feat_temporal), start=1):
            train_feat = df_feat_temporal.iloc[idx_train]
            test_feat  = df_feat_temporal.iloc[idx_test]

            X_train, y_train = train_feat[self.features], train_feat[TARGET]
            X_test,  y_test  = test_feat[self.features],  test_feat[TARGET]

            # 1) Réplica de la MISMA estrategia que el modelo de
            #    producción (forzada, no recalculada por pliegue).
            if self.estrategia == "lgbm":
                modelo_replica = lgb.LGBMRegressor(
                    objective="regression_l1",
                    n_estimators=700, learning_rate=0.05,
                    num_leaves=63, min_child_samples=20,
                    subsample=0.8, colsample_bytree=0.8,
                    reg_alpha=0.1, reg_lambda=0.1,
                    random_state=42, verbose=-1,
                )
                modelo_replica.fit(X_train, y_train)
                pred_restoiq = np.maximum(0, modelo_replica.predict(X_test))
            else:
                pred_restoiq = self._predecir_promedio_movil_holdout(train_feat, test_feat)

            # 2) Modelo de referencia: Regresión Lineal, mismas features.
            baseline = LinearRegression()
            baseline.fit(X_train, y_train)
            pred_baseline = np.maximum(0, baseline.predict(X_test))

            # 3) Naive estacional: "lo que vendió este mismo producto
            #    hace 7 días". Cero entrenamiento, cero features — es
            #    literalmente la columna lag_7 que ya se calcula para
            #    decidir la estrategia de producción. El piso mínimo
            #    que cualquier modelo real debe superar.
            pred_naive = test_feat["lag_7"].values

            y_reales_todos.append(y_test.values)
            pred_restoiq_todos.append(pred_restoiq)
            pred_baseline_todos.append(pred_baseline)
            pred_naive_todos.append(pred_naive)

            detalle = test_feat[["fecha", "producto"]].copy()
            detalle["real"]     = y_test.values
            detalle["restoiq"]  = np.round(pred_restoiq).astype(int)
            detalle["baseline"] = np.round(pred_baseline).astype(int)
            detalle["naive"]    = np.round(pred_naive).astype(int)
            detalle["fold"]     = fold_i
            detalle_folds.append(detalle)

            if verbose:
                print(f"  Pliegue {fold_i}/{n_folds}: train={len(idx_train)} filas, test={len(idx_test)} filas "
                      f"({str(test_feat['fecha'].min().date())} → {str(test_feat['fecha'].max().date())})")

        # ── Métricas agrupadas (pooled) sobre todos los pliegues ──
        y_reales      = np.concatenate(y_reales_todos)
        pred_restoiq  = np.concatenate(pred_restoiq_todos)
        pred_baseline = np.concatenate(pred_baseline_todos)
        pred_naive    = np.concatenate(pred_naive_todos)

        mae_restoiq   = round(float(mean_absolute_error(y_reales, pred_restoiq)), 2)
        mape_restoiq  = round(float(mean_absolute_percentage_error(y_reales, pred_restoiq)) * 100, 2)
        wape_restoiq  = self._wape(y_reales, pred_restoiq)
        rmse_restoiq  = round(float(np.sqrt(np.mean((y_reales - pred_restoiq) ** 2))), 2)

        mae_baseline  = round(float(mean_absolute_error(y_reales, pred_baseline)), 2)
        mape_baseline = round(float(mean_absolute_percentage_error(y_reales, pred_baseline)) * 100, 2)
        wape_baseline = self._wape(y_reales, pred_baseline)
        rmse_baseline = round(float(np.sqrt(np.mean((y_reales - pred_baseline) ** 2))), 2)

        mae_naive     = round(float(mean_absolute_error(y_reales, pred_naive)), 2)
        wape_naive    = self._wape(y_reales, pred_naive)
        rmse_naive    = round(float(np.sqrt(np.mean((y_reales - pred_naive) ** 2))), 2)

        # NRMSE = RMSE normalizado (÷ promedio de la demanda real) —
        # pone el RMSE en escala 0-1 (o %), comparable entre negocios
        # de distinto volumen, igual que WAPE ya hace con el MAE.
        media_real = float(np.mean(y_reales)) or 1
        nrmse_restoiq  = round(rmse_restoiq  / media_real, 3)
        nrmse_baseline = round(rmse_baseline / media_real, 3)
        nrmse_naive    = round(rmse_naive    / media_real, 3)
        nmae_restoiq   = round(mae_restoiq   / media_real, 3)
        nmae_baseline  = round(mae_baseline  / media_real, 3)
        nmae_naive     = round(mae_naive     / media_real, 3)

        # R² — cuánta varianza de la demanda real explica cada modelo.
        # Mismo dato agrupado (pooled) que el resto de las métricas.
        r2_restoiq  = round(float(r2_score(y_reales, pred_restoiq)), 3)
        r2_baseline = round(float(r2_score(y_reales, pred_baseline)), 3)
        r2_naive    = round(float(r2_score(y_reales, pred_naive)), 3)

        # % de mejora de RestoIQ sobre cada referencia (por WAPE — la
        # métrica principal de confiabilidad ahora, ver docstring)
        mejora_pct           = round((1 - wape_restoiq / wape_baseline) * 100, 1) if wape_baseline else None
        mejora_vs_naive_pct  = round((1 - wape_restoiq / wape_naive) * 100, 1) if wape_naive else None

        # Modelo ganador por WAPE (para el scatter Real vs. Predicho)
        candidatos = {
            "RestoIQ":                       (wape_restoiq, pred_restoiq),
            "Regresión Lineal":               (wape_baseline, pred_baseline),
            "Naive estacional (lag-7)":       (wape_naive, pred_naive),
        }
        modelo_ganador, (_, pred_ganador) = min(candidatos.items(), key=lambda kv: kv[1][0])

        # Scatter Real vs. Predicho del modelo ganador — se limita a
        # una muestra para que el gráfico se mantenga legible con
        # datasets grandes (miles de filas por producto-día-pliegue).
        n_total = len(y_reales)
        if n_total > 500:
            idx_muestra = np.linspace(0, n_total - 1, 500).astype(int)
        else:
            idx_muestra = np.arange(n_total)
        scatter = [
            {"real": round(float(y_reales[i]), 1), "predicho": round(float(pred_ganador[i]), 1)}
            for i in idx_muestra
        ]

        detalle_completo = pd.concat(detalle_folds, ignore_index=True)
        serie_diaria = (
            detalle_completo.groupby("fecha")
            .agg(real=("real", "sum"), restoiq=("restoiq", "sum"),
                 baseline=("baseline", "sum"), naive=("naive", "sum"))
            .reset_index()
        )
        serie_diaria["fecha"] = serie_diaria["fecha"].dt.strftime("%Y-%m-%d")

        self.metricas["holdout"] = {
            "n_folds":          n_folds,
            "esquema":          "walk-forward (TimeSeriesSplit, ventana expansiva)",
            "dias_evaluados":   int(detalle_completo["fecha"].nunique()),
            "fecha_desde":      str(detalle_completo["fecha"].min().date()),
            "fecha_hasta":      str(detalle_completo["fecha"].max().date()),
            "mae_restoiq":      mae_restoiq,
            "mape_restoiq":     mape_restoiq,
            "wape_restoiq":     wape_restoiq,
            "rmse_restoiq":     rmse_restoiq,
            "nrmse_restoiq":    nrmse_restoiq,
            "nmae_restoiq":     nmae_restoiq,
            "r2_restoiq":       r2_restoiq,
            "mae_baseline":     mae_baseline,
            "mape_baseline":    mape_baseline,
            "wape_baseline":    wape_baseline,
            "rmse_baseline":    rmse_baseline,
            "nrmse_baseline":   nrmse_baseline,
            "nmae_baseline":    nmae_baseline,
            "r2_baseline":      r2_baseline,
            "mae_naive":        mae_naive,
            "wape_naive":       wape_naive,
            "rmse_naive":       rmse_naive,
            "nrmse_naive":      nrmse_naive,
            "nmae_naive":       nmae_naive,
            "r2_naive":         r2_naive,
            "mejora_pct":            mejora_pct,
            "mejora_vs_naive_pct":   mejora_vs_naive_pct,
            "baseline_nombre":  "Regresión Lineal",
            "naive_nombre":     "Naive estacional (lag-7)",
            "modelo_ganador":   modelo_ganador,
            "scatter":          scatter,
            "serie_diaria":     serie_diaria.to_dict(orient="records"),
        }

        # Métrica única de confiabilidad para toda la app: la del
        # walk-forward agrupado reemplaza cualquier otra. WAPE es la
        # métrica principal (más robusta ante productos de baja
        # demanda); MAPE se conserva como referencia secundaria.
        self.metricas["mae"]  = mae_restoiq
        self.metricas["mape"] = mape_restoiq
        self.metricas["wape"] = wape_restoiq
        self.metricas["rmse"]  = rmse_restoiq
        self.metricas["nrmse"] = nrmse_restoiq
        self.metricas["nmae"]  = nmae_restoiq

        if verbose:
            print(f" Walk-forward ({n_folds} pliegues): RestoIQ MAE={mae_restoiq} WAPE={wape_restoiq}% "
                  f"· Baseline MAE={mae_baseline} WAPE={wape_baseline}% "
                  f"· Naive MAE={mae_naive} WAPE={wape_naive}%")
            print(f"  Mejora vs. baseline: {mejora_pct}% · Mejora vs. naive: {mejora_vs_naive_pct}% "
                  f"· Ganador: {modelo_ganador}\n")

    # ── Predicción ────────────────────────────────────────────────────────

    def predecir(self, dias=7, dias_operacion=None) -> dict:
        """
        Retorna:
          por_producto : DataFrame con fecha, producto, cantidad_pred, ingreso_pred
          diario       : DataFrame con totales por día
          pivote       : tabla fecha × producto (cantidad)
          resumen      : dict con totales y top productos
        """
        fechas_futuras = self._fechas_prediccion(dias, dias_operacion)

        if self.estrategia == "lgbm":
            por_producto = self._predecir_lgbm(fechas_futuras)
        else:
            por_producto = self._predecir_promedio_movil(fechas_futuras)

        # Calcular ingreso
        por_producto["precio"]       = por_producto["producto"].map(self.precio_promedio).fillna(0)
        por_producto["ingreso_pred"] = (por_producto["cantidad_pred"] * por_producto["precio"]).round(2)
        por_producto = por_producto.drop(columns=["precio"])

        # Totales diarios
        diario = (
            por_producto.groupby("fecha")
            .agg(
                ingreso_total_pred  =("ingreso_pred",   "sum"),
                cantidad_total_pred =("cantidad_pred",  "sum"),
            )
            .reset_index()
        )
        diario["ingreso_total_pred"] = diario["ingreso_total_pred"].round(2)

        # Tabla pivote fecha × producto
        pivote = por_producto.pivot_table(
            index="fecha",
            columns="producto",
            values="cantidad_pred",
            fill_value=0,
        ).reset_index()
        pivote.columns.name = None

        # Resumen
        top5 = (
            por_producto.groupby("producto")["cantidad_pred"]
            .sum()
            .sort_values(ascending=False)
            .head(5)
            .to_dict()
        )
        resumen = {
            "estrategia":           self.estrategia,
            "periodo":              f"{por_producto['fecha'].min()} → {por_producto['fecha'].max()}",
            "ingreso_total_pred":   round(float(diario["ingreso_total_pred"].sum()), 2),
            "cantidad_total_pred":  int(diario["cantidad_total_pred"].sum()),
            "dia_mayor_venta":      diario.loc[diario["ingreso_total_pred"].idxmax(), "fecha"],
            "dia_menor_venta":      diario.loc[diario["ingreso_total_pred"].idxmin(), "fecha"],
            "top_5_productos":      top5,
        }

        return {
            "por_producto": por_producto,
            "diario":       diario,
            "pivote":       pivote,
            "resumen":      resumen,
        }

    def _fechas_prediccion(self, dias: int, dias_operacion=None) -> list:
        """
        Ancla la predicción en la fecha real de HOY (el momento en que el
        usuario usa la app) — nunca en fechas pasadas del CSV histórico.

        Si el historial está al día (p. ej. se resubieron datos ayer),
        predice desde el día siguiente al último dato real, como antes.

        Si el historial está desactualizado (p. ej. el CSV solo tiene
        datos de enero y hoy es julio), predice igual a partir de HOY.
        Los lags/rolling que no encuentren una fecha exacta en el
        historial ya caen automáticamente al promedio del producto
        (ver _features_para_fecha_producto → get_lag), así que esto
        nunca revienta, solo reduce precisión si el historial es viejo.

        Si `dias_operacion` viene dado (set de weekday(): 0=Lunes...
        6=Domingo), los días cerrados se SALTAN — se sigue devolviendo
        `dias` fechas hábiles, no `dias` fechas de calendario.
        """
        hoy = pd.Timestamp(datetime.now().date())
        ancla = max(self.fecha_ultimo_dato, hoy)

        fechas = []
        cursor = ancla
        intentos = 0
        limite_intentos = dias * 4 + 14  # margen de seguridad ante negocios con pocos días abiertos

        while len(fechas) < dias and intentos < limite_intentos:
            cursor = cursor + timedelta(days=1)
            intentos += 1
            if dias_operacion is None or cursor.weekday() in dias_operacion:
                fechas.append(cursor)

        return fechas

    def _predecir_lgbm(self, fechas_futuras: list) -> pd.DataFrame:
        filas = []
        for fecha in fechas_futuras:
            for producto in self.productos:
                feat = self._features_para_fecha_producto(fecha, producto)
                filas.append(feat)

        df_pred = pd.DataFrame(filas)
        cantidades = np.maximum(0, self.modelo.predict(df_pred[self.features])).round()
        df_pred["cantidad_pred"] = cantidades.astype(int)

        df_pred["fecha"] = df_pred["fecha"].dt.strftime("%Y-%m-%d")
        return df_pred[["fecha", "producto", "cantidad_pred"]]

    def _predecir_promedio_movil(self, fechas_futuras: list) -> pd.DataFrame:
        filas = []
        for fecha in fechas_futuras:
            f_dia      = self.factor_dia.get(fecha.weekday(), 1.0)
            f_feriado  = 1.25 if fecha.strftime("%Y-%m-%d") in self.feriados else 1.0
            for producto in self.productos:
                base     = self.promedio_producto.get(producto, 1)
                cantidad = max(0, round(base * f_dia * f_feriado))
                filas.append({
                    "fecha":         fecha.strftime("%Y-%m-%d"),
                    "producto":      producto,
                    "cantidad_pred": cantidad,
                })
        return pd.DataFrame(filas)

    def _features_para_fecha_producto(self, fecha: pd.Timestamp, producto: str) -> dict:
        feat = self._features_fecha(fecha)
        hist_prod = (
            self.df_historial[self.df_historial["producto"] == producto]
            .set_index("fecha")["cantidad"]
        )
        def get_lag(d):
            lf = fecha - timedelta(days=d)
            return float(hist_prod.get(lf, hist_prod.mean() if len(hist_prod) else 0))

        reciente = hist_prod.sort_index().tail(14)
        feat.update({
            "fecha":            fecha,
            "producto":         producto,
            "producto_encoded": int(self.label_encoder.transform([producto])[0]),
            "lag_7":            get_lag(7),
            "lag_14":           get_lag(14),
            "lag_28":           get_lag(28),
            "rolling_7_mean":   float(reciente.tail(7).mean()) if len(reciente) >= 1 else 0,
            "rolling_14_mean":  float(reciente.mean())          if len(reciente) >= 1 else 0,
            "rolling_7_std":    float(reciente.tail(7).std())  if len(reciente) >= 2 else 0,
        })

        # Features de negocio opcionales: para fechas FUTURAS no hay
        # forma de saber si habrá promoción/evento — se asume que no
        # (0), salvo que en una versión futura el usuario las declare
        # explícitamente para fechas puntuales.
        for col in ("promocion", "es_evento_especial"):
            if col in self.features:
                feat[col] = 0
        if "descuento_pct" in self.features:
            feat["descuento_pct"] = 0.0

        return feat

    # ── Persistencia ──────────────────────────────────────────────────────

    def guardar(self, ruta="models/sales_model.pkl"):
        os.makedirs(os.path.dirname(ruta) if os.path.dirname(ruta) else ".", exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(self, f)
        print(f"✅ Modelo guardado en: {ruta}")

    @classmethod
    def cargar(cls, ruta="models/sales_model.pkl") -> "SalesModel":
        with open(ruta, "rb") as f:
            modelo = pickle.load(f)
        print(f" Modelo cargado desde: {ruta}")
        return modelo

    # ── Métricas ──────────────────────────────────────────────────────────

    def _imprimir_metricas(self):
        m = self.metricas
        print(f"\n{'='*52}")
        print(f" MODELO ENTRENADO  [{m['estrategia']}]")
        print(f"{'='*52}")
        print(f"Período:    {m.get('fecha_desde')} → {m.get('fecha_hasta')}")
        print(f"Productos:  {m['productos']}")
        print(f"Historial:  {m.get('dias_historial')} / {m.get('umbral_dias_ml')} días (umbral LightGBM)")
        if m.get("mae") is not None:
            print(f"MAE:        {m['mae']} unidades  (walk-forward, validado)")
            print(f"MAPE:       {m['mape']}%")
            print(f"WAPE:       {m.get('wape')}%")
            if m.get("top_features"):
                print(f"Top feat.:  {', '.join(m['top_features'])}")
        elif m.get("mae_provisional") is not None:
            print(f"MAE:        {m['mae_provisional']} unidades  (PROVISIONAL, sin validar — walk-forward no corrió)")
        else:
            print(f"Nota:       {m.get('nota', 'Sin métricas — historial insuficiente para evaluar')}")

        h = m.get("holdout")
        if h:
            print(f"Validación: {h['n_folds']} pliegues walk-forward · {h['dias_evaluados']} días evaluados "
                  f"({h['fecha_desde']} → {h['fecha_hasta']})")
            print(f"  RestoIQ ({m['estrategia']}):  MAE={h['mae_restoiq']}  WAPE={h['wape_restoiq']}%")
            print(f"  Baseline ({h['baseline_nombre']}): MAE={h['mae_baseline']}  WAPE={h['wape_baseline']}%")
            print(f"  Mejora sobre baseline: {h['mejora_pct']}%")
        else:
            print("Validación: no disponible (historial < 60 días)")
        print(f"{'='*52}\n")


# ════════════════════════════════════════════════════════════
# EJECUCIÓN DIRECTA
# ════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

    print(" RestoIQ — Prueba del modelo de predicción\n")

    ruta_csv = "data/train_restaurante.csv"
    if not os.path.exists(ruta_csv):
        print(f" No se encontró {ruta_csv}")
        print("   Ejecuta primero: python services/data_generator.py restaurante")
        sys.exit(1)

    df = pd.read_csv(ruta_csv)
    print(f" Dataset: {len(df):,} filas, {df['producto'].nunique()} productos\n")

    model   = SalesModel()
    model.entrenar(df, verbose=True)

    print(" Predicciones próximos 7 días...\n")
    resultado = model.predecir(dias=7)

    # ── Detalle por producto y día ──
    print(" DETALLE POR PRODUCTO Y DÍA:")
    print(resultado["por_producto"].to_string(index=False))

    # ── Totales diarios ──
    print(" TOTALES DIARIOS:")
    print(resultado["diario"].to_string(index=False))

    # ── Tabla pivote ──
    print(" TABLA PIVOTE (fecha × producto):")
    print(resultado["pivote"].to_string(index=False))

    # ── Resumen ──
    r = resultado["resumen"]
    print(f" RESUMEN SEMANAL [{r['estrategia']}]:")
    print(f"  Período:       {r['periodo']}")
    print(f"  Ingreso total: ${r['ingreso_total_pred']:,.2f}")
    print(f"  Unidades:      {r['cantidad_total_pred']:,}")
    print(f"  Mejor día:     {r['dia_mayor_venta']}")
    print(f"  Top productos:")
    for prod, cant in r["top_5_productos"].items():
        print(f"    • {prod:<28} {cant:>6} unidades")

    # Exportar detalle a CSV
    resultado["por_producto"].to_csv("data/predicciones_detalle.csv", index=False)
    print(" Detalle exportado: data/predicciones_detalle.csv")

    model.guardar("models/sales_model.pkl")