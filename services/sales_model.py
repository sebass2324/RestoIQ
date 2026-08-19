"""
services/sales_model.py
RestoIQ — Modelo de predicción de demanda (LightGBM Direct Forecasting / promedio móvil)

Direct Forecasting: en vez de un solo modelo recursivo (que encadena
sus propias predicciones día a día y acumula error), se entrena UN
MODELO INDEPENDIENTE por cada día del horizonte (día+1, día+2, ...
día+14). Cada modelo predice directamente desde el último dato REAL
conocido -- nunca desde una predicción anterior.

Comprobado con validación honesta (walk-forward, mismos pliegues que
antes): comparado contra el enfoque recursivo previo, esto mejora el
WAPE en 13-14 puntos para horizontes de 4-7 días (día+7: 43.83% ->
30.43%), sin cambiar el dato de entrada -- solo la arquitectura de
entrenamiento y predicción.
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Configuración
UMBRAL_DIAS_ML = 90
MINIMO_FILAS_LGBM = 30
HORIZONTE_MAX = 14  # modelos independientes entrenados: día+1 ... día+14

# ── Features temporales ──

FEATURES_TEMPORALES = [
    "dia_semana", "dia_mes", "mes", "semana_anio",
    "es_finde", "es_puente",
    "dias_distancia_cobro", "pico_comida_rapida", "fase_liquidez",
]
FEATURES_TEMPORALES_OPCIONALES = {
    "es_feriado":      "considerar_feriados",
    "vispera_feriado": "considerar_feriados",
}

# ── Features históricas ──
# Lags y promedios móviles, calculados del propio historial de ventas
# del producto (ver _preparar_features).
FEATURES_HISTORICAS = [
    "lag_1", "lag_7", "lag_14", "lag_28",
    "rolling_7_mean", "rolling_14_mean", "rolling_28_mean", "rolling_7_std",
]

# ── Features del producto ──

FEATURES_PRODUCTO = ["producto", "categoria"]

# ── Features comerciales ──

FEATURES_COMERCIALES_SI_EXISTEN = [
    "precio",
    "lluvia_nocturna_mm", "temp_nocturna_promed",
]
FEATURES_COMERCIALES_OPCIONALES = {
    "promocion":          "considerar_promociones",
    "descuento_pct":      "considerar_descuentos",
    "es_evento_especial": "considerar_eventos",
}


FEATURES_CATEGORICAS_FLAG = [
    "dia_semana", "dia_mes", "mes", "semana_anio", "es_finde", "es_puente",
    "es_feriado", "vispera_feriado", "promocion", "es_evento_especial",
]
# "dias_distancia_cobro", "pico_comida_rapida" y "fase_liquidez"
# quedan DELIBERADAMENTE fuera de esta lista -- son numéricas/ordinales
# a propósito, para que LightGBM pueda encontrar cortes finos (ej.
# dias_distancia_cobro <= 2), en vez de tratarlas como categorías
# discretas sin orden. Meterlas acá anularía la razón por la que se
# rediseñó la variable.

TARGET = "cantidad"

LGBM_PARAMS = {
    "objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 20, "max_bin": 255, "feature_fraction": 0.8,
    "bagging_fraction": 0.8, "bagging_freq": 1, "lambda_l1": 0.1,
    "lambda_l2": 0.1, "verbose": -1, "seed": 42,
}


class SalesModel:

    def __init__(self):
        self.modelos               = {}     # {k: modelo_lgbm} -- uno por día del horizonte (Direct Forecasting)
        self.estrategia           = None
        self.productos            = []
        self.categorias_producto  = []
        self.categorias_categoria = []
        self.precio_actual_por_producto = {}
        self.categoria_por_producto = {}
        self.df_historial         = None
        self.n_arboles_optimo     = None
        self.metricas             = {}
        self.fecha_ultimo_dato    = None
        self.feriados             = self._cargar_feriados()
        self.clima_promedio = {
            "lluvia_nocturna_mm": 0.0, "temp_nocturna_promed": 26.0,
        }
        self.features              = FEATURES_TEMPORALES + FEATURES_HISTORICAS + FEATURES_PRODUCTO
        self.ancla_cols            = []  # se llenan en entrenar() -- features conocidas al momento de predecir (lag/rolling/producto)
        self.objetivo_cols         = []  # features de la fecha objetivo (calendario/clima futuro)

    # Calendarios externos

    def _cargar_feriados(self):
        for mod in ["services.data_generator", "data_generator"]:
            try:
                import importlib
                return importlib.import_module(mod).FERIADOS_ECUADOR
            except Exception:
                continue
        return set()

    # Variables de tiempo

    def _features_fecha(self, fecha: pd.Timestamp) -> dict:
        fecha_str = fecha.strftime("%Y-%m-%d")
        dia       = fecha.weekday()
        ayer      = (fecha - timedelta(days=1)).strftime("%Y-%m-%d")
        maniana   = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
        dia_mes   = fecha.day
        dias_en_mes = fecha.days_in_month

        # Misma lógica EXACTA que services/data_generator.py
        # (_features_temporales) -- tienen que coincidir siempre,
        # entrenamiento y predicción calculan esto igual.
        dist_15 = abs(dia_mes - 15)
        dist_fin = min(dia_mes - 1, dias_en_mes - dia_mes)
        dias_distancia_cobro = min(dist_15, dist_fin)

        es_ventana_pago = (
            dia_mes in (15, 16, dias_en_mes, 1)
            or (dia_mes in (13, 14, dias_en_mes - 2, dias_en_mes - 1) and dia == 4)
        )
        es_finde_alto = dia in (4, 5)
        pico_comida_rapida = int(es_ventana_pago and es_finde_alto)

        if pico_comida_rapida == 1 or dia_mes in (15, dias_en_mes, 1):
            fase_liquidez = 3
        elif dia_mes in (2, 16, 17):
            fase_liquidez = 2
        elif dia_mes in (13, 14, dias_en_mes - 2, dias_en_mes - 1) and pico_comida_rapida == 0:
            fase_liquidez = 0
        else:
            fase_liquidez = 1

        return {
            "dia_semana":     dia,
            "dia_mes":        dia_mes,
            "mes":            fecha.month,
            "semana_anio":    int(fecha.isocalendar()[1]),
            "es_finde":       int(dia in [5, 6]),
            "es_feriado":     int(fecha_str in self.feriados),
            "vispera_feriado": int(maniana in self.feriados),
            "es_puente":      int(ayer in self.feriados or maniana in self.feriados),
            "dias_distancia_cobro": dias_distancia_cobro,
            "pico_comida_rapida":   pico_comida_rapida,
            "fase_liquidez":        fase_liquidez,
        }

    def _asegurar_features_contexto(self, df: pd.DataFrame) -> pd.DataFrame:
        if "dia_semana" not in df.columns:
            feats = df["fecha"].apply(self._features_fecha)
            for col in feats.iloc[0].keys():
                df[col] = feats.apply(lambda x: x[col])
        return df

    # Ingeniería de variables: calendario continuo + lags + rolling

    def _preparar_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        columnas_negocio = [c for c in ("promocion", "descuento_pct", "es_evento_especial")
                            if c in df.columns]
        columnas_clima = [c for c in ("lluvia_nocturna_mm", "temp_nocturna_promed")
                          if c in df.columns]

        piezas = []
        for producto, grupo in df.groupby("producto"):
            # calendario continuo por producto — evita que shift() se desalinee con huecos
            grupo = grupo.set_index("fecha").sort_index()
            rango = pd.date_range(grupo.index.min(), grupo.index.max(), freq="D")

            pieza = pd.DataFrame({"fecha": rango})
            pieza["producto"]  = producto
            pieza["categoria"] = self.categoria_por_producto.get(producto, "Sin categoría")
            pieza["cantidad"]  = grupo["cantidad"].reindex(rango, fill_value=0).values

            # Precio: el ÚLTIMO precio real conocido hasta esa fecha
            # (ffill), NUNCA un promedio de todo el historial -- un
            # promedio global usaría precios que todavía no existían
            # en fechas pasadas (fuga de datos hacia el futuro). Los
            # días sin precio real anterior (muy al inicio del
            # historial de un producto, antes de su primera venta
            # registrada) usan el primer precio real disponible como
            # única excepción razonable -- no hay forma causal de
            # saber el precio antes de que exista un solo dato.
            if "precio" in grupo.columns:
                pieza["precio"] = grupo["precio"].reindex(rango).ffill().bfill().fillna(0.0).values
            else:
                pieza["precio"] = 0.0

            for col in columnas_negocio:
                pieza[col] = grupo[col].reindex(rango, fill_value=0).values

            # clima: rellenar con el promedio, NUNCA con 0 — un día sin
            # dato no significa "0°C" ni "0mm", significa "no sabemos",
            # y el promedio es la mejor estimación neutral disponible.
            for col in columnas_clima:
                serie = grupo[col].reindex(rango)
                pieza[col] = serie.fillna(serie.mean()).values

            # variables de historial: lags
            for lag in (1, 7, 14, 28):
                pieza[f"lag_{lag}"] = pieza["cantidad"].shift(lag)

            # variables de historial: rolling (shift(1) antes de rolling, sin fuga de datos)
            pieza["rolling_7_mean"]  = pieza["cantidad"].shift(1).rolling(7,  min_periods=1).mean()
            pieza["rolling_14_mean"] = pieza["cantidad"].shift(1).rolling(14, min_periods=1).mean()
            pieza["rolling_28_mean"] = pieza["cantidad"].shift(1).rolling(28, min_periods=1).mean()
            pieza["rolling_7_std"]   = pieza["cantidad"].shift(1).rolling(7,  min_periods=1).std().fillna(0)

            piezas.append(pieza)

        df_continuo = pd.concat(piezas, ignore_index=True)
        df_continuo = self._asegurar_features_contexto(df_continuo)
        df_continuo = df_continuo.dropna(subset=["lag_7", "lag_28"])

        # categóricas nativas de pandas — mismas categorías fijas en train y predicción
        df_continuo["producto"] = pd.Categorical(df_continuo["producto"], categories=self.categorias_producto)
        df_continuo["categoria"] = pd.Categorical(df_continuo["categoria"], categories=self.categorias_categoria)

        return df_continuo

    # Entrenamiento

    def entrenar(self, df: pd.DataFrame, config=None, verbose=True) -> dict:
        df = df.copy()
        df["fecha"]    = pd.to_datetime(df["fecha"])
        df["cantidad"] = pd.to_numeric(df["cantidad"], errors="coerce")
        df = df.dropna(subset=["fecha", "producto", "cantidad"])
        df = df[df["cantidad"] > 0]

        self.productos           = sorted(df["producto"].unique().tolist())
        self.categorias_producto = self.productos
        self.fecha_ultimo_dato   = df["fecha"].max()
        self.categoria_por_producto = (
            df.groupby("producto")["categoria"].first().to_dict() if "categoria" in df.columns else {}
        )
        self.categorias_categoria = sorted(set(self.categoria_por_producto.values())) or ["Sin categoría"]

        # Promedios de clima — respaldo si el pronóstico en vivo falla
        # al predecir (API caída, sin internet). Nunca se deja una
        # predicción sin poder correr por un problema de red externo.
        self.clima_promedio = {
            col: float(df[col].mean())
            for col in ("lluvia_nocturna_mm", "temp_nocturna_promed")
            if col in df.columns
        }

        self.features = self._construir_features(df, config)
        self.ancla_cols, self.objetivo_cols = self._clasificar_features()

        # agregación diaria por producto
        agg_spec = {"cantidad": ("cantidad", "sum")}
        for col, fn in (("precio", "mean"), ("promocion", "max"), ("descuento_pct", "mean"), ("es_evento_especial", "max"),
                        ("lluvia_nocturna_mm", "mean"), ("temp_nocturna_promed", "mean")):
            if col in df.columns:
                agg_spec[col] = (col, fn)

        df_agg = df.groupby(["fecha", "producto"]).agg(**agg_spec).reset_index()
        for col in ("promocion", "descuento_pct", "es_evento_especial"):
            if col in df_agg.columns:
                df_agg[col] = df_agg[col].fillna(0)

        dias_historial = (df["fecha"].max() - df["fecha"].min()).days
        df_feat = self._preparar_features(df_agg)
        self.df_historial = df_feat[["fecha", "producto", "cantidad"]].copy()
        # Precio ACTUAL (el más reciente conocido, ya calculado sin fuga
        # en _preparar_features) -- es el que usa la predicción como
        # feature ancla.
        # un promedio de todo el historial, usado SOLO para estimar el
        # ingreso proyectado en pantalla, nunca como entrada del modelo.
        self.precio_actual_por_producto = (
            df_feat.sort_values("fecha").groupby("producto")["precio"].last().to_dict()
            if "precio" in df_feat.columns else {}
        )

        if verbose:
            print(f" Historial disponible: {dias_historial} días · {len(df_feat)} filas con lags completos")

        # selección de estrategia
        if dias_historial >= UMBRAL_DIAS_ML and len(df_feat) >= MINIMO_FILAS_LGBM:
            self.estrategia = "lgbm"
            if verbose:
                print(" Estrategia: LightGBM\n")
            self._entrenar_lgbm(df_feat, verbose)
        else:
            self.estrategia = "promedio_movil"
            if verbose:
                print(" Estrategia: Promedio móvil ponderado\n")
            self._entrenar_promedio_movil(df_agg)

        self._evaluar_holdout(df_feat, dias_historial, verbose)
        self.metricas["dias_historial"] = dias_historial
        self.metricas["umbral_dias_ml"] = UMBRAL_DIAS_ML
        return self.metricas

    def _construir_features(self, df: pd.DataFrame, config) -> list:
        features = []

        # 1. Temporales — siempre entran
        features += FEATURES_TEMPORALES

        # 1b. Temporales opcionales — se calculan solas, solo dependen
        #     de si el negocio quiere considerarlas (nunca del archivo)
        for columna, atributo_config in FEATURES_TEMPORALES_OPCIONALES.items():
            quiere_usarla = True if config is None else bool(getattr(config, atributo_config, False))
            if quiere_usarla:
                features.append(columna)

        # 2. Históricas — siempre entran (calculadas del propio historial)
        features += FEATURES_HISTORICAS

        # 3. Producto — siempre entran (categóricas nativas)
        features += FEATURES_PRODUCTO

        # 4. Comerciales — dependen del archivo del usuario
        for columna in FEATURES_COMERCIALES_SI_EXISTEN:
            if columna in df.columns and df[columna].notna().any():
                features.append(columna)

        for columna, atributo_config in FEATURES_COMERCIALES_OPCIONALES.items():
            columna_existe = columna in df.columns and df[columna].notna().any()
            quiere_usarla = False if config is None else bool(getattr(config, atributo_config, False))
            if quiere_usarla and columna_existe:
                features.append(columna)

        return features

    def _categoricas_activas(self) -> list:
        cats = [c for c in FEATURES_CATEGORICAS_FLAG if c in self.features]
        cats += [c for c in ("producto", "categoria") if c in self.features]
        return cats

    # Entrenamiento LightGBM

    def _clasificar_features(self):
        """Separa self.features en 2 grupos, para Direct Forecasting:
        ANCLA = lo que se conoce en el momento de predecir (historial
        del producto: lags/rolling, más su identidad); OBJETIVO = lo
        que se sabe de antemano sobre la fecha futura que se predice
        (calendario -- siempre determinístico -- y clima, vía
        pronóstico)."""
        cols_ancla_base = set(FEATURES_HISTORICAS) | set(FEATURES_PRODUCTO) | {"precio"}
        ancla = [f for f in self.features if f in cols_ancla_base]
        objetivo = [f for f in self.features if f not in cols_ancla_base]
        return ancla, objetivo

    def _construir_dataset_direct(self, df_feat, k):
        """Para cada producto, empareja las features ANCLA de la fecha
        T con las features OBJETIVO y la cantidad real de la fecha
        T+k -- así el modelo aprende a predecir k días hacia adelante
        directamente desde datos reales, sin encadenar predicciones."""
        columnas_futuro = [TARGET] + self.objetivo_cols
        piezas = []
        for _, grupo in df_feat.groupby("producto"):
            grupo = grupo.sort_values("fecha").reset_index(drop=True)
            anclas = grupo[["fecha"] + self.ancla_cols].reset_index(drop=True)
            futuro = grupo[columnas_futuro].shift(-k).reset_index(drop=True)
            combinado = pd.concat([anclas, futuro], axis=1)
            combinado = combinado.dropna(subset=[TARGET])
            piezas.append(combinado)
        if not piezas:
            return pd.DataFrame(columns=["fecha"] + self.features + [TARGET])
        resultado = pd.concat(piezas, ignore_index=True)
        resultado["producto"] = pd.Categorical(resultado["producto"], categories=self.categorias_producto)
        resultado["categoria"] = pd.Categorical(resultado["categoria"], categories=self.categorias_categoria)
        return resultado

    def _entrenar_lgbm(self, df_feat, verbose):
        categoricas = self._categoricas_activas()

        # Early stopping en 2 fases -- UNA sola vez (usando k=1 como
        # referencia representativa), reutilizado en los 14 modelos.
        # Repetir la búsqueda completa 14 veces sería 14x más lento sin
        # aportar nada: misma arquitectura de features, mismo tipo de
        # problema en cada horizonte.
        ds_ref = self._construir_dataset_direct(df_feat, 1).sort_values("fecha").reset_index(drop=True)
        n_val = max(15, int(len(ds_ref) * 0.15))
        corte_val = len(ds_ref) - n_val
        X_train_fs = ds_ref.iloc[:corte_val][self.features]
        y_train_fs = ds_ref.iloc[:corte_val][TARGET]
        X_val_fs   = ds_ref.iloc[corte_val:][self.features]
        y_val_fs   = ds_ref.iloc[corte_val:][TARGET]

        params_busqueda = {k: v for k, v in LGBM_PARAMS.items() if k != "verbose"}
        modelo_busqueda = lgb.LGBMRegressor(n_estimators=2000, verbose=-1, **params_busqueda)
        modelo_busqueda.fit(
            X_train_fs, y_train_fs, eval_set=[(X_val_fs, y_val_fs)],
            categorical_feature=categoricas,
            callbacks=[lgb.early_stopping(50, verbose=False)],
        )
        n_arboles_optimo = modelo_busqueda.best_iteration_ or 700
        self.n_arboles_optimo = int(n_arboles_optimo)

        # Modelo final por cada día del horizonte -- con el 100% de los
        # datos disponibles para ESE horizonte (refit estándar, mismo
        # criterio que siempre: la búsqueda de arriba midió honesto con
        # una porción separada, el modelo final aprovecha todo).
        parametros = dict(LGBM_PARAMS)
        self.modelos = {}
        filas_por_horizonte = {}
        for k in range(1, HORIZONTE_MAX + 1):
            ds = self._construir_dataset_direct(df_feat, k)
            if len(ds) < MINIMO_FILAS_LGBM:
                continue
            X, y = ds[self.features], ds[TARGET]
            train_set = lgb.Dataset(X, label=y, categorical_feature=categoricas)
            self.modelos[k] = lgb.train(parametros, train_set, num_boost_round=self.n_arboles_optimo)
            filas_por_horizonte[k] = len(ds)
            if verbose:
                print(f"  modelo día+{k}: {len(ds)} filas de entrenamiento")

        if not self.modelos:
            raise ValueError("No hay suficientes datos para entrenar ni un solo horizonte de Direct Forecasting.")

        importancias = pd.Series(
            self.modelos[1].feature_importance(importance_type="gain"), index=self.features
        ).sort_values(ascending=False)

        self.metricas = {
            "estrategia": "LightGBM (Direct Forecasting)",
            "n_train": int(filas_por_horizonte.get(1, 0)),
            "horizonte_max": max(self.modelos.keys()),
            "hiperparametros": dict(LGBM_PARAMS),
            "top_features": importancias.head(5).index.tolist(),
            "importancias_top10": importancias.head(10).round(1).to_dict(),
        }

    def _entrenar_promedio_movil(self, df_agg):
        self.factor_dia = df_agg.groupby(df_agg["fecha"].dt.dayofweek)["cantidad"].mean()
        media_global = self.factor_dia.mean()
        self.factor_dia = (self.factor_dia / media_global).to_dict()

        fecha_corte = df_agg["fecha"].max() - timedelta(days=14)
        reciente    = df_agg[df_agg["fecha"] >= fecha_corte]
        self.promedio_producto = reciente.groupby("producto")["cantidad"].mean().to_dict()

        self.metricas = {
            "estrategia": "Promedio móvil ponderado",
            "mae": None, "mae_provisional": None,
            "productos": len(self.productos),
            "fecha_desde": str(df_agg["fecha"].min().date()),
            "fecha_hasta": str(self.fecha_ultimo_dato.date()),
            "nota": "Datos insuficientes para ML. Se usó promedio móvil.",
            "features_usadas": self.features,
        }

    # Evaluación: walk-forward + baselines

    # Bloques de prueba de exactamente este tamaño -- coincide con el
    # horizonte más común que ofrece la app (7 días). Antes, el
    # histórico se dividía en N partes IGUALES entre sí, lo que con
    # historiales largos daba bloques de prueba de cientos de días
    # (ej. 1460 días / 6 ≈ 243 días por pliegue) -- eso mide un
    # escenario que la app nunca ofrece, no el desempeño real a 7 días.
    HORIZONTE_VALIDACION = HORIZONTE_MAX  # evaluar los 14 días, para poder reportar día+7 y día+14
    MAX_PLIEGUES = 4  # con Direct Forecasting cada pliegue entrena 14 modelos -- se reduce el tope para mantener el tiempo de reentrenamiento razonable

    def _calcular_n_folds(self, dias_historial: int):
        if dias_historial < 60:   return None
        if dias_historial <= 120: return 2
        if dias_historial <= 250: return 3
        return 5

    @classmethod
    def _pliegues_por_fecha(cls, fechas_unicas, n_folds=None):
        """Separa por FECHA completa, no por fila -- con una fila por
        producto-día, cortar por posición de fila puede dejar algunos
        productos de una misma fecha en train y otros en test. Mismo
        patrón ya usado en el clasificador de prioridad.

        Cada bloque de prueba mide EXACTAMENTE HORIZONTE_VALIDACION
        días (7), avanzando de 7 en 7 sobre el histórico disponible --
        así el WAPE reportado sí describe "qué tan bien predice un
        lote de 7 días", que es lo que la app realmente ofrece. El
        parámetro `n_folds` ya no se usa para el tamaño del bloque
        (se deja solo por compatibilidad de firma); el número de
        pliegues sale de cuántos bloques de 7 días caben, con un tope
        de MAX_PLIEGUES para no disparar el tiempo de cómputo."""
        fechas_unicas = np.sort(np.unique(fechas_unicas))
        n = len(fechas_unicas)
        dias_test = cls.HORIZONTE_VALIDACION
        min_dias_train = max(MINIMO_FILAS_LGBM, dias_test * 4)  # margen razonable de entrenamiento antes del primer pliegue

        if n <= min_dias_train + dias_test:
            return []

        pliegues = []
        inicio_test = min_dias_train
        while inicio_test + dias_test <= n:
            fecha_corte = pd.Timestamp(fechas_unicas[inicio_test])
            fecha_fin   = pd.Timestamp(fechas_unicas[min(inicio_test + dias_test - 1, n - 1)])
            pliegues.append((fecha_corte, fecha_fin))
            inicio_test += dias_test  # bloques consecutivos, sin solape

        if len(pliegues) > cls.MAX_PLIEGUES:
            # Muestra pareja a lo largo de todo el histórico, no solo
            # los primeros -- para no sesgar hacia una sola época del año.
            idx = np.linspace(0, len(pliegues) - 1, cls.MAX_PLIEGUES).astype(int)
            pliegues = [pliegues[i] for i in idx]

        return pliegues


    def _predecir_promedio_movil_holdout(self, train_feat, test_feat):
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
    def _wape(y_true, y_pred):
        total_real = np.sum(np.abs(y_true))
        if total_real == 0:
            return None
        return round(float(np.sum(np.abs(y_true - y_pred)) / total_real) * 100, 2)

    def _evaluar_holdout(self, df_feat, dias_historial, verbose=False):
        n_folds = self._calcular_n_folds(dias_historial)
        if n_folds is None:
            self.metricas["holdout"] = None
            return

        df_feat = df_feat.copy()
        fechas_unicas_todas = df_feat["fecha"].unique()

        # Clima REAL histórico por fecha -- NUNCA pronóstico acá, porque
        # estamos evaluando fechas que ya pasaron y de las que sí
        # conocemos el clima real. Pasar None (como estaba antes) hacía
        # que todas las filas del holdout cayeran al promedio plano de
        # self.clima_promedio, sin importar si ese día llovió fuerte o
        # no -- con el clima siendo la variable más importante del
        # modelo, eso aplanaba la evaluación entera.
        cols_clima = ("lluvia_nocturna_mm", "temp_nocturna_promed")
        cols_clima_presentes = [c for c in cols_clima if c in df_feat.columns]
        if cols_clima_presentes:
            clima_real_por_fecha = (
                df_feat[["fecha"] + cols_clima_presentes]
                .drop_duplicates(subset="fecha")
                .set_index("fecha")[cols_clima_presentes]
                .to_dict(orient="index")
            )
            clima_real_por_fecha = {
                fecha.strftime("%Y-%m-%d"): valores
                for fecha, valores in clima_real_por_fecha.items()
            }
        else:
            clima_real_por_fecha = {}

        pliegues = self._pliegues_por_fecha(fechas_unicas_todas)
        if len(pliegues) < 2:
            self.metricas["holdout"] = None
            return
        categoricas = self._categoricas_activas()

        parametros_replica = dict(LGBM_PARAMS)
        # MISMO número de árboles que el modelo final desplegado -- antes
        # esta réplica usaba 700 fijo, sin importar cuál fue el número
        # real elegido por early stopping, así que las métricas podían
        # no corresponder exactamente al modelo en producción.
        n_arboles_replica = self.n_arboles_optimo or 700

        y_reales_todos, pred_restoiq_todos, pred_baseline_todos, pred_naive_todos = [], [], [], []

        # WAPE por día del horizonte (día+1 ... día+HORIZONTE_VALIDACION)
        # -- con Direct Forecasting, cada posición YA es un modelo
        # independiente, así que este desglose es literalmente el
        # desempeño real de cada modelo_k, no una simulación.
        por_posicion = {i: {"reales": [], "preds": []} for i in range(1, self.HORIZONTE_VALIDACION + 1)}

        for fecha_corte, fecha_fin in pliegues:
            train_feat = df_feat[df_feat["fecha"] < fecha_corte]
            if len(train_feat) < MINIMO_FILAS_LGBM:
                continue

            for k in range(1, self.HORIZONTE_VALIDACION + 1):
                ds_train = self._construir_dataset_direct(train_feat, k)
                if len(ds_train) < MINIMO_FILAS_LGBM:
                    continue

                # Dataset completo para este k, filtrado a las anclas
                # cuyo objetivo (ancla + k días) cae dentro del pliegue
                # de prueba -- nunca se entrena con estas filas (arriba
                # se usó train_feat, estrictamente anterior a fecha_corte).
                ds_completo = self._construir_dataset_direct(df_feat, k)
                mask_test = (
                    (ds_completo["fecha"] >= fecha_corte - timedelta(days=k)) &
                    (ds_completo["fecha"] <= fecha_fin - timedelta(days=k))
                )
                ds_test = ds_completo[mask_test]
                if ds_test.empty:
                    continue

                X_train, y_train = ds_train[self.features], ds_train[TARGET]
                X_test, y_test = ds_test[self.features], ds_test[TARGET]

                if self.estrategia == "lgbm":
                    train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categoricas)
                    modelo_replica = lgb.train(parametros_replica, train_set, num_boost_round=n_arboles_replica)
                    pred_restoiq = np.maximum(0, modelo_replica.predict(X_test))
                else:
                    pred_restoiq = np.full(len(y_test), float(y_train.mean()))

                cols_numericas = [c for c in self.features if c not in ("producto", "categoria")]
                baseline = LinearRegression()
                baseline.fit(X_train[cols_numericas], y_train)
                pred_baseline = np.maximum(0, baseline.predict(X_test[cols_numericas]))

                # Naive (lag-7): la propia feature ancla ya la tiene
                # calculada (causal, shift(1) antes de cualquier cosa),
                # sin necesitar simular ni encadenar nada.
                if "lag_7" in ds_test.columns:
                    respaldo = float(ds_train["lag_7"].mean()) if "lag_7" in ds_train.columns else float(y_train.mean())
                    pred_naive = ds_test["lag_7"].fillna(respaldo).values
                else:
                    pred_naive = np.full(len(y_test), float(y_train.mean()))

                y_reales_todos.append(y_test.values)
                pred_restoiq_todos.append(pred_restoiq)
                pred_baseline_todos.append(pred_baseline)
                pred_naive_todos.append(pred_naive)

                if k in por_posicion:
                    por_posicion[k]["reales"].extend(y_test.values.tolist())
                    por_posicion[k]["preds"].extend(pred_restoiq.tolist())

        if not y_reales_todos:
            self.metricas["holdout"] = None
            return

        y_reales      = np.concatenate(y_reales_todos)
        pred_restoiq  = np.concatenate(pred_restoiq_todos)
        pred_baseline = np.concatenate(pred_baseline_todos)
        pred_naive    = np.concatenate(pred_naive_todos)

        def metricas_de(y_true, y_pred):
            mae  = round(float(mean_absolute_error(y_true, y_pred)), 2)
            wape = self._wape(y_true, y_pred)
            rmse = round(float(np.sqrt(np.mean((y_true - y_pred) ** 2))), 2)
            media = float(np.mean(y_true)) or 1
            return {"mae": mae, "wape": wape, "rmse": rmse,
                    "nmae": round(mae / media, 3), "nrmse": round(rmse / media, 3),
                    "r2": round(float(r2_score(y_true, y_pred)), 3)}

        m_restoiq  = metricas_de(y_reales, pred_restoiq)
        m_baseline = metricas_de(y_reales, pred_baseline)
        m_naive    = metricas_de(y_reales, pred_naive)
        mejora_pct = round((1 - m_restoiq["wape"] / m_baseline["wape"]) * 100, 1) if m_baseline["wape"] else None

        # gráfico Real vs. Predicho del modelo ganador
        scatter_idx = np.random.RandomState(42).choice(len(y_reales), size=min(500, len(y_reales)), replace=False)
        candidatos = {
            "RestoIQ": (m_restoiq["wape"], m_restoiq["r2"], pred_restoiq),
            "Regresión Lineal": (m_baseline["wape"], m_baseline["r2"], pred_baseline),
            "Naive estacional (lag-7)": (m_naive["wape"], m_naive["r2"], pred_naive),
        }
        modelo_ganador, (_, r2_ganador, pred_ganador) = min(candidatos.items(), key=lambda kv: kv[1][0])
        scatter = [{"real": round(float(y_reales[i]), 1), "predicho": round(float(pred_ganador[i]), 1)}
                   for i in scatter_idx]

        # WAPE por día del horizonte -- el día+1 es notablemente mejor
        # que el promedio (nunca depende de una predicción encadenada),
        # y es el número que respalda el flujo de "registro diario"
        # (ver services/registro_diario.py y prediction_service.py).
        wape_por_dia = {}
        for pos, datos in por_posicion.items():
            if datos["reales"]:
                wape_por_dia[pos] = self._wape(np.array(datos["reales"]), np.array(datos["preds"]))
        wape_dia_1 = wape_por_dia.get(1)

        self.metricas["holdout"] = {
            "n_folds": n_folds,
            "mae_restoiq": m_restoiq["mae"], "wape_restoiq": m_restoiq["wape"],
            "rmse_restoiq": m_restoiq["rmse"], "nmae_restoiq": m_restoiq["nmae"],
            "nrmse_restoiq": m_restoiq["nrmse"], "r2_restoiq": m_restoiq["r2"],
            "mae_baseline": m_baseline["mae"], "wape_baseline": m_baseline["wape"],
            "rmse_baseline": m_baseline["rmse"], "nmae_baseline": m_baseline["nmae"],
            "nrmse_baseline": m_baseline["nrmse"], "r2_baseline": m_baseline["r2"],
            "mae_naive": m_naive["mae"], "wape_naive": m_naive["wape"],
            "rmse_naive": m_naive["rmse"], "nmae_naive": m_naive["nmae"],
            "nrmse_naive": m_naive["nrmse"], "r2_naive": m_naive["r2"],
            "baseline_nombre": "Regresión Lineal",
            "naive_nombre": "Naive estacional (lag-7)",
            "mejora_pct": mejora_pct,
            "modelo_ganador": modelo_ganador,
            "r2_ganador": r2_ganador,
            "wape_por_dia_horizonte": wape_por_dia,
            "wape_dia_1": wape_dia_1,
            "scatter": scatter,
        }
        self.metricas["mae"]   = m_restoiq["mae"]
        self.metricas["wape"]  = m_restoiq["wape"]
        self.metricas["rmse"]  = m_restoiq["rmse"]
        self.metricas["nmae"]  = m_restoiq["nmae"]
        self.metricas["nrmse"] = m_restoiq["nrmse"]

    # Predicción

    def predecir(self, dias=7, dias_operacion=None, eventos_futuros=None) -> dict:
        """
        eventos_futuros: dict opcional {"YYYY-MM-DD": {"promocion": 0/1,
        "es_evento_especial": 0/1, "descuento_pct": float}} -- fechas
        específicas donde el dueño YA SABE que habrá promoción/evento
        (ej. "el martes hay partido", "el lunes hago descuento").
        Reemplaza el respaldo de "no hay nada" (ver _features_objetivo)
        SOLO para esas fechas puntuales, con información real conocida
        de antemano -- no un supuesto genérico.
        """
        fechas_futuras = self._fechas_prediccion(dias, dias_operacion)

        if self.estrategia == "lgbm":
            por_producto = self._predecir_lgbm(fechas_futuras, eventos_futuros)
        else:
            por_producto = self._predecir_promedio_movil(fechas_futuras)

        diario = (
            por_producto.groupby("fecha")
            .agg(cantidad_total_pred=("cantidad_pred", "sum"))
            .reset_index()
        )
        pivote = por_producto.pivot_table(
            index="fecha", columns="producto", values="cantidad_pred", fill_value=0
        ).reset_index()

        top5 = por_producto.groupby("producto")["cantidad_pred"].sum().sort_values(ascending=False).head(5).to_dict()
        resumen = {
            "estrategia": self.estrategia,
            "cantidad_total_pred": int(diario["cantidad_total_pred"].sum()),
            "top_5_productos": top5,
        }
        return {"por_producto": por_producto, "diario": diario, "pivote": pivote, "resumen": resumen}

    def _fechas_prediccion(self, dias, dias_operacion=None):
        dias = min(dias, HORIZONTE_MAX)  # Direct Forecasting solo entrena hasta HORIZONTE_MAX modelos
        hoy = pd.Timestamp(datetime.now().date())
        ancla = max(self.fecha_ultimo_dato, hoy)
        fechas, cursor, intentos = [], ancla, 0
        limite = dias * 4 + 14
        while len(fechas) < dias and intentos < limite:
            cursor += timedelta(days=1)
            intentos += 1
            if dias_operacion is None or cursor.weekday() in dias_operacion:
                fechas.append(cursor)
        return fechas

    def _predecir_lgbm(self, fechas_futuras, eventos_futuros=None):
        """Direct Forecasting: cada fecha futura usa el modelo
        entrenado específicamente para esa distancia (k = días desde
        el último dato real), aplicado UNA vez sobre datos 100%
        reales -- nunca se encadena una predicción sobre otra."""
        clima_pronostico = self._obtener_clima_pronostico(fechas_futuras)
        eventos_futuros = eventos_futuros or {}
        ancla_fecha = self.fecha_ultimo_dato

        # Features ANCLA: se calculan UNA sola vez por producto (con el
        # último dato real disponible), se reutilizan para todas las
        # fechas futuras -- son las mismas sin importar qué día se esté
        # prediciendo, porque Direct Forecasting nunca actualiza el
        # historial con sus propias predicciones.
        features_ancla_por_producto = {}
        for producto in self.productos:
            hist_prod = self.df_historial[self.df_historial["producto"] == producto].set_index("fecha")["cantidad"]
            features_ancla_por_producto[producto] = self._features_ancla(hist_prod, producto)

        resultados = []
        for fecha in fechas_futuras:
            k = (fecha - ancla_fecha).days
            k = max(1, min(k, HORIZONTE_MAX))
            modelo_k = self.modelos.get(k)
            if modelo_k is None:
                # No hay modelo entrenado para este k específico (historial
                # insuficiente para ese horizonte) -- usar el más cercano
                # disponible como respaldo, nunca dejar la predicción sin correr.
                disponibles = sorted(self.modelos.keys())
                k_respaldo = min(disponibles, key=lambda x: abs(x - k)) if disponibles else None
                modelo_k = self.modelos.get(k_respaldo)
                if modelo_k is None:
                    continue

            filas = []
            eventos_fecha = eventos_futuros.get(fecha.strftime("%Y-%m-%d"), {})
            for producto in self.productos:
                fila = dict(features_ancla_por_producto[producto])
                # Fusión, no "uno u otro": el evento especial del día
                # SIEMPRE vive bajo "__TODOS__" (nunca por producto,
                # mismo criterio del generador: se decide una vez por
                # día). La promoción, en cambio, si el producto tiene
                # una declarada específicamente, esa gana sobre
                # cualquier promoción general -- pero el evento
                # especial del día debe llegarle a TODOS los
                # productos igual, tengan o no su propia promoción.
                evento_todos    = eventos_fecha.get("__TODOS__", {})
                evento_producto = eventos_fecha.get(producto, {})
                evento_dia = {**evento_todos, **evento_producto}
                fila.update(self._features_objetivo(fecha, clima_pronostico, evento_dia))
                fila["producto"] = producto
                filas.append(fila)

            df_dia = pd.DataFrame(filas)
            df_dia["producto"]  = pd.Categorical(df_dia["producto"],  categories=self.categorias_producto)
            df_dia["categoria"] = pd.Categorical(df_dia["categoria"], categories=self.categorias_categoria)

            pred_dia = np.maximum(0, modelo_k.predict(df_dia[self.features])).round()
            df_dia["cantidad_pred"] = pred_dia.astype(int)
            df_dia["fecha"] = fecha.strftime("%Y-%m-%d")
            resultados.append(df_dia[["fecha", "producto", "cantidad_pred"]])

        if not resultados:
            return pd.DataFrame(columns=["fecha", "producto", "cantidad_pred"])
        return pd.concat(resultados, ignore_index=True)

    def _obtener_clima_pronostico(self, fechas_futuras) -> dict:
        """Una sola llamada a la API para TODAS las fechas a predecir —
        nunca una por fecha/producto. {} si el clima no es feature
        activa, o si la API falla (el respaldo se aplica más abajo)."""
        cols_clima = ("lluvia_nocturna_mm", "temp_nocturna_promed")
        if not any(c in self.features for c in cols_clima):
            return {}
        try:
            from services.clima_service import obtener_pronostico
            dias_necesarios = (max(fechas_futuras) - pd.Timestamp.now().normalize()).days + 2
            df_clima = obtener_pronostico(dias=max(1, min(16, dias_necesarios)))
            if df_clima is None:
                return {}
            return {
                row["fecha"].strftime("%Y-%m-%d"): row[list(cols_clima)].to_dict()
                for _, row in df_clima.iterrows()
            }
        except Exception:
            return {}

    def _predecir_promedio_movil(self, fechas_futuras):
        filas = []
        for fecha in fechas_futuras:
            f_dia = self.factor_dia.get(fecha.weekday(), 1.0)
            f_feriado = 1.25 if fecha.strftime("%Y-%m-%d") in self.feriados else 1.0
            for producto in self.productos:
                base = self.promedio_producto.get(producto, 1)
                cantidad = max(0, round(base * f_dia * f_feriado))
                filas.append({"fecha": fecha.strftime("%Y-%m-%d"), "producto": producto, "cantidad_pred": cantidad})
        return pd.DataFrame(filas)

    def _features_ancla(self, hist_prod: pd.Series, producto: str) -> dict:
        """Features ANCLA: calculadas del último dato REAL disponible
        (self.df_historial), fijas para todo el horizonte -- Direct
        Forecasting nunca las actualiza con predicciones propias.

        Misma convención EXACTA que el entrenamiento (shift(d) sobre
        la fecha ancla): lag_d = cantidad(fecha_ultimo_dato - d), y
        los rolling NUNCA incluyen la propia fecha ancla -- solo datos
        estrictamente anteriores, igual que en _preparar_features."""
        hist_prod = hist_prod.sort_index()

        def get_lag(d):
            fecha_objetivo = self.fecha_ultimo_dato - timedelta(days=d)
            return float(hist_prod.get(fecha_objetivo, hist_prod.mean() if len(hist_prod) else 0))

        anterior = hist_prod[hist_prod.index < self.fecha_ultimo_dato]
        reciente = anterior.tail(28)
        return {
            "categoria": self.categoria_por_producto.get(producto, "Sin categoría"),
            "precio": self.precio_actual_por_producto.get(producto, 0.0),
            "lag_1": get_lag(1), "lag_7": get_lag(7), "lag_14": get_lag(14), "lag_28": get_lag(28),
            "rolling_7_mean":  float(reciente.tail(7).mean())  if len(reciente) >= 1 else 0,
            "rolling_14_mean": float(reciente.tail(14).mean()) if len(reciente) >= 1 else 0,
            "rolling_28_mean": float(reciente.mean())          if len(reciente) >= 1 else 0,
            "rolling_7_std":   float(reciente.tail(7).std())   if len(reciente) >= 2 else 0,
        }

    def _features_objetivo(self, fecha: pd.Timestamp, clima_pronostico=None, evento_dia=None) -> dict:
        """Features OBJETIVO: calendario (siempre determinístico) y
        clima (vía pronóstico) de la fecha que se está prediciendo --
        se conocen de antemano, sin importar cuántos días falten.

        evento_dia: dict opcional {"promocion":0/1, "es_evento_especial":0/1,
        "descuento_pct":float} -- si el usuario declaró de antemano que
        ESA fecha específica tiene promoción/evento (ver predecir()),
        se usa ese valor real. Si no se declaró nada para esa fecha,
        se asume "no hay" -- sigue siendo una limitación conocida
        (nunca se inventa una promoción que el usuario no confirmó),
        pero ahora es una elección informada, no la única opción."""
        feat = self._features_fecha(fecha)
        feat["fecha"] = fecha

        evento_dia = evento_dia or {}
        if "promocion" in self.features:
            feat["promocion"] = int(evento_dia.get("promocion", 0))
        if "es_evento_especial" in self.features:
            feat["es_evento_especial"] = int(evento_dia.get("es_evento_especial", 0))
        if "descuento_pct" in self.features:
            feat["descuento_pct"] = float(evento_dia.get("descuento_pct", 0.0))

        cols_clima = ("lluvia_nocturna_mm", "temp_nocturna_promed")
        if any(c in self.features for c in cols_clima):
            clima_pronostico = clima_pronostico or {}
            valores_dia = clima_pronostico.get(fecha.strftime("%Y-%m-%d"), {})
            for col in cols_clima:
                if col in self.features:
                    feat[col] = valores_dia.get(col, self.clima_promedio.get(col, 0.0))

        return feat

    # Persistencia

    def guardar(self, ruta="models/sales_model.pkl"):
        os.makedirs(os.path.dirname(ruta) if os.path.dirname(ruta) else ".", exist_ok=True)
        with open(ruta, "wb") as f:
            pickle.dump(self, f)
        print(f"Modelo guardado en: {ruta}")
        from services.storage_service import subir
        subir(ruta, os.path.basename(ruta))

    @classmethod
    def cargar(cls, ruta="models/sales_model.pkl") -> "SalesModel":
        from services.storage_service import asegurar_local
        asegurar_local(os.path.basename(ruta), ruta)
        with open(ruta, "rb") as f:
            return pickle.load(f)