"""
services/sales_model.py
RestoIQ — Modelo de predicción de demanda (LightGBM / promedio móvil)
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import lightgbm as lgb
from datetime import datetime, timedelta
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

# Configuración
UMBRAL_DIAS_ML = 90
MINIMO_FILAS_LGBM = 30

# ── Features temporales ──
# Calculadas de la fecha. dia_semana/dia_mes/mes/semana_anio/es_finde/
# es_puente/es_quincena siempre entran. es_feriado/vispera_feriado son
# opcionales (el negocio decide si le interesan), pero NUNCA dependen
# de una columna del archivo del usuario — se calculan internamente
# con la lista de feriados, no hace falta que el dueño tenga esa
# columna en su Excel.
FEATURES_TEMPORALES = [
    "dia_semana", "dia_mes", "mes", "semana_anio",
    "es_finde", "es_puente", "es_quincena",
]
FEATURES_TEMPORALES_OPCIONALES = {
    "es_feriado":      "considerar_feriados",
    "vispera_feriado": "considerar_feriados",
}

# ── Features históricas ──
# Lags y promedios móviles, calculados del propio historial de ventas
# del producto (ver _preparar_features).
FEATURES_HISTORICAS = [
    "lag_1", "lag_7", "lag_28",
    "rolling_7_mean", "rolling_14_mean", "rolling_28_mean", "rolling_7_std",
]

# ── Features del producto ──
# Identidad del producto. NO son "encoded" con LabelEncoder — son
# categóricas NATIVAS de pandas/LightGBM (dtype 'category'), el
# algoritmo las agrupa solo, sin necesidad de convertirlas a número
# a mano primero (ver _preparar_features).
FEATURES_PRODUCTO = ["producto", "categoria"]

# ── Features comerciales ──
# precio: se agrega si el archivo del usuario la trae, sin necesidad
# de configuración (no es una preferencia, es un dato que existe o no).
# promocion/descuento_pct/es_evento_especial: además de existir en el
# archivo, el negocio tiene que haber pedido considerarlas.
FEATURES_COMERCIALES_SI_EXISTEN = ["precio"]
FEATURES_COMERCIALES_OPCIONALES = {
    "promocion":          "considerar_promociones",
    "descuento_pct":      "considerar_descuentos",
    "es_evento_especial": "considerar_eventos",
}

# Variables categóricas "de bandera" — se le avisan a LightGBM
# explícitamente (no son cantidades continuas reales). producto y
# categoria NO están acá: ya son dtype 'category', LightGBM las
# detecta solas.
FEATURES_CATEGORICAS_FLAG = [
    "dia_semana", "dia_mes", "mes", "semana_anio", "es_finde", "es_puente", "es_quincena",
    "es_feriado", "vispera_feriado", "promocion", "es_evento_especial",
]

TARGET = "cantidad"


class SalesModel:

    def __init__(self):
        self.modelo               = None
        self.estrategia           = None
        self.productos            = []
        self.categorias_producto  = []
        self.categorias_categoria = []
        self.precio_promedio      = {}
        self.categoria_por_producto = {}
        self.df_historial         = None
        self.metricas             = {}
        self.fecha_ultimo_dato    = None
        self.feriados             = self._cargar_feriados()
        self.features              = FEATURES_TEMPORALES + FEATURES_HISTORICAS + FEATURES_PRODUCTO

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
        return {
            "dia_semana":     dia,
            "dia_mes":        dia_mes,
            "mes":            fecha.month,
            "semana_anio":    int(fecha.isocalendar()[1]),
            "es_finde":       int(dia in [5, 6]),
            "es_feriado":     int(fecha_str in self.feriados),
            "vispera_feriado": int(maniana in self.feriados),
            "es_puente":      int(ayer in self.feriados or maniana in self.feriados),
            "es_quincena":    int(1 <= dia_mes <= 7 or 15 <= dia_mes <= 21),
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

        piezas = []
        for producto, grupo in df.groupby("producto"):
            # calendario continuo por producto — evita que shift() se desalinee con huecos
            grupo = grupo.set_index("fecha").sort_index()
            rango = pd.date_range(grupo.index.min(), grupo.index.max(), freq="D")

            pieza = pd.DataFrame({"fecha": rango})
            pieza["producto"]  = producto
            pieza["categoria"] = self.categoria_por_producto.get(producto, "Sin categoría")
            pieza["precio"]    = self.precio_promedio.get(producto, 0.0)
            pieza["cantidad"]  = grupo["cantidad"].reindex(rango, fill_value=0).values

            for col in columnas_negocio:
                pieza[col] = grupo[col].reindex(rango, fill_value=0).values

            # variables de historial: lags
            for lag in (1, 7, 28):
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
        self.precio_promedio = (
            df.groupby("producto")["precio"].mean().to_dict() if "precio" in df.columns else {}
        )
        self.categoria_por_producto = (
            df.groupby("producto")["categoria"].first().to_dict() if "categoria" in df.columns else {}
        )
        self.categorias_categoria = sorted(set(self.categoria_por_producto.values())) or ["Sin categoría"]

        self.features = self._construir_features(df, config)

        # agregación diaria por producto
        agg_spec = {"cantidad": ("cantidad", "sum")}
        for col, fn in (("promocion", "max"), ("descuento_pct", "mean"), ("es_evento_especial", "max")):
            if col in df.columns:
                agg_spec[col] = (col, fn)

        df_agg = df.groupby(["fecha", "producto"]).agg(**agg_spec).reset_index()
        for col in ("promocion", "descuento_pct", "es_evento_especial"):
            if col in df_agg.columns:
                df_agg[col] = df_agg[col].fillna(0)

        dias_historial = (df["fecha"].max() - df["fecha"].min()).days
        df_feat = self._preparar_features(df_agg)
        self.df_historial = df_feat[["fecha", "producto", "cantidad"]].copy()

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

    def _entrenar_lgbm(self, df_feat, verbose):
        df_feat_temporal = df_feat.sort_values("fecha").reset_index(drop=True)
        X = df_feat_temporal[self.features]
        y = df_feat_temporal[TARGET]
        categoricas = self._categoricas_activas()

        parametros = {
            "objective": "regression_l1",
            "learning_rate": 0.05,
            "num_leaves": 63,
            "min_data_in_leaf": 20,
            "max_bin":255,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 1,
            "lambda_l1": 0.1,
            "lambda_l2": 0.1,
            "verbose": -1,
            "seed": 42,
        }

        # validación cruzada interna (MAE provisional)
        tscv = TimeSeriesSplit(n_splits=3)
        maes = []
        for train_idx, val_idx in tscv.split(X):
            train_set = lgb.Dataset(X.iloc[train_idx], label=y.iloc[train_idx], categorical_feature=categoricas)
            val_set   = lgb.Dataset(X.iloc[val_idx],   label=y.iloc[val_idx],   categorical_feature=categoricas, reference=train_set)
            m = lgb.train(parametros, train_set, num_boost_round=500, valid_sets=[val_set],
                          callbacks=[lgb.early_stopping(50, verbose=False)])
            preds = np.maximum(0, m.predict(X.iloc[val_idx], num_iteration=m.best_iteration))
            maes.append(mean_absolute_error(y.iloc[val_idx], preds))

        # búsqueda de n_arboles óptimo (early stopping)
        n_val = min(max(15, int(len(X) * 0.15)), len(X) // 3)
        corte_val = len(X) - n_val
        X_train_fs, X_val_fs = X.iloc[:corte_val], X.iloc[corte_val:]
        y_train_fs, y_val_fs = y.iloc[:corte_val], y.iloc[corte_val:]

        train_set_fs = lgb.Dataset(X_train_fs, label=y_train_fs, categorical_feature=categoricas)
        val_set_fs   = lgb.Dataset(X_val_fs, label=y_val_fs, categorical_feature=categoricas, reference=train_set_fs)
        modelo_busqueda = lgb.train(parametros, train_set_fs, num_boost_round=2000, valid_sets=[val_set_fs],
                                    callbacks=[lgb.early_stopping(50, verbose=False)])
        n_arboles_optimo = modelo_busqueda.best_iteration or 700

        # modelo final, 100% de los datos
        train_set_final = lgb.Dataset(X, label=y, categorical_feature=categoricas)
        self.modelo = lgb.train(parametros, train_set_final, num_boost_round=n_arboles_optimo)

        importancias = pd.Series(
            self.modelo.feature_importance(importance_type="gain"),
            index=self.modelo.feature_name(),
        ).sort_values(ascending=False)

        self.metricas = {
            "estrategia":        "LightGBM",
            "mae": None,
            "mae_provisional":  round(float(np.mean(maes)), 2),
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

    def _calcular_n_folds(self, dias_historial: int):
        if dias_historial < 60:   return None
        if dias_historial <= 120: return 2
        if dias_historial <= 250: return 3
        return 5

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

        while n_folds >= 2:
            if len(df_feat) // (n_folds + 1) >= MINIMO_FILAS_LGBM:
                break
            n_folds -= 1
        if n_folds < 2:
            self.metricas["holdout"] = None
            return

        df_feat_temporal = df_feat.sort_values("fecha").reset_index(drop=True)
        tscv = TimeSeriesSplit(n_splits=n_folds)
        categoricas = self._categoricas_activas()

        parametros_replica = {
            "objective": "regression_l1", "learning_rate": 0.05, "num_leaves": 63, "max_bin":255,
            "min_data_in_leaf": 20, "feature_fraction": 0.8, "bagging_fraction": 0.8,
            "bagging_freq": 1, "lambda_l1": 0.1, "lambda_l2": 0.1, "verbose": -1, "seed": 42,
        }

        y_reales_todos, pred_restoiq_todos, pred_baseline_todos, pred_naive_todos = [], [], [], []

        for idx_train, idx_test in tscv.split(df_feat_temporal):
            train_feat = df_feat_temporal.iloc[idx_train]
            test_feat  = df_feat_temporal.iloc[idx_test]
            X_train, y_train = train_feat[self.features], train_feat[TARGET]
            X_test,  y_test  = test_feat[self.features],  test_feat[TARGET]

            if self.estrategia == "lgbm":
                train_set = lgb.Dataset(X_train, label=y_train, categorical_feature=categoricas)
                modelo_replica = lgb.train(parametros_replica, train_set, num_boost_round=700)
                pred_restoiq = np.maximum(0, modelo_replica.predict(X_test))
            else:
                pred_restoiq = self._predecir_promedio_movil_holdout(train_feat, test_feat)

            # baseline: Regresión Lineal (solo columnas numéricas)
            cols_numericas = [c for c in self.features if c not in ("producto", "categoria")]
            baseline = LinearRegression()
            baseline.fit(X_train[cols_numericas], y_train)
            pred_baseline = np.maximum(0, baseline.predict(X_test[cols_numericas]))

            # naive: lo mismo que hace 7 días
            pred_naive = test_feat["lag_7"].values

            y_reales_todos.append(y_test.values)
            pred_restoiq_todos.append(pred_restoiq)
            pred_baseline_todos.append(pred_baseline)
            pred_naive_todos.append(pred_naive)

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
            "scatter": scatter,
        }
        self.metricas["mae"]   = m_restoiq["mae"]
        self.metricas["wape"]  = m_restoiq["wape"]
        self.metricas["rmse"]  = m_restoiq["rmse"]
        self.metricas["nmae"]  = m_restoiq["nmae"]
        self.metricas["nrmse"] = m_restoiq["nrmse"]

    # Predicción

    def predecir(self, dias=7, dias_operacion=None) -> dict:
        fechas_futuras = self._fechas_prediccion(dias, dias_operacion)

        if self.estrategia == "lgbm":
            por_producto = self._predecir_lgbm(fechas_futuras)
        else:
            por_producto = self._predecir_promedio_movil(fechas_futuras)

        por_producto["precio"] = por_producto["producto"].map(self.precio_promedio).fillna(0)
        por_producto["ingreso_pred"] = (por_producto["cantidad_pred"] * por_producto["precio"]).round(2)
        por_producto = por_producto.drop(columns=["precio"])

        diario = (
            por_producto.groupby("fecha")
            .agg(ingreso_total_pred=("ingreso_pred", "sum"), cantidad_total_pred=("cantidad_pred", "sum"))
            .reset_index()
        )
        pivote = por_producto.pivot_table(
            index="fecha", columns="producto", values="cantidad_pred", fill_value=0
        ).reset_index()

        top5 = por_producto.groupby("producto")["cantidad_pred"].sum().sort_values(ascending=False).head(5).to_dict()
        resumen = {
            "estrategia": self.estrategia,
            "ingreso_total_pred": round(float(diario["ingreso_total_pred"].sum()), 2),
            "cantidad_total_pred": int(diario["cantidad_total_pred"].sum()),
            "top_5_productos": top5,
        }
        return {"por_producto": por_producto, "diario": diario, "pivote": pivote, "resumen": resumen}

    def _fechas_prediccion(self, dias, dias_operacion=None):
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

    def _predecir_lgbm(self, fechas_futuras):
        filas = [self._features_para_fecha_producto(fecha, producto)
                 for fecha in fechas_futuras for producto in self.productos]
        df_pred = pd.DataFrame(filas)

        df_pred["producto"] = pd.Categorical(df_pred["producto"], categories=self.categorias_producto)
        df_pred["categoria"] = pd.Categorical(df_pred["categoria"], categories=self.categorias_categoria)

        cantidades = np.maximum(0, self.modelo.predict(df_pred[self.features])).round()
        df_pred["cantidad_pred"] = cantidades.astype(int)
        df_pred["fecha"] = df_pred["fecha"].dt.strftime("%Y-%m-%d")
        return df_pred[["fecha", "producto", "cantidad_pred"]]

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

    def _features_para_fecha_producto(self, fecha, producto):
        feat = self._features_fecha(fecha)
        hist_prod = self.df_historial[self.df_historial["producto"] == producto].set_index("fecha")["cantidad"]

        def get_lag(d):
            lf = fecha - timedelta(days=d)
            return float(hist_prod.get(lf, hist_prod.mean() if len(hist_prod) else 0))

        reciente = hist_prod.sort_index().tail(28)
        feat.update({
            "fecha": fecha, "producto": producto,
            "categoria": self.categoria_por_producto.get(producto, "Sin categoría"),
            "precio": self.precio_promedio.get(producto, 0.0),
            "lag_1": get_lag(1), "lag_7": get_lag(7), "lag_28": get_lag(28),
            "rolling_7_mean":  float(reciente.tail(7).mean())  if len(reciente) >= 1 else 0,
            "rolling_14_mean": float(reciente.tail(14).mean()) if len(reciente) >= 1 else 0,
            "rolling_28_mean": float(reciente.mean())          if len(reciente) >= 1 else 0,
            "rolling_7_std":   float(reciente.tail(7).std())   if len(reciente) >= 2 else 0,
        })
        for col in ("promocion", "es_evento_especial"):
            if col in self.features:
                feat[col] = 0
        if "descuento_pct" in self.features:
            feat["descuento_pct"] = 0.0
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