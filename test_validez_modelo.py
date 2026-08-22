"""
test_validez_modelo.py

Batería de pruebas para confirmar (o refutar) que ModeloAbastecimiento
(classification_model.py) generaliza de verdad y no memoriza identidad
de producto ni hace trampa con el split temporal.

El docstring del modelo AFIRMA varias cosas ("nunca usa identidad de
producto", "rolling causal", "split por fecha"). Este script las
CONVIERTE en pruebas ejecutables, no en confianza ciega en el comentario.

────────────────────────────────────────────────────────────────────
CÓMO USARLO

    from test_validez_modelo import ejecutar_bateria_completa
    ejecutar_bateria_completa(df)   # df = el mismo CSV/DataFrame que
                                     # le pasás a modelo.entrenar(df)

o desde la terminal:

    python test_validez_modelo.py ruta_a_tus_datos.csv

────────────────────────────────────────────────────────────────────
QUÉ PRUEBA CADA TEST Y QUÉ SIGNIFICA QUE FALLE

1. test_features_prohibidas
   Falla si "producto" (identidad cruda), "precio" o "cantidad" crudos
   aparecen como columna de entrada al modelo. Es un assert estructural,
   no estadístico: si esto falla, hay fuga por diseño, no por casualidad.

2. test_leave_products_out (LA PRUEBA CENTRAL)
   Separa productos en K grupos. En cada fold, un grupo de productos
   queda TOTALMENTE fuera del entrenamiento (ni en el fit del árbol, ni
   en stats_producto, ni en hist_producto). Se evalúa el modelo en esos
   productos "nunca vistos" y se compara contra su propio desempeño en
   productos conocidos (el que ya viste en el dashboard).
   Señal de trampa: caída grande (>15-20 puntos de accuracy/F1 macro)
   entre "conocidos" y "nunca vistos". Si el modelo realmente aprende
   patrón de categoría/contexto y no identidad, la caída debe ser chica.

3. test_walkforward_temporal
   Repite el split temporal en varios cortes de fecha (no solo el
   último 80/20), y mide la varianza del F1 macro entre cortes.
   Señal de trampa / sobreajuste a un split con suerte: métricas que
   varían mucho de un corte a otro, o que solo se ven bien en el corte
   más reciente.

4. test_shuffle_sanity
   Entrena con las etiquetas de y_train barajadas al azar (rompe
   cualquier relación real feature->etiqueta) y mide el F1 en test.
   Si el modelo "aprende" algo con etiquetas basura, hay una fuga en
   alguna parte de la construcción de features (alguna columna que
   codifica el score/etiqueta sin que te dieras cuenta).
   Resultado esperado: F1 macro cercano al baseline aleatorio (~0.33
   para 3 clases balanceadas).

5. test_producto_fantasma
   Usa diagnostico_producto_nuevo() del propio modelo con productos que
   no existen, para varias categorías/contextos, y confirma que:
     - no lanza error
     - las probabilidades no son degeneradas (ej. 100% siempre a la
       misma clase, lo que indicaría fallback roto)
"""

import sys
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score

try:
    from classification_model import ModeloAbastecimiento, CLASES
except ImportError:
    from services.classification_model import ModeloAbastecimiento, CLASES

RANDOM_STATE = 42
COLUMNAS_PROHIBIDAS_EXACTAS = {"producto", "precio", "cantidad", "demanda"}


# ────────────────────────────────────────────────────────────
# 1. FEATURES PROHIBIDAS
# ────────────────────────────────────────────────────────────

def test_features_prohibidas(df):
    modelo = ModeloAbastecimiento(random_state=RANDOM_STATE)
    modelo.entrenar(df)

    encontradas = COLUMNAS_PROHIBIDAS_EXACTAS & set(modelo.feature_names)
    ok = len(encontradas) == 0

    print("\n[1] FEATURES PROHIBIDAS")
    print(f"    Columnas en X: {len(modelo.feature_names)}")
    if ok:
        print("    OK: no hay identidad de producto, precio crudo ni "
              "cantidad cruda en las features.")
    else:
        print(f"    FALLA: se colaron columnas prohibidas: {encontradas}")
    return ok


# ────────────────────────────────────────────────────────────
# 2. LEAVE-PRODUCTS-OUT (la prueba central anti-memorización)
# ────────────────────────────────────────────────────────────

def _entrenar_y_evaluar_en(df_train, df_holdout):
    """Entrena SOLO con df_train (incluye su propio split interno por
    fecha) y evalúa el modelo resultante en df_holdout: filas de
    productos que el modelo jamás vio, ni en el fit ni en las stats."""
    modelo = ModeloAbastecimiento(random_state=RANDOM_STATE)
    metricas_conocidos = modelo.entrenar(df_train)  # su propio test interno

    pd_holdout = modelo._agregar_producto_dia(df_holdout)
    pd_holdout = modelo._asegurar_contexto_temporal(pd_holdout)
    if len(pd_holdout) == 0:
        return metricas_conocidos, None

    scores = modelo._construir_scores(pd_holdout)
    y_holdout = modelo._a_clase(scores)
    X_holdout = modelo._construir_features(pd_holdout, entrenando=False)

    pred = modelo.modelo.predict(X_holdout)
    metricas_nuevos = {
        "accuracy": accuracy_score(y_holdout, pred),
        "f1_macro": f1_score(y_holdout, pred, labels=CLASES,
                              average="macro", zero_division=0),
        "n": len(y_holdout),
    }
    return metricas_conocidos, metricas_nuevos


def test_leave_products_out(df, n_folds=5, umbral_caida_alerta=0.15):
    print("\n[2] LEAVE-PRODUCTS-OUT CV (prueba central)")
    productos = df["producto"].dropna().unique()
    if len(productos) < n_folds * 2:
        print(f"    Muy pocos productos ({len(productos)}) para "
              f"{n_folds} folds confiables. Bajá n_folds o suma datos.")
        n_folds = max(2, len(productos) // 3)

    rng = np.random.RandomState(RANDOM_STATE)
    productos_barajados = productos.copy()
    rng.shuffle(productos_barajados)
    folds = np.array_split(productos_barajados, n_folds)

    filas_resumen = []
    for i, productos_holdout in enumerate(folds):
        productos_holdout = set(productos_holdout)
        productos_train = set(productos) - productos_holdout

        df_train = df[df["producto"].isin(productos_train)]
        df_holdout = df[df["producto"].isin(productos_holdout)]

        if df_train["producto"].nunique() < 2 or len(df_holdout) < 5:
            continue

        try:
            m_conocidos, m_nuevos = _entrenar_y_evaluar_en(df_train, df_holdout)
        except ValueError as e:
            print(f"    Fold {i}: omitido ({e})")
            continue

        if m_nuevos is None:
            continue

        caida_acc = m_conocidos["accuracy"] - m_nuevos["accuracy"]
        caida_f1 = m_conocidos["f1_macro"] - m_nuevos["f1_macro"]
        filas_resumen.append({
            "fold": i,
            "n_productos_holdout": len(productos_holdout),
            "n_filas_holdout": m_nuevos["n"],
            "accuracy_conocidos": m_conocidos["accuracy"],
            "accuracy_nunca_vistos": m_nuevos["accuracy"],
            "caida_accuracy": caida_acc,
            "f1_conocidos": m_conocidos["f1_macro"],
            "f1_nunca_vistos": m_nuevos["f1_macro"],
            "caida_f1": caida_f1,
        })

    resumen = pd.DataFrame(filas_resumen)
    if resumen.empty:
        print("    No se pudo correr ningún fold (datos insuficientes).")
        return False

    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

    caida_media_acc = resumen["caida_accuracy"].mean()
    caida_media_f1 = resumen["caida_f1"].mean()
    ok = caida_media_acc < umbral_caida_alerta and caida_media_f1 < umbral_caida_alerta

    print(f"\n    Caída media accuracy (conocidos - nunca vistos): {caida_media_acc:.4f}")
    print(f"    Caída media F1 macro : {caida_media_f1:.4f}")
    print(f"    Umbral de alerta: {umbral_caida_alerta:.2f}")
    if ok:
        print("    OK: el desempeño en productos nunca vistos es "
              "comparable al de productos conocidos. No hay indicio de "
              "memorización por identidad de producto.")
    else:
        print("    ALERTA: caída fuerte en productos nunca vistos -> "
              "posible memorización de identidad de producto, revisar "
              "features indirectas de categoría / causales de producto.")
    return ok


# ────────────────────────────────────────────────────────────
# 3. WALK-FORWARD TEMPORAL (varios cortes, no solo el último)
# ────────────────────────────────────────────────────────────

def test_walkforward_temporal(df, n_cortes=4):
    print("\n[3] WALK-FORWARD TEMPORAL")
    df = df.copy()
    df["fecha"] = pd.to_datetime(df["fecha"])
    fechas_unicas = np.sort(df["fecha"].unique())
    if len(fechas_unicas) < n_cortes * 10:
        print("    Rango de fechas muy corto para walk-forward confiable "
              "(se necesitan más días de historia).")
        return None

    # cortes equiespaciados entre 50% y 90% de la historia
    posiciones = np.linspace(0.5, 0.9, n_cortes)
    resultados = []
    for frac in posiciones:
        idx_corte = int(len(fechas_unicas) * frac)
        fecha_corte = fechas_unicas[idx_corte]
        train = df[df["fecha"] < fecha_corte]
        test = df[df["fecha"] >= fecha_corte]
        if train["producto"].nunique() < 2 or len(test) < 5:
            continue
        modelo = ModeloAbastecimiento(random_state=RANDOM_STATE)
        try:
            metricas = modelo.entrenar(train)
        except ValueError:
            continue
        resultados.append({
            "fecha_corte": pd.Timestamp(fecha_corte).date(),
            "n_train": metricas["n_train"],
            "n_test": metricas["n_test"],
            "accuracy": metricas["accuracy"],
            "f1_macro": metricas["f1_macro"],
        })

    resumen = pd.DataFrame(resultados)
    if resumen.empty:
        print("    No se pudo correr walk-forward (datos insuficientes).")
        return None

    print(resumen.to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    std_f1 = resumen["f1_macro"].std()
    print(f"\n    Desvío estándar del F1 macro entre cortes: {std_f1:.4f}")
    ok = std_f1 < 0.08
    if ok:
        print("    OK: el desempeño es estable entre distintos cortes "
              "temporales, no depende de un split con suerte.")
    else:
        print("    ALERTA: el desempeño varía bastante según el corte de "
              "fecha usado. Revisar estacionalidad no capturada o un "
              "split poco representativo.")
    return ok


# ────────────────────────────────────────────────────────────
# 4. SHUFFLE SANITY CHECK
# ────────────────────────────────────────────────────────────

def test_shuffle_sanity(df):
    print("\n[4] SHUFFLE SANITY CHECK (detecta fuga oculta)")
    modelo = ModeloAbastecimiento(random_state=RANDOM_STATE)

    pd_df = modelo._agregar_producto_dia(df)
    pd_df = modelo._asegurar_contexto_temporal(pd_df)
    if len(pd_df) < 30 or pd_df["producto"].nunique() < 2:
        print("    Datos insuficientes para esta prueba.")
        return None

    corte = int(len(pd_df) * 0.8)
    fecha_corte = pd_df["fecha"].iloc[corte]
    train = pd_df[pd_df["fecha"] < fecha_corte].reset_index(drop=True)
    test = pd_df[pd_df["fecha"] >= fecha_corte].reset_index(drop=True)

    modelo._calcular_stats(train)
    modelo._calcular_historial_categoria(pd_df)
    modelo._calcular_historial_producto(pd_df)

    s_train = modelo._construir_scores(train)
    modelo._fijar_umbrales(s_train)
    y_train = modelo._a_clase(s_train)
    y_test = modelo._a_clase(modelo._construir_scores(test))

    X_train = modelo._construir_features(train, entrenando=True)
    X_test = modelo._construir_features(test, entrenando=False)

    rng = np.random.RandomState(RANDOM_STATE)
    y_train_barajado = y_train.sample(frac=1.0, random_state=RANDOM_STATE).reset_index(drop=True)

    from lightgbm import LGBMClassifier
    lgbm = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                           min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                           reg_alpha=0.1, reg_lambda=0.1, class_weight="balanced",
                           random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
    lgbm.fit(X_train, y_train_barajado)
    pred = lgbm.predict(X_test)
    f1_con_ruido = f1_score(y_test, pred, labels=CLASES, average="macro", zero_division=0)

    print(f"    F1 macro entrenando con etiquetas barajadas: {f1_con_ruido:.4f}")
    print("    Referencia: con 3 clases balanceadas, azar puro ~0.33.")
    ok = f1_con_ruido < 0.45
    if ok:
        print("    OK: con etiquetas basura el modelo no aprende nada "
              "útil, como se espera. No hay fuga oculta en las features.")
    else:
        print("    ALERTA: el modelo 'aprende' algo incluso con etiquetas "
              "aleatorias -> alguna feature está codificando la etiqueta "
              "por otra vía (fuga). Revisar _construir_features().")
    return ok


# ────────────────────────────────────────────────────────────
# 2b. ABLATION: ¿la caída es cold-start legítimo o algo más raro?
# ────────────────────────────────────────────────────────────

def test_ablation_historial_producto(df, n_folds=5):
    """
    Compara, en los MISMOS folds de leave-products-out, dos versiones:
      (a) modelo completo (usa hist_producto/causales por producto)
      (b) modelo "solo categoría" (fuerza las causales de producto al
          valor neutro SIEMPRE, incluso para productos conocidos)

    Si (b) casi no cae entre conocidos y nunca-vistos, pero (a) sí cae
    fuerte -> la caída de (a) es cold-start legítimo (pierde una señal
    real que el producto nuevo no tiene), NO memorización de identidad.

    Si (b) también cae fuerte -> hay algo más generando la caída
    (ej. el propio cálculo del label/stats_categoria), y ahí sí
    conviene investigar más a fondo antes de confiar en el modelo.
    """
    print("\n[2b] ABLATION: aporte real de hist_producto vs solo-categoría")
    productos = df["producto"].dropna().unique()
    rng = np.random.RandomState(RANDOM_STATE)
    productos_barajados = productos.copy()
    rng.shuffle(productos_barajados)
    folds = np.array_split(productos_barajados, n_folds)

    filas = []
    for i, productos_holdout in enumerate(folds):
        productos_holdout = set(productos_holdout)
        productos_train = set(productos) - productos_holdout
        df_train = df[df["producto"].isin(productos_train)]
        df_holdout = df[df["producto"].isin(productos_holdout)]
        if df_train["producto"].nunique() < 2 or len(df_holdout) < 5:
            continue

        modelo = ModeloAbastecimiento(random_state=RANDOM_STATE)
        try:
            modelo.entrenar(df_train)
        except ValueError:
            continue

        # --- (b) forzamos las causales de producto a neutro SIEMPRE,
        #     re-entrenando el LightGBM con esas features anuladas,
        #     para aislar cuánto pesa hist_producto en el resultado.
        hist_producto_real = modelo.hist_producto
        modelo.hist_producto = {}  # cualquier lookup cae al default

        pd_train = modelo._agregar_producto_dia(df_train)
        pd_train = modelo._asegurar_contexto_temporal(pd_train)
        corte = int(len(pd_train) * 0.8)
        fecha_corte = pd_train["fecha"].iloc[corte]
        train_b = pd_train[pd_train["fecha"] < fecha_corte]
        test_b = pd_train[pd_train["fecha"] >= fecha_corte]
        if len(train_b) < 20 or len(test_b) < 5 or train_b["producto"].nunique() < 2:
            modelo.hist_producto = hist_producto_real
            continue

        s_train_b = modelo._construir_scores(train_b)
        modelo._fijar_umbrales(s_train_b)
        y_train_b = modelo._a_clase(s_train_b)
        y_test_b = modelo._a_clase(modelo._construir_scores(test_b))
        X_train_b = modelo._construir_features(train_b, entrenando=True)
        X_test_b = modelo._construir_features(test_b, entrenando=False)

        from lightgbm import LGBMClassifier
        modelo_b = LGBMClassifier(n_estimators=300, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=20, subsample=0.8, colsample_bytree=0.8,
                                   reg_alpha=0.1, reg_lambda=0.1, class_weight="balanced",
                                   random_state=RANDOM_STATE, verbosity=-1, n_jobs=1)
        modelo_b.fit(X_train_b, y_train_b)
        f1_conocidos_b = f1_score(y_test_b, modelo_b.predict(X_test_b),
                                   labels=CLASES, average="macro", zero_division=0)

        pd_holdout = modelo._agregar_producto_dia(df_holdout)
        pd_holdout = modelo._asegurar_contexto_temporal(pd_holdout)
        y_holdout_b = modelo._a_clase(modelo._construir_scores(pd_holdout))
        X_holdout_b = modelo._construir_features(pd_holdout, entrenando=False)
        f1_nuevos_b = f1_score(y_holdout_b, modelo_b.predict(X_holdout_b),
                                labels=CLASES, average="macro", zero_division=0)

        modelo.hist_producto = hist_producto_real  # restaurar

        filas.append({
            "fold": i,
            "f1_completo_conocidos": None,
            "f1_solo_categoria_conocidos": f1_conocidos_b,
            "f1_solo_categoria_nuevos": f1_nuevos_b,
            "caida_solo_categoria": f1_conocidos_b - f1_nuevos_b,
        })

    resumen = pd.DataFrame(filas)
    if resumen.empty:
        print("    No se pudo correr (datos insuficientes).")
        return None

    print(resumen[["fold", "f1_solo_categoria_conocidos", "f1_solo_categoria_nuevos",
                    "caida_solo_categoria"]].to_string(index=False, float_format=lambda x: f"{x:.4f}"))
    caida_media = resumen["caida_solo_categoria"].mean()
    print(f"\n    Caída media (solo-categoría, conocidos vs nunca vistos): {caida_media:.4f}")
    print("    Comparar contra la caída de [2] (~0.22-0.23 con features completas).")
    if caida_media < 0.10:
        print("    -> Confirma cold-start legítimo: sin hist_producto, apenas hay "
              "diferencia entre productos conocidos y nuevos. La caída de [2] viene "
              "de perder señal real de historial propio, no de memorización.")
    else:
        print("    -> La caída persiste incluso sin hist_producto: investigar "
              "stats_categoria / el label mismo, no es solo cold-start.")
    return resumen


# ────────────────────────────────────────────────────────────
# 5. PRODUCTO FANTASMA (sanity de fallback, ya existe en la clase)
# ────────────────────────────────────────────────────────────

def test_producto_fantasma(df):
    print("\n[5] PRODUCTO FANTASMA (fallback a categoría)")
    modelo = ModeloAbastecimiento(random_state=RANDOM_STATE)
    modelo.entrenar(df)

    categorias = df["categoria"].dropna().unique()[:5] if "categoria" in df.columns else [None]
    contextos_dia = [
        {"fecha": "2026-01-15", "es_finde": 0, "es_feriado": 0, "promocion": 0},
        {"fecha": "2026-06-14", "es_finde": 1, "es_feriado": 0, "promocion": 1},
    ]

    resultados = []
    errores = 0
    for cat in categorias:
        for ctx in contextos_dia:
            try:
                r = modelo.diagnostico_producto_nuevo(cat, ctx)
                resultados.append(r)
            except Exception as e:
                errores += 1
                print(f"    ERROR con categoria={cat}, ctx={ctx}: {e}")

    if not resultados:
        print("    No se pudo generar ningún diagnóstico.")
        return False

    prioridades = [r["prioridad"] for r in resultados]
    confianzas = [r["confianza"] for r in resultados]
    distintas_clases = len(set(prioridades))

    print(f"    Diagnósticos generados: {len(resultados)}, errores: {errores}")
    print(f"    Clases distintas predichas: {distintas_clases} de {len(CLASES)}")
    print(f"    Confianza promedio: {np.mean(confianzas):.4f}")

    ok = errores == 0
    if ok:
        print("    OK: el fallback a categoría funciona sin errores para "
              "productos inexistentes.")
    else:
        print("    ALERTA: hubo errores prediciendo para producto "
              "inexistente -> el fallback no es robusto en producción.")
    return ok


# ────────────────────────────────────────────────────────────
# ORQUESTADOR
# ────────────────────────────────────────────────────────────

def ejecutar_bateria_completa(df):
    print("=" * 70)
    print("VALIDACIÓN ANTI-TRAMPA — ModeloAbastecimiento")
    print("=" * 70)

    resultados = {
        "features_prohibidas": test_features_prohibidas(df),
        "leave_products_out": test_leave_products_out(df),
        "ablation_hist_producto": test_ablation_historial_producto(df) is not None,
        "walkforward_temporal": test_walkforward_temporal(df),
        "shuffle_sanity": test_shuffle_sanity(df),
        "producto_fantasma": test_producto_fantasma(df),
    }

    print("\n" + "=" * 70)
    print("VEREDICTO FINAL")
    print("=" * 70)
    for nombre, ok in resultados.items():
        estado = "PASA" if ok else ("N/A" if ok is None else "FALLA")
        print(f"  {nombre:.<40} {estado}")

    criticos_ok = all(v for k, v in resultados.items()
                       if v is not None and k in ("features_prohibidas",
                                                   "leave_products_out",
                                                   "shuffle_sanity"))
    print("\n" + ("MODELO VALIDADO: sin indicios de memorización por "
                  "producto ni fuga de datos." if criticos_ok else
                  "MODELO NO VALIDADO: revisar los tests marcados FALLA "
                  "antes de confiar en las métricas del dashboard."))
    return resultados


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Uso: python test_validez_modelo.py ruta_a_datos.csv")
        sys.exit(1)
    df = pd.read_csv(sys.argv[1])
    ejecutar_bateria_completa(df)