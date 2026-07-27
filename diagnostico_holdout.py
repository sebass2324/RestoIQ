"""
Diagnóstico del bug "7 de 30 días en el holdout".
Correr desde la raíz del proyecto: python diagnostico_holdout.py <tu_user_id>
"""
import sys
sys.path.insert(0, ".")

from app import create_app  # ajusta si tu factory se llama distinto
from models.venta import Venta
from models.modelo_ml import ModeloML
from models.dataset_usuario import DatasetUsuario
import pandas as pd

app = create_app()

with app.app_context():
    user_id = int(sys.argv[1]) if len(sys.argv) > 1 else 1

    # 1) ¿El CSV/dataset en MySQL tiene huecos de calendario?
    ventas = Venta.query.filter_by(user_id=user_id).all()
    fechas = sorted(set(v.fecha for v in ventas))
    print(f"Filas en MySQL (tabla ventas): {len(ventas)}")
    print(f"Fechas únicas con AL MENOS una venta: {len(fechas)}")
    print(f"Rango: {fechas[0]} -> {fechas[-1]}")
    rango_calendario = (fechas[-1] - fechas[0]).days + 1
    print(f"Días de calendario en ese rango: {rango_calendario}")
    print(f"¿Coinciden? {'SÍ, sin huecos' if len(fechas) == rango_calendario else f'NO — faltan {rango_calendario - len(fechas)} días de calendario sin NINGUNA venta'}")

    # 2) ¿DatasetUsuario.filas coincide con lo que hay REALMENTE en ventas?
    #    Si NO coincide, el DELETE de un upload anterior no limpió del
    #    todo la tabla, y hay datos acumulados de más de una subida.
    dataset = DatasetUsuario.query.filter_by(user_id=user_id).first()
    if dataset:
        print(f"\nDatasetUsuario.filas (lo que el ÚLTIMO upload dice que subió): {dataset.filas}")
        print(f"Filas reales en tabla ventas ahora mismo: {len(ventas)}")
        if dataset.filas != len(ventas):
            print("🚨 NO COINCIDEN — hay datos acumulados de más de un upload (el DELETE no limpió todo).")
        else:
            print("✅ Coinciden — la tabla ventas solo tiene el último upload, no hay acumulación.")
        print(f"Nombre del último archivo subido: {dataset.nombre_archivo}")
        print(f"Fecha del último upload: {dataset.fecha_subida}")

    # 3) Distribución de filas por año — si hay "islas" separadas de
    #    años sueltos (ej. mucho en 2020-2021, casi nada 2022-2023,
    #    mucho de nuevo en 2024-2026), confirma que son datasets
    #    distintos pegados, no uno continuo.
    df_fechas = pd.DataFrame({"fecha": fechas})
    df_fechas["anio"] = pd.to_datetime(df_fechas["fecha"]).dt.year
    print("\nFechas únicas con venta, por año:")
    print(df_fechas.groupby("anio").size().to_string())

    # 5) ¿Cuáles son las fechas raras exactamente? Primeras/últimas 15,
    #    y los saltos (gaps) más grandes entre fechas consecutivas —
    #    esto debería mostrar 1 o 2 fechas sueltas muy alejadas del
    #    resto (outliers), no un patrón disperso.
    print("\nPrimeras 15 fechas (las más antiguas):")
    for f in fechas[:15]:
        print(" ", f)
    print("\nÚltimas 15 fechas (las más recientes):")
    for f in fechas[-15:]:
        print(" ", f)

    diffs = [(fechas[i+1] - fechas[i]).days for i in range(len(fechas) - 1)]
    saltos = sorted(zip(diffs, fechas, fechas[1:]), reverse=True)[:10]
    print("\nLos 10 saltos (gaps) más grandes entre fechas consecutivas:")
    for dias, f1, f2 in saltos:
        print(f"  {f1} -> {f2}  ({dias} días de salto)")
    registro = ModeloML.query.filter_by(user_id=user_id).first()
    if registro:
        print(f"\nModelo registrado: estrategia={registro.estrategia}, mae={registro.mae}, mape={registro.mape}")
        print(f"Entrenado: {registro.fecha_entrenamiento}")
    else:
        print("\nNo hay ModeloML registrado para este user_id.")