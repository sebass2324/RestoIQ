import pandas as pd

df = pd.read_csv("data/train_restaurante.csv")
df["fecha"] = pd.to_datetime(df["fecha"])
fechas_unicas = df["fecha"].dt.date.nunique()
rango = (df["fecha"].max() - df["fecha"].min()).days + 1

print(f"Filas totales: {len(df)}")
print(f"Fechas únicas en el CSV crudo: {fechas_unicas}")
print(f"Rango de calendario: {rango}")
print(f"¿Coinciden? {'SÍ' if fechas_unicas == rango else f'NO — faltan {rango - fechas_unicas} en el CSV mismo'}")