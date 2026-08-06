import pandas as pd
from services.classification_model import ModeloAbastecimiento

df = pd.read_csv("data/el_chamo_burger.csv")
modelo = ModeloAbastecimiento()
resultado = modelo.entrenar(df)

print("accuracy:", resultado["accuracy"])
print("f1_macro:", resultado["f1_macro"])
print("precision_macro:", resultado["precision_macro"])
print("recall_macro:", resultado["recall_macro"])
print()
print("Ganador:", resultado["algoritmo_ganador"])
for algo, met in resultado["comparacion_algoritmos"].items():
    print(f"{algo}: accuracy={met['accuracy']}  f1_macro={met['f1_macro']}")
print()
print("por_clase:")
for c, met in resultado["por_clase"].items():
    print(f"  {c}: precision={met['precision']} recall={met['recall']} f1={met['f1']} soporte={met['soporte']}")
print()
print("matriz_confusion:", resultado["matriz_confusion"])