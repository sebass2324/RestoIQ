"""
diagnostico_dispersion_generador.py

Compara el ruido TEORICO que el generador inyecta (via la dispersion
de la distribucion binomial negativa) contra el CV (coeficiente de
variacion) REAL observado en el CSV ya generado -- para saber si el
generador esta metiendo mas ruido del que aparenta el dato final, o
si la variabilidad observada ya reflejaba fielmente la dispersion
configurada.

Ejecutar desde la raiz del proyecto (no necesita Flask ni base de
datos, solo el CSV):
    python diagnostico_dispersion_generador.py ruta/al/archivo.csv
"""

import sys
import numpy as np
import pandas as pd
from services.data_generator import PERFILES, DISPERSION_NB_BASE


def main():
    if len(sys.argv) < 2:
        print("Uso: python diagnostico_dispersion_generador.py ruta/al/csv")
        return
    ruta_csv = sys.argv[1]

    df = pd.read_csv(ruta_csv)

    perfil = PERFILES["elchamoburger"]
    productos_perfil = {nombre: demanda_base for nombre, _, demanda_base, _ in perfil["productos"]}
    demanda_base_promedio = np.mean(list(productos_perfil.values()))

    print(f"DISPERSION_NB_BASE = {DISPERSION_NB_BASE}")
    print(f"demanda_base_promedio (todo el perfil) = {demanda_base_promedio:.2f}\n")

    filas = []
    for producto, demanda_base in productos_perfil.items():
        dispersion = max(2.0, DISPERSION_NB_BASE * (demanda_base / demanda_base_promedio) ** 0.5)

        datos_producto = df[df["producto"] == producto]["cantidad"]
        if datos_producto.empty:
            continue
        media_real = datos_producto.mean()
        std_real = datos_producto.std()
        cv_real = std_real / media_real if media_real else None

        # CV teórico que predice la fórmula de dispersión NB, usando la
        # media REAL observada como aproximación de mu (ya incluye el
        # efecto agregado de día_semana/clima/quincena/etc., no solo NB puro)
        cv_teorico = np.sqrt(1 / media_real + 1 / dispersion) if media_real else None

        filas.append({
            "producto": producto,
            "demanda_base": demanda_base,
            "dispersion_usada": round(dispersion, 2),
            "media_real": round(media_real, 2),
            "cv_real": round(cv_real, 3) if cv_real else None,
            "cv_teorico_nb_solo": round(cv_teorico, 3) if cv_teorico else None,
        })

    tabla = pd.DataFrame(filas).sort_values("demanda_base")
    print(tabla.to_string(index=False))

    print("\n=== INTERPRETACIÓN ===")
    print("Si 'cv_real' es MUCHO MAYOR que 'cv_teorico_nb_solo': hay ruido extra")
    print("viniendo de otro lado (ej. la interacción de muchos multiplicadores),")
    print("no solo de la dispersión NB -- vale la pena revisar esos multiplicadores.")
    print("Si son parecidos o cv_real es MENOR: la dispersión NB está bien calibrada,")
    print("el ruido observado es principalmente el que se configuró a propósito.")


if __name__ == "__main__":
    main()
