"""
cargar_recetas.py

Carga el Excel de recetas (Recetas_BOM) a la tabla recetas_insumos.
Corre UNA vez por usuario (o de nuevo si el Excel cambió -- hace
upsert, no duplica).

Uso:
    python cargar_recetas.py
"""

import pandas as pd
from app import create_app
from models import db
from models.receta_insumo import RecetaInsumo

USER_ID = 6  # <-- ajusta al user_id correcto
RUTA_EXCEL = "El_Chamo_Burger_Recetas_Insumos_v3.xlsx"  # <-- ajusta la ruta si hace falta


def main():
    df = pd.read_excel(RUTA_EXCEL, sheet_name="Recetas_BOM", skiprows=3)
    df.columns = ["categoria", "producto", "ingrediente", "cantidad", "unidad"]
    df = df.dropna(subset=["producto", "ingrediente"])

    app = create_app()
    with app.app_context():
        cargadas, actualizadas = 0, 0
        for _, fila in df.iterrows():
            existente = RecetaInsumo.query.filter_by(
                user_id=USER_ID, producto=fila["producto"], ingrediente=fila["ingrediente"]
            ).first()
            if existente:
                existente.cantidad_por_unidad = float(fila["cantidad"])
                existente.unidad = str(fila["unidad"])
                actualizadas += 1
            else:
                db.session.add(RecetaInsumo(
                    user_id=USER_ID, producto=fila["producto"], ingrediente=fila["ingrediente"],
                    cantidad_por_unidad=float(fila["cantidad"]), unidad=str(fila["unidad"]),
                ))
                cargadas += 1
        db.session.commit()
        print(f"Recetas nuevas: {cargadas}  |  Actualizadas: {actualizadas}")
        print(f"Productos con receta: {df['producto'].nunique()}")


if __name__ == "__main__":
    main()