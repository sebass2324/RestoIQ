"""
services/insumos_service.py

Convierte el resultado de ejecutar_prediccion() (cantidad_pred por
producto y día) en previsión de insumos: cuánto de cada ingrediente
hace falta, sumando sobre todos los productos que lo usan.

necesidad_insumo = SUMA( cantidad_pred_producto x cantidad_por_unidad_receta )

Las cantidades de receta son estimaciones (ver models/receta_insumo.py)
-- este módulo nunca las presenta como medición real, siempre marca el
resultado como estimado en el dict que devuelve, para que la UI lo
etiquete correctamente.
"""

from models.receta_insumo import RecetaInsumo


def tiene_recetas_cargadas(user_id: int) -> bool:
    return RecetaInsumo.query.filter_by(user_id=user_id).first() is not None


def calcular_prevision_insumos(user_id: int, por_producto: list) -> dict:
    """
    por_producto: la lista "por_producto" que ya devuelve
    ejecutar_prediccion() -- [{"fecha":..., "producto":..., "cantidad_pred":...}, ...]

    Devuelve: {
        "es_estimado": True,  -- SIEMPRE True hoy; ver docstring del módulo
        "por_ingrediente": [{"ingrediente":..., "unidad":..., "cantidad_necesaria":...,
                              "productos": [...]}, ...],  ordenado de mayor a menor
        "productos_sin_receta": [...],  -- para avisar qué falta cargar
    }
    """
    recetas = RecetaInsumo.query.filter_by(user_id=user_id).all()
    if not recetas:
        return {
            "es_estimado": True,
            "por_ingrediente": [],
            "productos_sin_receta": sorted({f["producto"] for f in por_producto}),
        }

    # producto -> [(ingrediente, cantidad_por_unidad, unidad), ...]
    receta_por_producto = {}
    for r in recetas:
        receta_por_producto.setdefault(r.producto, []).append(
            (r.ingrediente, r.cantidad_por_unidad, r.unidad)
        )

    total_por_producto = {}
    for fila in por_producto:
        total_por_producto[fila["producto"]] = (
            total_por_producto.get(fila["producto"], 0) + fila["cantidad_pred"]
        )

    acumulado = {}  # ingrediente -> {"unidad":..., "cantidad":..., "productos": set()}
    productos_sin_receta = []

    for producto, cantidad_total in total_por_producto.items():
        receta = receta_por_producto.get(producto)
        if not receta:
            productos_sin_receta.append(producto)
            continue
        for ingrediente, cantidad_por_unidad, unidad in receta:
            entrada = acumulado.setdefault(ingrediente, {"unidad": unidad, "cantidad": 0.0, "productos": set()})
            entrada["cantidad"] += cantidad_total * cantidad_por_unidad
            entrada["productos"].add(producto)

    por_ingrediente = sorted(
        [
            {
                "ingrediente": ing,
                "unidad": datos["unidad"],
                "cantidad_necesaria": round(datos["cantidad"], 1),
                "productos": sorted(datos["productos"]),
            }
            for ing, datos in acumulado.items()
        ],
        key=lambda x: x["cantidad_necesaria"],
        reverse=True,
    )

    return {
        "es_estimado": True,  # las cantidades de receta son estimadas -- ver docstring
        "por_ingrediente": por_ingrediente,
        "productos_sin_receta": sorted(set(productos_sin_receta)),
    }
