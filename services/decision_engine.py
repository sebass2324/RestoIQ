"""
services/decision_engine.py

Decision Engine — capa de inteligencia de NEGOCIO, no un tercer modelo
de Machine Learning. Interpreta las salidas ya calculadas por los dos
modelos existentes (Predicción de Demanda y Clasificación de
Prioridad de Abastecimiento) y las traduce en recomendaciones
operativas. Nunca entrena, nunca ejecuta ML — solo consume resultados.

    Predicción de Demanda  ─┐
                             ├──▶ Decision Engine ──▶ Insights
    Clasificación           ─┘

Organización:
    generar_insights_predictivos()    — SOLO regresión
    generar_insights_operativos()     — SOLO clasificación
    generar_insights_cruzados()       — cruza ambos modelos
    generar_insights_confiabilidad()  — cruza métricas de ambos
    priorizar_insights()              — ordena y recorta a 5-8
    calcular_resumen()                — panel lateral: conteos + acción urgente
    calcular_puntuacion_riesgo()      — score 0-100, reglas simples, no IA nueva

Cada insight trae:
    tipo, prioridad, polaridad ("riesgo"/"oportunidad"/"informativa"),
    icono, titulo, descripcion (qué pasó + por qué), accion (qué
    hacer), impacto (qué se gana si se actúa), evidencia.

No todo es alerta: hay reglas que devuelven insights de polaridad
"oportunidad" o "informativa" (ahorro posible, sin riesgos, alta
confianza) — el equilibrio es intencional, no todo debe sonar a peligro.
"""

from collections import defaultdict


_ORDEN_PRIORIDAD = {"Alta": 0, "Media": 1, "Baja": 2}

MAX_INSIGHTS = 8
MIN_INSIGHTS = 5
MIN_PRODUCTOS_PATRON = 2


# ════════════════════════════════════════════════════════════
# 1. GENERAR_INSIGHTS_PREDICTIVOS — SOLO regresión
# ════════════════════════════════════════════════════════════

def generar_insights_predictivos(datos_prediccion: dict) -> list:
    insights = []
    dia_critico = datos_prediccion.get("dia_critico")
    diario = datos_prediccion.get("diario", [])

    pct = (dia_critico or {}).get("pct_sobre_promedio")
    if pct is not None and pct >= 15:
        insights.append({
            "tipo": "Predictivo", "polaridad": "riesgo",
            "prioridad": "Alta" if pct >= 25 else "Media",
            "icono": "calendar-clock",
            "titulo": f"{dia_critico['fecha']} será tu día más exigente",
            "descripcion": (
                f"La demanda pronosticada para ese día está {pct:.0f}% por encima del "
                f"promedio histórico real de ese mismo día de la semana."
            ),
            "accion": "Reforzar personal e inventario general para esa fecha específica.",
            "impacto": "Evita quiebres de stock y sobrecarga de personal ese día puntual.",
            "evidencia": {"fecha": dia_critico["fecha"], "pct_sobre_promedio": pct,
                         "cantidad_total": dia_critico.get("cantidad")},
        })

    if diario and len(diario) >= 4:
        mitad = len(diario) // 2
        dias_primera, dias_segunda = diario[:mitad], diario[mitad:]
        primera_prom = sum(d["cantidad_total_pred"] for d in dias_primera) / len(dias_primera)
        segunda_prom = sum(d["cantidad_total_pred"] for d in dias_segunda) / len(dias_segunda)
        if primera_prom > 0:
            pct_tend = (segunda_prom - primera_prom) / primera_prom * 100
            if abs(pct_tend) >= 15:
                creciendo = pct_tend > 0
                insights.append({
                    "tipo": "Predictivo", "polaridad": "riesgo" if creciendo else "oportunidad",
                    "prioridad": "Media",
                    "icono": "trending-up" if creciendo else "trending-down",
                    "titulo": f"La demanda viene {'subiendo' if creciendo else 'bajando'} dentro del período",
                    "descripcion": (
                        f"El promedio diario de la segunda mitad del horizonte pronosticado es "
                        f"{abs(pct_tend):.0f}% {'más alto' if creciendo else 'más bajo'} que el de "
                        f"la primera mitad — comparando promedios, no totales, para que no influya "
                        f"que una mitad tenga un día más que la otra."
                    ),
                    "accion": (
                        "Planificar compras al alza para los próximos días." if creciendo
                        else "Podés reducir compras sin afectar ventas — la demanda viene en baja."
                    ),
                    "impacto": (
                        "Anticipar la compra evita quiebres de stock en la segunda mitad del período."
                        if creciendo else
                        "Menos capital inmovilizado en inventario que no vas a necesitar."
                    ),
                    "evidencia": {"pct_cambio": round(pct_tend, 1),
                                 "promedio_diario_primera_mitad": round(primera_prom, 1),
                                 "promedio_diario_segunda_mitad": round(segunda_prom, 1)},
                })

    return insights


# ════════════════════════════════════════════════════════════
# 2. GENERAR_INSIGHTS_OPERATIVOS — SOLO clasificación
# ════════════════════════════════════════════════════════════

def generar_insights_operativos(datos_clasificacion: dict, categoria_excluir: str = None) -> list:
    insights = []
    dias = datos_clasificacion.get("dias", [])
    if not dias:
        return insights
    dia_hoy = dias[0]

    # Concentración de prioridad Alta por categoría (riesgo)
    grupos_alta = defaultdict(list)
    conteo_categoria = defaultdict(lambda: {"total": 0, "alta": 0})
    for p in dia_hoy.get("productos", []):
        cat = p.get("categoria") or "Sin categoría"
        conteo_categoria[cat]["total"] += 1
        if p["prioridad"] == "Alta":
            conteo_categoria[cat]["alta"] += 1
            grupos_alta[cat].append(p["producto"])
    grupos_alta.pop(categoria_excluir, None)

    if grupos_alta:
        categoria, productos = max(grupos_alta.items(), key=lambda kv: len(kv[1]))
        if len(productos) >= MIN_PRODUCTOS_PATRON:
            insights.append({
                "tipo": "Operativo", "polaridad": "riesgo",
                "prioridad": "Alta",
                "icono": "package-x",
                "titulo": f"Riesgo de desabastecimiento en {categoria}",
                "descripcion": (
                    f"{len(productos)} productos de {categoria} están marcados como prioridad "
                    f"Alta hoy por el clasificador — considerando rotación, volatilidad y el "
                    f"contexto del día."
                ),
                "accion": f"Verificar existencias de {categoria} hoy mismo antes de que falte stock.",
                "impacto": f"Evita quiebre de stock en {categoria}, tu categoría más expuesta hoy.",
                "evidencia": {"categoria": categoria, "productos": productos[:6]},
            })

    # Equilibrio: categoría SIN ningún producto de prioridad Alta (buena noticia)
    sin_riesgo = [c for c, d in conteo_categoria.items()
                  if d["alta"] == 0 and d["total"] >= MIN_PRODUCTOS_PATRON and c != categoria_excluir]
    if sin_riesgo:
        categoria_ok = max(sin_riesgo, key=lambda c: conteo_categoria[c]["total"])
        insights.append({
            "tipo": "Operativo", "polaridad": "informativa",
            "prioridad": "Baja",
            "icono": "shield-check",
            "titulo": f"Sin riesgos detectados en {categoria_ok}",
            "descripcion": f"Ningún producto de {categoria_ok} está en prioridad Alta hoy — comportamiento estable, sin señales de alerta.",
            "accion": "No se requiere ninguna acción especial en esta categoría.",
            "impacto": "Podés enfocar tu atención en las categorías que sí lo necesitan.",
            "evidencia": {"categoria": categoria_ok},
        })

    # Concentración de alertas en fin de semana / feriado
    total_alta = en_finde = 0
    for dia in dias:
        for p in dia["productos"]:
            if p["prioridad"] != "Alta":
                continue
            total_alta += 1
            if "Fin de semana" in p.get("factores", []) or "Feriado" in p.get("factores", []):
                en_finde += 1
    if total_alta >= MIN_PRODUCTOS_PATRON:
        pct = en_finde / total_alta * 100
        if pct >= 60:
            insights.append({
                "tipo": "Operativo", "polaridad": "riesgo",
                "prioridad": "Media",
                "icono": "users",
                "titulo": "Las alertas de prioridad se concentran en fin de semana / feriados",
                "descripcion": (
                    f"{pct:.0f}% de las alertas de prioridad Alta del período caen en fin de "
                    f"semana o feriado — patrón de contexto detectado por el clasificador."
                ),
                "accion": "Planificar personal con foco en esos días, no distribuir el esfuerzo parejo toda la semana.",
                "impacto": "Mejor cobertura de personal justo cuando más se necesita.",
                "evidencia": {"pct_en_finde_feriado": round(pct, 1), "total_alertas_altas": total_alta},
            })

    return insights


# ════════════════════════════════════════════════════════════
# 3. GENERAR_INSIGHTS_CRUZADOS — combinan ambos modelos
# ════════════════════════════════════════════════════════════

def _cruzar_senales(niveles_producto: dict, dia_clasificacion: dict) -> dict:
    cruce = {}
    for producto, info in (niveles_producto or {}).items():
        cruce[producto] = {"nivel_demanda": info.get("nivel"), "prioridad": None,
                           "confianza": None, "categoria": None, "factores": []}
    for p in (dia_clasificacion or {}).get("productos", []):
        if p["producto"] in cruce:
            cruce[p["producto"]].update({
                "prioridad": p.get("prioridad"), "confianza": p.get("confianza"),
                "categoria": p.get("categoria"), "factores": p.get("factores", []),
            })
    return {k: v for k, v in cruce.items() if v["prioridad"] is not None}


def _agrupar(cruce: dict, nivel_demanda: str, prioridad: str) -> dict:
    grupos = defaultdict(list)
    for producto, info in cruce.items():
        if info["nivel_demanda"] == nivel_demanda and info["prioridad"] == prioridad:
            grupos[info["categoria"] or "Sin categoría"].append(producto)
    return dict(grupos)


def generar_insights_cruzados(datos_prediccion: dict, datos_clasificacion: dict):
    """Retorna (insights, categoria_cubierta)."""
    niveles_producto = datos_prediccion.get("niveles_producto", {})
    dias = datos_clasificacion.get("dias", [])
    dia_hoy = dias[0] if dias else None
    cruce = _cruzar_senales(niveles_producto, dia_hoy)

    insights = []
    categoria_cubierta = None

    # ALTO + Alta → reforzar inventario (riesgo)
    grupos = _agrupar(cruce, "ALTO", "Alta")
    if grupos:
        categoria, productos = max(grupos.items(), key=lambda kv: len(kv[1]))
        if len(productos) >= MIN_PRODUCTOS_PATRON:
            categoria_cubierta = categoria
            insights.append({
                "tipo": "Cruzado", "polaridad": "riesgo",
                "prioridad": "Alta",
                "icono": "trending-up",
                "titulo": f"Reforzar inventario de {categoria}",
                "descripcion": (
                    f"Se detectó alta demanda prevista (regresión) Y alta prioridad de "
                    f"abastecimiento (clasificación) en {len(productos)} productos de "
                    f"{categoria} — dos modelos independientes coinciden."
                ),
                "accion": "Incrementar el stock antes del inicio del turno.",
                "impacto": f"Reduce probabilidad de quiebre de stock en {categoria}, tu señal más fuerte hoy.",
                "evidencia": {"categoria": categoria, "productos": productos[:6]},
            })

    # ALTO + Baja → mantener operación habitual (informativa)
    grupos = _agrupar(cruce, "ALTO", "Baja")
    if grupos:
        categoria, productos = max(grupos.items(), key=lambda kv: len(kv[1]))
        if len(productos) >= MIN_PRODUCTOS_PATRON:
            insights.append({
                "tipo": "Cruzado", "polaridad": "informativa",
                "prioridad": "Baja",
                "icono": "check-circle",
                "titulo": f"{categoria}: demanda alta pero estable",
                "descripcion": (
                    f"{len(productos)} productos de {categoria} venden por encima de su "
                    f"promedio, pero el clasificador no los marca como prioritarios — "
                    f"comportamiento predecible, sin picos de volatilidad."
                ),
                "accion": "Mantener el abastecimiento habitual, sin intervención extra.",
                "impacto": "Libera tiempo de gestión — esta categoría no necesita atención especial.",
                "evidencia": {"categoria": categoria, "productos": productos[:6]},
            })

    # BAJO + Alta → poco vendido pero inestable, revisar (riesgo)
    grupos = _agrupar(cruce, "BAJO", "Alta")
    if grupos:
        categoria, productos = max(grupos.items(), key=lambda kv: len(kv[1]))
        insights.append({
            "tipo": "Cruzado", "polaridad": "riesgo",
            "prioridad": "Media",
            "icono": "eye",
            "titulo": f"Vigilar {categoria}: bajo volumen, alta sensibilidad",
            "descripcion": (
                f"{len(productos)} producto(s) de {categoria} tienen poco volumen, pero el "
                f"clasificador los marca como prioridad Alta — indica volatilidad o fuerte "
                f"reacción a promociones/fin de semana."
            ),
            "accion": "No descartarlos por bajo volumen — revisar su historial de picos.",
            "impacto": "Evita subestimar productos que parecen menores pero pueden agotarse rápido.",
            "evidencia": {"categoria": categoria, "productos": productos[:6]},
        })

    # BAJO + Baja → optimizar recursos (oportunidad)
    grupos = _agrupar(cruce, "BAJO", "Baja")
    total = sum(len(v) for v in grupos.values())
    if total >= 3:
        categoria, productos = max(grupos.items(), key=lambda kv: len(kv[1]))
        insights.append({
            "tipo": "Cruzado", "polaridad": "oportunidad",
            "prioridad": "Baja",
            "icono": "trending-down",
            "titulo": f"Oportunidad de ahorro en {categoria}",
            "descripcion": (
                f"{total} productos en total tienen demanda baja Y prioridad baja de forma "
                f"consistente entre ambos modelos."
            ),
            "accion": "Reducir la preparación anticipada de estos productos para bajar desperdicio.",
            "impacto": "Menos desperdicio y menos capital inmovilizado en productos de baja rotación.",
            "evidencia": {"categoria": categoria, "productos": productos[:6], "total_productos": total},
        })

    return insights, categoria_cubierta


# ════════════════════════════════════════════════════════════
# 4. GENERAR_INSIGHTS_CONFIABILIDAD — cruza métricas de ambos
# ════════════════════════════════════════════════════════════

def generar_insights_confiabilidad(datos_prediccion: dict, datos_clasificacion: dict) -> list:
    calidad = datos_prediccion.get("calidad_modelo", {})
    dias = datos_clasificacion.get("dias", [])

    wape = calidad.get("wape")
    wape_alto = wape is not None and wape > 30
    wape_bajo = wape is not None and wape <= 15

    confs = []
    if dias:
        confs = [p.get("confianza") for p in dias[0].get("productos", []) if p.get("confianza") is not None]
    n_conf_baja = sum(1 for c in confs if c < 0.5)
    pct_conf_baja = (n_conf_baja / len(confs) * 100) if confs else 0
    confianza_promedio = (sum(confs) / len(confs)) if confs else None

    # Caso negativo: algo no anda bien, avisar
    if wape_alto or pct_conf_baja >= 25:
        partes = []
        if wape_alto:
            partes.append(f"el error del modelo de predicción (WAPE) está en {wape:.0f}%, más alto de lo ideal")
        if pct_conf_baja >= 25:
            partes.append(f"{pct_conf_baja:.0f}% de los productos tienen confianza de clasificación baja")
        return [{
            "tipo": "Confiabilidad", "polaridad": "riesgo",
            "prioridad": "Media" if (wape_alto and pct_conf_baja >= 25) else "Baja",
            "icono": "alert-triangle",
            "titulo": "Revisar manualmente antes de decidir",
            "descripcion": "Esta semana " + " y ".join(partes) + " — los insights anteriores siguen siendo útiles, pero conviene confirmarlos con criterio propio.",
            "accion": "Usar estos insights como punto de partida, no como decisión automática, hasta que el historial crezca.",
            "impacto": "Evita decisiones de compra/personal basadas en una predicción poco confiable.",
            "evidencia": {"wape": wape, "pct_confianza_baja": round(pct_conf_baja, 1)},
        }]

    # Equilibrio: caso positivo, ambos modelos confiables (informativa)
    if wape_bajo and confianza_promedio is not None and confianza_promedio >= 0.7:
        return [{
            "tipo": "Confiabilidad", "polaridad": "informativa",
            "prioridad": "Baja",
            "icono": "shield-check",
            "titulo": "El modelo mantiene alta confianza esta semana",
            "descripcion": (
                f"El error de predicción (WAPE) está en {wape:.0f}% y la confianza promedio de "
                f"clasificación es alta — las recomendaciones de este análisis son confiables."
            ),
            "accion": "Podés seguir estos insights con mayor seguridad de lo habitual.",
            "impacto": "Menor necesidad de validación manual esta semana.",
            "evidencia": {"wape": wape, "confianza_promedio": round(confianza_promedio, 2)},
        }]

    return []


# ════════════════════════════════════════════════════════════
# 5. PRIORIZAR_INSIGHTS
# ════════════════════════════════════════════════════════════

def priorizar_insights(insights: list) -> list:
    ordenados = sorted(insights, key=lambda i: _ORDEN_PRIORIDAD.get(i["prioridad"], 3))
    return ordenados[:MAX_INSIGHTS]


# ════════════════════════════════════════════════════════════
# 6. PANEL LATERAL — resumen de los insights ya generados
#    Reglas simples, no IA nueva.
# ════════════════════════════════════════════════════════════

def calcular_resumen(insights: list) -> dict:
    """Para el panel lateral siempre visible: conteos + acción más
    urgente + categorías involucradas."""
    criticas = sum(1 for i in insights if i["prioridad"] == "Alta")
    oportunidades = sum(1 for i in insights if i.get("polaridad") == "oportunidad")
    informativas = max(0, len(insights) - criticas - oportunidades)

    categorias = sorted({
        (i.get("evidencia") or {}).get("categoria")
        for i in insights if (i.get("evidencia") or {}).get("categoria")
    })

    accion_urgente = next(
        (i["accion"] for i in sorted(insights, key=lambda x: _ORDEN_PRIORIDAD.get(x["prioridad"], 3))
         if i["prioridad"] == "Alta"),
        None,
    )

    return {
        "total": len(insights),
        "criticas": criticas,
        "oportunidades": oportunidades,
        "informativas": informativas,
        "categorias_afectadas": categorias,
        "accion_urgente": accion_urgente,
    }


# ════════════════════════════════════════════════════════════
# ORQUESTADOR
# ════════════════════════════════════════════════════════════

def generar_insights(datos_prediccion: dict, datos_clasificacion: dict) -> list:
    """Punto de entrada único del Decision Engine. Devuelve solo la
    lista de insights — para el panel lateral y la puntuación, llamar
    aparte a calcular_resumen()/calcular_puntuacion_riesgo() sobre
    el resultado (no cambia el contrato de esta función para no
    romper a quien ya la usa)."""
    predictivos = generar_insights_predictivos(datos_prediccion)
    cruzados, categoria_cubierta = generar_insights_cruzados(datos_prediccion, datos_clasificacion)
    operativos = generar_insights_operativos(datos_clasificacion, categoria_excluir=categoria_cubierta)
    confiabilidad = generar_insights_confiabilidad(datos_prediccion, datos_clasificacion)

    todos = predictivos + operativos + cruzados + confiabilidad
    return priorizar_insights(todos)