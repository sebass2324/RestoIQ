"""
prueba_quincena_vs_binaria.py

Comparación CORRECTA (la anterior comparaba "3 variables nuevas" vs.
"nada de quincena" -- esta compara "3 variables nuevas" vs. "la
bandera binaria vieja", que es la pregunta real). Mismo dataset
exacto en ambos casos.

Ejecutar desde la raiz del proyecto:
    python prueba_quincena_vs_binaria.py
"""

import pandas as pd

from app import create_app
from models.venta import Venta
from models.configuracion_analisis import ConfiguracionAnalisis
import services.sales_model as sm

USER_ID = 6  # <-- ajusta al user_id correcto


def cargar_ventas(user_id):
    ventas = Venta.query.filter_by(user_id=user_id).all()
    print(f"[debug] Ventas encontradas: {len(ventas)}")
    filas = []
    for v in ventas:
        filas.append({
            "fecha": v.fecha, "producto": v.producto, "cantidad": v.cantidad,
            "precio": v.precio, "total": v.total,
            "dia_semana": v.dia_semana, "mes": v.mes, "semana_anio": v.semana_anio,
            "es_finde": v.es_finde, "es_feriado": v.es_feriado,
            "es_puente": v.es_puente,
            "promocion":           int(v.promocion) if v.promocion is not None else 0,
            "descuento_pct":       v.descuento_pct,
            "es_evento_especial":  int(v.es_evento_especial) if v.es_evento_especial is not None else 0,
            "lluvia_nocturna_mm":    v.lluvia_nocturna_mm,
            "temp_nocturna_promed":  v.temp_nocturna_promed,
        })
    df = pd.DataFrame(filas)
    if df.empty:
        return df
    df["fecha"] = pd.to_datetime(df["fecha"])
    return df


def features_fecha_binaria(self, fecha):
    """Copia de _features_fecha, pero con la quincena VIEJA (binaria,
    ventana ancha 1-7/15-21, +10% en la fórmula de demanda original)
    en vez de las 3 variables nuevas -- para la comparación directa."""
    from datetime import timedelta
    fecha_str = fecha.strftime("%Y-%m-%d")
    dia       = fecha.weekday()
    ayer      = (fecha - timedelta(days=1)).strftime("%Y-%m-%d")
    maniana   = (fecha + timedelta(days=1)).strftime("%Y-%m-%d")
    dia_mes   = fecha.day
    return {
        "dia_semana":      dia,
        "dia_mes":         dia_mes,
        "mes":             fecha.month,
        "semana_anio":     int(fecha.isocalendar()[1]),
        "es_finde":        int(dia in [5, 6]),
        "es_feriado":      int(fecha_str in self.feriados),
        "vispera_feriado": int(maniana in self.feriados),
        "es_puente":       int(ayer in self.feriados or maniana in self.feriados),
        "es_quincena":     int(1 <= dia_mes <= 7 or 15 <= dia_mes <= 21),
    }


def entrenar_y_medir(df, config, etiqueta):
    modelo = sm.SalesModel()
    metricas = modelo.entrenar(df, config=config, verbose=False)
    holdout = metricas.get("holdout") or {}
    print(f"\n=== {etiqueta} ===")
    print(f"  Features usadas: {modelo.features}")
    print(f"  WAPE: {holdout.get('wape_restoiq')}")
    print(f"  R²:   {holdout.get('r2_restoiq')}")
    print(f"  Mejora vs. baseline: {holdout.get('mejora_pct')}%")
    return holdout.get("wape_restoiq"), holdout.get("r2_restoiq")


def main():
    app = create_app()
    with app.app_context():
        df = cargar_ventas(USER_ID)
        if df.empty:
            print(f"[!] No hay ventas para USER_ID={USER_ID}. Sube tu CSV en /upload primero.")
            return

        config = ConfiguracionAnalisis.query.filter_by(user_id=USER_ID).first()
        if config is None:
            print("No hay ConfiguracionAnalisis -- configura el negocio primero.")
            return

        temporales_original = list(sm.FEATURES_TEMPORALES)
        features_fecha_original = sm.SalesModel._features_fecha

        try:
            print("=== A) CON las 3 variables nuevas (como está ahora) ===")
            wape_nuevas, r2_nuevas = entrenar_y_medir(df, config, "3 variables nuevas")

            print("\n=== B) CON la quincena binaria vieja (es_quincena, ventana ancha) ===")
            sm.FEATURES_TEMPORALES = [
                f for f in temporales_original
                if f not in ("dias_distancia_cobro", "pico_comida_rapida", "fase_liquidez")
            ] + ["es_quincena"]
            sm.SalesModel._features_fecha = features_fecha_binaria
            wape_binaria, r2_binaria = entrenar_y_medir(df, config, "es_quincena binaria vieja")

            print("\n=== RESUMEN ===")
            print(f"  WAPE 3 variables nuevas: {wape_nuevas}   WAPE binaria vieja: {wape_binaria}   diferencia: {round(wape_nuevas - wape_binaria, 2):+.2f}")
            print(f"  R² 3 variables nuevas:   {r2_nuevas}   R² binaria vieja:   {r2_binaria}   diferencia: {round(r2_nuevas - r2_binaria, 4):+.4f}")
            print()
            if wape_nuevas < wape_binaria:
                print("  -> Las 3 variables nuevas SÍ superan a la binaria vieja.")
            elif wape_nuevas > wape_binaria:
                print("  -> La binaria vieja da MEJOR resultado que las 3 variables nuevas en este dataset.")
            else:
                print("  -> Prácticamente empatadas.")

        finally:
            sm.FEATURES_TEMPORALES = temporales_original
            sm.SalesModel._features_fecha = features_fecha_original


if __name__ == "__main__":
    main()