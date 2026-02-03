import pandas as pd
import numpy as np
from config import DATA_DIR, TABLES_DIR

# Entrada recomendada: tras corrección de errores (HU-1.3) y/o outliers (HU-1.4)
# Ajusta este nombre si tu pipeline guarda otro archivo intermedio.
INPUT_FILE = DATA_DIR / "interim/datos_dep_outliers_tratados.parquet"

# Salida: dataset final depurado
OUTPUT_FILE = DATA_DIR / "interim/datos_dep_alfa.parquet"

# Reporte para el informe
MISSING_REPORT = TABLES_DIR / "resumen_missing.csv"


def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(INPUT_FILE)


def resumen_missing(datos: pd.DataFrame) -> pd.DataFrame:
    """Resumen de valores perdidos por variable (N y %)."""
    res = datos.isna().sum().sort_values(ascending=False).to_frame(name="Nulos")
    res["PctNulos"] = (res["Nulos"] / len(datos) * 100).round(3)
    return res


def tratar_missing(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Tratamiento de valores perdidos con decisiones justificadas:

    - Explotaciones y Explotaciones_win: se mantienen como NaN si provienen de outliers excluidos.
    - PobChange_pct: eliminar filas con NaN (en tu caso eran 7, impacto despreciable).
    - Densidad: si aún hubiera NaN, recodificar a 'Desconocida' (categórica ordinal).
    - SUPERFICIE: si hubiera NaN, imputar con mediana (robusta).
    """

    # 1) PobChange_pct: eliminar filas con NA (casos marginales)
    if "PobChange_pct" in datos.columns:
        datos = datos.dropna(subset=["PobChange_pct"])

    # 2) Densidad: si quedan NA, recodificar
    if "Densidad" in datos.columns:
        # asegurar tipo string para fillna
        datos["Densidad"] = datos["Densidad"].astype("string")
        datos["Densidad"] = datos["Densidad"].fillna("Desconocida")

    # 3) SUPERFICIE: imputación robusta si hay NA
    if "SUPERFICIE" in datos.columns and datos["SUPERFICIE"].isna().any():
        mediana = datos["SUPERFICIE"].median()
        datos["SUPERFICIE"] = datos["SUPERFICIE"].fillna(mediana)

    # 4) Explotaciones / Explotaciones_win: NO imputar
    # 5) Categorización de Explotaciones según P75
    if "Explotaciones" in datos.columns:
        p75 = datos["Explotaciones"].quantile(0.75)

        def cat_explot(x):
            if pd.isna(x):
                return pd.NA
            elif x <= p75:
                return "≤137"
            else:
                return ">137"

        datos["Explotaciones_cat"] = datos["Explotaciones"].apply(cat_explot)

    # 6) Imputación aleatoria para Explotaciones_cat
    if "Explotaciones_cat" in datos.columns:
        dist_real = (
            datos["Explotaciones_cat"]
            .dropna()
            .value_counts(normalize=True)
        )

        mask_nan = datos["Explotaciones_cat"].isna()
        n_nan = mask_nan.sum()

        if n_nan > 0 and len(dist_real) > 0:
            np.random.seed(123)
            asignacion = np.random.choice(
                dist_real.index,
                size=n_nan,
                p=dist_real.values
            )
            datos.loc[mask_nan, "Explotaciones_cat"] = asignacion

    return datos


def main() -> None:
    datos = cargar_datos()

    # Reporte antes
    rep_antes = resumen_missing(datos)
    rep_antes.to_csv(TABLES_DIR / "resumen_missing_antes.csv")

    # Tratamiento
    datos_tratados = tratar_missing(datos)

    # Reporte después
    rep_despues = resumen_missing(datos_tratados)
    rep_despues.to_csv(MISSING_REPORT)

    # Guardar dataset final
    datos_tratados.to_parquet(OUTPUT_FILE, index=False)

    # Mostrar resumen principal
    print("Tratamiento de valores perdidos completado.")
    print("Top 15 variables con más nulos (después):")
    print(rep_despues.head(15))


if __name__ == "__main__":
    main()
