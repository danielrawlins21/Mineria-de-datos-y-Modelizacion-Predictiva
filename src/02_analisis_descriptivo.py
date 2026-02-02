import pandas as pd
from src.config import DEP_BASE_FILE, TABLES_DIR

def cargar_datos() -> pd.DataFrame:
    """Carga el dataset depurado base."""
    return pd.read_parquet(DEP_BASE_FILE)

def analisis_estructura(datos: pd.DataFrame) -> dict:
    """Devuelve información básica del dataset."""
    return {
        "n_observaciones": datos.shape[0],
        "n_variables": datos.shape[1],
        "tipos": datos.dtypes.value_counts()
    }

def descriptivos_numericos(datos: pd.DataFrame) -> pd.DataFrame:
    """Calcula estadísticos descriptivos para variables numéricas."""
    desc = datos.describe().T
    desc["Asimetria"] = datos[desc.index].skew()
    desc["Curtosis"] = datos[desc.index].kurtosis()
    return desc

def frecuencias_categoricas(datos: pd.DataFrame) -> dict:
    """Calcula frecuencias de variables categóricas."""
    cat_vars = datos.select_dtypes(include=["object", "string"]).columns
    return {var: datos[var].value_counts(dropna=False) for var in cat_vars}

def main() -> None:
    datos = cargar_datos()

    info = analisis_estructura(datos)
    print("Número de observaciones:", info["n_observaciones"])
    print("Número de variables:", info["n_variables"])
    print("\nTipos de variables:\n", info["tipos"])

    desc_num = descriptivos_numericos(datos)
    desc_num.to_csv(TABLES_DIR / "descriptivos_numericos.csv")

    frec_cat = frecuencias_categoricas(datos)
    for var, frec in frec_cat.items():
        frec.to_csv(TABLES_DIR / f"frecuencias_{var}.csv")

if __name__ == "__main__":
    main()
