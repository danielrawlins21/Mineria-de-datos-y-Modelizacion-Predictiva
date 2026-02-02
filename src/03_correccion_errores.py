import pandas as pd
from config import DATA_DIR

DEP_BASE_FILE = DATA_DIR / "interim/datos_dep_base.parquet"
DEP_ERR_FILE = DATA_DIR / "interim/datos_dep_err_corr.parquet"

def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(DEP_BASE_FILE)

def corregir_errores(datos: pd.DataFrame) -> pd.DataFrame:
    # Corrección de Densidad: categoría inválida
    if "Densidad" in datos.columns:
        datos["Densidad"] = datos["Densidad"].replace("?", pd.NA)
    return datos

def main() -> None:
    datos = cargar_datos()
    datos = corregir_errores(datos)
    datos.to_parquet(DEP_ERR_FILE, index=False)

if __name__ == "__main__":
    main()
