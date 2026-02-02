import pandas as pd
from config import DATA_FILE, DEP_BASE_FILE

def cargar_datos() -> pd.DataFrame:
    return pd.read_excel(DATA_FILE)

def corregir_tipos(datos: pd.DataFrame) -> pd.DataFrame:
    # Binarias a categóricas
    binarias_a_categoricas = ["AbstencionAlta", "Izquierda", "Derecha"]
    for v in binarias_a_categoricas:
        if v in datos.columns:
            datos[v] = datos[v].astype("string")

    # Densidad a numérica
    if "Densidad" in datos.columns:
        # Densidad es categórica ordinal
        datos["Densidad"] = datos["Densidad"].astype("string")
    return datos

def main() -> None:
    datos = cargar_datos()
    datos = corregir_tipos(datos)
    datos.to_parquet(DEP_BASE_FILE, index=False)

if __name__ == "__main__":
    main()
