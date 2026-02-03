import pandas as pd
from config import DATA_DIR, TABLES_DIR

INPUT_FILE = DATA_DIR / "interim/datos_dep_err_corr.parquet"
OUTPUT_FILE = DATA_DIR / "interim/datos_dep_outliers_tratados.parquet"
SUMMARY_FILE = TABLES_DIR / "winsorizacion_explotaciones.csv"


def winsorizar_explotaciones(datos: pd.DataFrame, p: float = 0.99) -> pd.DataFrame:
    """
    Aplica winsorización a Explotaciones usando el percentil p.
    Antes, recodifica posibles sentinelas (ej. 99999) como NaN.
    """
    # 1) Detectar y recodificar sentinela típico
    if "Explotaciones" not in datos.columns:
        raise KeyError("No existe la columna 'Explotaciones' en el dataset.")

    n_sentinela = (datos["Explotaciones"] == 99999).sum()
    datos["Explotaciones"] = datos["Explotaciones"].replace(99999, pd.NA)

    # 2) Calcular percentil ignorando NA
    limite_sup = datos["Explotaciones"].dropna().quantile(p)

    # 3) Winsorizar
    datos["Explotaciones_win"] = datos["Explotaciones"].clip(upper=limite_sup)

    resumen = {
        "Percentil": p,
        "Limite_superior": float(limite_sup),
        "N_sentinela_99999": int(n_sentinela),
        "N_valores_afectados": int((datos["Explotaciones"] > limite_sup).sum())
    }

    return datos, pd.DataFrame([resumen])



def main() -> None:
    datos = pd.read_parquet(INPUT_FILE)

    datos, resumen = winsorizar_explotaciones(datos, p=0.99)

    # Guardar dataset tratado
    datos.to_parquet(OUTPUT_FILE, index=False)

    # Guardar resumen para el informe
    resumen.to_csv(SUMMARY_FILE, index=False)

    print("Winsorización aplicada correctamente.")
    print(resumen)


if __name__ == "__main__":
    main()
