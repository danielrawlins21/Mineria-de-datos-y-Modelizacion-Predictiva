import pandas as pd
import numpy as np
from scipy.stats import zscore
from config import DATA_DIR, TABLES_DIR

INPUT_FILE = DATA_DIR / "interim/datos_dep_err_corr.parquet"
OUT_SUMMARY_FILE = TABLES_DIR / "resumen_outliers.csv"

OUTLIER_VARS = [
    "TotalCensus",
    "PobChange_pct",
    "Explotaciones",
    "SUPERFICIE",
    "UnemployLess25_Ptge",
    "UnemployMore40_Ptge",
    "AgricultureUnemploymentPtge",
    "IndustryUnemploymentPtge",
    "ConstructionUnemploymentPtge",
    "ServicesUnemploymentPtge",
    "DifComAutonPtge",
    "SameComAutonDiffProvPtge"
]


def detectar_outliers_iqr(serie: pd.Series) -> pd.Index:
    q1 = serie.quantile(0.25)
    q3 = serie.quantile(0.75)
    iqr = q3 - q1
    lim_inf = q1 - 1.5 * iqr
    lim_sup = q3 + 1.5 * iqr
    return serie[(serie < lim_inf) | (serie > lim_sup)].index


def detectar_outliers_zscore(serie: pd.Series, umbral: float = 3.0) -> pd.Index:
    z = zscore(serie.dropna())
    return serie.dropna().index[np.abs(z) > umbral]


def analizar_outliers(datos: pd.DataFrame) -> pd.DataFrame:
    resumen = []

    for var in OUTLIER_VARS:
        if var not in datos.columns:
            continue

        serie = datos[var]

        iqr_idx = detectar_outliers_iqr(serie.dropna())
        z_idx = detectar_outliers_zscore(serie)

        resumen.append({
            "Variable": var,
            "N_Observaciones": serie.notna().sum(),
            "Outliers_IQR": len(iqr_idx),
            "Outliers_Zscore": len(z_idx),
            "Pct_IQR": len(iqr_idx) / serie.notna().sum() * 100,
            "Pct_Zscore": len(z_idx) / serie.notna().sum() * 100
        })

    return pd.DataFrame(resumen).sort_values(
        by=["Outliers_IQR", "Outliers_Zscore"], ascending=False
    )


def main() -> None:
    datos = pd.read_parquet(INPUT_FILE)
    resumen_outliers = analizar_outliers(datos)
    resumen_outliers.to_csv(OUT_SUMMARY_FILE, index=False)
    print("Análisis de outliers completado.")
    print(resumen_outliers)


if __name__ == "__main__":
    main()
