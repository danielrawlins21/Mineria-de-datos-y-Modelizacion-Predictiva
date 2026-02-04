import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import DATA_DIR, TABLES_DIR

# Entrada recomendada: dataset ya depurado tras missing/outliers (Sprint 1)
# Ajusta si tu pipeline guarda otro archivo.
INPUT_FILE = DATA_DIR / "interim/datos_dep_alfa.parquet"

# Salidas: particiones para modelización lineal (preparación previa)
OUT_DIR = DATA_DIR / "processed/lineal"
X_TRAIN_FILE = OUT_DIR / "X_train_lineal.parquet"
X_TEST_FILE  = OUT_DIR / "X_test_lineal.parquet"
Y_TRAIN_FILE = OUT_DIR / "y_train_lineal.parquet"
Y_TEST_FILE  = OUT_DIR / "y_test_lineal.parquet"

# Reportes para el informe
PREP_REPORT = TABLES_DIR / "resumen_preparacion_lineal.json"
NULOS_VAR_REPORT = TABLES_DIR / "lineal_nulos_por_variable.csv"
NULOS_FILA_REPORT = TABLES_DIR / "lineal_nulos_por_fila.csv"
VARS_NUM_REPORT = TABLES_DIR / "lineal_vars_numericas.csv"
VARS_CAT_REPORT = TABLES_DIR / "lineal_vars_categoricas.csv"

# Parámetros reproducibles
RANDOM_STATE = 1234567
TEST_SIZE = 0.20

# Variables objetivo del enunciado
OBJETIVOS_CONT = ["AbstentionPtge", "Izda_Pct", "Dcha_Pct", "Otros_Pct"]
OBJETIVOS_BIN  = ["AbstencionAlta", "Izquierda", "Derecha"]
OTRAS_VARIABLES_EXCLUIDAS = ["Name", "Explotaciones", "Explotaciones_win", "CodigoProvincia"]

# Objetivo continuo seleccionado para regresión lineal (ajusta si tu elección final cambia)
Y_CONT_NAME = "AbstentionPtge"


def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(INPUT_FILE)


def construir_X_y(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Construye X e y para regresión lineal eliminando del set explicativo
    todas las variables objetivo no seleccionadas (continuas y binarias).
    """
    if Y_CONT_NAME not in datos.columns:
        raise ValueError(f"No existe la variable objetivo '{Y_CONT_NAME}' en el dataset.")

    targets_to_drop = [
        c for c in (OBJETIVOS_CONT + OBJETIVOS_BIN + OTRAS_VARIABLES_EXCLUIDAS)
        if (c != Y_CONT_NAME) and (c in datos.columns)
    ]

    X = datos.drop(columns=targets_to_drop + [Y_CONT_NAME], errors="ignore")
    y = datos[Y_CONT_NAME].copy()

    if Y_CONT_NAME in X.columns:
        raise RuntimeError("La variable objetivo quedó dentro de X. Revisa la lógica de eliminación.")

    return X, y, targets_to_drop


def separar_tipos(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    """Identifica variables numéricas y categóricas según dtype (sin transformar)."""
    vars_num = X.select_dtypes(include=[np.number]).columns.tolist()
    vars_cat = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return vars_num, vars_cat


def resumen_nulos(X: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Devuelve (nulos por variable) y (nulos por fila)."""
    nulos_var = X.isna().sum().sort_values(ascending=False).to_frame(name="Nulos")
    nulos_var["PctNulos"] = (nulos_var["Nulos"] / len(X) * 100).round(3)

    nulos_fila = X.isna().sum(axis=1).to_frame(name="NulosFila")
    return nulos_var, nulos_fila


def partir_train_test(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE
    )


def guardar_listas(vars_num: list[str], vars_cat: list[str]) -> None:
    pd.Series(vars_num, name="vars_num").to_csv(VARS_NUM_REPORT, index=False)
    pd.Series(vars_cat, name="vars_cat").to_csv(VARS_CAT_REPORT, index=False)


def guardar_metadata(meta: dict) -> None:
    with open(PREP_REPORT, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    # 1) Carga
    datos = cargar_datos()
    print(f"[INFO] Dataset cargado: {INPUT_FILE.name} | shape={datos.shape}")

    # 2) Construcción X/y eliminando objetivos no seleccionados
    X, y, targets_to_drop = construir_X_y(datos)
    print(f"[INFO] X shape={X.shape} | y shape={y.shape}")
    print(f"[INFO] Objetivos eliminados como explicativas ({len(targets_to_drop)}): {targets_to_drop}")

    # 3) Separación por tipo (informativo; no se transforma)
    vars_num, vars_cat = separar_tipos(X)
    print(f"[INFO] #vars_num={len(vars_num)} | #vars_cat={len(vars_cat)}")

    # 4) Reportes de nulos/duplicados (informativos para el informe)
    nulos_var, nulos_fila = resumen_nulos(X)
    dup_rows = int(X.duplicated().sum())

    nulos_var.to_csv(NULOS_VAR_REPORT)
    nulos_fila.to_csv(NULOS_FILA_REPORT, index=False)
    guardar_listas(vars_num, vars_cat)

    # 5) Split train/test
    X_train, X_test, y_train, y_test = partir_train_test(X, y)
    print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test : X={X_test.shape}, y={y_test.shape}")

    # 6) Guardar artefactos para los siguientes scripts (selección clásica, evaluación, etc.)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(X_TRAIN_FILE, index=False)
    X_test.to_parquet(X_TEST_FILE, index=False)
    y_train.to_frame(Y_CONT_NAME).to_parquet(Y_TRAIN_FILE, index=False)
    y_test.to_frame(Y_CONT_NAME).to_parquet(Y_TEST_FILE, index=False)

    # 7) Metadata
    meta = {
        "input_file": str(INPUT_FILE),
        "y_cont_name": Y_CONT_NAME,
        "targets_dropped": targets_to_drop,
        "n_rows": int(datos.shape[0]),
        "n_cols_total": int(datos.shape[1]),
        "n_cols_X": int(X.shape[1]),
        "n_vars_num": int(len(vars_num)),
        "n_vars_cat": int(len(vars_cat)),
        "test_size": float(TEST_SIZE),
        "random_state": int(RANDOM_STATE),
        "duplicated_rows_X": dup_rows,
        "top10_nulos_var": nulos_var.head(10).to_dict(orient="index"),
    }
    guardar_metadata(meta)

    print("[OK] Preparación para regresión lineal completada.")
    print(f"[OK] Artefactos: {X_TRAIN_FILE.name}, {X_TEST_FILE.name}, {Y_TRAIN_FILE.name}, {Y_TEST_FILE.name}")
    print(f"[OK] Reportes: {PREP_REPORT.name}, {NULOS_VAR_REPORT.name}, {NULOS_FILA_REPORT.name}")


if __name__ == "__main__":
    main()
