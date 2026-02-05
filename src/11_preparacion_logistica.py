import json
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split

from config import DATA_DIR, TABLES_DIR

# Entrada recomendada: dataset ya depurado tras missing/outliers (Sprint 1)
INPUT_FILE = DATA_DIR / "interim/datos_dep_alfa.parquet"

# Salidas: particiones para modelización logística (preparación previa)
OUT_DIR = DATA_DIR / "processed/logistica"
X_TRAIN_FILE = OUT_DIR / "X_train_logistica.parquet"
X_TEST_FILE  = OUT_DIR / "X_test_logistica.parquet"
Y_TRAIN_FILE = OUT_DIR / "y_train_logistica.parquet"
Y_TEST_FILE  = OUT_DIR / "y_test_logistica.parquet"

# Reportes para el informe
PREP_REPORT = TABLES_DIR / "resumen_preparacion_logistica.json"
NULOS_VAR_REPORT = TABLES_DIR / "logistica_nulos_por_variable.csv"
NULOS_FILA_REPORT = TABLES_DIR / "logistica_nulos_por_fila.csv"
VARS_NUM_REPORT = TABLES_DIR / "logistica_vars_numericas.csv"
VARS_CAT_REPORT = TABLES_DIR / "logistica_vars_categoricas.csv"
BALANCE_Y_REPORT = TABLES_DIR / "logistica_balance_y.csv"

# Parámetros reproducibles
RANDOM_STATE = 1234567
TEST_SIZE = 0.20

# Variables objetivo del enunciado
# OJO: en tu script lineal aparecen Izda_Pct/Dcha_Pct con guion bajo.
# En el enunciado original suelen venir como "Izda Pct", etc.
# Mantengo aquí el mismo naming que ya usas en tu proyecto.
OBJETIVOS_CONT = ["AbstentionPtge", "Izda_Pct", "Dcha_Pct", "Otros_Pct"]
OBJETIVOS_BIN  = ["AbstencionAlta", "Izquierda", "Derecha"]
OTRAS_VARIABLES_EXCLUIDAS = ["Name", "Explotaciones", "Explotaciones_win", "CodigoProvincia"]

# Objetivo binario seleccionado para regresión logística (ajusta si tu elección final cambia)
Y_BIN_NAME = "AbstencionAlta"


def cargar_datos() -> pd.DataFrame:
    return pd.read_parquet(INPUT_FILE)


def forzar_y_binaria(datos: pd.DataFrame) -> pd.DataFrame:
    """
    Corrige el tipo de la variable objetivo binaria si viene como string ('0'/'1').
    La regresión logística requiere y numérica (0/1).
    """
    if Y_BIN_NAME not in datos.columns:
        raise ValueError(f"No existe la variable objetivo '{Y_BIN_NAME}' en el dataset.")

    y = datos[Y_BIN_NAME]

    # Si viene como object/string: limpiamos y convertimos
    if (y.dtype == "object") or (str(y.dtype) == "string"):
        y2 = y.astype(str).str.strip()
        # tolera comillas, espacios, etc. pero exige 0/1
        y2 = pd.to_numeric(y2, errors="raise").astype(int)
        datos[Y_BIN_NAME] = y2

    # Validación final de dicotomía
    vals = set(pd.unique(datos[Y_BIN_NAME].dropna()))
    if not vals.issubset({0, 1}):
        raise ValueError(
            f"'{Y_BIN_NAME}' debe ser dicotómica 0/1. Valores encontrados: {sorted(list(vals))}"
        )

    return datos


def construir_X_y(datos: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    """
    Construye X e y para regresión logística eliminando del set explicativo
    todas las variables objetivo no seleccionadas (continuas y binarias).
    """
    if Y_BIN_NAME not in datos.columns:
        raise ValueError(f"No existe la variable objetivo '{Y_BIN_NAME}' en el dataset.")

    targets_to_drop = [
        c for c in (OBJETIVOS_CONT + OBJETIVOS_BIN + OTRAS_VARIABLES_EXCLUIDAS)
        if (c != Y_BIN_NAME) and (c in datos.columns)
    ]

    X = datos.drop(columns=targets_to_drop + [Y_BIN_NAME], errors="ignore")
    y = datos[Y_BIN_NAME].copy()

    if Y_BIN_NAME in X.columns:
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


def balance_clases(y: pd.Series) -> pd.DataFrame:
    """Tabla simple de balance (conteo y porcentaje)."""
    conteo = y.value_counts(dropna=False).sort_index()
    pct = (conteo / len(y) * 100).round(3)
    out = pd.DataFrame({"Conteo": conteo, "Pct": pct})
    out.index.name = Y_BIN_NAME
    return out


def partir_train_test(X: pd.DataFrame, y: pd.Series) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    # Estratificar para preservar proporciones de clase en train/test
    return train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y
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

    # 2) Corregir tipo de y binaria si aplica
    datos = forzar_y_binaria(datos)
    print(f"[INFO] Tipo de '{Y_BIN_NAME}': {datos[Y_BIN_NAME].dtype}")

    # 3) Construcción X/y eliminando objetivos no seleccionados
    X, y, targets_to_drop = construir_X_y(datos)
    print(f"[INFO] X shape={X.shape} | y shape={y.shape}")
    print(f"[INFO] Objetivos eliminados como explicativas ({len(targets_to_drop)}): {targets_to_drop}")

    # 4) Separación por tipo (informativo; no se transforma)
    vars_num, vars_cat = separar_tipos(X)
    print(f"[INFO] #vars_num={len(vars_num)} | #vars_cat={len(vars_cat)}")

    # 5) Reportes de nulos/duplicados (informativos para el informe)
    nulos_var, nulos_fila = resumen_nulos(X)
    dup_rows = int(X.duplicated().sum())

    nulos_var.to_csv(NULOS_VAR_REPORT)
    nulos_fila.to_csv(NULOS_FILA_REPORT, index=False)
    guardar_listas(vars_num, vars_cat)

    # 6) Balance de clases (informativo para el informe)
    bal = balance_clases(y)
    bal.to_csv(BALANCE_Y_REPORT)
    print(f"[INFO] Balance de clases guardado: {BALANCE_Y_REPORT.name}")
    print(f"[INFO] Balance (pct): {bal['Pct'].to_dict()}")

    # 7) Split train/test (estratificado)
    X_train, X_test, y_train, y_test = partir_train_test(X, y)
    print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test : X={X_test.shape}, y={y_test.shape}")

    # 8) Guardar artefactos para los siguientes scripts (selección clásica, punto de corte, etc.)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    X_train.to_parquet(X_TRAIN_FILE, index=False)
    X_test.to_parquet(X_TEST_FILE, index=False)
    y_train.to_frame(Y_BIN_NAME).to_parquet(Y_TRAIN_FILE, index=False)
    y_test.to_frame(Y_BIN_NAME).to_parquet(Y_TEST_FILE, index=False)

    # 9) Metadata
    meta = {
        "input_file": str(INPUT_FILE),
        "y_bin_name": Y_BIN_NAME,
        "targets_dropped": targets_to_drop,
        "n_rows": int(datos.shape[0]),
        "n_cols_total": int(datos.shape[1]),
        "n_cols_X": int(X.shape[1]),
        "n_vars_num": int(len(vars_num)),
        "n_vars_cat": int(len(vars_cat)),
        "test_size": float(TEST_SIZE),
        "random_state": int(RANDOM_STATE),
        "duplicated_rows_X": dup_rows,
        "balance_y_pct": bal["Pct"].to_dict(),
        "top10_nulos_var": nulos_var.head(10).to_dict(orient="index"),
    }
    guardar_metadata(meta)

    print("[OK] Preparación para regresión logística completada.")
    print(f"[OK] Artefactos: {X_TRAIN_FILE.name}, {X_TEST_FILE.name}, {Y_TRAIN_FILE.name}, {Y_TEST_FILE.name}")
    print(f"[OK] Reportes: {PREP_REPORT.name}, {NULOS_VAR_REPORT.name}, {NULOS_FILA_REPORT.name}, {BALANCE_Y_REPORT.name}")


if __name__ == "__main__":
    main()
