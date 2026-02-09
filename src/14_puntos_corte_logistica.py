import json
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix

from config import DATA_DIR, TABLES_DIR
from FuncionesMineria import crear_data_modelo

# ============================================================
# HU-3.4 — Determinación del punto de corte óptimo (Logística)
# (basado en el notebook corregido: alineación de dummies train/test)
# ============================================================

# Entradas
META_HU32_FILE   = TABLES_DIR / "logistica_hu_3_2_seleccion_clasica_meta.json"
GANADOR_HU33_FILE = TABLES_DIR / "logistica_hu_3_3_modelo_ganador.csv"

# Salidas
TABLA_CORTES_FILE = TABLES_DIR / "logistica_hu_3_4_tabla_puntos_corte.csv"
RESUMEN_FILE      = TABLES_DIR / "logistica_hu_3_4_resumen_punto_corte.json"

# Barrido de puntos de corte
CORTE_MIN = 0.10
CORTE_MAX = 0.90
N_CORTES  = 81

# Reentrenar SOLO el modelo ganador (mismas variables) para evitar dependencia de coef CSV
LOGIT_MAX_ITER = 2000
LOGIT_SOLVER   = "lbfgs"


def cargar_meta_hu32() -> dict:
    if not META_HU32_FILE.exists():
        raise FileNotFoundError(f"No existe {META_HU32_FILE}. Ejecuta antes HU-3.2.")
    with open(META_HU32_FILE, "r", encoding="utf-8") as f:
        return json.load(f)


def cargar_ganador_hu33() -> str:
    if not GANADOR_HU33_FILE.exists():
        raise FileNotFoundError(f"No existe {GANADOR_HU33_FILE}. Ejecuta antes HU-3.3.")
    g = pd.read_csv(GANADOR_HU33_FILE).iloc[0]
    if "Modelo" not in g.index:
        raise ValueError("El CSV del ganador HU-3.3 no contiene la columna 'Modelo'.")
    return str(g["Modelo"])


def extraer_variables_ganador(meta32: dict, nombre_modelo: str) -> tuple[list, list]:
    """
    En HU-3.2 guardamos 'resumen_completo' con registros que incluyen:
      - Modelo
      - Vars_cont
      - Vars_categ
    """
    if "resumen_completo" not in meta32:
        raise KeyError("No se encuentra 'resumen_completo' en el meta de HU-3.2. Revisa la estructura del JSON.")

    recs = meta32["resumen_completo"]
    rec = None
    for r in recs:
        if r.get("Modelo") == nombre_modelo:
            rec = r
            break

    if rec is None:
        modelos = [r.get("Modelo") for r in recs]
        raise ValueError(
            f"No encontré el modelo '{nombre_modelo}' dentro de 'resumen_completo'. "
            f"Modelos disponibles: {modelos}"
        )

    var_cont = rec.get("Vars_cont", [])
    var_categ = rec.get("Vars_categ", [])

    return var_cont, var_categ


def cargar_splits(meta32: dict):
    input_files = meta32.get("input_files", {})
    for k in ["X_train", "X_test", "y_train", "y_test"]:
        if k not in input_files:
            raise KeyError(f"Falta '{k}' en meta32['input_files'].")

    X_train = pd.read_parquet(input_files["X_train"])
    X_test  = pd.read_parquet(input_files["X_test"])
    y_train = pd.read_parquet(input_files["y_train"]).iloc[:, 0].astype(int)
    y_test  = pd.read_parquet(input_files["y_test"]).iloc[:, 0].astype(int)

    return X_train, X_test, y_train, y_test


def construir_matriz_modelo(X: pd.DataFrame, var_cont: list, var_categ: list) -> pd.DataFrame:
    X_m = crear_data_modelo(X, var_cont, var_categ, [])
    X_m = pd.get_dummies(X_m, drop_first=True)
    return X_m


def barrer_puntos_corte(modelo, probs: np.ndarray, y_true: pd.Series) -> pd.DataFrame:
    puntos = np.linspace(CORTE_MIN, CORTE_MAX, N_CORTES)
    rows = []
    yv = y_true.values

    for c in puntos:
        pred = (probs >= c).astype(int)
        tn, fp, fn, tp = confusion_matrix(yv, pred).ravel()

        acc = float((pred == yv).mean())
        sens = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
        esp  = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
        youden = float(sens + esp - 1) if (not np.isnan(sens) and not np.isnan(esp)) else np.nan

        rows.append({
            "PuntoCorte": float(c),
            "Accuracy": acc,
            "Sensibilidad": sens,
            "Especificidad": esp,
            "Youden": youden
        })

    return pd.DataFrame(rows)


def main() -> None:
    # 1) Cargar meta y ganador
    meta32 = cargar_meta_hu32()
    ganador_nombre = cargar_ganador_hu33()
    print(f"[INFO] Modelo ganador HU-3.3: {ganador_nombre}")

    # 2) Variables del ganador (según HU-3.2)
    var_cont, var_categ = extraer_variables_ganador(meta32, ganador_nombre)
    print(f"[INFO] #var_cont={len(var_cont)} | #var_categ={len(var_categ)}")

    # 3) Cargar splits HU-3.1 desde meta HU-3.2
    X_train, X_test, y_train, y_test = cargar_splits(meta32)
    print(f"[INFO] X_train={X_train.shape} | X_test={X_test.shape}")
    print(f"[INFO] Balance y_test: {y_test.value_counts(normalize=True).round(3).to_dict()}")

    # 4) Construir matrices de diseño (train/test) y alinear columnas
    X_train_m = construir_matriz_modelo(X_train, var_cont, var_categ)
    design_cols = X_train_m.columns.tolist()

    X_test_m = construir_matriz_modelo(X_test, var_cont, var_categ)
    X_test_m = X_test_m.reindex(columns=design_cols, fill_value=0)

    print(f"[INFO] Matriz train (dummies): {X_train_m.shape} | Matriz test (alineada): {X_test_m.shape}")

    # 5) Ajustar el modelo ganador (solo con sus variables)
    modelo = LogisticRegression(max_iter=LOGIT_MAX_ITER, solver=LOGIT_SOLVER)
    modelo.fit(X_train_m, y_train)

    # 6) Probabilidades en test
    probs_test = modelo.predict_proba(X_test_m)[:, 1]

    # 7) Barrido puntos de corte y métricas
    tabla_cortes = barrer_puntos_corte(modelo, probs_test, y_test)
    tabla_cortes.to_csv(TABLA_CORTES_FILE, index=False)
    print(f"[OK] Tabla de puntos de corte guardada: {TABLA_CORTES_FILE.name}")

    # 8) Extraer óptimos (accuracy y Youden)
    corte_acc = tabla_cortes.loc[tabla_cortes["Accuracy"].idxmax()].to_dict()
    corte_youden = tabla_cortes.loc[tabla_cortes["Youden"].idxmax()].to_dict()

    resumen = {
        "modelo_ganador": ganador_nombre,
        "rango_cortes": [CORTE_MIN, CORTE_MAX],
        "n_cortes": int(N_CORTES),
        "optimo_accuracy": corte_acc,
        "optimo_youden": corte_youden,
        "nota": "El punto de corte definitivo se elige en el informe según el criterio (p.ej. Youden si no hay costes asimétricos)."
    }

    with open(RESUMEN_FILE, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print(f"[OK] Resumen guardado: {RESUMEN_FILE.name}")
    print("\n[INFO] Óptimo por Accuracy:", {k: corte_acc[k] for k in ["PuntoCorte", "Accuracy", "Sensibilidad", "Especificidad", "Youden"]})
    print("[INFO] Óptimo por Youden  :", {k: corte_youden[k] for k in ["PuntoCorte", "Accuracy", "Sensibilidad", "Especificidad", "Youden"]})


if __name__ == "__main__":
    main()
