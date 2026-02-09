import json
import numpy as np
import pandas as pd

from joblib import load
from sklearn.metrics import confusion_matrix, accuracy_score

from config import TABLES_DIR, DATA_DIR, OUTPUT_DIR
from FuncionesMineria import crear_data_modelo

# ============================================================
# HU-3.5 — Evaluación del modelo logístico final (Youden=0.31)
# ============================================================
OUT_DIR = DATA_DIR / "processed/logistica"
# Entradas (generadas en HU-3.4)
MODELO_FILE = DATA_DIR / "processed/logistica/logistica_modelo_final.joblib"
META_FILE   = DATA_DIR / "processed/logistica/logistica_modelo_final_meta.json"
X_TEST_FILE  = OUT_DIR / "X_test_logistica.parquet"
Y_TEST_FILE  = OUT_DIR / "y_test_logistica.parquet"

# Salidas
TABLA_METRICAS_FILE = TABLES_DIR / "logistica_hu_3_5_metricas_finales.csv"
TABLA_MATRIZ_CONF_FILE = TABLES_DIR / "logistica_hu_3_5_matriz_confusion.csv"
TABLA_TOP_COEF_FILE = TABLES_DIR / "logistica_hu_3_5_top_coeficientes.csv"
RESUMEN_FILE = TABLES_DIR / "logistica_hu_3_5_resumen_evaluacion.json"

# Parámetros
TOP_K_COEF = 15


def cargar_modelo_y_meta():
    if not MODELO_FILE.exists():
        raise FileNotFoundError(f"No existe {MODELO_FILE}. Ejecuta HU-3.4 para generarlo.")
    if not META_FILE.exists():
        raise FileNotFoundError(f"No existe {META_FILE}. Ejecuta HU-3.4 para generarlo.")

    modelo = load(MODELO_FILE)

    with open(META_FILE, "r", encoding="utf-8") as f:
        meta = json.load(f)

    return modelo, meta


def cargar_test(meta: dict):
    X_test = pd.read_parquet(X_TEST_FILE)
    y_test = pd.read_parquet(Y_TEST_FILE).iloc[:, 0].astype(int)
    return X_test, y_test


def construir_X_test_m(X_test: pd.DataFrame, meta: dict) -> pd.DataFrame:
    var_cont = meta["variables"]["continuas"]
    var_categ = meta["variables"]["categoricas"]
    design_cols = meta["design_columns"]

    X_test_m = crear_data_modelo(X_test, var_cont, var_categ, [])
    X_test_m = pd.get_dummies(X_test_m, drop_first=True)
    X_test_m = X_test_m.reindex(columns=design_cols, fill_value=0)
    return X_test_m


def evaluar(modelo, X_test_m: pd.DataFrame, y_test: pd.Series, cutoff: float):
    probs = modelo.predict_proba(X_test_m)[:, 1]
    y_pred = (probs >= cutoff).astype(int)

    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()

    acc = float(accuracy_score(y_test, y_pred))
    sens = float(tp / (tp + fn)) if (tp + fn) > 0 else np.nan
    esp  = float(tn / (tn + fp)) if (tn + fp) > 0 else np.nan
    youden = float(sens + esp - 1) if (not np.isnan(sens) and not np.isnan(esp)) else np.nan

    return {
        "cutoff": float(cutoff),
        "accuracy": acc,
        "sensibilidad": sens,
        "especificidad": esp,
        "youden": youden,
        "tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp),
        "n_parametros_aprox": int(X_test_m.shape[1] + 1)
    }, probs, y_pred, cm


def top_coeficientes(modelo, feature_names, k=15):
    if not hasattr(modelo, "coef_"):
        return pd.DataFrame(columns=["Variable", "Coeficiente", "AbsCoeficiente"])

    coefs = pd.Series(modelo.coef_[0], index=feature_names, name="Coeficiente")
    df = coefs.to_frame()
    df["AbsCoeficiente"] = df["Coeficiente"].abs()
    df = df.sort_values("AbsCoeficiente", ascending=False).head(k).reset_index()
    df = df.rename(columns={"index": "Variable"})
    return df


def main():
    # 1) Cargar modelo final y metadata
    modelo, meta = cargar_modelo_y_meta()

    # 2) Cargar test
    X_test, y_test = cargar_test(meta)

    # 3) Construir matriz de diseño alineada
    X_test_m = construir_X_test_m(X_test, meta)

    # 4) Evaluar con punto de corte final (Youden)
    cutoff = float(meta.get("punto_corte", 0.31))
    metricas, probs, y_pred, cm = evaluar(modelo, X_test_m, y_test, cutoff)

    # 5) Guardar métricas (tabla de 1 fila)
    df_metricas = pd.DataFrame([{
        "PuntoCorte": metricas["cutoff"],
        "Accuracy": metricas["accuracy"],
        "Sensibilidad": metricas["sensibilidad"],
        "Especificidad": metricas["especificidad"],
        "Youden": metricas["youden"],
        "N_parametros_aprox": metricas["n_parametros_aprox"]
    }])
    df_metricas.to_csv(TABLA_METRICAS_FILE, index=False)

    # 6) Guardar matriz de confusión
    df_cm = pd.DataFrame(cm, index=["Real_0", "Real_1"], columns=["Pred_0", "Pred_1"])
    df_cm.to_csv(TABLA_MATRIZ_CONF_FILE)

    # 7) Guardar top coeficientes
    df_top = top_coeficientes(modelo, X_test_m.columns.tolist(), k=TOP_K_COEF)
    df_top.to_csv(TABLA_TOP_COEF_FILE, index=False)

    # 8) Guardar resumen JSON (para informe)
    resumen = {
        "modelo_ganador": meta.get("modelo_ganador", "N/A"),
        "criterio_punto_corte": meta.get("criterio_punto_corte", "Youden"),
        "punto_corte": metricas["cutoff"],
        "metricas": {
            "accuracy": metricas["accuracy"],
            "sensibilidad": metricas["sensibilidad"],
            "especificidad": metricas["especificidad"],
            "youden": metricas["youden"],
        },
        "matriz_confusion": {
            "tn": metricas["tn"], "fp": metricas["fp"],
            "fn": metricas["fn"], "tp": metricas["tp"]
        },
        "n_parametros_aprox": metricas["n_parametros_aprox"],
        "salidas": {
            "metricas_csv": str(TABLA_METRICAS_FILE),
            "matriz_confusion_csv": str(TABLA_MATRIZ_CONF_FILE),
            "top_coeficientes_csv": str(TABLA_TOP_COEF_FILE),
        }
    }

    with open(RESUMEN_FILE, "w", encoding="utf-8") as f:
        json.dump(resumen, f, ensure_ascii=False, indent=2)

    print("[OK] HU-3.5 completada.")
    print(f"[OK] Métricas: {TABLA_METRICAS_FILE.name}")
    print(f"[OK] Matriz confusión: {TABLA_MATRIZ_CONF_FILE.name}")
    print(f"[OK] Top coeficientes: {TABLA_TOP_COEF_FILE.name}")
    print(f"[OK] Resumen: {RESUMEN_FILE.name}")
    print("\n[INFO] Métricas finales:")
    print(df_metricas.to_string(index=False))
    print("\n[INFO] Matriz de confusión:")
    print(df_cm.to_string())


if __name__ == "__main__":
    main()
