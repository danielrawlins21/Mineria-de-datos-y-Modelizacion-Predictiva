import json
import numpy as np
import pandas as pd
from pathlib import Path

from config import DATA_DIR, TABLES_DIR

# ============================================================
# HU-3.2 — Selección clásica de variables (Regresión logística)
# + Guardado de parámetros/modelos candidatos para HU-3.3/3.4/3.5
# ============================================================

# Entradas (artefactos HU-3.1)
IN_DIR = DATA_DIR / "processed/logistica"
X_TRAIN_FILE = IN_DIR / "X_train_logistica.parquet"
X_TEST_FILE  = IN_DIR / "X_test_logistica.parquet"
Y_TRAIN_FILE = IN_DIR / "y_train_logistica.parquet"
Y_TEST_FILE  = IN_DIR / "y_test_logistica.parquet"

# Salidas (tablas/metadata para informe)
OUT_DIR = DATA_DIR / "results/logistica"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TABLA_RESUMEN_FILE = TABLES_DIR / "logistica_hu_3_2_seleccion_clasica_resumen.csv"
META_FILE          = TABLES_DIR / "logistica_hu_3_2_seleccion_clasica_meta.json"

# Guardado de parámetros para futuras secciones
PARAMS_DIR = OUT_DIR / "parametros"
PARAMS_DIR.mkdir(parents=True, exist_ok=True)

# Parámetros
CUTOFF_PROV = 0.50   # OJO: el punto de corte óptimo se determina en HU-3.4
TOP_K_CANDIDATOS = 4 # nº modelos candidatos que persistimos (para HU-3.3+)
EPS = 1e-12


# ------------------------------------------------------------
# Funciones del curso (reutilizadas)
# ------------------------------------------------------------
from FuncionesMineria import (
    crear_data_modelo,
    glm_forward, glm_backward, glm_stepwise,
    pseudoR2
)


def cargar_splits():
    X_train = pd.read_parquet(X_TRAIN_FILE)
    X_test  = pd.read_parquet(X_TEST_FILE)
    y_train = pd.read_parquet(Y_TRAIN_FILE).iloc[:, 0].astype(int)
    y_test  = pd.read_parquet(Y_TEST_FILE).iloc[:, 0].astype(int)
    return X_train, X_test, y_train, y_test


def separar_tipos(X: pd.DataFrame):
    var_cont = X.select_dtypes(include=[np.number]).columns.tolist()
    var_categ = X.select_dtypes(exclude=[np.number]).columns.tolist()
    return var_cont, var_categ


def design_test_alineado(X_test: pd.DataFrame, modelo_dict: dict) -> pd.DataFrame:
    """
    Construye la matriz de diseño de test con las variables del modelo y alinea columnas con train.
    modelo_dict viene de glm_forward/backward/stepwise.
    """
    cont = modelo_dict["Variables"]["cont"]
    categ = modelo_dict["Variables"]["categ"]

    Xte = crear_data_modelo(X_test, cont, categ, [])  # sin interacciones
    Xte = pd.get_dummies(Xte, drop_first=True)

    # Alinear con columnas usadas en entrenamiento por el modelo
    Xtr_cols = modelo_dict["X"].columns
    Xte = Xte.reindex(columns=Xtr_cols, fill_value=0)
    return Xte


def evaluar_modelo(nombre: str, modelo_dict: dict, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series):
    """
    Devuelve métricas comparables para tabla resumen.
    - pseudoR2 con función del curso.
    - accuracy a cutoff provisional 0.5 (HU-3.4 optimiza).
    """
    modelo = modelo_dict["Modelo"]
    Xtr = modelo_dict["X"]
    Xte = design_test_alineado(X_test, modelo_dict)

    # Pseudo-R2 (curso)
    pr2_train = float(pseudoR2(modelo, Xtr, y_train))
    pr2_test  = float(pseudoR2(modelo, Xte, y_test))

    # Predicción probabilística y accuracy provisional
    p_train = modelo.predict_proba(Xtr)[:, 1]
    p_test  = modelo.predict_proba(Xte)[:, 1]

    acc_train = float(((p_train >= CUTOFF_PROV).astype(int) == y_train.values).mean())
    acc_test  = float(((p_test >= CUTOFF_PROV).astype(int) == y_test.values).mean())

    # Nº parámetros aproximado (nº columnas tras dummies + intercepto)
    n_param = int(Xtr.shape[1] + 1)

    return {
        "Modelo": nombre,
        "PseudoR2_train": pr2_train,
        "PseudoR2_test": pr2_test,
        "Acc_train@0.5": acc_train,
        "Acc_test@0.5": acc_test,
        "N_parametros_aprox": n_param,
        "N_vars_cont": int(len(modelo_dict["Variables"]["cont"])),
        "N_vars_categ": int(len(modelo_dict["Variables"]["categ"])),
        # Estas listas no van en la tabla “compacta”, pero sí en metadata:
        "Vars_cont": modelo_dict["Variables"]["cont"],
        "Vars_categ": modelo_dict["Variables"]["categ"],
    }


def extraer_parametros_logit(modelo_dict: dict) -> pd.DataFrame:
    """
    Extrae coeficientes del modelo logístico (sklearn LogisticRegression dentro de FuncionesMineria)
    y los asocia a las columnas efectivas (tras dummies) usadas en entrenamiento.
    """
    modelo = modelo_dict["Modelo"]
    X_cols = list(modelo_dict["X"].columns)

    coef = modelo.coef_.ravel()
    intercept = float(modelo.intercept_.ravel()[0]) if hasattr(modelo, "intercept_") else np.nan

    df = pd.DataFrame({
        "Variable": X_cols,
        "Coeficiente": coef
    })
    df.insert(1, "Intercepto", intercept)  # repetido por fila para facilitar lectura/merge
    return df


def guardar_json(path: Path, obj: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)


def main():
    # 1) Carga splits
    X_train, X_test, y_train, y_test = cargar_splits()
    print(f"[INFO] X_train={X_train.shape} | X_test={X_test.shape}")
    print(f"[INFO] Balance y_train: {y_train.value_counts(normalize=True).round(3).to_dict()}")

    # 2) Separar tipos (sin transformaciones/interacciones)
    var_cont, var_categ = separar_tipos(X_train)
    print(f"[INFO] #var_cont={len(var_cont)} | #var_categ={len(var_categ)}")

    var_interac = []  # PROHIBIDO usar interacciones en esta práctica

    # 3) Ejecutar selección clásica (6 modelos)
    print("[INFO] Ejecutando selección clásica logística... (puede tardar si hay muchas dummies)")
    modelos = {}

    modelos["Forward_AIC"]  = glm_forward(y_train, X_train, var_cont, var_categ, var_interac, metodo="AIC")
    modelos["Forward_BIC"]  = glm_forward(y_train, X_train, var_cont, var_categ, var_interac, metodo="BIC")

    modelos["Backward_AIC"] = glm_backward(y_train, X_train, var_cont, var_categ, var_interac, metodo="AIC")
    modelos["Backward_BIC"] = glm_backward(y_train, X_train, var_cont, var_categ, var_interac, metodo="BIC")

    modelos["Stepwise_AIC"] = glm_stepwise(y_train, X_train, var_cont, var_categ, var_interac, metodo="AIC")
    modelos["Stepwise_BIC"] = glm_stepwise(y_train, X_train, var_cont, var_categ, var_interac, metodo="BIC")

    # 4) Evaluación y resumen
    filas = []
    for k, m in modelos.items():
        nombre_pretty = k.replace("_", " ")
        filas.append(evaluar_modelo(nombre_pretty, m, y_train, X_test, y_test))

    resumen = pd.DataFrame(filas)

    # Orden sugerido: rendimiento en test (pseudoR2), y luego parsimonia
    resumen_orden = resumen.sort_values(
        by=["PseudoR2_test", "N_parametros_aprox"],
        ascending=[False, True]
    ).reset_index(drop=True)

    # 5) Guardar tabla compacta para el informe
    cols_out = [
        "Modelo", "PseudoR2_train", "PseudoR2_test",
        "Acc_train@0.5", "Acc_test@0.5",
        "N_parametros_aprox", "N_vars_cont", "N_vars_categ"
    ]
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    resumen_orden[cols_out].to_csv(TABLA_RESUMEN_FILE, index=False)

    # 6) Seleccionar TOP-K candidatos y guardar parámetros/estructura
    #    (para HU-3.3/3.4/3.5 sin rehacer ni perder trazabilidad)
    top_k = min(TOP_K_CANDIDATOS, len(resumen_orden))
    candidatos_df = resumen_orden.head(top_k).copy()

    candidatos_meta = {}
    for _, row in candidatos_df.iterrows():
        # Recuperar la clave original de "modelos" desde el nombre bonito
        # "Stepwise BIC" -> "Stepwise_BIC"
        key = row["Modelo"].replace(" ", "_")
        modelo_dict = modelos[key]

        # Guardar coeficientes
        coef_df = extraer_parametros_logit(modelo_dict)
        coef_path = PARAMS_DIR / f"coeficientes_{key}.csv"
        coef_df.to_csv(coef_path, index=False)

        # Guardar variables seleccionadas (cont/categ) y columnas de diseño
        candidatos_meta[key] = {
            "pretty_name": row["Modelo"],
            "pseudoR2_test": float(row["PseudoR2_test"]),
            "acc_test_cutoff_0_5": float(row["Acc_test@0.5"]),
            "n_parametros_aprox": int(row["N_parametros_aprox"]),
            "variables": {
                "cont": list(modelo_dict["Variables"]["cont"]),
                "categ": list(modelo_dict["Variables"]["categ"]),
            },
            "design_columns": list(modelo_dict["X"].columns),
            "coeficientes_csv": str(coef_path),
        }

    # 7) Metadata global (incluye resumen completo + candidatos persistidos)
    meta = {
        "input_files": {
            "X_train": str(X_TRAIN_FILE),
            "X_test": str(X_TEST_FILE),
            "y_train": str(Y_TRAIN_FILE),
            "y_test": str(Y_TEST_FILE),
        },
        "cutoff_provisional": float(CUTOFF_PROV),
        "n_train": int(len(X_train)),
        "n_test": int(len(X_test)),
        "n_vars_cont_total": int(len(var_cont)),
        "n_vars_categ_total": int(len(var_categ)),
        "tabla_resumen_file": str(TABLA_RESUMEN_FILE),
        "resumen_completo": resumen_orden.to_dict(orient="records"),
        "top_k_candidatos": int(top_k),
        "candidatos": candidatos_meta,
        "parametros_dir": str(PARAMS_DIR),
        "nota": (
            "Los coeficientes guardados se usarán en HU-3.5 (interpretación). "
            "El punto de corte óptimo se calcula en HU-3.4; aquí se reporta accuracy@0.5 solo como referencia."
        ),
    }
    guardar_json(META_FILE, meta)

    print("[OK] HU-3.2 completada.")
    print(f"[OK] Tabla resumen: {TABLA_RESUMEN_FILE.name}")
    print(f"[OK] Metadata: {META_FILE.name}")
    print(f"[OK] Coeficientes candidatos en: {PARAMS_DIR}")
    print("\n[INFO] Top candidatos (para HU-3.3):")
    print(candidatos_df[cols_out].to_string(index=False))


if __name__ == "__main__":
    main()
