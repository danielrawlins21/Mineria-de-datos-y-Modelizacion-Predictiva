import pickle
import pandas as pd

from config import DATA_DIR, TABLES_DIR

# =========================
# Entradas (artefactos HU-2.1)
# =========================
X_TRAIN_FILE = DATA_DIR / "processed/lineal/X_train_lineal.parquet"
Y_TRAIN_FILE = DATA_DIR / "processed/lineal/y_train_lineal.parquet"

VARS_NUM_FILE = TABLES_DIR / "lineal_vars_numericas.csv"
VARS_CAT_FILE = TABLES_DIR / "lineal_vars_categoricas.csv"

# =========================
# Salidas (HU-2.2)
# =========================
TABLA_MODELOS_FILE = TABLES_DIR / "tabla_modelos_seleccion_clasica_lineal.csv"
MODELOS_PKL_FILE = DATA_DIR / "processed/lineal/modelos_seleccion_clasica_lineal.pkl"


def cargar_artefactos() -> tuple[pd.DataFrame, pd.Series, list[str], list[str]]:
    """Carga X_train, y_train y listas de variables numéricas y categóricas."""
    X_train = pd.read_parquet(X_TRAIN_FILE)

    y_train_df = pd.read_parquet(Y_TRAIN_FILE)
    # y_train guardado como DataFrame de 1 columna en HU-2.1
    y_train = y_train_df.iloc[:, 0]

    vars_num = pd.read_csv(VARS_NUM_FILE).iloc[:, 0].dropna().tolist()
    vars_cat = pd.read_csv(VARS_CAT_FILE).iloc[:, 0].dropna().tolist()

    return X_train, y_train, vars_num, vars_cat


def ajustar_modelos_seleccion_clasica(
    y_train: pd.Series,
    X_train: pd.DataFrame,
    vars_num: list[str],
    vars_cat: list[str],
) -> dict:
    """
    Ajusta 6 modelos (Forward/Backward/Stepwise con AIC/BIC)
    usando las funciones oficiales de clase (FuncionesMineria.py).
    """
    # Import local para que el script no falle si no estás ejecutando HU-2.2 todavía
    # y para dejar claro que esto depende de las funciones "de clase".
    from FuncionesMineria import lm_forward, lm_backward, lm_stepwise  # noqa: WPS433

    # Interacciones prohibidas/NO usadas en esta práctica
    var_interac = []

    modelos = {
        "Forward AIC": lm_forward(y_train, X_train, vars_num, vars_cat, var_interac, metodo="AIC"),
        "Forward BIC": lm_forward(y_train, X_train, vars_num, vars_cat, var_interac, metodo="BIC"),
        "Backward AIC": lm_backward(y_train, X_train, vars_num, vars_cat, var_interac, metodo="AIC"),
        "Backward BIC": lm_backward(y_train, X_train, vars_num, vars_cat, var_interac, metodo="BIC"),
        "Stepwise AIC": lm_stepwise(y_train, X_train, vars_num, vars_cat, var_interac, metodo="AIC"),
        "Stepwise BIC": lm_stepwise(y_train, X_train, vars_num, vars_cat, var_interac, metodo="BIC"),
    }

    return modelos


def tabla_resumen_modelos(modelos: dict) -> pd.DataFrame:
    """
    Construye una tabla resumen (solo TRAIN, como corresponde a HU-2.2):
    - Método
    - R2_train
    - Nº parámetros
    - Nº vars continuas / categóricas seleccionadas
    """
    filas = []

    for nombre, out in modelos.items():
        modelo = out["Modelo"]
        var_cont_sel = out.get("VariablesContinuas", [])
        var_cat_sel = out.get("VariablesCategoricas", [])
        var_inter_sel = out.get("VariablesInteraccion", [])

        filas.append(
            {
                "Metodo": nombre,
                "R2_train": float(modelo.rsquared),
                "N_parametros": int(modelo.df_model + 1),  # +1 por intercept
                "N_continuas_sel": int(len(var_cont_sel)),
                "N_categoricas_sel": int(len(var_cat_sel)),
                "N_interacciones_sel": int(len(var_inter_sel)),
            }
        )

    return pd.DataFrame(filas).sort_values(by="R2_train", ascending=False)


def guardar_modelos(modelos: dict) -> None:
    MODELOS_PKL_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(MODELOS_PKL_FILE, "wb") as f:
        pickle.dump(modelos, f)


def main() -> None:
    X_train, y_train, vars_num, vars_cat = cargar_artefactos()

    print(f"[INFO] X_train shape={X_train.shape} | y_train shape={y_train.shape}")
    print(f"[INFO] #vars_num={len(vars_num)} | #vars_cat={len(vars_cat)}")

    modelos = ajustar_modelos_seleccion_clasica(y_train, X_train, vars_num, vars_cat)
    tabla = tabla_resumen_modelos(modelos)

    tabla.to_csv(TABLA_MODELOS_FILE, index=False)
    guardar_modelos(modelos)

    print("[OK] HU-2.2 completada: modelos de selección clásica ajustados.")
    print(f"[OK] Tabla guardada en: {TABLA_MODELOS_FILE}")
    print(f"[OK] Modelos serializados en: {MODELOS_PKL_FILE}")
    print("\nTop 6 (train):")
    print(tabla)


if __name__ == "__main__":
    main()
