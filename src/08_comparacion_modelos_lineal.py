import matplotlib.pyplot as plt
import pickle
import pandas as pd
import numpy as np

from config import DATA_DIR, TABLES_DIR

# =========================
# Entradas 
# =========================
X_TRAIN_FILE = DATA_DIR / "processed/lineal/X_train_lineal.parquet"
X_TEST_FILE  = DATA_DIR / "processed/lineal/X_test_lineal.parquet"
Y_TRAIN_FILE = DATA_DIR / "processed/lineal/y_train_lineal.parquet"
Y_TEST_FILE  = DATA_DIR / "processed/lineal/y_test_lineal.parquet"

# Modelos HU-2.2
MODELOS_PKL_FILE = DATA_DIR / "processed/lineal/modelos_seleccion_clasica_lineal.pkl"

# =========================
# Salidas (HU-2.3)
# =========================
TABLA_COMP_FILE = TABLES_DIR / "tabla_comparacion_modelos_lineal_train_test.csv"


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Carga train/test (X,y) generados en HU-2.1."""
    X_train = pd.read_parquet(X_TRAIN_FILE)
    X_test = pd.read_parquet(X_TEST_FILE)

    y_train = pd.read_parquet(Y_TRAIN_FILE).iloc[:, 0]
    y_test = pd.read_parquet(Y_TEST_FILE).iloc[:, 0]

    return X_train, X_test, y_train, y_test


def cargar_modelos() -> dict:
    """Carga el diccionario de modelos generado en HU-2.2."""
    with open(MODELOS_PKL_FILE, "rb") as f:
        modelos = pickle.load(f)
    return modelos


def construir_X_test_alineado(out: dict, X_test_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Reconstruye la matriz de diseño de test para que tenga EXACTAMENTE
    las mismas columnas (dummies incluidas) que out["X"] (train).

    Esto evita KeyError en Rsq por columnas que existen en train (dummies)
    pero no en X_test crudo.
    """
    # Matriz de diseño usada en el ajuste (ya dumificada por FuncionesMineria)
    X_train_design = out["X"]
    cols_train = X_train_design.columns

    # Variables seleccionadas por el método
    var_cont = out["Variables"]["cont"]
    var_categ = out["Variables"]["categ"]

    # Subset crudo de test
    X_test_sub = X_test_raw[var_cont + var_categ].copy()

    # Dumificación replicando el procedimiento "de clase"
    if len(var_categ) > 0:
        X_test_sub = pd.get_dummies(X_test_sub, columns=var_categ, drop_first=True)

    # Alinear columnas al train: mismas columnas, mismo orden
    X_test_aligned = X_test_sub.reindex(columns=cols_train, fill_value=0)

    # Forzar numérico
    X_test_aligned = X_test_aligned.apply(pd.to_numeric, errors="coerce").astype(float)

    return X_test_aligned


def comparar_modelos(modelos: dict, X_test: pd.DataFrame, y_test: pd.Series) -> pd.DataFrame:
    """Calcula R2 train/test y nº de parámetros para cada modelo."""
    from FuncionesMineria import Rsq  # funciones de clase

    filas = []
    for nombre, out in modelos.items():
        m = out["Modelo"]

        X_test_aligned = construir_X_test_alineado(out, X_test)

        filas.append(
            {
                "Metodo": nombre,
                "R2_train": float(m.rsquared),
                "R2_test": float(Rsq(m, y_test, X_test_aligned)),
                "N_parametros": int(m.df_model + 1),  # +1 intercepto
                "N_continuas_sel": int(len(out["Variables"]["cont"])),
                "N_categoricas_sel": int(len(out["Variables"]["categ"])),
            }
        )

    tabla = pd.DataFrame(filas).sort_values(by="R2_test", ascending=False)
    return tabla


def main() -> None:
    # Validaciones mínimas
    if not MODELOS_PKL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {MODELOS_PKL_FILE}. Ejecuta antes HU-2.2 para generarlo."
        )

    X_train, X_test, y_train, y_test = cargar_datos()
    modelos = cargar_modelos()

    print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test : X={X_test.shape}, y={y_test.shape}")
    print(f"[INFO] Modelos cargados: {list(modelos.keys())}")

    tabla_comp = comparar_modelos(modelos, X_test, y_test)

    df = tabla_comp.copy()

    # Marcar criterio AIC/BIC a partir del nombre del método
    df["Criterio"] = df["Metodo"].apply(lambda s: "AIC" if "AIC" in s else "BIC")

    # Grafico de complejidad vs R2_test (opcional, pero útil para el informe)
    plt.figure(figsize=(8, 5))
    for crit in ["AIC", "BIC"]:
        sub = tabla_comp[tabla_comp["Metodo"].str.contains(crit)]
        plt.scatter(sub["N_parametros"], sub["R2_test"], s=80, label=crit)
    plt.title(r"Comparación de modelos: complejidad vs $R^2_{test}$ ")
    plt.xlabel("Número de parámetros (complejidad)")
    plt.ylabel(r"$R^2$  en test (rendimiento)")
    plt.legend(title="Criterio de selección")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig("../reports/figures/hu23_modelos_lineal_por_metodo.png", dpi=300)


    TABLA_COMP_FILE.parent.mkdir(parents=True, exist_ok=True)
    tabla_comp.to_csv(TABLA_COMP_FILE, index=False)

    print("[OK] HU-2.3 completada: tabla comparativa train/test generada.")
    print(f"[OK] Guardado: {TABLA_COMP_FILE}")
    print("\nTop resultados (ordenado por R2_test):")
    print(tabla_comp.head(10))


if __name__ == "__main__":
    main()
