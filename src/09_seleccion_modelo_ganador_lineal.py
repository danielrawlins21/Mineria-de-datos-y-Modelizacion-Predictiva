import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import KFold

from config import DATA_DIR, TABLES_DIR, FIG_DIR
from FuncionesMineria import validacion_cruzada_lm

# =========================
# Entradas (HU-2.1 / HU-2.2)
# =========================
X_TRAIN_FILE = DATA_DIR / "processed/lineal/X_train_lineal.parquet"
Y_TRAIN_FILE = DATA_DIR / "processed/lineal/y_train_lineal.parquet"
MODELOS_PKL_FILE = DATA_DIR / "processed/lineal/modelos_seleccion_clasica_lineal.pkl"

# =========================
# Salidas (HU-2.4)
# =========================
RESULTS_FILE = TABLES_DIR / "hu24_resultados_cv_repetida_4_modelos.csv"
SUMMARY_FILE = TABLES_DIR / "hu24_resumen_cv_repetida_4_modelos.csv"
DECISION_FILE = TABLES_DIR / "hu24_decision_modelo_ganador.json"


FIG_FILE = FIG_DIR / "hu24_boxplot_cv_4_modelos.png"

# =========================
# Parámetros CV repetida
# =========================
N_SPLITS = 5
N_REPS = 20
BASE_SEED = 123

# Empate técnico: si la diferencia de medias es menor que este umbral,
# se favorece un modelo BIC por parsimonia.
UMBRAL_EMPATE = 0.0002

# Modelos candidatos (claves exactas del diccionario)
CANDIDATOS_KEYS = [
    "Stepwise AIC",
    "Backward AIC",
    "Stepwise BIC",
    "Backward BIC",
]


def cargar_train() -> tuple[pd.DataFrame, pd.Series]:
    X_train = pd.read_parquet(X_TRAIN_FILE)
    y_train = pd.read_parquet(Y_TRAIN_FILE).iloc[:, 0]
    return X_train, y_train


def cargar_modelos() -> dict:
    if not MODELOS_PKL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {MODELOS_PKL_FILE}. Ejecuta antes HU-2.2."
        )
    with open(MODELOS_PKL_FILE, "rb") as f:
        return pickle.load(f)


def repeated_cv_scores(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    modelo_out: dict,
    n_splits: int,
    n_reps: int,
    base_seed: int,
) -> list[float]:
    """Valida con KFold (shuffle=True) repetido n_reps y devuelve todos los R2."""
    var_cont = modelo_out["Variables"]["cont"]
    var_categ = modelo_out["Variables"]["categ"]
    var_inter = modelo_out["Variables"].get("inter", [])

    all_scores: list[float] = []
    for rep in range(n_reps):
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=base_seed + rep)
        scores = validacion_cruzada_lm(kf, X_train, y_train, var_cont, var_categ, var_inter)
        all_scores.extend([float(s) for s in scores])

    return all_scores


def construir_results_df(scores_dict: dict[str, list[float]]) -> pd.DataFrame:
    frames = []
    for nombre, scores in scores_dict.items():
        frames.append(pd.DataFrame({"Modelo": nombre, "Rsquared": scores}))
    return pd.concat(frames, ignore_index=True)


def resumen_estadistico(results: pd.DataFrame) -> pd.DataFrame:
    return (
        results.groupby("Modelo")["Rsquared"]
        .agg(["count", "mean", "std", "median", "min", "max"])
        .sort_values(by="mean", ascending=False)
        .round(6)
    )


def elegir_ganador(summary: pd.DataFrame, umbral: float) -> tuple[str, pd.DataFrame]:
    """
    Regla:
    1) modelo con mayor media
    2) si hay empate técnico (diferencia <= umbral), priorizar BIC si está dentro del grupo cercano
    """
    best_model = summary.index[0]
    best_mean = float(summary.iloc[0]["mean"])

    near = summary[(best_mean - summary["mean"]) <= umbral].copy()

    if len(near) == 1:
        ganador = best_model
    else:
        bic_near = [m for m in near.index if "BIC" in m]
        ganador = bic_near[0] if len(bic_near) > 0 else best_model

    return ganador, near


def guardar_decision(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    import json
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def plot_boxplot(results: pd.DataFrame) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9, 5))
    results.boxplot(column="Rsquared", by="Modelo", rot=20)
    plt.suptitle("")
    plt.title(f"Validación cruzada repetida ({N_SPLITS}-fold x {N_REPS} repeticiones) — Modelos candidatos")
    plt.ylabel(r"$R^2$ (CV)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(FIG_FILE, dpi=300)
    plt.close()


def main() -> None:
    X_train, y_train = cargar_train()
    modelos = cargar_modelos()

    # Verificar que existan las claves de candidatos
    faltan = [k for k in CANDIDATOS_KEYS if k not in modelos]
    if len(faltan) > 0:
        raise KeyError(
            f"Faltan modelos candidatos en el diccionario: {faltan}\n"
            f"Disponibles: {list(modelos.keys())}"
        )

    candidatos = {k: modelos[k] for k in CANDIDATOS_KEYS}

    print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Candidatos: {list(candidatos.keys())}")
    print(f"[INFO] CV: n_splits={N_SPLITS}, n_reps={N_REPS}, base_seed={BASE_SEED}")

    # Ejecutar CV repetida
    scores_dict: dict[str, list[float]] = {}
    for nombre, out in candidatos.items():
        scores = repeated_cv_scores(X_train, y_train, out, N_SPLITS, N_REPS, BASE_SEED)
        scores_dict[nombre] = scores
        print(f"[OK] {nombre}: {len(scores)} scores")

    # Construir outputs
    results = construir_results_df(scores_dict)
    summary = resumen_estadistico(results)

    ganador, near = elegir_ganador(summary, UMBRAL_EMPATE)

    # Guardar outputs
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    results.to_csv(RESULTS_FILE, index=False)
    summary.to_csv(SUMMARY_FILE)

    plot_boxplot(results)

    decision_payload = {
        "ganador": ganador,
        "umbral_empate": UMBRAL_EMPATE,
        "best_mean": float(summary.iloc[0]["mean"]),
        "near_models": near.reset_index().to_dict(orient="records"),
        "cv": {
            "n_splits": N_SPLITS,
            "n_reps": N_REPS,
            "base_seed": BASE_SEED,
        },
        "outputs": {
            "results_csv": str(RESULTS_FILE),
            "summary_csv": str(SUMMARY_FILE),
            "figure": str(FIG_FILE),
        },
    }
    guardar_decision(DECISION_FILE, decision_payload)

    print("\n[OK] HU-2.4 completada.")
    print(f"[OK] Resultados: {RESULTS_FILE}")
    print(f"[OK] Resumen:    {SUMMARY_FILE}")
    print(f"[OK] Figura:     {FIG_FILE}")
    print(f"[OK] Decisión:   {DECISION_FILE}")
    print("\nResumen (ordenado por media de R2 CV):")
    print(summary)
    print("\nModelos cerca del mejor (<= umbral):")
    print(near)
    print(f"\nGANADOR (según regla): {ganador}")


if __name__ == "__main__":
    main()
