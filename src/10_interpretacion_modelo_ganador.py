import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm

from sklearn.metrics import mean_squared_error, mean_absolute_error

from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson

from config import DATA_DIR, TABLES_DIR, FIG_DIR
from FuncionesMineria import modelEffectSizes

# =========================
# Entradas
# =========================
X_TRAIN_FILE = DATA_DIR / "processed/lineal/X_train_lineal.parquet"
X_TEST_FILE = DATA_DIR / "processed/lineal/X_test_lineal.parquet"
Y_TRAIN_FILE = DATA_DIR / "processed/lineal/y_train_lineal.parquet"
Y_TEST_FILE = DATA_DIR / "processed/lineal/y_test_lineal.parquet"

MODELOS_PKL_FILE = DATA_DIR / "processed/lineal/modelos_seleccion_clasica_lineal.pkl"

# Dataset base (sin dummies) para modelEffectSizes.
# Ajusta este nombre si tu archivo se llama distinto.
DATOS_BASE_FILE = DATA_DIR / "interim/datos_dep_alfa.parquet"

# Modelo ganador
GANADOR_NOMBRE = "Backward AIC"

# =========================
# Salidas
# =========================

RESID_FIG = FIG_DIR / "hu25_residuos_vs_ajustados.png"
QQPLOT_FIG = FIG_DIR / "hu25_qqplot_residuos.png"
IMP_FIG = FIG_DIR / "hu25_importancia_variables.png"

SUMMARY_TXT = TABLES_DIR / "hu25_summary_modelo_ganador.txt"
METRICAS_CSV = TABLES_DIR / "Rendimiento Modelo Ganador.csv"
COEFS_CSV = TABLES_DIR / "Coefs.csv"
DIAGNOSTICOS_CSV = TABLES_DIR / "hu25_diagnosticos.csv"
INTERPRETACION_CSV = TABLES_DIR / "hu26_interpretacion_coeficientes.csv"
IMPORTANCIA_CSV = TABLES_DIR / "hu25_importancia_variables.csv"


def cargar_datos() -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    X_train = pd.read_parquet(X_TRAIN_FILE)
    X_test = pd.read_parquet(X_TEST_FILE)
    y_train = pd.read_parquet(Y_TRAIN_FILE).iloc[:, 0]
    y_test = pd.read_parquet(Y_TEST_FILE).iloc[:, 0]
    return X_train, X_test, y_train, y_test


def cargar_modelo_ganador() -> dict:
    if not MODELOS_PKL_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {MODELOS_PKL_FILE}. Ejecuta antes HU-2.2."
        )
    with open(MODELOS_PKL_FILE, "rb") as f:
        modelos = pickle.load(f)

    if GANADOR_NOMBRE not in modelos:
        raise KeyError(
            f"No se encontró el modelo '{GANADOR_NOMBRE}' en el diccionario. "
            f"Disponibles: {list(modelos.keys())}"
        )

    return modelos[GANADOR_NOMBRE]


def guardar_summary(m) -> None:
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    with open(SUMMARY_TXT, "w", encoding="utf-8") as f:
        f.write(str(m.summary()))


def alinear_X_test_al_modelo(m, X_test_raw: pd.DataFrame) -> pd.DataFrame:
    """
    Alinea X_test a las columnas del modelo (dummies incluidas) para poder predecir.
    """
    cols_modelo = m.model.exog_names[1:]  # excluye 'const'
    X_test_aligned = X_test_raw.reindex(columns=cols_modelo, fill_value=0)
    X_test_aligned = X_test_aligned.apply(pd.to_numeric, errors="coerce").astype(float)
    return X_test_aligned


def calcular_metricas(m, X_test_raw, y_train, y_test) -> pd.DataFrame:
    n_param = int(m.df_model + 1)
    r2_train = float(m.rsquared)

    X_test_aligned = alinear_X_test_al_modelo(m, X_test_raw)

    # Predicciones
    y_pred_train = m.predict(m.model.exog)  # ya tiene const
    y_pred_test = m.predict(sm.add_constant(X_test_aligned, has_constant="add"))

    # Métricas
    rmse_train = mean_squared_error(y_train, y_pred_train)
    rmse_test = mean_squared_error(y_test, y_pred_test)
    mae_train = mean_absolute_error(y_train, y_pred_train)
    mae_test = mean_absolute_error(y_test, y_pred_test)

    # R² test manual
    sst = float(np.sum((y_test - y_test.mean()) ** 2))
    sse = float(np.sum((y_test - y_pred_test) ** 2))
    r2_test = float(1 - sse / sst)

    df = pd.DataFrame([{
        "Modelo": GANADOR_NOMBRE,
        "N_parametros": n_param,
        "R2_train": r2_train,
        "R2_test": r2_test,
        "RMSE_train": rmse_train,
        "RMSE_test": rmse_test,
        "MAE_train": mae_train,
        "MAE_test": mae_test,
    }])
    return df


def exportar_coeficientes(m) -> pd.DataFrame:
    coefs = pd.DataFrame({
        "variable": m.params.index,
        "coef": m.params.values,
        "std_err": m.bse.values,
        "t": m.tvalues.values,
        "p_value": m.pvalues.values,
    }).sort_values("p_value", ascending=True)

    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    coefs.to_csv(COEFS_CSV, index=False)
    return coefs


def graficos_residuos(m) -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)

    resid = m.resid
    fitted = m.fittedvalues

    # Residuos vs Ajustados
    plt.figure(figsize=(7, 5))
    plt.scatter(fitted, resid)
    plt.axhline(0)
    plt.xlabel("Valores ajustados")
    plt.ylabel("Residuos")
    plt.title("Residuos vs Ajustados")
    plt.tight_layout()
    plt.savefig(RESID_FIG, dpi=300)
    plt.close()

    # QQ-Plot
    fig = sm.qqplot(resid, line="45")
    plt.title("QQ-Plot de residuos")
    plt.tight_layout()
    plt.savefig(QQPLOT_FIG, dpi=300)
    plt.close()


def diagnosticos(m) -> pd.DataFrame:
    resid = m.resid

    # Durbin-Watson
    dw = float(durbin_watson(resid))

    # Breusch-Pagan
    bp = het_breuschpagan(resid, m.model.exog)
    bp_labels = ["LM_stat", "LM_pvalue", "F_stat", "F_pvalue"]
    bp_dict = {k: float(v) for k, v in zip(bp_labels, bp)}

    out = {
        "Modelo": GANADOR_NOMBRE,
        "Durbin_Watson": dw,
        **bp_dict,
    }

    df = pd.DataFrame([out])
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(DIAGNOSTICOS_CSV, index=False)
    return df


def importancia_variables(modelo_out, X_train_idx, y_train: pd.Series) -> pd.DataFrame:
    """
    Calcula importance (aportación marginal al R2) y genera figura.
    Requiere datos base sin dummies.
    """
    if not DATOS_BASE_FILE.exists():
        raise FileNotFoundError(
            f"No se encontró {DATOS_BASE_FILE}. "
            f"Ajusta DATOS_BASE_FILE al nombre real de tu dataset base (sin dummies)."
        )

    datos_base = pd.read_parquet(DATOS_BASE_FILE)

    var_cont = modelo_out["Variables"]["cont"]
    var_categ = modelo_out["Variables"]["categ"]
    var_inter = modelo_out["Variables"].get("inter", [])

    imp = modelEffectSizes(
        modelo_out,
        y_train,
        datos_base.loc[X_train_idx],
        var_cont,
        var_categ,
        var_inter,
    )

    # Guardar CSV
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    imp.to_csv(IMPORTANCIA_CSV, index=False)

    # Figura (top variables)
    top = imp.sort_values(by="R2", ascending=True)

    FIG_DIR.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 5))
    plt.barh(top["Variables"], top["R2"])
    plt.xlabel("Aportación al $R^2$")
    plt.ylabel("Variable")
    plt.title("Importancia de variables (aportación marginal al $R^2$)")
    plt.tight_layout()
    plt.savefig(IMP_FIG, dpi=300)
    plt.close()

    return imp


def exportar_interpretacion(coefs: pd.DataFrame) -> pd.DataFrame:
    """
    Exporta coeficientes de una variable continua y una categórica para HU-2.6.
    Continuo: SUPERFICIE
    Categórico: CCAA_Cataluña (si existe)
    """
    filas = []

    # Continua: SUPERFICIE
    fila_sup = coefs[coefs["variable"] == "SUPERFICIE"]
    if not fila_sup.empty:
        r = fila_sup.iloc[0].to_dict()
        r["tipo"] = "continua"
        filas.append(r)

    # Categórica: CCAA_Cataluña
    fila_cat = coefs[coefs["variable"] == "CCAA_Cataluña"]
    if not fila_cat.empty:
        r = fila_cat.iloc[0].to_dict()
        r["tipo"] = "categórica (dummy vs referencia)"
        filas.append(r)

    df = pd.DataFrame(filas)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(INTERPRETACION_CSV, index=False)
    return df


def main() -> None:
    X_train, X_test, y_train, y_test = cargar_datos()
    modelo_out = cargar_modelo_ganador()
    m = modelo_out["Modelo"]

    print(f"[INFO] Modelo ganador: {GANADOR_NOMBRE}")
    print(f"[INFO] Train: X={X_train.shape}, y={y_train.shape}")
    print(f"[INFO] Test : X={X_test.shape}, y={y_test.shape}")

    # 1) Summary
    guardar_summary(m)
    print(f"[OK] Summary guardado en: {SUMMARY_TXT}")

    # 2) Métricas
    metricas = calcular_metricas(m, X_test, y_train, y_test)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)
    metricas.to_csv(METRICAS_CSV, index=False)
    print(f"[OK] Métricas guardadas en: {METRICAS_CSV}")
    print(metricas)

    # 3) Coeficientes
    coefs = exportar_coeficientes(m)
    print(f"[OK] Coeficientes guardados en: {COEFS_CSV}")

    # 4) Gráficos residuos
    graficos_residuos(m)
    print(f"[OK] Figuras guardadas: {RESID_FIG} y {QQPLOT_FIG}")

    # 5) Diagnósticos
    diag = diagnosticos(m)
    print(f"[OK] Diagnósticos guardados en: {DIAGNOSTICOS_CSV}")
    print(diag)

    # 6) Importancia de variables (requiere datos base sin dummies)
    try:
        imp = importancia_variables(modelo_out, X_train.index, y_train)
        print(f"[OK] Importancia guardada en: {IMPORTANCIA_CSV}")
        print(f"[OK] Figura importancia: {IMP_FIG}")
        print(imp.tail(10))
    except FileNotFoundError as e:
        print(f"[WARN] No se pudo calcular importancia: {e}")

    # 7) Export interpretación (HU-2.6)
    interp = exportar_interpretacion(coefs)
    print(f"[OK] Interpretación (HU-2.6) guardada en: {INTERPRETACION_CSV}")
    print(interp)


if __name__ == "__main__":
    main()
