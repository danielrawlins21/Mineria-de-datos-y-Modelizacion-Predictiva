import json
import pandas as pd
import numpy as np

from config import DATA_DIR, TABLES_DIR

# ============================================================
# HU-3.3 — Comparación y selección del modelo ganador (Logística)
# ============================================================

# Entrada: tabla resumen generada en HU-3.2
TABLA_HU32_FILE = TABLES_DIR / "logistica_hu_3_2_seleccion_clasica_resumen.csv"

# Salidas: candidatos + ganador + metadata (para informe / siguientes HUs)
TABLA_CANDIDATOS_FILE = TABLES_DIR / "logistica_hu_3_3_modelos_candidatos.csv"
TABLA_GANADOR_FILE    = TABLES_DIR / "logistica_hu_3_3_modelo_ganador.csv"
META_FILE             = TABLES_DIR / "logistica_hu_3_3_modelo_ganador_meta.json"

# Parámetros de decisión (ajustables, pero defendibles)
MAX_CANDIDATOS = 4
UMBRAL_GAP_ACC = 0.05  # estabilidad: |Acc_train - Acc_test| < 0.05
METRICA_PRINCIPAL = "PseudoR2_test"
METRICA_SECUNDARIA = "N_parametros_aprox"  # parsimonia

# NUEVO: tolerancia para tratar pseudoR2_test como "empate" (evitar micro-diferencias)
TOL_PR2 = 1e-4

# Punto de corte provisional usado en HU-3.2 (HU-3.4 lo optimiza)
CUTOFF_PROV = 0.5


def cargar_tabla_hu32() -> pd.DataFrame:
    if not TABLA_HU32_FILE.exists():
        raise FileNotFoundError(
            f"No existe {TABLA_HU32_FILE}. Ejecuta antes HU-3.2 para generar la tabla."
        )
    return pd.read_csv(TABLA_HU32_FILE)


def preparar_tabla(df: pd.DataFrame) -> pd.DataFrame:
    cols_req = {
        "Modelo",
        "PseudoR2_train", "PseudoR2_test",
        "Acc_train@0.5", "Acc_test@0.5",
        "N_parametros_aprox",
    }
    faltan = cols_req - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en la tabla HU-3.2: {sorted(list(faltan))}")

    df = df.copy()
    df["Gap_Acc"] = (df["Acc_train@0.5"] - df["Acc_test@0.5"]).abs()

    # Orden informativo (no determina ganador por sí solo)
    df = df.sort_values(
        by=[METRICA_PRINCIPAL, METRICA_SECUNDARIA],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


def seleccionar_candidatos(df: pd.DataFrame) -> pd.DataFrame:
    estables = df[df["Gap_Acc"] < UMBRAL_GAP_ACC].copy()
    if len(estables) == 0:
        # Evitar bloquear flujo si nadie cumple umbral
        estables = df.copy()

    candidatos = estables.head(MAX_CANDIDATOS).reset_index(drop=True)
    return candidatos


def seleccionar_ganador_con_empates(candidatos: pd.DataFrame) -> pd.Series:
    """
    Selección del ganador con criterio de empates:
      1) Max Acc_test@0.5
      2) Max PseudoR2_test dentro de tolerancia TOL_PR2
      3) Menos parámetros
      4) Preferir BIC (si sigue empate)
    """
    if len(candidatos) == 0:
        raise ValueError("No hay candidatos para seleccionar ganador.")

    top_acc = candidatos["Acc_test@0.5"].max()
    top_pr2 = candidatos["PseudoR2_test"].max()

    # Filtrar por máximos con tolerancia
    empate = candidatos[
        (candidatos["Acc_test@0.5"] >= top_acc - 1e-12) &
        (candidatos["PseudoR2_test"] >= top_pr2 - TOL_PR2)
    ].copy()

    # Si por cualquier motivo queda vacío, caer al mejor por pseudoR2
    if len(empate) == 0:
        empate = candidatos.copy()

    # Desempate por parsimonia
    empate = empate.sort_values(by=["N_parametros_aprox"], ascending=True)

    # Último desempate: preferir BIC
    def pref_bic(nombre: str) -> int:
        return 0 if "BIC" in str(nombre) else 1

    empate["pref_bic"] = empate["Modelo"].apply(pref_bic)
    empate = empate.sort_values(by=["N_parametros_aprox", "pref_bic"], ascending=[True, True])

    ganador = empate.iloc[0]
    return ganador


def guardar_outputs(tabla_ordenada: pd.DataFrame, candidatos: pd.DataFrame, ganador: pd.Series) -> None:
    candidatos.to_csv(TABLA_CANDIDATOS_FILE, index=False)
    ganador.to_frame().T.to_csv(TABLA_GANADOR_FILE, index=False)

    meta = {
        "input_table": str(TABLA_HU32_FILE),
        "params": {
            "max_candidatos": int(MAX_CANDIDATOS),
            "umbral_gap_acc": float(UMBRAL_GAP_ACC),
            "metrica_principal_orden": METRICA_PRINCIPAL,
            "metrica_secundaria_orden": METRICA_SECUNDARIA,
            "tolerancia_pseudoR2": float(TOL_PR2),
            "criterio_ganador": [
                "max Acc_test@0.5",
                f"PseudoR2_test dentro de {TOL_PR2}",
                "min N_parametros_aprox",
                "preferir BIC",
            ],
            "punto_corte_provisional": float(CUTOFF_PROV),
        },
        "n_modelos_hu32": int(len(tabla_ordenada)),
        "n_candidatos": int(len(candidatos)),
        "ganador": ganador.to_dict(),
        "top_resumen": tabla_ordenada.head(6).to_dict(orient="records"),
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    df = cargar_tabla_hu32()
    print(f"[INFO] Tabla HU-3.2 cargada: {TABLA_HU32_FILE.name} | filas={len(df)}")

    tabla_ordenada = preparar_tabla(df)
    print("[INFO] Modelos ordenados por rendimiento en test y parsimonia.")
    print(tabla_ordenada[
        ["Modelo", "PseudoR2_test", "Acc_test@0.5", "Gap_Acc", "N_parametros_aprox"]
    ].head(6).to_string(index=False))

    candidatos = seleccionar_candidatos(tabla_ordenada)
    print(f"[INFO] Candidatos seleccionados (top {MAX_CANDIDATOS} con estabilidad): {len(candidatos)}")
    print(candidatos[
        ["Modelo", "PseudoR2_test", "Acc_test@0.5", "Gap_Acc", "N_parametros_aprox"]
    ].to_string(index=False))

    ganador = seleccionar_ganador_con_empates(candidatos)
    print("\n[OK] Modelo ganador seleccionado (con criterio de empates):")
    print(pd.DataFrame([ganador])[[
        "Modelo", "PseudoR2_test", "Acc_test@0.5", "Gap_Acc", "N_parametros_aprox"
    ]].to_string(index=False))

    guardar_outputs(tabla_ordenada, candidatos, ganador)
    print(f"\n[OK] Guardado: {TABLA_CANDIDATOS_FILE.name}, {TABLA_GANADOR_FILE.name}, {META_FILE.name}")


if __name__ == "__main__":
    main()
