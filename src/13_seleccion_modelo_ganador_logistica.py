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


def cargar_tabla_hu32() -> pd.DataFrame:
    if not TABLA_HU32_FILE.exists():
        raise FileNotFoundError(
            f"No existe {TABLA_HU32_FILE}. Ejecuta antes HU-3.2 para generar la tabla."
        )
    df = pd.read_csv(TABLA_HU32_FILE)
    return df


def preparar_tabla(df: pd.DataFrame) -> pd.DataFrame:
    # Asegurar columnas esperadas (mínimas)
    cols_req = {
        "Modelo",
        "PseudoR2_train", "PseudoR2_test",
        "Acc_train@0.5", "Acc_test@0.5",
        "N_parametros_aprox",
    }
    faltan = cols_req - set(df.columns)
    if faltan:
        raise ValueError(f"Faltan columnas en la tabla HU-3.2: {sorted(list(faltan))}")

    # Gap de estabilidad
    df = df.copy()
    df["Gap_Acc"] = (df["Acc_train@0.5"] - df["Acc_test@0.5"]).abs()

    # Orden principal: mejor desempeño en test + más parsimonioso
    df = df.sort_values(
        by=[METRICA_PRINCIPAL, METRICA_SECUNDARIA],
        ascending=[False, True],
    ).reset_index(drop=True)

    return df


def seleccionar_candidatos(df: pd.DataFrame) -> pd.DataFrame:
    # Filtro de estabilidad
    estables = df[df["Gap_Acc"] < UMBRAL_GAP_ACC].copy()

    # Si ninguno pasa el umbral, cogemos los top-N igualmente (evita bloquear el flujo)
    if len(estables) == 0:
        estables = df.copy()

    candidatos = estables.head(MAX_CANDIDATOS).reset_index(drop=True)
    return candidatos


def seleccionar_ganador(candidatos: pd.DataFrame) -> pd.DataFrame:
    # Ganador = primer candidato tras ordenar por (PseudoR2_test desc, N_parametros asc)
    ganador = candidatos.head(1).copy()
    return ganador


def guardar_outputs(tabla_ordenada: pd.DataFrame, candidatos: pd.DataFrame, ganador: pd.DataFrame) -> None:
    candidatos.to_csv(TABLA_CANDIDATOS_FILE, index=False)
    ganador.to_csv(TABLA_GANADOR_FILE, index=False)

    meta = {
        "input_table": str(TABLA_HU32_FILE),
        "params": {
            "max_candidatos": int(MAX_CANDIDATOS),
            "umbral_gap_acc": float(UMBRAL_GAP_ACC),
            "metrica_principal": METRICA_PRINCIPAL,
            "metrica_secundaria": METRICA_SECUNDARIA,
            "punto_corte_provisional": 0.5,
        },
        "n_modelos_hu32": int(len(tabla_ordenada)),
        "n_candidatos": int(len(candidatos)),
        "ganador": ganador.to_dict(orient="records")[0],
        "top_resumen": tabla_ordenada.head(6).to_dict(orient="records"),
    }

    with open(META_FILE, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)


def main() -> None:
    # 1) Cargar tabla HU-3.2
    df = cargar_tabla_hu32()
    print(f"[INFO] Tabla HU-3.2 cargada: {TABLA_HU32_FILE.name} | filas={len(df)}")

    # 2) Preparar y ordenar
    tabla_ordenada = preparar_tabla(df)
    print("[INFO] Modelos ordenados por rendimiento en test y parsimonia.")
    print(tabla_ordenada[
        ["Modelo", "PseudoR2_test", "Acc_test@0.5", "Gap_Acc", "N_parametros_aprox"]
    ].head(6).to_string(index=False))

    # 3) Selección de candidatos
    candidatos = seleccionar_candidatos(tabla_ordenada)
    print(f"[INFO] Candidatos seleccionados (top {MAX_CANDIDATOS} con estabilidad): {len(candidatos)}")
    print(candidatos[
        ["Modelo", "PseudoR2_test", "Acc_test@0.5", "Gap_Acc", "N_parametros_aprox"]
    ].to_string(index=False))

    # 4) Selección de ganador
    ganador = seleccionar_ganador(candidatos)
    print("\n[OK] Modelo ganador seleccionado:")
    print(ganador[
        ["Modelo", "PseudoR2_test", "Acc_test@0.5", "Gap_Acc", "N_parametros_aprox"]
    ].to_string(index=False))

    # 5) Guardar outputs
    guardar_outputs(tabla_ordenada, candidatos, ganador)
    print(f"\n[OK] Guardado: {TABLA_CANDIDATOS_FILE.name}, {TABLA_GANADOR_FILE.name}, {META_FILE.name}")


if __name__ == "__main__":
    main()
