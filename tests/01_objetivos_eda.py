# 00_objetivos_eda.py
# -----------------------------------------------------------
# Sprint 0 - HU-0.4: EDA mínimo para seleccionar variables objetivo
# Subtareas:
#  - Analizar distribución de la variable continua candidata
#  - Analizar proporción de clases de la binaria candidata
#  - Detectar problemas (desbalance extremo, valores raros)
#  - Dejar lista una justificación basada en evidencias
# -----------------------------------------------------------

from pathlib import Path
from scipy import stats
import numpy as np
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt


# ====== CONFIGURA AQUÍ ======
VAR_OBJ_CONT = "AbstentionPtge"   # <-- cambia por tu candidata continua
VAR_OBJ_BIN  = "AbstencionAlta"   # <-- cambia por tu candidata binaria
EXCEL_NAME   = "raw/DatosEleccionesEspaña.xlsx"
SHEET_NAME   = None  # si necesitas hoja: "Hoja1"
# ============================


def basic_numeric_profile(s: pd.Series) -> dict:
    """Devuelve métricas clave para detectar rarezas y describir distribución."""
    s_clean = pd.to_numeric(s, errors="coerce").dropna()

    if s_clean.empty:
        return {"n": 0}

    q = s_clean.quantile([0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]).to_dict()
    return {
        "n": int(s_clean.shape[0]),
        "missing": int(s.isna().sum()),
        "mean": float(s_clean.mean()),
        "std": float(s_clean.std(ddof=1)),
        "min": float(s_clean.min()),
        "max": float(s_clean.max()),
        "skew": float(s_clean.skew()),
        "kurtosis": float(s_clean.kurtosis()),
        "q01": float(q.get(0.01, np.nan)),
        "q05": float(q.get(0.05, np.nan)),
        "q25": float(q.get(0.25, np.nan)),
        "q50": float(q.get(0.50, np.nan)),
        "q75": float(q.get(0.75, np.nan)),
        "q95": float(q.get(0.95, np.nan)),
        "q99": float(q.get(0.99, np.nan)),
    }


def check_continuous_issues(s: pd.Series) -> list:
    """Flags típicos para variable objetivo continua."""
    issues = []
    s_num = pd.to_numeric(s, errors="coerce")

    n_total = len(s_num)
    n_valid = s_num.notna().sum()

    if n_valid == 0:
        return ["Sin valores numéricos válidos (todo NaN o no numérico)."]

    unique = s_num.dropna().nunique()
    if unique <= 5:
        issues.append(f"Pocos valores distintos ({unique}). Podría comportarse como categórica/discreta.")

    # Chequeo "porcentaje" típico (0-100): no lo asumimos como regla, solo alertamos.
    below0 = (s_num < 0).sum(skipna=True)
    above100 = (s_num > 100).sum(skipna=True)
    if below0 > 0:
        issues.append(f"Hay {below0} valores < 0 (posibles errores).")
    if above100 > 0:
        issues.append(f"Hay {above100} valores > 100 (posibles errores si es porcentaje).")

    # Outliers extremos por IQR (solo alerta temprana; análisis formal va en Sprint 1)
    x = s_num.dropna()
    q1, q3 = x.quantile(0.25), x.quantile(0.75)
    iqr = q3 - q1
    if iqr > 0:
        lower, upper = q1 - 3 * iqr, q3 + 3 * iqr
        n_ext = ((x < lower) | (x > upper)).sum()
        if n_ext / len(x) > 0.02:
            issues.append(f"Posibles outliers extremos (regla 3*IQR): {n_ext} ({n_ext/len(x):.1%}).")

    miss_rate = (n_total - n_valid) / n_total if n_total > 0 else 0
    if miss_rate > 0.10:
        issues.append(f"Tasa de missing alta en objetivo continuo: {miss_rate:.1%}.")

    return issues


def check_binary_issues(s: pd.Series) -> tuple[list, pd.Series]:
    """
    Flags típicos para variable binaria.
    Devuelve (issues, series_bin_normalizada)
    """
    issues = []
    # Normalizamos: aceptamos {0,1} como numérico o strings "0"/"1"
    s2 = s.copy()

    # Intentar convertir a numérico si procede
    if not pd.api.types.is_numeric_dtype(s2):
        s2 = s2.astype(str).str.strip()
        # convertir "0.0" -> "0" etc
        s2 = s2.replace({"0.0": "0", "1.0": "1"})

    s_num = pd.to_numeric(s2, errors="coerce")

    valid = s_num.dropna()
    if valid.empty:
        return (["Sin valores binarios válidos (no convertible a 0/1)."], s_num)

    # Valores únicos (deberían ser subset de {0,1})
    uniq = sorted(valid.unique().tolist())
    allowed = set([0, 1])
    if any(v not in allowed for v in uniq):
        issues.append(f"Valores fuera de {{0,1}} detectados: {uniq}. Se requiere recodificación/limpieza.")

    # Proporción de clases (desbalance)
    counts = valid.value_counts().sort_index()
    # Nos quedamos con 0 y 1 si existen
    c0 = int(counts.get(0, 0))
    c1 = int(counts.get(1, 0))
    n = c0 + c1

    if n == 0:
        issues.append("No hay casos válidos en {0,1}.")
        return (issues, s_num)

    p1 = c1 / n
    p0 = c0 / n

    # Alerta de desbalance fuerte
    # (no es "prohibido", pero complica métricas y punto de corte)
    if p1 < 0.15 or p1 > 0.85:
        issues.append(f"Desbalance notable: P(1)={p1:.1%}, P(0)={p0:.1%}. Considerar impacto en métricas/umbral.")

    # Missing en objetivo binario
    miss_rate = (len(s_num) - s_num.notna().sum()) / len(s_num) if len(s_num) > 0 else 0
    if miss_rate > 0.10:
        issues.append(f"Tasa de missing alta en objetivo binario: {miss_rate:.1%}.")

    return (issues, s_num)


def save_text_report(path: Path, title: str, lines: list[str]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(title + "\n")
        f.write("=" * len(title) + "\n\n")
        for line in lines:
            f.write(line + "\n")


def main():
    ROOT = Path(__file__).resolve().parents[1]
    DATA_DIR = ROOT / "data"
    OUT_DIR = ROOT / "outputs"
    OUT_DIR.mkdir(exist_ok=True)

    file_path = DATA_DIR / EXCEL_NAME
    if not file_path.exists():
        raise FileNotFoundError(f"No se encontró {file_path}. Coloca el Excel en data/ o ajusta EXCEL_NAME.")

    df = pd.read_excel(file_path)

    # --- Validación de existencia de columnas ---
    for col in [VAR_OBJ_CONT, VAR_OBJ_BIN]:
        if col not in df.columns:
            raise KeyError(f"La columna '{col}' no existe en el dataset. Revisa nombre exacto.")

    # --- CONTINUA ---
    cont = df[VAR_OBJ_CONT]
    cont_profile = basic_numeric_profile(cont)
    cont_issues = check_continuous_issues(cont)

    # Plot histograma
    #cont_num = pd.to_numeric(cont, errors="coerce").dropna()
    plt.figure()
    sns.histplot(data=df, x=VAR_OBJ_CONT, bins=30, kde=True)
    plt.title(f"Histograma: {VAR_OBJ_CONT}")
    plt.xlabel(VAR_OBJ_CONT)
    plt.ylabel("Frecuencia")
    plt.tight_layout()
    cont_plot_path = OUT_DIR / f"HU04_hist_{VAR_OBJ_CONT}.png"
    plt.savefig(cont_plot_path, dpi=200)
    plt.close()

    # Plot boxplot
    plt.figure()
    sns.boxplot(x=cont, orient="h")
    plt.title(f"Boxplot: {VAR_OBJ_CONT}")
    plt.xlabel(VAR_OBJ_CONT)
    plt.tight_layout()
    boxplot_path = OUT_DIR / f"HU04_boxplot_{VAR_OBJ_CONT}.png"
    plt.savefig(boxplot_path, dpi=200)
    plt.close()

    # Plot QQ plot
    plt.figure()
    stats.probplot(cont.dropna(), dist="norm", plot=plt)
    plt.title(f"QQ Plot: {VAR_OBJ_CONT}")
    plt.tight_layout()
    qqplot_path = OUT_DIR / f"HU04_qqplot_{VAR_OBJ_CONT}.png"
    plt.savefig(qqplot_path, dpi=200)
    plt.close()
    
    

    # --- BINARIA ---
    bin_raw = df[VAR_OBJ_BIN]
    bin_issues, bin_num = check_binary_issues(bin_raw)
    bin_valid = bin_num.dropna()

    # Conteo y barras
    counts = bin_valid.value_counts().sort_index()
    plt.figure()
    plt.bar([str(i) for i in counts.index.tolist()], counts.values.tolist())
    plt.title(f"Distribución clases: {VAR_OBJ_BIN}")
    plt.xlabel("Clase")
    plt.ylabel("N")
    plt.tight_layout()
    bin_plot_path = OUT_DIR / f"HU04_barras_{VAR_OBJ_BIN}.png"
    plt.savefig(bin_plot_path, dpi=200)
    plt.close()

    # --- Guardar tabla resumen en CSV ---
    resumen = pd.DataFrame([
        {"variable": VAR_OBJ_CONT, "tipo_objetivo": "continua", **cont_profile},
        {
            "variable": VAR_OBJ_BIN,
            "tipo_objetivo": "binaria",
            "n": int(bin_valid.shape[0]),
            "missing": int(bin_num.isna().sum()),
            "p1": float((bin_valid == 1).mean()) if bin_valid.shape[0] else np.nan,
            "p0": float((bin_valid == 0).mean()) if bin_valid.shape[0] else np.nan,
            "valores_unicos": ",".join(map(str, sorted(bin_valid.unique().tolist())))
        }
    ])
    resumen_path = OUT_DIR / "HU04_resumen_objetivos.csv"
    resumen.to_csv(resumen_path, index=False, encoding="utf-8")

    # --- Guardar reporte textual para pegar al informe ---
    lines = []
    lines.append(f"Variable objetivo CONTINUA candidata: {VAR_OBJ_CONT}")
    lines.append(f"- n válidos: {cont_profile.get('n', 0)} | missing: {cont_profile.get('missing', 0)}")
    if cont_profile.get("n", 0) > 0:
        lines.append(f"- min={cont_profile['min']:.3f} | q25={cont_profile['q25']:.3f} | mediana={cont_profile['q50']:.3f} | q75={cont_profile['q75']:.3f} | max={cont_profile['max']:.3f}")
        lines.append(f"- asimetría={cont_profile['skew']:.3f} | curtosis={cont_profile['kurtosis']:.3f}")
    lines.append(f"- Figura guardada: {cont_plot_path.name}")
    lines.append("Posibles incidencias (alertas tempranas):")
    if cont_issues:
        for it in cont_issues:
            lines.append(f"  * {it}")
    else:
        lines.append("  * No se detectan incidencias relevantes en este EDA mínimo.")

    lines.append("")
    lines.append(f"Variable objetivo BINARIA candidata: {VAR_OBJ_BIN}")
    lines.append(f"- n válidos: {int(bin_valid.shape[0])} | missing: {int(bin_num.isna().sum())}")
    if bin_valid.shape[0] > 0:
        p1 = (bin_valid == 1).mean()
        p0 = (bin_valid == 0).mean()
        lines.append(f"- P(1)={p1:.1%} | P(0)={p0:.1%} | valores únicos válidos: {sorted(bin_valid.unique().tolist())}")
    lines.append(f"- Figura guardada: {bin_plot_path.name}")
    lines.append("Posibles incidencias (alertas tempranas):")
    if bin_issues:
        for it in bin_issues:
            lines.append(f"  * {it}")
    else:
        lines.append("  * No se detectan incidencias relevantes en este EDA mínimo.")

    report_path = OUT_DIR / "HU04_reporte_objetivos.txt"
    save_text_report(report_path, "HU-0.4 EDA mínimo de variables objetivo", lines)

    print("✅ HU-0.4 completada. Archivos generados:")
    print(f"- {resumen_path}")
    print(f"- {report_path}")
    print(f"- {cont_plot_path}")
    print(f"- {bin_plot_path}")


if __name__ == "__main__":
    main()
