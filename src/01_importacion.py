# 01_importacion.py
# --------------------------------------------
# Sprint 0 - HU-0.2: Carga inicial del dataset
# Subtareas:
#   - Cargar DatosEleccionesEspaña.xlsx
#   - Revisar dimensiones (shape)
#   - Mostrar primeras filas (head)
#   - Listar nombres de variables
#   - Guardar script base de importación + salidas
# --------------------------------------------

from pathlib import Path
import pandas as pd

def main():
    # --------------------------------------------
    # Configuración de rutas y archivos
    # --------------------------------------------
    # --- Rutas del proyecto ---
    # Estructura recomendada:
    # proyecto/
    # ├── data/
    # │   └── DatosEleccionesEspaña.xlsx
    # └── src/
    #     └── 01_importacion.py

    ROOT = Path(__file__).resolve().parents[1]     # /proyecto
    DATA_DIR = ROOT / "data/raw"
    OUT_DIR = ROOT / "results"
    OUT_DIR.mkdir(exist_ok=True)

    file_path = DATA_DIR / "DatosEleccionesEspaña.xlsx"

    if not file_path.exists():
        raise FileNotFoundError(
            f"No se encontró el archivo: {file_path}\n"
            f"👉 Coloca el Excel en {DATA_DIR} o ajusta la ruta en el script."
        )

    # --- 1) Cargar dataset ---
    # (si tiene varias hojas, puedes especificar sheet_name="Hoja1")
    df = pd.read_excel(file_path)

    # --- 2) Revisar dimensiones (shape) ---
    n_rows, n_cols = df.shape

    # --- 3) Mostrar primeras filas (head) ---
    head_df = df.head(10)

    # --- 4) Listar nombres de variables ---
    columnas = df.columns.tolist()

    # --- Salidas por consola (útil para ejecutar rápido) ---
    print("\n=== CARGA COMPLETADA ===")
    print(f"Archivo: {file_path.name}")
    print(f"Dimensiones (filas, columnas): {df.shape}")

    print("\n=== PRIMERAS 10 FILAS ===")
    print(head_df)

    print("\n=== NOMBRES DE VARIABLES ===")
    for i, col in enumerate(columnas, start=1):
        print(f"{i:02d}. {col}")

    # --- 5) Guardar salidas para el informe ---
    # 5.1 Guardar primeras filas como CSV (fácil de pegar/consultar)
    head_df.to_csv(OUT_DIR / "01_head_10.csv", index=False, encoding="utf-8")

    # 5.2 Guardar lista de variables y shape en un TXT (para el PDF)
    with open(OUT_DIR / "01_resumen_importacion.txt", "w", encoding="utf-8") as f:
        f.write("=== RESUMEN DE IMPORTACIÓN ===\n")
        f.write(f"Archivo: {file_path.name}\n")
        f.write(f"Dimensiones (filas, columnas): {n_rows}, {n_cols}\n\n")

        f.write("=== VARIABLES (COLUMNAS) ===\n")
        for i, col in enumerate(columnas, start=1):
            f.write(f"{i:02d}. {col}\n")

    print("\n✅ Salidas guardadas en:")
    print(f"- {OUT_DIR / '01_head_10.csv'}")
    print(f"- {OUT_DIR / '01_resumen_importacion.txt'}")


if __name__ == "__main__":
    main()
