from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
TABLES_DIR = OUTPUT_DIR / "tablas"
FIG_DIR = OUTPUT_DIR / "figuras"

TABLES_DIR.mkdir(parents=True, exist_ok=True)
FIG_DIR.mkdir(parents=True, exist_ok=True)

RANDOM_STATE = 1234567
DATA_FILE = DATA_DIR / "raw/DatosEleccionesEspaña.xlsx"
DEP_BASE_FILE = DATA_DIR / "interim/datos_dep_base.parquet"