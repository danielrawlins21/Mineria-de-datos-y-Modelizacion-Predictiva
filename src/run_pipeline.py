import subprocess
import sys
from pathlib import Path

# ============================================================
# Script maestro — Ejecuta todo el proyecto en orden
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
SRC_DIR = BASE_DIR 

# Lista ordenada de scripts
SCRIPTS = [
    # --- REGRESIÓN LINEAL ---
    "06_preparacion_lineal.py",
    "07_seleccion_clasica_lineal.py",
    "08_modelo_ganador_lineal.py",
    "09_evaluacion_modelo_lineal.py",
    "10_interpretacion_modelo_ganador.py",

    # --- REGRESIÓN LOGÍSTICA ---
    "11_preparacion_logistica.py",
    "12_seleccion_clasica_logistica.py",
    "13_modelo_ganador_logistica.py",
    "14_puntos_corte_logistica.py",
    "15_evaluacion_modelo_logistico.py",
]


def ejecutar_script(script_name: str):
    script_path = SRC_DIR / script_name

    if not script_path.exists():
        print(f"[WARNING] No se encontró: {script_name}")
        return

    print(f"\n{'='*60}")
    print(f"[RUNNING] {script_name}")
    print(f"{'='*60}")

    result = subprocess.run(
        [sys.executable, str(script_path)],
        cwd=BASE_DIR
    )

    if result.returncode != 0:
        print(f"[ERROR] Falló la ejecución de {script_name}")
        sys.exit(result.returncode)

    print(f"[OK] Finalizó correctamente: {script_name}")


def main():
    print("\n🚀 Iniciando ejecución completa del proyecto...\n")

    for script in SCRIPTS:
        ejecutar_script(script)

    print("\n✅ Pipeline completado correctamente.\n")


if __name__ == "__main__":
    main()
