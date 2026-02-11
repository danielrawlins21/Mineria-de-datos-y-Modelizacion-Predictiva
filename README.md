# 📊 Análisis del Comportamiento Electoral --- Regresión Lineal y Logística

Este proyecto construye y evalúa modelos de **regresión lineal** y
**regresión logística** para analizar el comportamiento electoral a
nivel municipal utilizando variables socioeconómicas y demográficas.

El flujo completo del análisis puede ejecutarse automáticamente mediante
un script maestro que garantiza la reproducibilidad de los resultados.

------------------------------------------------------------------------

# 📁 Estructura del proyecto

    .
    ├── data/                  # Datos originales
    ├── tables/                # Resultados generados automáticamente
    ├── src/                   # Scripts del proyecto
    │   ├── 06_preparacion_lineal.py
    │   ├── ...
    │   ├── 15_evaluacion_modelo_logistico.py
    ├── run_pipeline.py        # Script maestro de ejecución
    ├── requirements.txt       # Dependencias del proyecto
    └── README.md

------------------------------------------------------------------------

# 🚀 Cómo ejecutar el proyecto

## 1️⃣ Clonar el repositorio

``` bash
git clone https://github.com/USUARIO/NOMBRE-REPO.git
cd NOMBRE-REPO
```

## 2️⃣ Crear entorno virtual (recomendado)

### Windows

``` bash
python -m venv venv
venv\Scripts\activate
```

### Mac/Linux

``` bash
python3 -m venv venv
source venv/bin/activate
```

## 3️⃣ Instalar dependencias

``` bash
pip install -r requirements.txt
```

Si no existe `requirements.txt`, instalar manualmente:

``` bash
pip install pandas numpy scikit-learn statsmodels joblib pyarrow
```

------------------------------------------------------------------------

# ▶️ Ejecutar todo el pipeline automáticamente

Desde la raíz del proyecto:

``` bash
python run_pipeline.py
```

Este script ejecutará en orden:

-   Regresión lineal:
    -   Preparación de datos
    -   Selección clásica
    -   Modelo ganador
    -   Evaluación
    -   Interpretación
-   Regresión logística:
    -   Preparación de datos
    -   Selección clásica
    -   Modelo ganador
    -   Determinación del punto de corte
    -   Evaluación final

Los resultados se almacenarán automáticamente en la carpeta `tables/`.

------------------------------------------------------------------------

# 🔎 Ejecutar scripts individuales

``` bash
python src/14_puntos_corte_logistica.py
```

Cada script genera sus propios archivos de salida en `tables/`.

------------------------------------------------------------------------

# 🔁 Reproducibilidad

El proyecto está diseñado para ser completamente reproducible:

-   Los modelos finales se almacenan junto con su configuración.
-   El punto de corte óptimo queda guardado en la metadata.
-   El pipeline puede ejecutarse de principio a fin con un solo comando.

------------------------------------------------------------------------

# 📌 Requisitos

-   Python ≥ 3.9
-   pandas
-   numpy
-   scikit-learn
-   statsmodels
-   joblib
-   pyarrow

------------------------------------------------------------------------

# 👤 Autor

Nombre del estudiante: Daniel Rawlins \
Asignatura:  TAREA DE MINERÍA DE DATOS Y 
MODELIZACIÓN PREDICTIVA/ Máster: Máster Big Data, Data Science y Inteligencia Artificial\
Universidad Complutense de Madrid

