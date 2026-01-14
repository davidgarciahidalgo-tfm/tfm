# Predicción de Disponibilidad de Bicicletas en Chicago mediante Modelos VAR/VARX

## Trabajo de Fin de Máster

**Autor:** David García Hidalgo
**Email:** david.garcia926@comunidadunir.net
**Universidad:** UNIR (Universidad Internacional de La Rioja)
**Fecha:** Enero 2026

---

## Índice

1. [Resumen](#resumen)
2. [Dataset](#dataset)
3. [Metodología](#metodología)
4. [Estructura del Proyecto](#estructura-del-proyecto)
5. [Requisitos e Instalación](#requisitos-e-instalación)
6. [Uso](#uso)

---

## Resumen

Este Trabajo de Fin de Máster aborda el problema de **predecir la disponibilidad de bicicletas** en el sistema de bicicletas compartidas de Chicago (Divvy). El proyecto utiliza modelos **VAR (Vector AutoRegressive)** y **VARX (con variables exógenas)** para capturar las dependencias temporales entre estaciones y los flujos de movilidad urbana.

La predicción de disponibilidad de bicicletas es un problema relevante en el contexto de movilidad urbana sostenible, ya que permite:
- Optimizar la redistribución de bicicletas entre estaciones
- Mejorar la experiencia del usuario al reducir estaciones vacías o llenas
- Planificar recursos de manera eficiente

---

## Dataset

El proyecto utiliza datos del sistema de bicicletas compartidas Divvy de Chicago, dicho dataset puede descargarse en https://www.kaggle.com/datasets/yingwurenjian/chicago-divvy-bicycle-sharing-data

---

## Metodología

### 1. Análisis Exploratorio de Datos (EDA)

El análisis exploratorio se estructura en cuatro secciones:

1. **Análisis General**: Estadísticas descriptivas del dataset
2. **Variable Target**: Análisis de disponibilidad por estación, flujos de movilidad
3. **Variables Exógenas**: Patrones temporales, impacto climático, tipos de usuario
4. **Correlaciones**: Matriz de correlación con transformaciones cíclicas

### 2. Preprocesamiento

- Agregación temporal a nivel diario
- Filtrado de estaciones con actividad mínima (umbral: 10 viajes)
- Interpolación lineal para valores faltantes
- Transformación de variables cíclicas (seno/coseno para hora, día, mes)
- Codificación one-hot para eventos meteorológicos

### 3. Modelado VAR/VARX

**Configuración del modelo:**
- Máximo de lags: 14 (dos semanas)
- Criterio de selección: AIC (Akaike Information Criterion)
- Tendencia: Constante
- Variables exógenas: Temperatura, eventos meteorológicos

**División de datos:**
- Entrenamiento: 80%
- Test: 20%

### 4. Evaluación

- Horizonte de predicción: 1 día
- Nivel de confianza: 95%
- Métricas: RMSE, MAE, MAPE

---

## Estructura del Proyecto

```
tfm/
├── README.md                    # Este archivo
├── python_scripts/
│   ├── config.py               # Configuración global del proyecto
│   └── 1_eda_analysis.py       # Script de análisis exploratorio
├── data/
│   └── data.csv                # Dataset (no incluido en el repositorio)
└── outputs/
    ├── figures/                # Visualizaciones generadas
    │   ├── distributions_key_variables.png
    │   ├── temporal_patterns.png
    │   ├── station_analysis.png
    │   ├── weather_impact.png
    │   ├── user_comparison.png
    │   └── correlation_heatmap.png
    ├── flow_map_mobility.html  # Mapa interactivo de flujos
    ├── station_statistics.csv  # Estadísticas por estación
    └── correlation_matrix.csv  # Matriz de correlaciones
```

---

## Requisitos e Instalación

### Requisitos del Sistema
- Python 3.8 o superior
- Sistema operativo: Windows, Linux o macOS

### Dependencias

```txt
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.12.0
folium>=0.14.0
statsmodels>=0.14.0
```

### Instalación

1. Clonar el repositorio:
```bash
git clone https://github.com/usuario/tfm.git
cd tfm
```

2. Crear entorno virtual (recomendado):
```bash
python -m venv venv
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```

3. Instalar dependencias:
```bash
pip install pandas numpy matplotlib seaborn folium statsmodels
```

4. Colocar el archivo de datos `data.csv` en la carpeta `data/`

---

## Uso

### Ejecutar Análisis Exploratorio

```bash
cd python_scripts
python 1_eda_analysis.py
```

Este script generará:
- Estadísticas descriptivas en consola
- Visualizaciones en `outputs/figures/`
- Mapa interactivo en `outputs/flow_map_mobility.html`
- Archivos CSV con estadísticas y correlaciones

### Configuración Personalizada

Modificar `python_scripts/config.py` para ajustar:
- Parámetros del modelo VAR/VARX
- Variables exógenas a incluir
- Umbrales de filtrado
- Horizonte de predicción

---

## Licencia

Este proyecto se desarrolla con fines académicos como parte del Trabajo de Fin de Máster.

---

## Contacto

Para consultas sobre este proyecto:
- **Autor:** David García Hidalgo
- **Email:** david.garcia926@comunidadunir.net
