"""
Configuracion Global del Proyecto de Prediccion de Disponibilidad de Bicicletas
================================================================================

Trabajo Fin de Master: Prediccion de Disponibilidad de Bicicletas mediante
Modelos VAR con Variables Climaticas.

Este modulo centraliza todas las configuraciones del proyecto, incluyendo:
    - Rutas de directorios y archivos
    - Parametros del modelo VAR/VARX
    - Configuracion de preprocesamiento
    - Parametros de prediccion
    - Configuracion de visualizacion

Autor: David Garcia Hidalgo
Universidad: [Universidad]
Fecha: 2024

Uso:
    from config import VAR_CONFIG, PREPROCESSING_CONFIG, OUTPUTS_DIR
"""

import os
from pathlib import Path


# =============================================================================
# RUTAS Y DIRECTORIOS
# =============================================================================

BASE_DIR = Path(__file__).parent.parent
"""Path: Directorio raiz del proyecto."""

DATA_DIR = BASE_DIR
"""Path: Directorio que contiene los datos de entrada."""

MODELS_DIR = BASE_DIR / "models"
"""Path: Directorio para almacenar modelos entrenados."""

OUTPUTS_DIR = BASE_DIR / "outputs"
"""Path: Directorio para resultados y reportes."""

FIGURES_DIR = OUTPUTS_DIR / "figures"
"""Path: Directorio para graficos y visualizaciones."""

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

DATA_FILE = DATA_DIR / "data.csv"
"""Path: Ruta al archivo CSV con datos de viajes de Divvy Chicago."""


# =============================================================================
# CONFIGURACION DEL MODELO VAR
# =============================================================================

VAR_CONFIG = {
    "max_lags": 14,
    "ic": "aic",
    "trend": "c",
    "use_exog": True,
}
"""
dict: Parametros para el modelo VAR (Vector Autoregression).

Claves:
    max_lags (int): Numero maximo de lags a evaluar (14 = 2 semanas).
    ic (str): Criterio de informacion para seleccion de orden ('aic', 'bic').
    trend (str): Tipo de tendencia ('c'=constante, 'ct'=constante+tendencia, 'n'=ninguna).
    use_exog (bool): Si True, incluye variables exogenas (modelo VARX).
"""


# =============================================================================
# CONFIGURACION DE VARIABLES EXOGENAS
# =============================================================================

EXOG_CONFIG = {
    "include_temperature": True,
    "include_weather_events": True,
    "weather_events_encoding": "onehot",
}
"""
dict: Configuracion de variables exogenas climaticas.

Claves:
    include_temperature (bool): Incluir temperatura como variable exogena.
    include_weather_events (bool): Incluir eventos meteorologicos.
    weather_events_encoding (str): Codificacion de eventos ('onehot', 'label').
"""


# =============================================================================
# CONFIGURACION DE PREPROCESAMIENTO
# =============================================================================

PREPROCESSING_CONFIG = {
    "test_size": 0.2,
    "min_trips_threshold": 10,
    "interpolation_method": "linear",
    "aggregation": "daily",
}
"""
dict: Parametros de preprocesamiento de datos.

Claves:
    test_size (float): Proporcion de datos para conjunto de test (0.2 = 20%).
    min_trips_threshold (int): Umbral minimo de viajes/dia para incluir estacion.
    interpolation_method (str): Metodo para imputar valores faltantes ('linear').
    aggregation (str): Nivel de agregacion temporal ('daily', 'hourly').
"""


# =============================================================================
# CONFIGURACION DE PREDICCIONES
# =============================================================================

PREDICTION_CONFIG = {
    "forecast_horizon": 1,
    "confidence_level": 0.95,
    "enforce_global_constraint": True,
}
"""
dict: Parametros para generacion de predicciones.

Claves:
    forecast_horizon (int): Horizonte de prediccion en dias.
    confidence_level (float): Nivel de confianza para intervalos (0.95 = 95%).
    enforce_global_constraint (bool): Si True, aplica restriccion de suma total.
"""


# =============================================================================
# UMBRALES DE CONFIANZA
# =============================================================================

CONFIDENCE_THRESHOLDS = {
    "high": 10,
    "medium": 20,
}
"""
dict: Umbrales para clasificar nivel de confianza de predicciones.

Claves:
    high (int): Intervalo < 10 bicis = confianza alta.
    medium (int): Intervalo < 20 bicis = confianza media.
    (Intervalo >= 20 bicis = confianza baja)
"""


# =============================================================================
# CONFIGURACION DE VISUALIZACION
# =============================================================================

PLOT_CONFIG = {
    "figsize": (14, 8),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid",
    "color_palette": "husl",
}
"""
dict: Parametros para generacion de graficos.

Claves:
    figsize (tuple): Tamano de figura por defecto (ancho, alto) en pulgadas.
    dpi (int): Resolucion de graficos exportados.
    style (str): Estilo de matplotlib a utilizar.
    color_palette (str): Paleta de colores de seaborn.
"""


# =============================================================================
# SEMILLA Y LOGGING
# =============================================================================

RANDOM_SEED = 42
"""int: Semilla para reproducibilidad de resultados."""

LOG_LEVEL = "INFO"
"""str: Nivel de logging (DEBUG, INFO, WARNING, ERROR, CRITICAL)."""

LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
"""str: Formato de mensajes de log."""
