"""
Configuración global para el proyecto de predicción de disponibilidad de bicicletas
"""
import os
from pathlib import Path

# Rutas base
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR
MODELS_DIR = BASE_DIR / "models"
OUTPUTS_DIR = BASE_DIR / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

# Crear directorios si no existen
MODELS_DIR.mkdir(exist_ok=True)
OUTPUTS_DIR.mkdir(exist_ok=True)
FIGURES_DIR.mkdir(exist_ok=True)

# Archivo de datos
DATA_FILE = DATA_DIR / "data.csv"

# Configuración del modelo VAR/VARX
VAR_CONFIG = {
    "max_lags": 14,  # Máximo número de lags a probar (2 semanas)
    "ic": "aic",  # Criterio de información: 'aic' o 'bic'
    "trend": "c",  # Tendencia: 'c' (constante), 'ct' (constante + tendencia), 'n' (ninguna)
    "use_exog": True,  # Usar variables exógenas (VARX)
}

# Configuración de variables exógenas (clima)
EXOG_CONFIG = {
    "include_temperature": True,  # Incluir temperatura como variable exógena
    "include_weather_events": True,  # Incluir eventos climáticos
    "weather_events_encoding": "onehot",  # 'onehot' o 'label'
}

# Configuración de preprocesamiento
PREPROCESSING_CONFIG = {
    "test_size": 0.2,  # 20% para test
    "min_trips_threshold": 10,  # Estaciones con menos de N viajes/día se excluyen
    "interpolation_method": "linear",  # Método para valores faltantes
    "aggregation": "daily",  # Agregación temporal: 'daily', 'hourly'
}

# Configuración de predicciones
PREDICTION_CONFIG = {
    "forecast_horizon": 1,  # Número de días a predecir
    "confidence_level": 0.95,  # Nivel de confianza para intervalos
    "enforce_global_constraint": True,  # Aplicar restricción de bicis totales
}

# Umbrales de confianza
CONFIDENCE_THRESHOLDS = {
    "high": 10,  # Intervalo de confianza < 10 bicis
    "medium": 20,  # Intervalo de confianza < 20 bicis
    # > 20 bicis = low confidence
}

# Configuración de visualización
PLOT_CONFIG = {
    "figsize": (14, 8),
    "dpi": 100,
    "style": "seaborn-v0_8-darkgrid",
    "color_palette": "husl",
}

# Semillas para reproducibilidad
RANDOM_SEED = 42

# Configuración de logging
LOG_LEVEL = "INFO"
LOG_FORMAT = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"

print(f"[OK] Configuracion cargada correctamente")
print(f"   - Directorio de datos: {DATA_DIR}")
print(f"   - Archivo de datos: {DATA_FILE}")
print(f"   - Directorio de modelos: {MODELS_DIR}")
print(f"   - Directorio de outputs: {OUTPUTS_DIR}")
