"""
Script 4: Generación de Predicciones

Genera predicciones de disponibilidad de bicicletas para el próximo día
usando el modelo VAR Optimizado (con variables climáticas via Two-Stage).

Pasos:
1. Cargar modelo VAR Optimizado entrenado
2. Cargar últimos datos disponibles y variables exógenas
3. Generar variables exógenas futuras (promedios históricos por día del año)
4. Generar predicciones para el próximo día (VAR + efecto climático)
5. Aplicar restricción global (suma = total de bicis)
6. Calcular intervalos de confianza
7. Determinar nivel de confianza (alto/medio/bajo)
8. Exportar predicciones en formato CSV y JSON
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta
import json
import pickle

# Configurar paths
sys.path.append(str(Path(__file__).parent))
from config import (
    OUTPUTS_DIR, MODELS_DIR, PREDICTION_CONFIG, CONFIDENCE_THRESHOLDS,
    LOG_LEVEL, LOG_FORMAT
)
from utils.data_loader import load_model, load_processed_data
from utils.constraints import enforce_global_constraint, validate_constraint

# Configurar logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def load_latest_model() -> dict:
    """
    Carga el último modelo VAR Optimizado entrenado

    Returns:
        Diccionario con modelo y metadata
    """
    logger.info("Cargando modelo VAR Optimizado...")

    # Intentar cargar VAR Optimizado primero, luego VARX, luego VAR como fallback
    model_path = MODELS_DIR / "var_optimized_model.pkl"

    if not model_path.exists():
        model_path = MODELS_DIR / "varx_model.pkl"
        if not model_path.exists():
            model_path = MODELS_DIR / "var_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"No se encontró modelo en {MODELS_DIR}")

    model_data = load_model(model_path)

    logger.info(f"   Modelo cargado correctamente")
    logger.info(f"   Entrenado en: {model_data.get('trained_at', 'N/A')}")
    logger.info(f"   Version: {model_data.get('model_version', 'N/A')}")
    logger.info(f"   Tipo: {model_data.get('model_type', 'VAR')}")
    logger.info(f"   Total de bicis: {model_data.get('total_bikes', 'N/A')}")
    logger.info(f"   Numero de estaciones: {model_data.get('num_stations', 'N/A')}")
    logger.info(f"   Variables exogenas: {model_data.get('exog_columns', [])}")
    logger.info(f"   Usa PCA: {model_data.get('use_pca', False)}")
    logger.info(f"   Usa clima: {model_data.get('use_climate', False)}")

    return model_data


def get_future_exog_from_history(
    historical_exog: pd.DataFrame,
    forecast_days: int,
    start_date: datetime
) -> pd.DataFrame:
    """
    Genera variables exógenas futuras usando promedios históricos por día del año

    Args:
        historical_exog: DataFrame con datos históricos de clima
        forecast_days: Número de días a predecir
        start_date: Fecha de inicio de la predicción

    Returns:
        DataFrame con variables exógenas para los días futuros
    """
    logger.info(f"Generando variables exogenas futuras usando promedios historicos...")

    future_dates = [
        start_date + timedelta(days=i)
        for i in range(forecast_days)
    ]

    exog_list = []

    for target_date in future_dates:
        day_of_year = target_date.timetuple().tm_yday

        # Crear copia para no modificar el original
        hist_copy = historical_exog.copy()
        hist_copy['doy'] = pd.to_datetime(hist_copy.index).dayofyear
        same_day = hist_copy[hist_copy['doy'] == day_of_year]

        if len(same_day) == 0:
            # Si no hay datos para ese día, usar promedio general
            same_day = hist_copy

        # Temperatura: promedio histórico para ese día del año
        avg_temp = same_day['temperature'].mean()

        # Eventos: moda histórica (evento más frecuente para ese día)
        weather_cols = [c for c in historical_exog.columns if c.startswith('weather_')]

        record = {'temperature': avg_temp}

        if weather_cols:
            # Encontrar el evento más común
            event_mode = same_day[weather_cols].mean().idxmax()
            for col in weather_cols:
                record[col] = 1 if col == event_mode else 0

        exog_list.append(pd.DataFrame([record], index=[target_date]))

    future_exog = pd.concat(exog_list)
    future_exog.index = pd.to_datetime(future_exog.index)

    logger.info(f"   Variables exogenas futuras generadas:")
    logger.info(f"      Días: {len(future_exog)}")
    logger.info(f"      Temperatura promedio: {future_exog['temperature'].mean():.1f}")

    return future_exog


def generate_forecast(
    model_data: dict,
    recent_data: pd.DataFrame,
    steps: int = 1,
    future_exog: pd.DataFrame = None
) -> np.ndarray:
    """
    Genera predicciones usando el modelo VAR Optimizado (Two-Stage) o VAR/VARX

    Args:
        model_data: Diccionario con modelo y metadata
        recent_data: Datos recientes para forecast
        steps: Número de días a predecir
        future_exog: Variables exógenas para el período a predecir (opcional)

    Returns:
        Array con predicciones (shape: [steps, num_stations])
    """
    logger.info(f"Generando predicciones para {steps} dia(s)...")

    # VAR Optimizado usa 'var_model', otros usan 'model'
    fitted_model = model_data.get('var_model') or model_data.get('model')
    lag_order = model_data['lag_order']
    is_differenced = model_data['is_differenced']
    model_type = model_data.get('model_type', 'VAR')

    # Componentes del modelo VAR Optimizado
    use_pca = model_data.get('use_pca', False)
    use_climate = model_data.get('use_climate', False)
    pca = model_data.get('pca_model', None)  # Guardado como 'pca_model'
    scaler = model_data.get('scaler', None)
    climate_model = model_data.get('climate_model', None)
    n_pca_components = model_data.get('n_pca_components', None)  # Número de componentes usados

    # Preparar datos
    if is_differenced:
        recent_data_processed = recent_data.diff().dropna()
    else:
        recent_data_processed = recent_data

    # Generar predicciones según el tipo de modelo
    if model_type == 'VAR_Optimizado_Clima':
        # Modelo VAR Optimizado con Two-Stage approach
        logger.info(f"   Usando modelo VAR Optimizado (Two-Stage)")

        # Si usamos PCA, transformar los datos
        if use_pca and pca is not None and scaler is not None:
            logger.info(f"   Aplicando transformacion PCA ({n_pca_components} componentes)...")
            data_scaled = scaler.transform(recent_data_processed.values)
            data_pca = pca.transform(data_scaled)
            # Solo usar los primeros n_pca_components (el VAR fue entrenado con estos)
            data_pca = data_pca[:, :n_pca_components]
            last_obs = data_pca[-lag_order:]
        else:
            last_obs = recent_data_processed.iloc[-lag_order:].values

        # Generar forecast del VAR (en espacio PCA o original)
        forecast_var = fitted_model.forecast(last_obs, steps=steps)

        # Si usamos PCA, invertir la transformación
        if use_pca and pca is not None and scaler is not None:
            logger.info(f"   Invirtiendo transformacion PCA...")
            # Expandir forecast a todas las componentes PCA (rellenar con ceros)
            forecast_full = np.zeros((steps, pca.n_components_))
            forecast_full[:, :n_pca_components] = forecast_var
            forecast_scaled = pca.inverse_transform(forecast_full)
            forecast = scaler.inverse_transform(forecast_scaled)
        else:
            forecast = forecast_var

        # Añadir efecto climático si tenemos modelo de clima y variables exógenas
        if use_climate and climate_model is not None and future_exog is not None:
            logger.info(f"   Anadiendo efecto climatico...")
            # climate_model es un dict con 'exog_scaler' y 'models' (uno por estación)
            exog_scaled = climate_model['exog_scaler'].transform(future_exog.values)
            climate_effect = np.zeros_like(forecast)
            station_columns = model_data.get('station_columns', [])
            for idx, col in enumerate(station_columns):
                if col in climate_model['models']:
                    climate_effect[:, idx] = climate_model['models'][col].predict(exog_scaled)
            forecast = forecast + climate_effect

    elif model_type == 'VARX' and future_exog is not None:
        # Modelo VARX con variables exógenas
        logger.info(f"   Usando modelo VARX con variables exógenas")
        try:
            # Para VARMAX, usamos get_forecast
            forecast_result = fitted_model.get_forecast(
                steps=steps,
                exog=future_exog.values
            )
            forecast = forecast_result.predicted_mean.values
        except Exception as e:
            logger.warning(f"   Error con get_forecast: {e}")
            logger.info(f"   Usando metodo forecast alternativo")
            forecast = fitted_model.forecast(
                steps=steps,
                exog=future_exog.values
            )
    else:
        # Modelo VAR sin variables exógenas
        logger.info(f"   Usando modelo VAR sin variables exógenas")
        last_obs = recent_data_processed.iloc[-lag_order:].values
        forecast = fitted_model.forecast(last_obs, steps=steps)

    # Si se aplicó diferenciación, invertir la transformación
    if is_differenced:
        last_value = recent_data.iloc[-1].values
        forecast_original = np.zeros_like(forecast)

        for i in range(steps):
            if i == 0:
                forecast_original[i] = last_value + forecast[i]
            else:
                forecast_original[i] = forecast_original[i-1] + forecast[i]

        forecast = forecast_original

    # Asegurar que forecast sea 2D
    if len(forecast.shape) == 1:
        forecast = forecast.reshape(1, -1)

    logger.info(f"   Predicciones generadas")
    logger.info(f"   Shape: {forecast.shape}")
    logger.info(f"   Suma total (sin ajustar): {forecast.sum(axis=1)}")

    return forecast


def calculate_confidence_intervals(
    model_data: dict,
    forecast: np.ndarray,
    alpha: float = 0.05
) -> tuple:
    """
    Calcula intervalos de confianza para las predicciones

    Args:
        model_data: Diccionario con modelo y metadata
        forecast: Predicciones del modelo
        alpha: Nivel de significancia (0.05 para 95% confianza)

    Returns:
        Tupla (lower_bound, upper_bound)
    """
    logger.info(f"Calculando intervalos de confianza ({(1-alpha)*100}%)...")

    # VAR Optimizado usa 'var_model', otros usan 'model'
    fitted_model = model_data.get('var_model') or model_data.get('model')

    try:
        # Obtener matriz de covarianza de los residuales
        residuals = fitted_model.resid
        residuals_std = residuals.std().values

        # Calcular intervalos usando distribución normal
        from scipy import stats
        z_score = stats.norm.ppf(1 - alpha/2)

        lower = forecast - z_score * residuals_std
        upper = forecast + z_score * residuals_std

        logger.info(f"   Intervalos calculados")

        return lower, upper

    except Exception as e:
        logger.warning(f"   Error al calcular intervalos: {e}")
        logger.warning(f"   Usando intervalos simplificados (+/-20%)")

        lower = forecast * 0.8
        upper = forecast * 1.2

        return lower, upper


def determine_confidence_level(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    """
    Determina el nivel de confianza (high/medium/low) basado en amplitud del intervalo

    Args:
        lower: Límite inferior del intervalo
        upper: Límite superior del intervalo

    Returns:
        Array con niveles de confianza
    """
    interval_width = upper - lower

    # Si es 2D, tomar la última fila (último día predicho)
    if len(interval_width.shape) > 1:
        interval_width = interval_width[-1]

    confidence_levels = np.full(len(interval_width), 'low')

    confidence_levels[interval_width < CONFIDENCE_THRESHOLDS['high']] = 'high'
    confidence_levels[(interval_width >= CONFIDENCE_THRESHOLDS['high']) &
                      (interval_width < CONFIDENCE_THRESHOLDS['medium'])] = 'medium'

    return confidence_levels


def create_predictions_dataframe(
    forecast: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    confidence_levels: np.ndarray,
    station_columns: list,
    station_name_map: dict,
    prediction_date: str
) -> pd.DataFrame:
    """
    Crea DataFrame con las predicciones en formato estructurado

    Args:
        forecast: Predicciones ajustadas
        lower: Límite inferior del intervalo
        upper: Límite superior del intervalo
        confidence_levels: Niveles de confianza
        station_columns: Lista de IDs de estaciones
        station_name_map: Mapeo de ID a nombre
        prediction_date: Fecha de la predicción

    Returns:
        DataFrame con predicciones
    """
    # Si forecast es 2D, tomar última fila
    if len(forecast.shape) > 1:
        forecast = forecast[-1]
        lower = lower[-1]
        upper = upper[-1]

    predictions_df = pd.DataFrame({
        'station_id': station_columns,
        'station_name': [station_name_map.get(sid, f'Station {sid}') for sid in station_columns],
        'prediction_date': prediction_date,
        'predicted_bikes': forecast.astype(int),
        'confidence_lower': lower.astype(int),
        'confidence_upper': upper.astype(int),
        'confidence_level': confidence_levels,
    })

    return predictions_df


def export_predictions(
    predictions_df: pd.DataFrame,
    output_dir: Path
) -> None:
    """
    Exporta predicciones a CSV y JSON

    Args:
        predictions_df: DataFrame con predicciones
        output_dir: Directorio de salida
    """
    logger.info("Exportando predicciones...")

    # CSV
    csv_path = output_dir / 'predictions_daily.csv'
    predictions_df.to_csv(csv_path, index=False)
    logger.info(f"   CSV guardado: {csv_path}")

    # JSON (formato para API)
    json_data = {
        'generated_at': datetime.now().isoformat(),
        'prediction_date': predictions_df['prediction_date'].iloc[0],
        'total_stations': len(predictions_df),
        'total_bikes_predicted': int(predictions_df['predicted_bikes'].sum()),
        'predictions': predictions_df.to_dict(orient='records')
    }

    json_path = output_dir / 'predictions_daily.json'
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, indent=2, ensure_ascii=False)

    logger.info(f"   JSON guardado: {json_path}")


def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO GENERACION DE PREDICCIONES")
    logger.info("=" * 80)

    # 1. Cargar modelo
    model_data = load_latest_model()

    # 2. Cargar datos recientes
    logger.info("\nCargando datos recientes...")
    data_path = OUTPUTS_DIR / "processed_data"
    train, test = load_processed_data(data_path)

    # Usar todas las datos disponibles (train + test) para predicción
    all_data = pd.concat([train, test])
    logger.info(f"   Datos cargados: {all_data.shape}")

    # Cargar metadata
    metadata_path = data_path.parent / "processed_data_metadata.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    total_bikes = metadata['total_bikes']
    station_name_map = metadata['station_name_map']

    # 3. Cargar variables exógenas históricas (si el modelo usa clima)
    model_type = model_data.get('model_type', 'VAR')
    exog_columns = model_data.get('exog_columns', [])
    use_climate = model_data.get('use_climate', False)
    future_exog = None

    if (model_type in ['VARX', 'VAR_Optimizado_Clima'] or use_climate) and exog_columns:
        logger.info("\nCargando variables exogenas historicas...")
        exog_path = OUTPUTS_DIR / "exog_data"

        try:
            exog_train = pd.read_pickle(f"{exog_path}_train.pkl")
            exog_test = pd.read_pickle(f"{exog_path}_test.pkl")
            historical_exog = pd.concat([exog_train, exog_test])

            logger.info(f"   Variables exogenas cargadas: {historical_exog.shape}")

            # 4. Generar variables exógenas para el período de predicción
            forecast_steps = PREDICTION_CONFIG['forecast_horizon']
            last_date = pd.to_datetime(all_data.index[-1])
            prediction_start = last_date + timedelta(days=1)

            future_exog = get_future_exog_from_history(
                historical_exog=historical_exog,
                forecast_days=forecast_steps,
                start_date=prediction_start
            )

            # Asegurar que las columnas estén en el mismo orden que el modelo
            future_exog = future_exog[exog_columns]

        except FileNotFoundError:
            logger.warning("   No se encontraron datos exogenos, usando modelo VAR sin exog")
            future_exog = None

    # 5. Generar predicciones
    forecast_steps = PREDICTION_CONFIG['forecast_horizon']
    forecast = generate_forecast(
        model_data,
        all_data,
        steps=forecast_steps,
        future_exog=future_exog
    )

    # 6. Aplicar restricción global
    if PREDICTION_CONFIG['enforce_global_constraint']:
        logger.info("\nAplicando restriccion global...")
        forecast_adjusted = enforce_global_constraint(
            forecast,
            total_bikes=total_bikes,
            method='proportional'
        )

        # Validar restricción
        validate_constraint(forecast_adjusted, total_bikes)
    else:
        forecast_adjusted = np.round(forecast).astype(int)

    # 7. Calcular intervalos de confianza
    lower, upper = calculate_confidence_intervals(
        model_data,
        forecast_adjusted,
        alpha=1 - PREDICTION_CONFIG['confidence_level']
    )

    # Asegurar que los intervalos respeten valores no negativos
    lower = np.maximum(lower, 0)

    # 8. Determinar niveles de confianza
    confidence_levels = determine_confidence_level(lower, upper)

    # 9. Crear DataFrame con predicciones
    prediction_date = (pd.to_datetime(all_data.index[-1]) + timedelta(days=1)).strftime('%Y-%m-%d')

    predictions_df = create_predictions_dataframe(
        forecast_adjusted,
        lower,
        upper,
        confidence_levels,
        station_columns=list(all_data.columns),
        station_name_map=station_name_map,
        prediction_date=prediction_date
    )

    # 10. Mostrar resumen
    logger.info("\n" + "=" * 80)
    logger.info("RESUMEN DE PREDICCIONES")
    logger.info("=" * 80)
    logger.info(f"Fecha de prediccion: {prediction_date}")
    logger.info(f"Total de estaciones: {len(predictions_df)}")
    logger.info(f"Total de bicis predichas: {predictions_df['predicted_bikes'].sum()}")
    logger.info(f"Tipo de modelo: {model_type}")
    if model_data.get('use_pca', False):
        logger.info(f"PCA activo: {model_data.get('n_components', 'N/A')} componentes")
    if model_data.get('use_climate', False):
        logger.info(f"Ajuste climatico: Si")

    if future_exog is not None:
        logger.info(f"Temperatura prevista: {future_exog['temperature'].mean():.1f}C")

    logger.info(f"Distribucion de confianza:")
    logger.info(f"   Alta: {(confidence_levels == 'high').sum()} estaciones")
    logger.info(f"   Media: {(confidence_levels == 'medium').sum()} estaciones")
    logger.info(f"   Baja: {(confidence_levels == 'low').sum()} estaciones")

    # Top 10 estaciones con más bicis predichas
    top_10 = predictions_df.nlargest(10, 'predicted_bikes')[['station_name', 'predicted_bikes', 'confidence_level']]
    logger.info(f"\nTop 10 estaciones con mas bicis predichas:")
    for idx, row in top_10.iterrows():
        logger.info(f"   {row['station_name']}: {row['predicted_bikes']} bicis ({row['confidence_level']} confidence)")

    # 11. Exportar predicciones
    export_predictions(predictions_df, OUTPUTS_DIR)

    logger.info("\n" + "=" * 80)
    logger.info("GENERACION DE PREDICCIONES COMPLETADA")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
