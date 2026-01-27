"""
Script 2: Preprocesamiento de Datos

Transforma los datos raw de viajes individuales en series temporales de
disponibilidad de bicicletas por estación y día.

Pasos:
1. Cargar datos raw
2. Calcular disponibilidad diaria por estación (llegadas - salidas)
3. Crear matriz estación × tiempo
4. Manejo de valores faltantes
5. Validar restricción global (total de bicis constante)
6. División train/test temporal
7. Guardar datos procesados
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime, timedelta

# Configurar paths
sys.path.append(str(Path(__file__).parent))
from config import (
    DATA_FILE, MODELS_DIR, OUTPUTS_DIR, PREPROCESSING_CONFIG,
    RANDOM_SEED, LOG_LEVEL, LOG_FORMAT, EXOG_CONFIG
)
from utils.data_loader import load_raw_data, save_processed_data
from utils.constraints import calculate_total_bikes, validate_constraint

# Configurar logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configurar semilla
np.random.seed(RANDOM_SEED)


def calculate_daily_station_activity(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula la actividad diaria (salidas y llegadas) por estación

    Args:
        df: DataFrame con datos raw de viajes

    Returns:
        DataFrame con actividad diaria por estación
    """
    logger.info("Calculando actividad diaria por estacion...")

    # Extraer fecha del starttime
    df['date'] = pd.to_datetime(df['starttime']).dt.date

    # Contar salidas por estación por día
    departures = df.groupby(['date', 'from_station_id', 'from_station_name']).size().reset_index(name='departures')

    # Contar llegadas por estación por día
    arrivals = df.groupby(['date', 'to_station_id', 'to_station_name']).size().reset_index(name='arrivals')

    # Renombrar columnas para merge
    departures.columns = ['date', 'station_id', 'station_name', 'departures']
    arrivals.columns = ['date', 'station_id', 'station_name', 'arrivals']

    # Merge salidas y llegadas
    activity = pd.merge(
        departures,
        arrivals,
        on=['date', 'station_id', 'station_name'],
        how='outer'
    ).fillna(0)

    # Calcular balance neto (llegadas - salidas)
    # Nota: Esto representa el cambio neto de bicis en la estación
    activity['net_change'] = activity['arrivals'] - activity['departures']

    logger.info(f"Actividad calculada:")
    logger.info(f"   Días únicos: {activity['date'].nunique()}")
    logger.info(f"   Estaciones únicas: {activity['station_id'].nunique()}")
    logger.info(f"   Registros totales: {len(activity):,}")

    return activity


def create_availability_time_series(activity: pd.DataFrame) -> pd.DataFrame:
    """
    Crea series temporales de disponibilidad de bicicletas por estación

    La disponibilidad se calcula asumiendo que cada estación comienza con un
    número base de bicis y luego se ajusta según el flujo neto diario.

    Args:
        activity: DataFrame con actividad diaria

    Returns:
        DataFrame con disponibilidad (estaciones como columnas, fechas como filas)
    """
    logger.info("Creando series temporales de disponibilidad...")

    # Pivot: filas=fechas, columnas=estaciones, valores=net_change
    availability_changes = activity.pivot_table(
        index='date',
        columns='station_id',
        values='net_change',
        fill_value=0
    )

    # Ordenar por fecha
    availability_changes = availability_changes.sort_index()

    # Crear mapeo station_id -> station_name
    station_names = activity[['station_id', 'station_name']].drop_duplicates()
    station_name_map = dict(zip(station_names['station_id'], station_names['station_name']))

    # Calcular disponibilidad acumulada
    # Asumimos que cada estación comienza con un número proporcional de bicis
    # basado en su capacidad promedio
    total_stations = len(availability_changes.columns)
    initial_bikes_per_station = 20  # Estimación inicial

    # Inicializar disponibilidad
    availability = pd.DataFrame(
        initial_bikes_per_station,
        index=availability_changes.index,
        columns=availability_changes.columns
    )

    # Aplicar cambios acumulativos
    availability = availability + availability_changes.cumsum()

    # Asegurar valores no negativos (mínimo 0 bicis por estación)
    availability = availability.clip(lower=0)

    # Convertir a enteros
    availability = availability.round().astype(int)

    logger.info(f"Series temporales creadas:")
    logger.info(f"   Período: {availability.index.min()} a {availability.index.max()}")
    logger.info(f"   Días: {len(availability)}")
    logger.info(f"   Estaciones: {len(availability.columns)}")
    logger.info(f"   Disponibilidad promedio por estación: {availability.mean().mean():.2f} bicis")
    logger.info(f"   Total de bicis en sistema (promedio): {availability.sum(axis=1).mean():.0f}")

    return availability, station_name_map


def filter_low_activity_stations(
    availability: pd.DataFrame,
    min_threshold: int = 10
) -> pd.DataFrame:
    """
    Filtra estaciones con actividad muy baja

    Args:
        availability: DataFrame con disponibilidad
        min_threshold: Mínimo promedio de bicis para mantener estación

    Returns:
        DataFrame filtrado
    """
    logger.info(f"Filtrando estaciones con actividad < {min_threshold} bicis/dia...")

    # Calcular promedio por estación
    station_averages = availability.mean()

    # Seleccionar estaciones que superan el umbral
    active_stations = station_averages[station_averages >= min_threshold].index

    # Filtrar
    availability_filtered = availability[active_stations]

    logger.info(f"   Estaciones antes del filtrado: {len(availability.columns)}")
    logger.info(f"   Estaciones después del filtrado: {len(availability_filtered.columns)}")
    logger.info(f"   Estaciones eliminadas: {len(availability.columns) - len(availability_filtered.columns)}")

    return availability_filtered


def handle_missing_values(
    availability: pd.DataFrame,
    method: str = 'linear'
) -> pd.DataFrame:
    """
    Maneja valores faltantes mediante interpolación

    Args:
        availability: DataFrame con disponibilidad
        method: Método de interpolación ('linear', 'ffill', 'bfill')

    Returns:
        DataFrame sin valores faltantes
    """
    logger.info(f"Manejando valores faltantes (metodo: {method})...")

    missing_before = availability.isnull().sum().sum()

    if missing_before > 0:
        if method == 'linear':
            availability = availability.interpolate(method='linear', axis=0)
        elif method == 'ffill':
            availability = availability.fillna(method='ffill')
        elif method == 'bfill':
            availability = availability.fillna(method='bfill')

        # Llenar cualquier valor faltante restante con 0
        availability = availability.fillna(0)

    missing_after = availability.isnull().sum().sum()

    logger.info(f"   Valores faltantes antes: {missing_before}")
    logger.info(f"   Valores faltantes después: {missing_after}")

    return availability


def split_train_test(
    availability: pd.DataFrame,
    test_size: float = 0.2
) -> tuple:
    """
    Divide datos en train y test (división temporal)

    Args:
        availability: DataFrame con disponibilidad
        test_size: Proporción para test

    Returns:
        Tupla (train_df, test_df)
    """
    logger.info(f"Dividiendo datos (test_size={test_size})...")

    # División temporal
    split_idx = int(len(availability) * (1 - test_size))

    train = availability.iloc[:split_idx]
    test = availability.iloc[split_idx:]

    logger.info(f"   Train: {len(train)} días ({train.index.min()} a {train.index.max()})")
    logger.info(f"   Test: {len(test)} días ({test.index.min()} a {test.index.max()})")

    return train, test


def add_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Agrega características temporales al DataFrame

    Args:
        df: DataFrame con índice de fecha

    Returns:
        DataFrame con características temporales agregadas
    """
    logger.info("Agregando caracteristicas temporales...")

    # Convertir índice a datetime si no lo es
    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)

    # Características temporales
    df['day_of_week'] = df.index.dayofweek
    df['day_of_month'] = df.index.day
    df['month'] = df.index.month
    df['year'] = df.index.year
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

    logger.info(f"   {5} caracteristicas temporales agregadas")

    return df


def extract_climate_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extrae y prepara variables climáticas diarias para usar como variables exógenas

    Args:
        df: DataFrame con datos raw de viajes (debe contener 'starttime', 'temperature', 'events')

    Returns:
        DataFrame con temperatura media diaria y eventos climáticos one-hot encoded
        Índice: fechas, Columnas: temperature, weather_rain, weather_cloudy, etc.
    """
    logger.info("Extrayendo variables climaticas...")

    # Extraer fecha
    df['date'] = pd.to_datetime(df['starttime']).dt.date

    # Temperatura media diaria
    temp_daily = df.groupby('date')['temperature'].mean()
    temp_daily.name = 'temperature'

    logger.info(f"   Temperatura: min={temp_daily.min():.1f}, max={temp_daily.max():.1f}, mean={temp_daily.mean():.1f}")

    # Eventos climáticos - normalizar
    df['events_clean'] = df['events'].fillna('normal').str.lower().str.strip()

    # Obtener el evento más frecuente por día (moda)
    events_daily = df.groupby('date')['events_clean'].agg(
        lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 'normal'
    )

    # One-hot encoding de eventos
    if EXOG_CONFIG['weather_events_encoding'] == 'onehot':
        events_dummies = pd.get_dummies(events_daily, prefix='weather')
        logger.info(f"   Eventos climáticos (one-hot): {list(events_dummies.columns)}")
    else:
        # Label encoding
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        events_dummies = pd.DataFrame({
            'weather_encoded': le.fit_transform(events_daily)
        }, index=events_daily.index)
        logger.info(f"   Eventos climáticos (label): {dict(zip(le.classes_, range(len(le.classes_))))}")

    # Combinar temperatura y eventos
    climate_features = pd.concat([temp_daily, events_dummies], axis=1)

    # Convertir índice a DatetimeIndex
    climate_features.index = pd.to_datetime(climate_features.index)

    # Ordenar por fecha
    climate_features = climate_features.sort_index()

    logger.info(f"   Variables climaticas extraidas:")
    logger.info(f"      Período: {climate_features.index.min()} a {climate_features.index.max()}")
    logger.info(f"      Días: {len(climate_features)}")
    logger.info(f"      Variables: {list(climate_features.columns)}")

    return climate_features


def split_exog_train_test(
    exog: pd.DataFrame,
    train_index,
    test_index
) -> tuple:
    """
    Divide las variables exógenas en train y test según los índices proporcionados

    Args:
        exog: DataFrame con variables exógenas
        train_index: Índice del conjunto de entrenamiento
        test_index: Índice del conjunto de test

    Returns:
        Tupla (exog_train, exog_test)
    """
    # Convertir índices a datetime si es necesario
    train_index = pd.to_datetime(train_index)
    test_index = pd.to_datetime(test_index)

    # Filtrar exog según los índices
    exog_train = exog[exog.index.isin(train_index)]
    exog_test = exog[exog.index.isin(test_index)]

    return exog_train, exog_test


def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO PREPROCESAMIENTO DE DATOS")
    logger.info("=" * 80)

    # 1. Cargar datos raw
    logger.info("\nCargando datos raw...")
    df = load_raw_data(DATA_FILE)

    # 2. Calcular actividad diaria por estación
    activity = calculate_daily_station_activity(df)

    # 3. Crear series temporales de disponibilidad
    availability, station_name_map = create_availability_time_series(activity)

    # 4. Filtrar estaciones con baja actividad
    availability = filter_low_activity_stations(
        availability,
        min_threshold=PREPROCESSING_CONFIG['min_trips_threshold']
    )

    # 5. Manejo de valores faltantes
    availability = handle_missing_values(
        availability,
        method=PREPROCESSING_CONFIG['interpolation_method']
    )

    # 6. Calcular total de bicis en el sistema
    total_bikes = calculate_total_bikes(availability)

    # 7. Validar restricción global
    logger.info("\nValidando restriccion global...")
    daily_totals = availability.sum(axis=1)
    logger.info(f"   Total de bicis por día: min={daily_totals.min()}, max={daily_totals.max()}, "
                f"mean={daily_totals.mean():.0f}, std={daily_totals.std():.2f}")

    # 8. Extraer variables climáticas (exógenas)
    logger.info("\nExtrayendo variables climaticas para modelo VARX...")
    climate_features = extract_climate_features(df)

    # Alinear índices de climate_features con availability
    climate_features.index = pd.to_datetime(climate_features.index)
    availability.index = pd.to_datetime(availability.index)

    # Filtrar climate_features para que coincida con availability
    climate_features = climate_features[climate_features.index.isin(availability.index)]

    # 9. División train/test
    train, test = split_train_test(
        availability,
        test_size=PREPROCESSING_CONFIG['test_size']
    )

    # 10. Dividir variables exógenas en train/test
    exog_train, exog_test = split_exog_train_test(
        climate_features,
        train.index,
        test.index
    )

    logger.info(f"   Exog train: {len(exog_train)} días")
    logger.info(f"   Exog test: {len(exog_test)} días")

    # 11. Guardar datos procesados
    output_path = OUTPUTS_DIR / "processed_data"
    metadata = {
        'total_bikes': total_bikes,
        'num_stations': len(availability.columns),
        'train_period': (train.index.min(), train.index.max()),
        'test_period': (test.index.min(), test.index.max()),
        'station_name_map': station_name_map,
        'preprocessing_config': PREPROCESSING_CONFIG,
        'exog_config': EXOG_CONFIG,
        'exog_columns': list(climate_features.columns),
        'processed_at': datetime.now().isoformat(),
    }

    save_processed_data(train, test, output_path, metadata)

    # 12. Guardar variables exógenas
    import pickle
    exog_path = OUTPUTS_DIR / "exog_data"
    exog_train.to_pickle(f"{exog_path}_train.pkl")
    exog_test.to_pickle(f"{exog_path}_test.pkl")
    logger.info(f"   Variables exogenas guardadas en: {exog_path}_train.pkl, {exog_path}_test.pkl")

    # 13. Guardar mapeo de nombres de estaciones
    station_mapping_df = pd.DataFrame([
        {'station_id': sid, 'station_name': sname}
        for sid, sname in station_name_map.items()
        if sid in availability.columns
    ])
    station_mapping_df.to_csv(OUTPUTS_DIR / 'station_mapping.csv', index=False)

    logger.info("\n" + "=" * 80)
    logger.info("PREPROCESAMIENTO COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"Datos guardados en: {OUTPUTS_DIR}")
    logger.info(f"Total de bicis en sistema: {total_bikes}")
    logger.info(f"Numero de estaciones: {len(availability.columns)}")
    logger.info(f"Variables exogenas: {list(climate_features.columns)}")
    logger.info(f"Periodo total: {availability.index.min()} a {availability.index.max()}")


if __name__ == "__main__":
    main()
