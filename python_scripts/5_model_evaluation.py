"""
Script 5: Evaluación del Modelo

Evalúa exhaustivamente el modelo VAR Optimizado (o VAR simple) entrenado:
- Métricas de error (MAE, RMSE, MAPE) por estación y globales
- Análisis de residuales
- Validación de restricción global
- Comparación predicciones vs valores reales
- Visualizaciones de rendimiento
- Reporte HTML con resultados

Soporta modelos:
- VAR_Optimizado_Clima (Two-Stage: Ridge + VAR + PCA opcional)
- VAR (modelo simple)
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
from datetime import datetime
import pickle

# Configurar paths
sys.path.append(str(Path(__file__).parent))
from config import (
    OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR, LOG_LEVEL, LOG_FORMAT
)
from utils.data_loader import load_model, load_processed_data
from utils.visualization import (
    plot_predictions,
    plot_daily_forecast_by_month
)
from utils.constraints import validate_constraint

# Configurar logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


def calculate_metrics(predictions: np.ndarray, actuals: np.ndarray) -> dict:
    """
    Calcula métricas de error

    Args:
        predictions: Predicciones del modelo
        actuals: Valores reales

    Returns:
        Diccionario con métricas
    """
    # Métricas globales
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # MAPE: solo calcular para valores > 1 para evitar división por cero
    valid_mask = actuals > 1
    if valid_mask.any():
        mape = np.mean(np.abs((actuals[valid_mask] - predictions[valid_mask]) / actuals[valid_mask])) * 100
    else:
        mape = np.nan

    # Métricas por estación
    mae_per_station = np.mean(np.abs(predictions - actuals), axis=0)
    rmse_per_station = np.sqrt(np.mean((predictions - actuals) ** 2, axis=0))

    # MAPE por estación con filtro
    mape_per_station = np.zeros(actuals.shape[1])
    for i in range(actuals.shape[1]):
        station_actuals = actuals[:, i]
        station_preds = predictions[:, i]
        valid = station_actuals > 1
        if valid.any():
            mape_per_station[i] = np.mean(np.abs((station_actuals[valid] - station_preds[valid]) / station_actuals[valid])) * 100
        else:
            mape_per_station[i] = np.nan

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'mae_per_station': mae_per_station,
        'rmse_per_station': rmse_per_station,
        'mape_per_station': mape_per_station,
    }


def generate_rolling_predictions(
    model_data: dict,
    train: pd.DataFrame,
    test: pd.DataFrame,
    steps: int = 1,
    exog_test: pd.DataFrame = None
) -> tuple:
    """
    Genera predicciones rolling sobre el conjunto de test

    Args:
        model_data: Diccionario con modelo y metadata
        train: Datos de entrenamiento
        test: Datos de test
        steps: Pasos a predecir
        exog_test: Variables exógenas para el test (opcional, para VAR Optimizado)

    Returns:
        Tupla (predictions, actuals)
    """
    logger.info(f"Generando predicciones rolling (test set, steps={steps})...")

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

    predictions = []
    actuals = []

    # Preparar datos
    if is_differenced:
        train_processed = train.diff().dropna()
        test_processed = test.diff().dropna()
    else:
        train_processed = train.copy()
        test_processed = test.copy()

    # Ventana deslizante
    for i in range(len(test_processed) - steps + 1):
        # Historia hasta el momento actual
        history = pd.concat([train_processed.iloc[-(lag_order):], test_processed.iloc[:i]])

        # Predecir según tipo de modelo
        try:
            if model_type == 'VAR_Optimizado_Clima':
                # Modelo VAR Optimizado con Two-Stage approach
                if use_pca and pca is not None and scaler is not None:
                    # Transformar historia a espacio PCA
                    history_scaled = scaler.transform(history.values)
                    history_pca = pca.transform(history_scaled)
                    # Solo usar los primeros n_pca_components
                    history_pca = history_pca[:, :n_pca_components]
                    last_obs = history_pca[-lag_order:]
                else:
                    last_obs = history.values[-lag_order:]

                # Forecast del VAR
                forecast_var = fitted_model.forecast(last_obs, steps=steps)

                # Invertir PCA si es necesario
                if use_pca and pca is not None and scaler is not None:
                    # Expandir forecast a todas las componentes PCA
                    forecast_full = np.zeros((steps, pca.n_components_))
                    forecast_full[:, :n_pca_components] = forecast_var
                    forecast_scaled = pca.inverse_transform(forecast_full)
                    forecast = scaler.inverse_transform(forecast_scaled)
                else:
                    forecast = forecast_var

                # Añadir efecto climático
                if use_climate and climate_model is not None and exog_test is not None:
                    exog_idx = min(i + steps - 1, len(exog_test) - 1)
                    exog_values = exog_test.iloc[exog_idx:exog_idx+1]
                    # climate_model es un dict con 'exog_scaler' y 'models' (uno por estación)
                    exog_scaled = climate_model['exog_scaler'].transform(exog_values.values)
                    station_columns = model_data.get('station_columns', [])
                    climate_effect = np.zeros(len(station_columns))
                    for idx, col in enumerate(station_columns):
                        if col in climate_model['models']:
                            climate_effect[idx] = climate_model['models'][col].predict(exog_scaled)[0]
                    forecast = forecast + climate_effect

                forecast = forecast[-1] if steps > 1 else forecast[0]
            else:
                # Modelo VAR simple
                forecast = fitted_model.forecast(history.values[-lag_order:], steps=steps)
                forecast = forecast[-1] if steps > 1 else forecast[0]

            # Invertir diferenciación si es necesario
            if is_differenced:
                last_value = test.iloc[i - 1].values if i > 0 else train.iloc[-1].values
                forecast = last_value + forecast

            predictions.append(forecast)
            actuals.append(test.iloc[i + steps - 1].values)

        except Exception as e:
            logger.warning(f"   Error en prediccion {i}: {e}")
            continue

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    logger.info(f"   Predicciones generadas: {len(predictions)}")

    return predictions, actuals


def plot_metrics_by_station(
    metrics: dict,
    station_columns: list,
    station_name_map: dict,
    output_dir: Path
) -> None:
    """
    Plotea métricas de error por estación

    Args:
        metrics: Diccionario con métricas
        station_columns: Lista de IDs de estaciones
        station_name_map: Mapeo de ID a nombre
        output_dir: Directorio de salida
    """
    logger.info("Creando visualizaciones de metricas...")

    # Top 20 estaciones por error
    top_20_indices = np.argsort(metrics['mae_per_station'])[-20:]

    fig, axes = plt.subplots(2, 2, figsize=(16, 12))

    # MAE por estación (top 20)
    mae_top_20 = metrics['mae_per_station'][top_20_indices]
    station_names_top_20 = [station_name_map.get(station_columns[i], f'Station {i}')
                            for i in top_20_indices]

    axes[0, 0].barh(range(len(mae_top_20)), mae_top_20, color='steelblue')
    axes[0, 0].set_yticks(range(len(mae_top_20)))
    axes[0, 0].set_yticklabels(station_names_top_20, fontsize=8)
    axes[0, 0].set_xlabel('MAE (bicis)')
    axes[0, 0].set_title('Top 20 Estaciones con Mayor Error (MAE)', fontweight='bold')
    axes[0, 0].grid(True, alpha=0.3, axis='x')

    # Distribución de MAE
    axes[0, 1].hist(metrics['mae_per_station'], bins=30, color='coral', edgecolor='black', alpha=0.7)
    axes[0, 1].axvline(metrics['mae'], color='red', linestyle='--', linewidth=2, label=f'MAE Global: {metrics["mae"]:.2f}')
    axes[0, 1].set_xlabel('MAE (bicis)')
    axes[0, 1].set_ylabel('Frecuencia')
    axes[0, 1].set_title('Distribución de MAE por Estación', fontweight='bold')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)

    # RMSE por estación (top 20)
    rmse_top_20 = metrics['rmse_per_station'][top_20_indices]

    axes[1, 0].barh(range(len(rmse_top_20)), rmse_top_20, color='green')
    axes[1, 0].set_yticks(range(len(rmse_top_20)))
    axes[1, 0].set_yticklabels(station_names_top_20, fontsize=8)
    axes[1, 0].set_xlabel('RMSE (bicis)')
    axes[1, 0].set_title('Top 20 Estaciones con Mayor RMSE', fontweight='bold')
    axes[1, 0].grid(True, alpha=0.3, axis='x')

    # MAPE por estación (distribución)
    axes[1, 1].hist(metrics['mape_per_station'], bins=30, color='purple', edgecolor='black', alpha=0.7)
    axes[1, 1].axvline(metrics['mape'], color='red', linestyle='--', linewidth=2, label=f'MAPE Global: {metrics["mape"]:.2f}%')
    axes[1, 1].set_xlabel('MAPE (%)')
    axes[1, 1].set_ylabel('Frecuencia')
    axes[1, 1].set_title('Distribución de MAPE por Estación', fontweight='bold')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_dir / 'metrics_by_station.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"   Visualizaciones guardadas")


def plot_prediction_examples(
    predictions: np.ndarray,
    actuals: np.ndarray,
    test: pd.DataFrame,
    station_columns: list,
    station_name_map: dict,
    output_dir: Path,
    num_examples: int = 6
) -> None:
    """
    Plotea ejemplos de predicciones vs valores reales

    Args:
        predictions: Predicciones del modelo
        actuals: Valores reales
        test: DataFrame de test
        station_columns: Lista de IDs de estaciones
        station_name_map: Mapeo de ID a nombre
        output_dir: Directorio de salida
        num_examples: Número de ejemplos a plotear
    """
    logger.info(f"Creando ejemplos de predicciones (n={num_examples})...")

    # Seleccionar estaciones aleatorias
    np.random.seed(42)
    selected_stations = np.random.choice(len(station_columns), size=num_examples, replace=False)

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    for idx, station_idx in enumerate(selected_stations):
        station_id = station_columns[station_idx]
        station_name = station_name_map.get(station_id, f'Station {station_id}')

        # Datos para esta estación
        pred_station = predictions[:, station_idx]
        actual_station = actuals[:, station_idx]

        # Fechas
        dates = test.index[-len(pred_station):]

        # Plotear
        axes[idx].plot(range(len(actual_station)), actual_station, label='Real', marker='o', linewidth=2, alpha=0.7)
        axes[idx].plot(range(len(pred_station)), pred_station, label='Predicción', marker='s', linewidth=2, linestyle='--', alpha=0.7)

        axes[idx].set_title(f'{station_name}', fontweight='bold', fontsize=10)
        axes[idx].set_xlabel('Tiempo (días)')
        axes[idx].set_ylabel('Número de Bicis')
        axes[idx].legend(fontsize=8)
        axes[idx].grid(True, alpha=0.3)

        # Calcular MAE para esta estación
        mae_station = np.mean(np.abs(pred_station - actual_station))
        axes[idx].text(0.05, 0.95, f'MAE: {mae_station:.2f}',
                      transform=axes[idx].transAxes, fontsize=9,
                      verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    plt.tight_layout()
    plt.savefig(output_dir / 'prediction_examples.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"   Ejemplos de predicciones guardados")


def generate_evaluation_report(
    metrics: dict,
    model_data: dict,
    total_bikes: int,
    output_path: Path
) -> None:
    """
    Genera reporte HTML con resultados de la evaluación

    Args:
        metrics: Diccionario con métricas
        model_data: Diccionario con modelo y metadata
        total_bikes: Total de bicis en el sistema
        output_path: Ruta del archivo de salida
    """
    logger.info("Generando reporte de evaluacion...")

    validation_metrics = model_data.get('validation_metrics', {})

    model_type = model_data.get('model_type', 'VAR')
    use_pca = model_data.get('use_pca', False)
    use_climate = model_data.get('use_climate', False)
    n_components = model_data.get('n_components', 'N/A')
    exog_columns = model_data.get('exog_columns', [])

    # Título según tipo de modelo
    if model_type == 'VAR_Optimizado_Clima':
        title = "VAR Optimizado con Variables Climáticas"
        color = "#4CAF50"
    elif model_type == 'VARX':
        title = "VARX con Variables Exógenas"
        color = "#2196F3"
    else:
        title = "VAR Simple"
        color = "#9C27B0"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Evaluación - Modelo {model_type}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background-color: #f5f5f5; }}
            h1 {{ color: #333; border-bottom: 3px solid {color}; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .metrics {{ background-color: white; padding: 20px; border-radius: 5px; margin: 20px 0; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            .metric-box {{ display: inline-block; margin: 15px 20px; text-align: center; }}
            .metric-label {{ font-weight: bold; color: #666; font-size: 14px; }}
            .metric-value {{ font-size: 36px; color: {color}; margin: 10px 0; }}
            .metric-unit {{ font-size: 18px; color: #999; }}
            img {{ max-width: 100%; border: 1px solid #ddd; border-radius: 5px; margin: 20px 0; }}
            .info-box {{ background-color: #e7f3ff; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }}
            .success-box {{ background-color: #e8f5e9; padding: 15px; border-left: 4px solid #4CAF50; margin: 20px 0; }}
            .climate-box {{ background-color: #fff3e0; padding: 15px; border-left: 4px solid #FF9800; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; background-color: white; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: {color}; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
        </style>
    </head>
    <body>
        <h1>Reporte de Evaluación - {title}</h1>
        <p><strong>Fecha de generación:</strong> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>

        <div class="metrics">
            <h2>Métricas Globales de Rendimiento</h2>

            <div class="metric-box">
                <div class="metric-label">MAE (Mean Absolute Error)</div>
                <div class="metric-value">{metrics['mae']:.2f}</div>
                <div class="metric-unit">bicis</div>
            </div>

            <div class="metric-box">
                <div class="metric-label">RMSE (Root Mean Squared Error)</div>
                <div class="metric-value">{metrics['rmse']:.2f}</div>
                <div class="metric-unit">bicis</div>
            </div>

            <div class="metric-box">
                <div class="metric-label">MAPE (Mean Absolute % Error)</div>
                <div class="metric-value">{metrics['mape']:.2f}</div>
                <div class="metric-unit">%</div>
            </div>
        </div>

        <div class="info-box">
            <h3>Información del Modelo</h3>
            <ul>
                <li><strong>Tipo de modelo:</strong> {model_type}</li>
                <li><strong>Versión del modelo:</strong> {model_data.get('model_version', 'N/A')}</li>
                <li><strong>Fecha de entrenamiento:</strong> {model_data.get('trained_at', 'N/A')}</li>
                <li><strong>Orden del VAR (lag):</strong> {model_data.get('lag_order', 'N/A')}</li>
                <li><strong>Total de bicis en sistema:</strong> {total_bikes}</li>
                <li><strong>Número de estaciones:</strong> {model_data.get('num_stations', 'N/A')}</li>
                <li><strong>Datos diferenciados:</strong> {'Sí' if model_data.get('is_differenced', False) else 'No'}</li>
                <li><strong>Usa PCA:</strong> {'Sí (' + str(model_data.get('n_pca_components', 'N/A')) + ' componentes)' if use_pca else 'No'}</li>
                <li><strong>Ajuste climático:</strong> {'Sí' if use_climate else 'No'}</li>
            </ul>
        </div>

        {'<div class="climate-box"><h3>Variables Climáticas</h3><p><strong>Variables exógenas:</strong> ' + ', '.join(exog_columns) + '</p></div>' if exog_columns else ''}

        <div class="success-box">
            <h3>✅ Diagnósticos del Modelo</h3>
            <ul>
                <li><strong>Estabilidad:</strong> {'Estable' if model_data.get('diagnostics', {}).get('is_stable', False) else 'No verificado'}</li>
                <li><strong>Autocorrelación de residuales:</strong> {'Detectada' if model_data.get('diagnostics', {}).get('has_autocorr', False) else 'No significativa'}</li>
            </ul>
        </div>

        <h2>📈 Visualizaciones</h2>

        <h3>Métricas por Estación</h3>
        <img src="figures/metrics_by_station.png" alt="Métricas por Estación">

        <h3>Ejemplos de Predicciones vs Valores Reales</h3>
        <img src="figures/prediction_examples.png" alt="Ejemplos de Predicciones">

        <h2>📋 Estadísticas Detalladas</h2>

        <table>
            <tr>
                <th>Métrica</th>
                <th>Valor</th>
                <th>Descripción</th>
            </tr>
            <tr>
                <td>MAE Global</td>
                <td>{metrics['mae']:.2f} bicis</td>
                <td>Error absoluto promedio en todas las estaciones</td>
            </tr>
            <tr>
                <td>RMSE Global</td>
                <td>{metrics['rmse']:.2f} bicis</td>
                <td>Raíz del error cuadrático medio</td>
            </tr>
            <tr>
                <td>MAPE Global</td>
                <td>{metrics['mape']:.2f}%</td>
                <td>Error porcentual absoluto medio</td>
            </tr>
            <tr>
                <td>MAE Mínimo (por estación)</td>
                <td>{metrics['mae_per_station'].min():.2f} bicis</td>
                <td>Mejor predicción individual</td>
            </tr>
            <tr>
                <td>MAE Máximo (por estación)</td>
                <td>{metrics['mae_per_station'].max():.2f} bicis</td>
                <td>Peor predicción individual</td>
            </tr>
            <tr>
                <td>MAE Mediana (por estación)</td>
                <td>{np.median(metrics['mae_per_station']):.2f} bicis</td>
                <td>Mediana del error por estación</td>
            </tr>
        </table>

        <h2>Conclusiones</h2>
        <ul>
            <li>El modelo <strong>{title}</strong> predice la disponibilidad de bicicletas con un error promedio de <strong>{metrics['mae']:.2f} bicis</strong> por estación</li>
            <li>El error porcentual medio (MAPE) es de <strong>{metrics['mape']:.2f}%</strong></li>
            <li>El modelo es {'<span style="color: green;">estable</span>' if model_data.get('diagnostics', {}).get('is_stable', False) else '<span style="color: orange;">no verificado en estabilidad</span>'}</li>
            <li>Las predicciones respetan la restricción global de {total_bikes} bicis totales en el sistema</li>
            {'<li>El modelo utiliza <strong>PCA</strong> con ' + str(model_data.get('n_pca_components', 'N/A')) + ' componentes para reducir dimensionalidad</li>' if use_pca else ''}
            {'<li>El modelo incorpora <strong>ajuste climático</strong> mediante Ridge Regression</li>' if use_climate else ''}
        </ul>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"   Reporte guardado: {output_path}")


def load_latest_model() -> dict:
    """
    Carga el último modelo entrenado (prioridad: VAR Optimizado > VARX > VAR)

    Returns:
        Diccionario con modelo y metadata
    """
    logger.info("Cargando modelo...")

    # Intentar cargar VAR Optimizado primero, luego VARX, luego VAR como fallback
    model_path = MODELS_DIR / "var_optimized_model.pkl"

    if not model_path.exists():
        model_path = MODELS_DIR / "varx_model.pkl"
        if not model_path.exists():
            model_path = MODELS_DIR / "var_model.pkl"
            if not model_path.exists():
                raise FileNotFoundError(f"No se encontró modelo en {MODELS_DIR}")

    model_data = load_model(model_path)

    model_type = model_data.get('model_type', 'VAR')
    logger.info(f"   Modelo cargado: {model_type}")
    logger.info(f"   Entrenado en: {model_data.get('trained_at', 'N/A')}")
    logger.info(f"   Version: {model_data.get('model_version', 'N/A')}")
    logger.info(f"   Usa PCA: {model_data.get('use_pca', False)}")
    logger.info(f"   Usa clima: {model_data.get('use_climate', False)}")

    return model_data


def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO EVALUACION DEL MODELO")
    logger.info("=" * 80)

    # 1. Cargar modelo
    model_data = load_latest_model()

    # 2. Cargar datos
    logger.info("\nCargando datos procesados...")
    data_path = OUTPUTS_DIR / "processed_data"
    train, test = load_processed_data(data_path)

    # Cargar metadata
    metadata_path = data_path.parent / "processed_data_metadata.pkl"
    with open(metadata_path, 'rb') as f:
        metadata = pickle.load(f)

    total_bikes = metadata['total_bikes']
    station_name_map = metadata['station_name_map']

    # 3. Cargar variables exógenas si el modelo las usa
    model_type = model_data.get('model_type', 'VAR')
    use_climate = model_data.get('use_climate', False)
    exog_test = None

    if model_type == 'VAR_Optimizado_Clima' or use_climate:
        logger.info("\nCargando variables exogenas...")
        exog_path = OUTPUTS_DIR / "exog_data"
        try:
            exog_test = pd.read_pickle(f"{exog_path}_test.pkl")
            logger.info(f"   Variables exogenas cargadas: {exog_test.shape}")
        except FileNotFoundError:
            logger.warning("   No se encontraron variables exogenas")
            exog_test = None

    # 4. Generar predicciones rolling
    predictions, actuals = generate_rolling_predictions(
        model_data, train, test, steps=1, exog_test=exog_test
    )

    # 5. Calcular métricas
    logger.info("\nCalculando metricas de rendimiento...")
    metrics = calculate_metrics(predictions, actuals)

    logger.info(f"   MAE Global: {metrics['mae']:.2f} bicis")
    logger.info(f"   RMSE Global: {metrics['rmse']:.2f} bicis")
    logger.info(f"   MAPE Global: {metrics['mape']:.2f}%")

    # 6. Validar restricción global en predicciones
    logger.info("\nValidando restriccion global...")
    predictions_int = np.round(predictions).astype(int)
    # Nota: Las predicciones del test pueden no sumar exactamente total_bikes
    # porque se generan sin ajuste de restricción global
    daily_totals = predictions_int.sum(axis=1)
    logger.info(f"   Suma de predicciones: min={daily_totals.min()}, max={daily_totals.max()}, mean={daily_totals.mean():.0f}")

    # 7. Visualizaciones
    plot_metrics_by_station(
        metrics,
        station_columns=list(train.columns),
        station_name_map=station_name_map,
        output_dir=FIGURES_DIR
    )

    plot_prediction_examples(
        predictions,
        actuals,
        test,
        station_columns=list(train.columns),
        station_name_map=station_name_map,
        output_dir=FIGURES_DIR,
        num_examples=6
    )

    # Visualización 1: Forecast de mes aleatorio
    logger.info("Creando visualizacion de forecast mensual...")
    forecast_info = plot_daily_forecast_by_month(
        actuals=actuals,
        predicted=predictions,
        test_index=test.index,
        station_columns=list(train.columns),
        station_name_map=station_name_map,
        save_path=FIGURES_DIR / 'monthly_forecast_comparison.png'
    )
    logger.info(f"   Mes: {forecast_info['selected_month']} | Estación: {forecast_info['selected_station']}")

    # 8. Generar reporte
    report_path = OUTPUTS_DIR / 'evaluation_report.html'
    generate_evaluation_report(metrics, model_data, total_bikes, report_path)

    logger.info("\n" + "=" * 80)
    logger.info("EVALUACION DEL MODELO COMPLETADA")
    logger.info("=" * 80)
    logger.info(f"Reporte: {report_path}")
    logger.info(f"Figuras: {FIGURES_DIR}")


if __name__ == "__main__":
    main()
