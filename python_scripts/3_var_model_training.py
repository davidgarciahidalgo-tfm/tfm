"""
Script 3: Entrenamiento del Modelo VAR (Vector Autoregression)

Entrena un modelo VAR para predecir la disponibilidad de bicicletas en múltiples
estaciones simultáneamente.

Pasos:
1. Cargar datos procesados (train/test)
2. Verificar estacionariedad (ADF test)
3. Aplicar diferenciación si es necesario
4. Seleccionar orden óptimo del VAR (lag) usando AIC/BIC
5. Entrenar modelo VAR
6. Diagnósticos del modelo (residuales, estabilidad)
7. Validación en conjunto de test
8. Guardar modelo entrenado
"""
import pandas as pd
import numpy as np
from pathlib import Path
import logging
import sys
from datetime import datetime
import pickle
import warnings
warnings.filterwarnings('ignore')

# Statsmodels para VAR
from statsmodels.tsa.stattools import adfuller, grangercausalitytests
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.metrics import r2_score

# Configurar paths
sys.path.append(str(Path(__file__).parent))
from config import (
    OUTPUTS_DIR, MODELS_DIR, FIGURES_DIR, VAR_CONFIG,
    RANDOM_SEED, LOG_LEVEL, LOG_FORMAT
)
from utils.data_loader import load_processed_data, save_model
from utils.visualization import plot_residuals

# Configurar logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configurar semilla
np.random.seed(RANDOM_SEED)


def test_stationarity(data: pd.DataFrame, max_to_test: int = 20) -> dict:
    """
    Test de estacionariedad (Augmented Dickey-Fuller) para cada serie

    Args:
        data: DataFrame con series temporales
        max_to_test: Máximo número de series a testear

    Returns:
        Diccionario con resultados
    """
    logger.info("Testeando estacionariedad (ADF test)...")

    results = {}
    columns_to_test = data.columns[:max_to_test]

    stationary_count = 0
    non_stationary_count = 0

    for col in columns_to_test:
        try:
            result = adfuller(data[col].dropna(), autolag='AIC')
            adf_statistic = result[0]
            p_value = result[1]

            is_stationary = p_value < 0.05

            results[col] = {
                'adf_statistic': adf_statistic,
                'p_value': p_value,
                'is_stationary': is_stationary
            }

            if is_stationary:
                stationary_count += 1
            else:
                non_stationary_count += 1

        except Exception as e:
            logger.warning(f"   No se pudo testear estacion {col}: {e}")

    logger.info(f"   Estacionarias: {stationary_count}/{len(columns_to_test)}")
    logger.info(f"   No estacionarias: {non_stationary_count}/{len(columns_to_test)}")

    if non_stationary_count > stationary_count:
        logger.warning("   La mayoria de las series NO son estacionarias")
        logger.warning("   Se recomienda aplicar diferenciacion")
        return {'mostly_stationary': False, 'results': results}
    else:
        logger.info("   La mayoria de las series son estacionarias")
        return {'mostly_stationary': True, 'results': results}


def apply_differencing(data: pd.DataFrame, order: int = 1) -> pd.DataFrame:
    """
    Aplica diferenciación a las series para hacerlas estacionarias

    Args:
        data: DataFrame original
        order: Orden de diferenciación

    Returns:
        DataFrame diferenciado
    """
    logger.info(f"Aplicando diferenciacion de orden {order}...")

    differenced = data.diff(order).dropna()

    logger.info(f"   Forma original: {data.shape}")
    logger.info(f"   Forma diferenciada: {differenced.shape}")

    return differenced


def inverse_differencing(
    diff_predictions: np.ndarray,
    last_original_values: np.ndarray
) -> np.ndarray:
    """
    Invierte la diferenciación para obtener predicciones en escala original.

    Para predicciones de 1 paso adelante:
    y_t = y_{t-1} + diff_t

    Args:
        diff_predictions: Array de predicciones diferenciadas (cambios)
        last_original_values: Últimos valores conocidos en escala original

    Returns:
        Predicciones en escala original
    """
    return diff_predictions + last_original_values


def select_var_order(data: pd.DataFrame, max_lags: int = 14) -> int:
    """
    Selecciona el orden óptimo del VAR usando criterios de información

    Args:
        data: DataFrame con series temporales
        max_lags: Máximo número de lags a probar

    Returns:
        Orden óptimo seleccionado
    """
    logger.info(f"Seleccionando orden optimo del VAR (max_lags={max_lags})...")

    # Crear modelo VAR
    model = VAR(data)

    # Buscar orden óptimo
    logger.info("   Probando diferentes órdenes (esto puede tardar)...")

    try:
        # Seleccionar orden usando AIC, BIC, FPE, HQIC
        results = model.select_order(maxlags=max_lags)

        logger.info(f"   Criterios de informacion:")
        logger.info(f"      AIC: {results.aic}")
        logger.info(f"      BIC: {results.bic}")
        logger.info(f"      FPE: {results.fpe}")
        logger.info(f"      HQIC: {results.hqic}")

        # Usar criterio configurado (por defecto AIC)
        if VAR_CONFIG['ic'] == 'aic':
            optimal_lag = results.aic
        elif VAR_CONFIG['ic'] == 'bic':
            optimal_lag = results.bic
        elif VAR_CONFIG['ic'] == 'fpe':
            optimal_lag = results.fpe
        else:
            optimal_lag = results.hqic

        logger.info(f"   Orden optimo seleccionado ({VAR_CONFIG['ic'].upper()}): {optimal_lag}")

        return optimal_lag

    except Exception as e:
        logger.warning(f"   Error al seleccionar orden: {e}")
        logger.info("   Usando orden por defecto: 7")
        return 7


def train_var_model(data: pd.DataFrame, lag_order: int, trend: str = 'c') -> VAR:
    """
    Entrena el modelo VAR

    Args:
        data: DataFrame con series temporales
        lag_order: Orden del VAR (número de lags)
        trend: Tipo de tendencia ('c', 'ct', 'n')

    Returns:
        Modelo VAR entrenado
    """
    logger.info(f"Entrenando modelo VAR...")
    logger.info(f"   Orden (lag): {lag_order}")
    logger.info(f"   Tendencia: {trend}")
    logger.info(f"   Número de variables: {data.shape[1]}")
    logger.info(f"   Observaciones: {len(data)}")

    # Crear y entrenar modelo
    model = VAR(data)
    fitted_model = model.fit(lag_order, trend=trend)

    logger.info(f"   Modelo entrenado correctamente")

    # Intentar obtener criterios de información (puede fallar si matriz no es positiva definida)
    try:
        logger.info(f"   AIC: {fitted_model.aic:.2f}")
        logger.info(f"   BIC: {fitted_model.bic:.2f}")
    except np.linalg.LinAlgError:
        logger.warning(f"   No se pudieron calcular AIC/BIC (matriz no positiva definida)")

    return fitted_model


def diagnose_model(fitted_model, data: pd.DataFrame) -> dict:
    """
    Diagnósticos del modelo VAR

    Args:
        fitted_model: Modelo VAR ajustado
        data: Datos de entrenamiento

    Returns:
        Diccionario con resultados de diagnósticos
    """
    logger.info("Realizando diagnosticos del modelo...")

    diagnostics = {}

    # 1. Residuales
    residuals = fitted_model.resid

    logger.info(f"   Analisis de residuales:")
    logger.info(f"      Media: {residuals.mean().mean():.4f}")
    logger.info(f"      Std: {residuals.std().mean():.4f}")

    diagnostics['residuals'] = residuals

    # 2. Test de estabilidad
    # Verificar que las raíces características estén dentro del círculo unitario
    try:
        # Obtener raíces del polinomio característico
        eigenvalues = fitted_model.stability()
        is_stable = np.all(eigenvalues < 1)

        logger.info(f"   Test de estabilidad:")
        logger.info(f"      Raíces dentro del círculo unitario: {'Sí' if is_stable else 'No'}")

        if is_stable:
            logger.info(f"      El modelo es ESTABLE")
        else:
            logger.warning(f"      El modelo podria ser INESTABLE")

        diagnostics['is_stable'] = is_stable

    except Exception as e:
        logger.warning(f"   No se pudo verificar estabilidad: {e}")
        diagnostics['is_stable'] = None

    # 3. Autocorrelación de residuales (prueba de Ljung-Box)
    try:
        lb_test = acorr_ljungbox(residuals.iloc[:, 0], lags=10, return_df=True)
        has_autocorr = any(lb_test['lb_pvalue'] < 0.05)

        logger.info(f"   Test de Ljung-Box (autocorrelacion de residuales):")
        logger.info(f"      Autocorrelación significativa: {'Sí' if has_autocorr else 'No'}")

        if not has_autocorr:
            logger.info(f"      Los residuales NO presentan autocorrelacion significativa")
        else:
            logger.warning(f"      Los residuales presentan autocorrelacion")

        diagnostics['has_autocorr'] = has_autocorr

    except Exception as e:
        logger.warning(f"   No se pudo realizar test de Ljung-Box: {e}")
        diagnostics['has_autocorr'] = None

    # 4. Visualizar residuales
    plot_residuals(
        residuals.values,
        title="Diagnóstico de Residuales - Modelo VAR",
        save_path=FIGURES_DIR / 'var_residuals.png'
    )

    return diagnostics


def generate_training_report(
    model_metadata: dict,
    validation_results: dict,
    diagnostics: dict,
    output_path: Path
) -> None:
    """
    Genera un reporte HTML con los resultados del entrenamiento del modelo VAR

    Args:
        model_metadata: Metadatos del modelo entrenado
        validation_results: Resultados de validación
        diagnostics: Resultados de diagnósticos
        output_path: Ruta donde guardar el reporte HTML
    """
    logger.info("Generando reporte HTML de entrenamiento...")

    # Extraer métricas
    mae = validation_results.get('mae', 0)
    rmse = validation_results.get('rmse', 0)
    mape = validation_results.get('mape', 0)
    r2_avg = validation_results.get('r2_avg', 0)
    r2_per_station = validation_results.get('r2_per_station', {})

    # Generar tabla de R² por estación (top 10 mejores y peores)
    r2_sorted = sorted(
        r2_per_station.items(),
        key=lambda x: x[1] if not np.isnan(x[1]) else -999,
        reverse=True
    )

    top_stations_html = ""
    for station, r2 in r2_sorted[:10]:
        r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        top_stations_html += f"<tr><td>{station}</td><td>{r2_display}</td></tr>\n"

    bottom_stations_html = ""
    for station, r2 in r2_sorted[-10:]:
        r2_display = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        bottom_stations_html += f"<tr><td>{station}</td><td>{r2_display}</td></tr>\n"

    # Estado de estabilidad
    is_stable = diagnostics.get('is_stable', None)
    stability_status = "Estable" if is_stable else ("Inestable" if is_stable is False else "No verificado")
    stability_color = "#4CAF50" if is_stable else ("#f44336" if is_stable is False else "#ff9800")

    # Autocorrelación
    has_autocorr = diagnostics.get('has_autocorr', None)
    autocorr_status = "Sí" if has_autocorr else ("No" if has_autocorr is False else "No verificado")
    autocorr_color = "#f44336" if has_autocorr else ("#4CAF50" if has_autocorr is False else "#ff9800")

    # Formatear valores
    mae_display = f"{mae:.4f}" if not np.isnan(mae) else "N/A"
    rmse_display = f"{rmse:.4f}" if not np.isnan(rmse) else "N/A"
    mape_display = f"{mape:.2f}%" if not np.isnan(mape) else "N/A"
    r2_display = f"{r2_avg:.4f}" if not np.isnan(r2_avg) else "N/A"

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte de Entrenamiento - Modelo VAR</title>
        <meta charset="UTF-8">
        <style>
            body {{
                font-family: Arial, sans-serif;
                margin: 40px;
                background-color: #f5f5f5;
            }}
            h1 {{
                color: #333;
                border-bottom: 3px solid #2196F3;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #555;
                margin-top: 30px;
                border-bottom: 2px solid #ddd;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #666;
                margin-top: 15px;
            }}
            .container {{
                max-width: 1200px;
                margin: 0 auto;
                background-color: white;
                padding: 30px;
                border-radius: 10px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .metrics {{
                display: flex;
                flex-wrap: wrap;
                gap: 20px;
                margin: 20px 0;
            }}
            .metric-box {{
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                color: white;
                padding: 20px;
                border-radius: 10px;
                text-align: center;
                min-width: 150px;
                flex: 1;
            }}
            .metric-box h3 {{
                margin: 0 0 10px 0;
                font-size: 14px;
                opacity: 0.9;
                color: white;
            }}
            .metric-box .value {{
                font-size: 28px;
                font-weight: bold;
            }}
            .info-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
                gap: 20px;
                margin: 20px 0;
            }}
            .info-box {{
                background-color: #f8f9fa;
                padding: 20px;
                border-radius: 8px;
                border-left: 4px solid #2196F3;
            }}
            .info-box h3 {{
                margin-top: 0;
                color: #333;
            }}
            .info-box p {{
                margin: 8px 0;
                color: #666;
            }}
            .info-box strong {{
                color: #333;
            }}
            .status-badge {{
                padding: 5px 15px;
                border-radius: 20px;
                color: white;
                font-weight: bold;
                display: inline-block;
            }}
            .figure-container {{
                text-align: center;
                margin: 20px 0;
            }}
            .figure-container img {{
                max-width: 100%;
                border-radius: 8px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.1);
            }}
            .timestamp {{
                color: #999;
                font-size: 12px;
                text-align: right;
                margin-top: 30px;
            }}
            table {{
                width: 100%;
                border-collapse: collapse;
                margin: 15px 0;
            }}
            th, td {{
                padding: 12px;
                text-align: left;
                border-bottom: 1px solid #ddd;
            }}
            th {{
                background-color: #2196F3;
                color: white;
            }}
            tr:hover {{
                background-color: #f5f5f5;
            }}
            .two-columns {{
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 20px;
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🚴 Reporte de Entrenamiento - Modelo VAR</h1>

            <h2>📊 Métricas de Validación</h2>
            <div class="metrics">
                <div class="metric-box">
                    <h3>MAE</h3>
                    <div class="value">{mae_display}</div>
                </div>
                <div class="metric-box">
                    <h3>RMSE</h3>
                    <div class="value">{rmse_display}</div>
                </div>
                <div class="metric-box">
                    <h3>MAPE</h3>
                    <div class="value">{mape_display}</div>
                </div>
                <div class="metric-box">
                    <h3>R² Promedio</h3>
                    <div class="value">{r2_display}</div>
                </div>
            </div>

            <h2>⚙️ Información del Modelo</h2>
            <div class="info-grid">
                <div class="info-box">
                    <h3>Configuración</h3>
                    <p><strong>Tipo de modelo:</strong> VAR (Vector Autoregression)</p>
                    <p><strong>Orden (lag):</strong> {model_metadata.get('lag_order', 'N/A')}</p>
                    <p><strong>Diferenciado:</strong> {'Sí' if model_metadata.get('is_differenced', False) else 'No'}</p>
                    <p><strong>Tendencia:</strong> {model_metadata.get('var_config', {}).get('trend', 'c')}</p>
                    <p><strong>Versión:</strong> {model_metadata.get('model_version', 'N/A')}</p>
                </div>
                <div class="info-box">
                    <h3>Datos</h3>
                    <p><strong>Total de bicicletas:</strong> {model_metadata.get('total_bikes', 'N/A'):,}</p>
                    <p><strong>Número de estaciones:</strong> {model_metadata.get('num_stations', 'N/A')}</p>
                </div>
                <div class="info-box">
                    <h3>Diagnósticos</h3>
                    <p><strong>Estabilidad:</strong> <span class="status-badge" style="background-color: {stability_color};">{stability_status}</span></p>
                    <p><strong>Autocorrelación residuales:</strong> <span class="status-badge" style="background-color: {autocorr_color};">{autocorr_status}</span></p>
                </div>
            </div>

            <h2>🎯 Rendimiento por Estación (R²)</h2>
            <div class="two-columns">
                <div>
                    <h3>📈 Top 10 Mejores Estaciones</h3>
                    <table>
                        <tr><th>Estación</th><th>R²</th></tr>
                        {top_stations_html}
                    </table>
                </div>
                <div>
                    <h3>📉 Top 10 Peores Estaciones</h3>
                    <table>
                        <tr><th>Estación</th><th>R²</th></tr>
                        {bottom_stations_html}
                    </table>
                </div>
            </div>

            <h2>📈 Análisis de Residuales</h2>
            <div class="figure-container">
                <img src="figures/var_residuals.png" alt="Análisis de Residuales">
            </div>

            <div class="timestamp">
                <p>Generado: {model_metadata.get('trained_at', datetime.now().isoformat())}</p>
            </div>
        </div>
    </body>
    </html>
    """

    # Guardar reporte
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"   Reporte HTML guardado en: {output_path}")


def validate_on_test(
    fitted_model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    steps: int = 1,
    is_differenced: bool = False,
    original_train: pd.DataFrame = None,
    original_test: pd.DataFrame = None
) -> dict:
    """
    Valida el modelo en el conjunto de test

    Args:
        fitted_model: Modelo VAR entrenado
        train: Datos de entrenamiento (pueden estar diferenciados)
        test: Datos de test (pueden estar diferenciados)
        steps: Número de pasos a predecir
        is_differenced: Si True, los datos están diferenciados y se invertirá
        original_train: Datos originales de train (sin diferenciar)
        original_test: Datos originales de test (sin diferenciar)

    Returns:
        Diccionario con métricas de validación
    """
    logger.info(f"Validando modelo en conjunto de test (steps={steps})...")

    if is_differenced:
        logger.info("   Datos diferenciados - se invertira para calcular metricas")

    # Hacer predicciones rolling
    predictions_diff = []
    predictions_original = []
    actuals_original = []

    # Ventana deslizante sobre el conjunto de test
    for i in range(len(test) - steps):
        # Usar últimos 'lag_order' datos del train + test hasta i
        history = pd.concat([train.iloc[-(fitted_model.k_ar):], test.iloc[:i]])

        # Predecir 'steps' pasos adelante
        forecast = fitted_model.forecast(history.values[-fitted_model.k_ar:], steps=steps)
        pred_diff = forecast[-1]  # Última predicción (diferenciada si is_differenced)
        predictions_diff.append(pred_diff)

        if is_differenced and original_test is not None:
            # Índice en test original: necesitamos alinear con test diferenciado
            # test diferenciado pierde primera fila, así que índice i en test_diff = índice i+1 en original
            # Para predecir el valor en i+steps-1 del test diferenciado,
            # necesitamos el valor anterior en original_test
            original_idx = i + steps  # índice en original_test (compensando por diff)

            if original_idx > 0 and original_idx < len(original_test):
                # Valor anterior en escala original
                last_original = original_test.iloc[original_idx - 1].values
                # Invertir diferenciación: pred_original = last_original + pred_diff
                pred_original = inverse_differencing(pred_diff, last_original)
                predictions_original.append(pred_original)
                # Valor real en escala original
                actuals_original.append(original_test.iloc[original_idx].values)
        else:
            # Sin diferenciación: usar directamente
            predictions_original.append(pred_diff)
            actuals_original.append(test.iloc[i + steps - 1].values)

    predictions = np.array(predictions_original)
    actuals = np.array(actuals_original)

    if len(predictions) == 0:
        logger.warning("   No se generaron predicciones validas")
        return {
            'mae': np.nan, 'rmse': np.nan, 'mape': np.nan,
            'r2_avg': np.nan, 'r2_per_station': {},
            'predictions': np.array([]), 'actuals': np.array([])
        }

    # Calcular métricas en escala ORIGINAL
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # MAPE mejorado: solo para valores positivos significativos
    valid_mask = actuals > 1.0  # Solo valores > 1 bici
    if valid_mask.any():
        mape = np.mean(np.abs((actuals[valid_mask] - predictions[valid_mask]) / actuals[valid_mask])) * 100
    else:
        mape = np.nan
        logger.warning("   No hay valores validos para calcular MAPE")

    # Calcular R² por estación
    # Usar columnas originales si están disponibles
    columns = original_train.columns if original_train is not None else train.columns
    r2_scores = {}
    for i, col in enumerate(columns):
        if i < predictions.shape[1]:
            try:
                r2_scores[col] = r2_score(actuals[:, i], predictions[:, i])
            except:
                r2_scores[col] = np.nan
    avg_r2 = np.nanmean(list(r2_scores.values()))

    logger.info(f"   Metricas de validacion (escala original):")
    logger.info(f"      MAE (Mean Absolute Error): {mae:.2f} bicis")
    logger.info(f"      RMSE (Root Mean Squared Error): {rmse:.2f} bicis")
    if not np.isnan(mape):
        logger.info(f"      MAPE (Mean Absolute Percentage Error): {mape:.2f}%")
    logger.info(f"      R² promedio: {avg_r2:.4f}")

    # Mostrar top 5 estaciones con mejor/peor R²
    r2_sorted = sorted(r2_scores.items(), key=lambda x: x[1] if not np.isnan(x[1]) else -999, reverse=True)
    logger.info(f"   Top 5 estaciones con mejor R2:")
    for station, r2 in r2_sorted[:5]:
        logger.info(f"      {station}: {r2:.4f}")
    logger.info(f"   Top 5 estaciones con peor R2:")
    for station, r2 in r2_sorted[-5:]:
        logger.info(f"      {station}: {r2:.4f}")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2_avg': avg_r2,
        'r2_per_station': r2_scores,
        'predictions': predictions,
        'actuals': actuals
    }


def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO ENTRENAMIENTO DEL MODELO VAR")
    logger.info("=" * 80)

    # 1. Cargar datos procesados
    logger.info("\nCargando datos procesados...")
    data_path = OUTPUTS_DIR / "processed_data"
    train, test = load_processed_data(data_path)

    # Cargar metadata
    with open(data_path.parent / "processed_data_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"   Total de bicis en sistema: {metadata['total_bikes']}")
    logger.info(f"   Número de estaciones: {metadata['num_stations']}")

    # 2. Test de estacionariedad
    stationarity_results = test_stationarity(train)

    # 3. Aplicar diferenciación si es necesario
    if not stationarity_results['mostly_stationary']:
        train_processed = apply_differencing(train, order=1)
        test_processed = apply_differencing(test, order=1)
        is_differenced = True
    else:
        train_processed = train.copy()
        test_processed = test.copy()
        is_differenced = False

    # 4. Seleccionar orden óptimo
    optimal_lag = select_var_order(train_processed, max_lags=VAR_CONFIG['max_lags'])

    # 5. Entrenar modelo VAR
    fitted_model = train_var_model(
        train_processed,
        lag_order=optimal_lag,
        trend=VAR_CONFIG['trend']
    )

    # 6. Diagnósticos del modelo
    diagnostics = diagnose_model(fitted_model, train_processed)

    # 7. Validación en test (con inversión de diferenciación si aplica)
    validation_results = validate_on_test(
        fitted_model,
        train_processed,
        test_processed,
        steps=1,
        is_differenced=is_differenced,
        original_train=train,
        original_test=test
    )

    # 8. Guardar modelo
    model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_DIR / f"var_model_{model_timestamp}.pkl"

    model_metadata = {
        'model': fitted_model,
        'lag_order': optimal_lag,
        'is_differenced': is_differenced,
        'total_bikes': metadata['total_bikes'],
        'num_stations': metadata['num_stations'],
        'station_columns': list(train.columns),
        'var_config': VAR_CONFIG,
        'diagnostics': {
            'is_stable': diagnostics.get('is_stable'),
            'has_autocorr': diagnostics.get('has_autocorr'),
        },
        'validation_metrics': {
            'mae': validation_results['mae'],
            'rmse': validation_results['rmse'],
            'mape': validation_results['mape'],
            'r2_avg': validation_results.get('r2_avg', 0),
        },
        'trained_at': datetime.now().isoformat(),
        'model_version': 'v1.0',
    }

    save_model(model_metadata, model_path)

    # Guardar también como modelo de producción (último modelo entrenado)
    production_model_path = MODELS_DIR / "var_model.pkl"
    save_model(model_metadata, production_model_path)

    # Generar reporte HTML
    report_path = OUTPUTS_DIR / "var_training_report.html"
    generate_training_report(
        model_metadata=model_metadata,
        validation_results=validation_results,
        diagnostics=diagnostics,
        output_path=report_path
    )

    logger.info("\n" + "=" * 80)
    logger.info("ENTRENAMIENTO DEL MODELO VAR COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"Modelo guardado en: {model_path}")
    logger.info(f"Modelo de produccion: {production_model_path}")
    logger.info(f"Reporte HTML: {report_path}")
    logger.info(f"Orden del modelo (lag): {optimal_lag}")
    logger.info(f"MAE en test: {validation_results['mae']:.2f}")
    logger.info(f"RMSE en test: {validation_results['rmse']:.2f}")
    logger.info(f"MAPE en test: {validation_results['mape']:.2f}%")
    logger.info(f"R2 promedio: {validation_results.get('r2_avg', 0):.4f}")


if __name__ == "__main__":
    main()
