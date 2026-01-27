"""
Script 3: Entrenamiento del Modelo VAR Optimizado con Variables Climáticas

Versión optimizada que incorpora variables climáticas sin la complejidad de VARMAX.

Estrategia: Two-Stage Approach
1. Reducir dimensionalidad con PCA (opcional)
2. Ajustar efecto climático mediante regresión
3. Entrenar VAR sobre residuales o datos ajustados
4. Combinar predicciones

Ventajas:
- Mucho más rápido que VARMAX
- Escalable a muchas estaciones
- Incorpora información climática
- Mantiene interpretabilidad
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
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.vector_ar.var_model import VAR
from statsmodels.stats.diagnostic import acorr_ljungbox
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
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

# ============================================================================
# CONFIGURACIÓN OPTIMIZADA
# ============================================================================
OPTIMIZED_CONFIG = {
    'use_pca': True,              # Usar PCA para reducir dimensionalidad
    'pca_variance_ratio': 0.95,   # Varianza explicada objetivo
    'max_pca_components': 20,     # Máximo componentes PCA
    'use_climate_adjustment': True,  # Ajustar por clima
    'climate_model': 'ridge',     # Modelo para ajuste climático
    'ridge_alpha': 1.0,           # Regularización Ridge
    'max_lags': 7,                # Máximo lags para VAR
    'default_lag': 3,             # Lag por defecto si falla selección
}


def test_stationarity(data: pd.DataFrame, max_to_test: int = 20) -> dict:
    """Test de estacionariedad (ADF) para cada serie"""
    logger.info("Testeando estacionariedad (ADF test)...")

    results = {}
    columns_to_test = data.columns[:max_to_test]
    stationary_count = 0
    non_stationary_count = 0

    for col in columns_to_test:
        try:
            result = adfuller(data[col].dropna(), autolag='AIC')
            is_stationary = result[1] < 0.05
            results[col] = {
                'adf_statistic': result[0],
                'p_value': result[1],
                'is_stationary': is_stationary
            }
            if is_stationary:
                stationary_count += 1
            else:
                non_stationary_count += 1
        except Exception as e:
            logger.warning(f"   No se pudo testear {col}: {e}")

    logger.info(f"   Estacionarias: {stationary_count}/{len(columns_to_test)}")
    logger.info(f"   No estacionarias: {non_stationary_count}/{len(columns_to_test)}")

    mostly_stationary = stationary_count >= non_stationary_count
    if not mostly_stationary:
        logger.warning("   La mayoria de las series NO son estacionarias")

    return {'mostly_stationary': mostly_stationary, 'results': results}


def apply_differencing(data: pd.DataFrame, order: int = 1) -> pd.DataFrame:
    """Aplica diferenciación a las series"""
    logger.info(f"Aplicando diferenciacion de orden {order}...")
    differenced = data.diff(order).dropna()
    logger.info(f"   Forma original: {data.shape} -> diferenciada: {differenced.shape}")
    return differenced


def inverse_differencing(
    diff_predictions: np.ndarray,
    last_original_values: np.ndarray
) -> np.ndarray:
    """
    Invierte la diferenciación para obtener predicciones en escala original.

    Args:
        diff_predictions: Array de predicciones diferenciadas (cambios)
        last_original_values: Últimos valores conocidos en escala original

    Returns:
        Predicciones en escala original
    """
    return diff_predictions + last_original_values


def apply_pca_reduction(
    train: pd.DataFrame,
    test: pd.DataFrame,
    variance_ratio: float = 0.95,
    max_components: int = 20
) -> tuple:
    """
    Reduce dimensionalidad usando PCA

    Returns:
        train_pca, test_pca, pca_model, scaler
    """
    logger.info(f"Aplicando PCA (varianza objetivo: {variance_ratio*100}%)...")

    # Escalar datos
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(train)
    test_scaled = scaler.transform(test)

    # Determinar número de componentes
    n_components = min(max_components, train.shape[1], train.shape[0] - 1)

    # Ajustar PCA
    pca = PCA(n_components=n_components)
    train_pca = pca.fit_transform(train_scaled)
    test_pca = pca.transform(test_scaled)

    # Encontrar componentes que explican varianza objetivo
    cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
    n_components_final = np.argmax(cumulative_variance >= variance_ratio) + 1
    n_components_final = max(3, n_components_final)  # Mínimo 3 componentes

    logger.info(f"   Componentes originales: {train.shape[1]}")
    logger.info(f"   Componentes PCA: {n_components_final}")
    logger.info(f"   Varianza explicada: {cumulative_variance[n_components_final-1]*100:.1f}%")

    # Crear DataFrames con componentes seleccionados
    pca_columns = [f'PC{i+1}' for i in range(n_components_final)]
    train_pca_df = pd.DataFrame(
        train_pca[:, :n_components_final],
        index=train.index,
        columns=pca_columns
    )
    test_pca_df = pd.DataFrame(
        test_pca[:, :n_components_final],
        index=test.index,
        columns=pca_columns
    )

    return train_pca_df, test_pca_df, pca, scaler, n_components_final


def fit_climate_adjustment(
    endog_train: pd.DataFrame,
    exog_train: pd.DataFrame,
    alpha: float = 1.0
) -> tuple:
    """
    Ajusta modelo de regresión para capturar efecto climático

    Returns:
        modelo_clima, residuales_train
    """
    logger.info("Ajustando modelo de efecto climatico...")

    # Alinear índices
    common_idx = endog_train.index.intersection(exog_train.index)
    endog_aligned = endog_train.loc[common_idx]
    exog_aligned = exog_train.loc[common_idx]

    logger.info(f"   Variables climaticas: {list(exog_aligned.columns)}")
    logger.info(f"   Observaciones alineadas: {len(common_idx)}")

    # Escalar exógenas
    exog_scaler = StandardScaler()
    exog_scaled = exog_scaler.fit_transform(exog_aligned)

    # Ajustar Ridge regression para cada variable endógena
    climate_models = {}
    residuals = pd.DataFrame(index=common_idx, columns=endog_aligned.columns)
    climate_effects = pd.DataFrame(index=common_idx, columns=endog_aligned.columns)

    r2_scores = []
    for col in endog_aligned.columns:
        model = Ridge(alpha=alpha)
        model.fit(exog_scaled, endog_aligned[col])

        prediction = model.predict(exog_scaled)
        residuals[col] = endog_aligned[col] - prediction
        climate_effects[col] = prediction
        climate_models[col] = model

        r2 = r2_score(endog_aligned[col], prediction)
        r2_scores.append(r2)

    avg_r2 = np.mean(r2_scores)
    logger.info(f"   R2 promedio del ajuste climatico: {avg_r2:.4f}")

    return {
        'models': climate_models,
        'exog_scaler': exog_scaler,
        'avg_r2': avg_r2
    }, residuals.astype(float), climate_effects.astype(float)


def predict_climate_effect(
    climate_model: dict,
    exog: pd.DataFrame
) -> pd.DataFrame:
    """Predice el efecto climático para nuevos datos"""
    exog_scaled = climate_model['exog_scaler'].transform(exog)

    predictions = pd.DataFrame(index=exog.index)
    for col, model in climate_model['models'].items():
        predictions[col] = model.predict(exog_scaled)

    return predictions


def select_var_order(data: pd.DataFrame, max_lags: int = 7) -> int:
    """Selecciona el orden óptimo del VAR"""
    logger.info(f"Seleccionando orden optimo del VAR (max_lags={max_lags})...")

    model = VAR(data)

    try:
        results = model.select_order(maxlags=max_lags)

        logger.info(f"   Criterios: AIC={results.aic}, BIC={results.bic}")

        # Usar criterio configurado
        ic = VAR_CONFIG.get('ic', 'aic')
        if ic == 'aic':
            optimal_lag = results.aic
        elif ic == 'bic':
            optimal_lag = results.bic
        else:
            optimal_lag = results.aic

        # Asegurar que sea al menos 1
        optimal_lag = max(1, optimal_lag)

        logger.info(f"   Orden optimo seleccionado: {optimal_lag}")
        return optimal_lag

    except Exception as e:
        logger.warning(f"   Error al seleccionar orden: {e}")
        logger.info(f"   Usando orden por defecto: {OPTIMIZED_CONFIG['default_lag']}")
        return OPTIMIZED_CONFIG['default_lag']


def train_var_model(data: pd.DataFrame, lag_order: int, trend: str = 'c'):
    """Entrena el modelo VAR"""
    logger.info(f"Entrenando modelo VAR...")
    logger.info(f"   Orden (lag): {lag_order}")
    logger.info(f"   Variables: {data.shape[1]}")
    logger.info(f"   Observaciones: {len(data)}")

    model = VAR(data)
    fitted_model = model.fit(lag_order, trend=trend)

    logger.info(f"   Modelo entrenado correctamente")

    try:
        logger.info(f"   AIC: {fitted_model.aic:.2f}")
        logger.info(f"   BIC: {fitted_model.bic:.2f}")
    except:
        pass

    return fitted_model


def diagnose_model(fitted_model, data: pd.DataFrame) -> dict:
    """Diagnósticos del modelo VAR"""
    logger.info("Realizando diagnosticos del modelo...")

    diagnostics = {}
    residuals = fitted_model.resid

    if isinstance(residuals, pd.DataFrame):
        residuals_array = residuals.values
    else:
        residuals_array = residuals

    logger.info(f"   Media residuales: {np.mean(residuals_array):.4f}")
    logger.info(f"   Std residuales: {np.std(residuals_array):.4f}")

    diagnostics['residuals'] = residuals_array

    # Test de estabilidad
    try:
        eigenvalues = fitted_model.stability()
        is_stable = np.all(eigenvalues < 1)
        logger.info(f"   Estabilidad: {'ESTABLE' if is_stable else 'INESTABLE'}")
        diagnostics['is_stable'] = is_stable
    except Exception as e:
        logger.warning(f"   No se pudo verificar estabilidad: {e}")
        diagnostics['is_stable'] = None

    # Test de autocorrelación
    try:
        first_col = residuals_array[:, 0] if residuals_array.ndim > 1 else residuals_array
        lb_test = acorr_ljungbox(first_col, lags=10, return_df=True)
        has_autocorr = any(lb_test['lb_pvalue'] < 0.05)
        logger.info(f"   Autocorrelacion significativa: {'Si' if has_autocorr else 'No'}")
        diagnostics['has_autocorr'] = has_autocorr
    except Exception as e:
        logger.warning(f"   No se pudo realizar test Ljung-Box: {e}")
        diagnostics['has_autocorr'] = None

    # Visualizar residuales
    try:
        plot_residuals(
            residuals_array,
            title="Diagnostico de Residuales - Modelo VAR Optimizado",
            save_path=FIGURES_DIR / 'var_optimized_residuals.png'
        )
    except Exception as e:
        logger.warning(f"   No se pudo generar grafico de residuales: {e}")

    return diagnostics


def validate_model(
    fitted_model,
    train: pd.DataFrame,
    test: pd.DataFrame,
    climate_model: dict = None,
    exog_test: pd.DataFrame = None,
    pca_model = None,
    scaler = None,
    n_pca_components: int = None,
    original_train: pd.DataFrame = None,
    original_test: pd.DataFrame = None,
    is_differenced: bool = False,
    steps: int = 1
) -> dict:
    """
    Valida el modelo combinado (clima + VAR) en el conjunto de test

    Args:
        fitted_model: Modelo VAR entrenado
        train: Datos de entrenamiento (pueden estar diferenciados/PCA)
        test: Datos de test (pueden estar diferenciados/PCA)
        climate_model: Modelo de ajuste climático (opcional)
        exog_test: Variables exógenas de test (opcional)
        pca_model: Modelo PCA (opcional)
        scaler: Escalador para PCA (opcional)
        n_pca_components: Número de componentes PCA
        original_train: Datos originales de train (sin diferenciar)
        original_test: Datos originales de test (sin diferenciar)
        is_differenced: Si True, los datos están diferenciados
        steps: Número de pasos a predecir
    """
    logger.info(f"Validando modelo en conjunto de test...")

    if is_differenced:
        logger.info("   Datos diferenciados - se invertira para calcular metricas")

    predictions_original = []
    actuals_original = []

    # Determinar columnas para métricas
    if original_test is not None:
        metric_columns = original_test.columns
    else:
        metric_columns = test.columns

    # Predicciones rolling
    max_iterations = min(len(test) - steps, 100)  # Limitar para velocidad

    for i in range(max_iterations):
        try:
            # Preparar historia para VAR
            history = pd.concat([train.iloc[-(fitted_model.k_ar):], test.iloc[:i]])

            # Predicción VAR (puede estar en espacio PCA y/o diferenciado)
            var_forecast = fitted_model.forecast(
                history.values[-fitted_model.k_ar:],
                steps=steps
            )

            # Si usamos PCA, transformar de vuelta al espacio original
            if pca_model is not None and scaler is not None:
                var_forecast_full = np.zeros((steps, pca_model.n_components_))
                var_forecast_full[:, :n_pca_components] = var_forecast
                var_forecast_original = pca_model.inverse_transform(var_forecast_full)
                var_forecast_original = scaler.inverse_transform(var_forecast_original)
                final_forecast = var_forecast_original[-1]
            else:
                final_forecast = var_forecast[-1]

            # Añadir efecto climático si existe
            if climate_model is not None and exog_test is not None:
                test_idx = test.index[i + steps - 1] if i + steps - 1 < len(test) else test.index[-1]
                if test_idx in exog_test.index:
                    climate_effect = predict_climate_effect(
                        climate_model,
                        exog_test.loc[[test_idx]]
                    ).values[0]
                    final_forecast = final_forecast + climate_effect

            # Invertir diferenciación si aplica
            if is_differenced and original_test is not None:
                original_idx = i + steps  # índice en original_test
                if original_idx > 0 and original_idx < len(original_test):
                    last_original = original_test.iloc[original_idx - 1].values
                    # Invertir: pred_original = last_original + pred_diff
                    final_forecast = inverse_differencing(final_forecast, last_original)
                    predictions_original.append(final_forecast)
                    actuals_original.append(original_test.iloc[original_idx].values)
            else:
                predictions_original.append(final_forecast)
                if original_test is not None and i + steps - 1 < len(original_test):
                    actuals_original.append(original_test.iloc[i + steps - 1].values)
                elif i + steps - 1 < len(test):
                    actuals_original.append(test.iloc[i + steps - 1].values)

        except Exception as e:
            continue

    if len(predictions_original) == 0:
        logger.warning("   No se pudieron generar predicciones")
        return {'mae': np.nan, 'rmse': np.nan, 'mape': np.nan, 'r2_avg': np.nan,
                'r2_per_station': {}, 'predictions': np.array([]), 'actuals': np.array([])}

    predictions = np.array(predictions_original)
    actuals = np.array(actuals_original)

    # Asegurar misma forma
    min_len = min(len(predictions), len(actuals))
    predictions = predictions[:min_len]
    actuals = actuals[:min_len]

    # Calcular métricas en escala ORIGINAL
    mae = np.mean(np.abs(predictions - actuals))
    rmse = np.sqrt(np.mean((predictions - actuals) ** 2))

    # MAPE mejorado: solo para valores positivos significativos
    valid_mask = actuals > 1.0
    if valid_mask.any():
        mape = np.mean(np.abs((actuals[valid_mask] - predictions[valid_mask]) / actuals[valid_mask])) * 100
    else:
        mape = np.nan

    # R² por columna
    r2_scores = {}
    n_cols = min(predictions.shape[1], len(metric_columns))
    for i in range(n_cols):
        col = metric_columns[i]
        try:
            r2_scores[col] = r2_score(actuals[:, i], predictions[:, i])
        except:
            r2_scores[col] = np.nan

    avg_r2 = np.nanmean(list(r2_scores.values()))

    logger.info(f"   MAE: {mae:.4f} bicis")
    logger.info(f"   RMSE: {rmse:.4f} bicis")
    if not np.isnan(mape):
        logger.info(f"   MAPE: {mape:.2f}%")
    logger.info(f"   R2 promedio: {avg_r2:.4f}")

    return {
        'mae': mae,
        'rmse': rmse,
        'mape': mape,
        'r2_avg': avg_r2,
        'r2_per_station': r2_scores,
        'predictions': predictions,
        'actuals': actuals
    }


def generate_training_report(
    model_metadata: dict,
    validation_results: dict,
    diagnostics: dict,
    output_path: Path
) -> None:
    """Genera un reporte HTML con los resultados del entrenamiento"""
    logger.info("Generando reporte HTML de entrenamiento...")

    mae = validation_results.get('mae', 0)
    rmse = validation_results.get('rmse', 0)
    mape = validation_results.get('mape', 0)
    r2_avg = validation_results.get('r2_avg', 0)
    r2_per_station = validation_results.get('r2_per_station', {})

    # Top estaciones
    r2_sorted = sorted(
        r2_per_station.items(),
        key=lambda x: x[1] if not np.isnan(x[1]) else -999,
        reverse=True
    )

    top_html = ""
    for station, r2 in r2_sorted[:10]:
        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        top_html += f"<tr><td>{station}</td><td>{r2_str}</td></tr>\n"

    bottom_html = ""
    for station, r2 in r2_sorted[-10:]:
        r2_str = f"{r2:.4f}" if not np.isnan(r2) else "N/A"
        bottom_html += f"<tr><td>{station}</td><td>{r2_str}</td></tr>\n"

    is_stable = diagnostics.get('is_stable')
    stability_status = "Estable" if is_stable else ("Inestable" if is_stable is False else "No verificado")
    stability_color = "#4CAF50" if is_stable else ("#f44336" if is_stable is False else "#ff9800")

    has_autocorr = diagnostics.get('has_autocorr')
    autocorr_status = "Si" if has_autocorr else ("No" if has_autocorr is False else "No verificado")
    autocorr_color = "#f44336" if has_autocorr else ("#4CAF50" if has_autocorr is False else "#ff9800")

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>Reporte - VAR Optimizado con Clima</title>
        <meta charset="UTF-8">
        <style>
            body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); }}
            h1 {{ color: #333; border-bottom: 3px solid #4CAF50; padding-bottom: 10px; }}
            h2 {{ color: #555; margin-top: 30px; }}
            .metrics {{ display: flex; flex-wrap: wrap; gap: 20px; margin: 20px 0; }}
            .metric-box {{ background: linear-gradient(135deg, #4CAF50 0%, #45a049 100%); color: white; padding: 20px; border-radius: 10px; text-align: center; min-width: 150px; flex: 1; }}
            .metric-box h3 {{ margin: 0 0 10px 0; font-size: 14px; opacity: 0.9; }}
            .metric-box .value {{ font-size: 28px; font-weight: bold; }}
            .info-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(300px, 1fr)); gap: 20px; margin: 20px 0; }}
            .info-box {{ background: #f8f9fa; padding: 20px; border-radius: 8px; border-left: 4px solid #4CAF50; }}
            .info-box h3 {{ margin-top: 0; color: #333; }}
            .info-box p {{ margin: 8px 0; color: #666; }}
            table {{ width: 100%; border-collapse: collapse; margin: 15px 0; }}
            th, td {{ padding: 12px; text-align: left; border-bottom: 1px solid #ddd; }}
            th {{ background: #4CAF50; color: white; }}
            .two-columns {{ display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }}
            .status-badge {{ padding: 5px 15px; border-radius: 20px; color: white; font-weight: bold; }}
            .timestamp {{ color: #999; font-size: 12px; text-align: right; margin-top: 30px; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>Reporte - Modelo VAR Optimizado con Variables Climaticas</h1>

            <h2>Metricas de Validacion</h2>
            <div class="metrics">
                <div class="metric-box">
                    <h3>MAE</h3>
                    <div class="value">{mae:.4f}</div>
                </div>
                <div class="metric-box">
                    <h3>RMSE</h3>
                    <div class="value">{rmse:.4f}</div>
                </div>
                <div class="metric-box">
                    <h3>MAPE</h3>
                    <div class="value">{mape:.2f}%</div>
                </div>
                <div class="metric-box">
                    <h3>R2 Promedio</h3>
                    <div class="value">{r2_avg:.4f}</div>
                </div>
            </div>

            <h2>Informacion del Modelo</h2>
            <div class="info-grid">
                <div class="info-box">
                    <h3>Configuracion</h3>
                    <p><strong>Tipo:</strong> {model_metadata.get('model_type', 'VAR Optimizado')}</p>
                    <p><strong>Orden (lag):</strong> {model_metadata.get('lag_order', 'N/A')}</p>
                    <p><strong>Usa PCA:</strong> {'Si' if model_metadata.get('use_pca') else 'No'}</p>
                    <p><strong>Componentes PCA:</strong> {model_metadata.get('n_pca_components', 'N/A')}</p>
                    <p><strong>Ajuste climatico:</strong> {'Si' if model_metadata.get('use_climate') else 'No'}</p>
                </div>
                <div class="info-box">
                    <h3>Datos</h3>
                    <p><strong>Total bicicletas:</strong> {model_metadata.get('total_bikes', 'N/A'):,}</p>
                    <p><strong>Estaciones:</strong> {model_metadata.get('num_stations', 'N/A')}</p>
                    <p><strong>Variables exogenas:</strong> {', '.join(model_metadata.get('exog_columns', []))}</p>
                </div>
                <div class="info-box">
                    <h3>Diagnosticos</h3>
                    <p><strong>Estabilidad:</strong> <span class="status-badge" style="background:{stability_color}">{stability_status}</span></p>
                    <p><strong>Autocorrelacion:</strong> <span class="status-badge" style="background:{autocorr_color}">{autocorr_status}</span></p>
                </div>
            </div>

            <h2>Rendimiento por Estacion (R2)</h2>
            <div class="two-columns">
                <div>
                    <h3>Top 10 Mejores</h3>
                    <table>
                        <tr><th>Estacion</th><th>R2</th></tr>
                        {top_html}
                    </table>
                </div>
                <div>
                    <h3>Top 10 Peores</h3>
                    <table>
                        <tr><th>Estacion</th><th>R2</th></tr>
                        {bottom_html}
                    </table>
                </div>
            </div>

            <div class="timestamp">
                <p>Generado: {model_metadata.get('trained_at', datetime.now().isoformat())}</p>
            </div>
        </div>
    </body>
    </html>
    """

    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)

    logger.info(f"   Reporte guardado en: {output_path}")


def main():
    """Función principal"""
    logger.info("=" * 80)
    logger.info("INICIANDO ENTRENAMIENTO VAR OPTIMIZADO CON CLIMA")
    logger.info("=" * 80)

    # 1. Cargar datos procesados
    logger.info("\n[1/8] Cargando datos procesados...")
    data_path = OUTPUTS_DIR / "processed_data"
    train, test = load_processed_data(data_path)

    with open(data_path.parent / "processed_data_metadata.pkl", 'rb') as f:
        metadata = pickle.load(f)

    logger.info(f"   Total bicis: {metadata['total_bikes']}")
    logger.info(f"   Estaciones: {metadata['num_stations']}")
    logger.info(f"   Train: {len(train)} dias, Test: {len(test)} dias")

    # Guardar datos originales para métricas finales
    original_train = train.copy()
    original_test = test.copy()

    # 2. Cargar variables exógenas (clima)
    logger.info("\n[2/8] Cargando variables exogenas (clima)...")
    exog_path = OUTPUTS_DIR / "exog_data"
    try:
        exog_train = pd.read_pickle(f"{exog_path}_train.pkl")
        exog_test = pd.read_pickle(f"{exog_path}_test.pkl")
        has_exog = True
        logger.info(f"   Variables: {list(exog_train.columns)}")
    except FileNotFoundError:
        logger.warning("   No se encontraron datos exogenos, continuando sin clima")
        exog_train = None
        exog_test = None
        has_exog = False

    # 3. Test de estacionariedad
    logger.info("\n[3/8] Test de estacionariedad...")
    stationarity_results = test_stationarity(train)

    # 4. Aplicar diferenciación si es necesario
    if not stationarity_results['mostly_stationary']:
        logger.info("\n[4/8] Aplicando diferenciacion...")
        train_processed = apply_differencing(train, order=1)
        test_processed = apply_differencing(test, order=1)
        if has_exog:
            exog_train = exog_train.iloc[1:]
            exog_test = exog_test.iloc[1:]
        is_differenced = True
    else:
        logger.info("\n[4/8] Series estacionarias, sin diferenciacion")
        train_processed = train.copy()
        test_processed = test.copy()
        is_differenced = False

    # 5. Ajuste climático (Two-Stage)
    climate_model = None
    if has_exog and OPTIMIZED_CONFIG['use_climate_adjustment']:
        logger.info("\n[5/8] Ajustando efecto climatico...")
        climate_model, train_residuals, _ = fit_climate_adjustment(
            train_processed,
            exog_train,
            alpha=OPTIMIZED_CONFIG['ridge_alpha']
        )
        # Usar residuales para VAR
        train_for_var = train_residuals
        # Alinear test
        common_test_idx = test_processed.index.intersection(exog_test.index)
        test_for_var = test_processed.loc[common_test_idx]
    else:
        logger.info("\n[5/8] Sin ajuste climatico")
        train_for_var = train_processed
        test_for_var = test_processed

    # 6. Reducción PCA (opcional)
    pca_model = None
    scaler = None
    n_pca_components = None

    if OPTIMIZED_CONFIG['use_pca'] and train_for_var.shape[1] > OPTIMIZED_CONFIG['max_pca_components']:
        logger.info("\n[6/8] Reduciendo dimensionalidad con PCA...")
        train_pca, test_pca, pca_model, scaler, n_pca_components = apply_pca_reduction(
            train_for_var,
            test_for_var,
            variance_ratio=OPTIMIZED_CONFIG['pca_variance_ratio'],
            max_components=OPTIMIZED_CONFIG['max_pca_components']
        )
        train_for_var = train_pca
        test_for_var = test_pca
    else:
        logger.info("\n[6/8] Sin reduccion PCA (pocas variables)")

    # 7. Seleccionar orden óptimo y entrenar VAR
    logger.info("\n[7/8] Entrenando modelo VAR...")
    optimal_lag = select_var_order(train_for_var, max_lags=OPTIMIZED_CONFIG['max_lags'])

    fitted_model = train_var_model(
        train_for_var,
        lag_order=optimal_lag,
        trend=VAR_CONFIG.get('trend', 'c')
    )

    # 8. Diagnósticos
    diagnostics = diagnose_model(fitted_model, train_for_var)

    # 9. Validación (con inversión de diferenciación si aplica)
    logger.info("\n[8/8] Validando modelo...")
    validation_results = validate_model(
        fitted_model=fitted_model,
        train=train_for_var,
        test=test_for_var,
        climate_model=climate_model,
        exog_test=exog_test if has_exog else None,
        pca_model=pca_model,
        scaler=scaler,
        n_pca_components=n_pca_components,
        original_train=original_train,
        original_test=original_test,
        is_differenced=is_differenced
    )

    # 10. Guardar modelo
    model_timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    model_path = MODELS_DIR / f"var_optimized_{model_timestamp}.pkl"

    model_metadata = {
        'var_model': fitted_model,
        'climate_model': climate_model,
        'pca_model': pca_model,
        'scaler': scaler,
        'n_pca_components': n_pca_components,
        'model_type': 'VAR_Optimizado_Clima',
        'lag_order': optimal_lag,
        'is_differenced': is_differenced,
        'use_pca': pca_model is not None,
        'use_climate': climate_model is not None,
        'total_bikes': metadata['total_bikes'],
        'num_stations': metadata['num_stations'],
        'station_columns': list(original_train.columns),
        'exog_columns': list(exog_train.columns) if has_exog else [],
        'config': OPTIMIZED_CONFIG,
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
        'model_version': 'v3.0_optimized',
    }

    save_model(model_metadata, model_path)

    # Guardar como modelo de producción
    production_path = MODELS_DIR / "var_optimized_model.pkl"
    save_model(model_metadata, production_path)

    # 11. Generar reporte
    report_path = OUTPUTS_DIR / "var_optimized_training_report.html"
    generate_training_report(
        model_metadata=model_metadata,
        validation_results=validation_results,
        diagnostics=diagnostics,
        output_path=report_path
    )

    # Resumen final
    logger.info("\n" + "=" * 80)
    logger.info("ENTRENAMIENTO COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"Modelo guardado: {model_path}")
    logger.info(f"Modelo produccion: {production_path}")
    logger.info(f"Reporte HTML: {report_path}")
    logger.info(f"Orden (lag): {optimal_lag}")
    logger.info(f"Usa PCA: {'Si (' + str(n_pca_components) + ' componentes)' if pca_model else 'No'}")
    logger.info(f"Usa clima: {'Si' if climate_model else 'No'}")
    logger.info(f"MAE: {validation_results['mae']:.4f}")
    logger.info(f"RMSE: {validation_results['rmse']:.4f}")
    logger.info(f"MAPE: {validation_results['mape']:.2f}%")
    logger.info(f"R2 promedio: {validation_results.get('r2_avg', 0):.4f}")


if __name__ == "__main__":
    main()
