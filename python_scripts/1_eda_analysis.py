"""
Script 1: Análisis Exploratorio de Datos (EDA)

Analiza el dataset de alquiler de bicicletas de Chicago, organizando el análisis en:

SECCION 1: ANALISIS GENERAL DEL DATASET
    - Estadísticas básicas (registros, estaciones, usuarios)

SECCION 2: VARIABLE TARGET (Disponibilidad de bicicletas)
    - Análisis de estaciones (salidas, llegadas, flujo neto)
    - Flow map de movilidad entre estaciones
    - Matriz de disponibilidad

SECCION 3: VARIABLES EXOGENAS
    - Variables temporales: hora, día de semana, mes, año
    - Variables climáticas: temperatura, eventos meteorológicos
    - Variables de usuario: tipo de usuario (Subscriber/Customer)

SECCION 4: CORRELACIONES
    - Matriz de correlación entre variables numéricas

SECCION 5: ANALISIS DE ESTACIONARIEDAD (preparación para ADF)
    - Serie temporal agregada con tendencia
    - Descomposición de la serie (tendencia, estacionalidad, residuos)
    - Funciones de autocorrelación (ACF y PACF)
    - Rolling statistics (media y desviación estándar móviles)

Outputs generados:
    - outputs/figures/: Visualizaciones PNG y HTML
    - outputs/: CSVs con estadísticas de estaciones y correlaciones
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import logging
import sys
import time
import folium
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Configurar paths
sys.path.append(str(Path(__file__).parent))
from config import DATA_FILE, OUTPUTS_DIR, FIGURES_DIR, RANDOM_SEED, LOG_LEVEL, LOG_FORMAT
from utils.data_loader import load_raw_data

# Configurar logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Configurar semilla
np.random.seed(RANDOM_SEED)


def analyze_distributions(df: pd.DataFrame) -> dict:
    """
    Analiza la distribucion de variables clave del dataset.

    Args:
        df: DataFrame con los datos de viajes.

    Returns:
        dict: Diccionario con estadisticas de distribucion por variable.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE DISTRIBUCIÓN DE VARIABLES CLAVE")
    logger.info("=" * 80)

    distribution_stats = {}

    # Variables a analizar
    cols_to_plot = ['tripduration', 'temperature']
    cols_to_plot = [c for c in cols_to_plot if c in df.columns]

    # Estadísticas descriptivas
    logger.info("\n[INFO] ESTADISTICAS DE VARIABLES CLAVE:")
    for col in cols_to_plot:
        stats = df[col].describe()
        skewness = df[col].skew()
        kurtosis = df[col].kurtosis()
        logger.info(f"   {col}:")
        logger.info(f"      Media: {stats['mean']:.2f}, Mediana: {stats['50%']:.2f}")
        logger.info(f"      Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
        logger.info(f"      Skewness: {skewness:.2f}, Kurtosis: {kurtosis:.2f}")
        distribution_stats[col] = {
            'mean': stats['mean'],
            'median': stats['50%'],
            'std': stats['std'],
            'skewness': skewness,
            'kurtosis': kurtosis
        }

    # Visualización de distribuciones
    if len(cols_to_plot) > 0:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        for idx, col in enumerate(cols_to_plot):
            ax = axes[idx]
            df[col].hist(bins=50, ax=ax, color='steelblue', edgecolor='black', alpha=0.7)
            ax.set_title(f'Distribución de {col}', fontweight='bold')
            ax.set_xlabel(col)
            ax.set_ylabel('Frecuencia')
            ax.grid(True, alpha=0.3)

            # Añadir líneas de media y mediana
            mean_val = df[col].mean()
            median_val = df[col].median()
            ax.axvline(mean_val, color='red', linestyle='--', label=f'Media: {mean_val:.1f}')
            ax.axvline(median_val, color='green', linestyle='-', label=f'Mediana: {median_val:.1f}')
            ax.legend(fontsize=8)

        plt.tight_layout()
        plt.savefig(FIGURES_DIR / 'distributions_key_variables.png', dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"[OK] Grafico guardado: distributions_key_variables.png")

    return distribution_stats


def analyze_user_comparison(df: pd.DataFrame) -> dict:
    """
    Compara patrones temporales de uso entre tipos de usuarios.

    Args:
        df: DataFrame con los datos de viajes.

    Returns:
        dict: Diccionario con estadisticas por tipo de usuario.
    """
    logger.info("\n" + "=" * 80)
    logger.info("COMPARACIÓN ENTRE TIPOS DE USUARIOS")
    logger.info("=" * 80)

    user_stats = {}

    if 'usertype' not in df.columns:
        logger.warning("Columna 'usertype' no encontrada en el dataset")
        return user_stats

    user_types = df['usertype'].unique()
    logger.info(f"\n[INFO] Tipos de usuario encontrados: {list(user_types)}")

    # Estadísticas por tipo de usuario
    for user_type in user_types:
        user_df = df[df['usertype'] == user_type]
        user_stats[user_type] = {
            'count': len(user_df),
            'percentage': len(user_df) / len(df) * 100,
        }
        logger.info(f"\n[INFO] {user_type}:")
        logger.info(f"   Total viajes: {user_stats[user_type]['count']:,} ({user_stats[user_type]['percentage']:.1f}%)")

    # Visualización de patrones temporales
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # 1. Patrón horario por tipo de usuario
    for ut in user_types:
        hourly = df[df['usertype'] == ut].groupby('hour').size()
        hourly_pct = hourly / hourly.sum() * 100
        axes[0].plot(hourly_pct.index, hourly_pct.values, marker='o', label=ut, linewidth=2)
    axes[0].set_title('Patrón Horario por Tipo de Usuario', fontweight='bold')
    axes[0].set_xlabel('Hora del Día')
    axes[0].set_ylabel('Porcentaje de Viajes (%)')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticks(range(0, 24))

    # 2. Patron semanal por tipo de usuario
    day_names = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    for ut in user_types:
        weekly = df[df['usertype'] == ut].groupby('day').size()
        weekly_pct = weekly / weekly.sum() * 100
        axes[1].plot(weekly_pct.index, weekly_pct.values, marker='s', label=ut, linewidth=2)
    axes[1].set_title('Patrón Semanal por Tipo de Usuario', fontweight='bold')
    axes[1].set_xlabel('Día de la Semana')
    axes[1].set_ylabel('Porcentaje de Viajes (%)')
    axes[1].set_xticks(range(7))
    axes[1].set_xticklabels(day_names)
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'user_comparison.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Grafico guardado: user_comparison.png")

    return user_stats


def analyze_correlations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula y visualiza la matriz de correlacion entre variables numericas.

    Aplica transformaciones ciclicas (seno/coseno) a variables temporales
    (hour, day, month) para capturar correctamente su naturaleza ciclica.

    Args:
        df: DataFrame con los datos de viajes.

    Returns:
        pd.DataFrame: Matriz de correlacion de Pearson.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE CORRELACIONES")
    logger.info("=" * 80)

    # Crear copia para no modificar el DataFrame original
    df_corr = df.copy()

    # Aplicar transformaciones cíclicas a variables temporales
    # Fórmula: sin(2*pi*x/periodo) y cos(2*pi*x/periodo)
    logger.info("\n[INFO] Aplicando transformaciones cíclicas a variables temporales:")

    # Hora del día (período = 24)
    if 'hour' in df_corr.columns:
        df_corr['hour_sin'] = np.sin(2 * np.pi * df_corr['hour'] / 24)
        df_corr['hour_cos'] = np.cos(2 * np.pi * df_corr['hour'] / 24)
        logger.info("   - hour -> hour_sin, hour_cos (período=24)")

    # Día de la semana (período = 7)
    if 'day' in df_corr.columns:
        df_corr['day_sin'] = np.sin(2 * np.pi * df_corr['day'] / 7)
        df_corr['day_cos'] = np.cos(2 * np.pi * df_corr['day'] / 7)
        logger.info("   - day -> day_sin, day_cos (período=7)")

    # Mes del año (período = 12)
    if 'month' in df_corr.columns:
        df_corr['month_sin'] = np.sin(2 * np.pi * df_corr['month'] / 12)
        df_corr['month_cos'] = np.cos(2 * np.pi * df_corr['month'] / 12)
        logger.info("   - month -> month_sin, month_cos (período=12)")

    # Seleccionar variables para correlación (sin las originales cíclicas)
    # Variables no cíclicas
    non_cyclic_cols = ['tripduration', 'temperature', 'humidity', 'wind_speed',
                       'visibility', 'dew_point']
    # Variables cíclicas transformadas
    cyclic_cols = ['hour_sin', 'hour_cos', 'day_sin', 'day_cos', 'month_sin', 'month_cos']

    corr_cols = [c for c in non_cyclic_cols if c in df_corr.columns]
    corr_cols += [c for c in cyclic_cols if c in df_corr.columns]

    if len(corr_cols) < 2:
        logger.warning("No hay suficientes columnas numéricas para correlación")
        return pd.DataFrame()

    # Calcular matriz de correlación
    corr_matrix = df_corr[corr_cols].corr()

    # Mostrar correlaciones significativas
    logger.info("\n[INFO] CORRELACIONES SIGNIFICATIVAS (|r| > 0.3):")
    for i in range(len(corr_cols)):
        for j in range(i+1, len(corr_cols)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.3:
                logger.info(f"   {corr_cols[i]} <-> {corr_cols[j]}: {corr_val:.3f}")

    # Crear heatmap de correlaciones
    fig, ax = plt.subplots(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f',
                cmap='RdBu_r', center=0, ax=ax,
                square=True, linewidths=0.5)
    ax.set_title('Matriz de Correlación de Variables Numéricas', fontweight='bold', fontsize=14)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'correlation_heatmap.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Grafico guardado: correlation_heatmap.png")

    return corr_matrix


def analyze_stationarity(df: pd.DataFrame) -> dict:
    """
    Analiza la estacionariedad de la serie temporal de viajes.

    Genera visualizaciones clave para evaluar estacionariedad antes del test ADF:
    - Serie temporal agregada con tendencia
    - Descomposicion de la serie (tendencia, estacionalidad, residuos)
    - Funciones de autocorrelacion (ACF y PACF)
    - Rolling statistics (media y desviacion estandar moviles)

    Args:
        df: DataFrame con los datos de viajes.

    Returns:
        dict: Diccionario con estadisticas de estacionariedad.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE ESTACIONARIEDAD (Preparación para Test ADF)")
    logger.info("=" * 80)

    stationarity_stats = {}

    # Crear serie temporal diaria de viajes
    if 'date' not in df.columns:
        df['date'] = pd.to_datetime(df['starttime']).dt.date

    daily_trips = df.groupby('date').size()
    daily_trips.index = pd.to_datetime(daily_trips.index)
    daily_trips = daily_trips.sort_index()

    logger.info(f"\n[INFO] Serie temporal creada:")
    logger.info(f"   Periodo: {daily_trips.index.min().strftime('%Y-%m-%d')} a {daily_trips.index.max().strftime('%Y-%m-%d')}")
    logger.info(f"   Observaciones: {len(daily_trips)}")
    logger.info(f"   Media: {daily_trips.mean():.2f} viajes/día")
    logger.info(f"   Desviación estándar: {daily_trips.std():.2f}")

    stationarity_stats['n_observations'] = len(daily_trips)
    stationarity_stats['mean'] = daily_trips.mean()
    stationarity_stats['std'] = daily_trips.std()

    # =========================================================================
    # FIGURA 1: Serie temporal con rolling statistics
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Análisis de Estacionariedad - Serie Temporal de Viajes Diarios',
                 fontweight='bold', fontsize=14)

    # (a) Serie temporal original con rolling mean y std
    ax1 = axes[0, 0]
    window = 30  # Ventana de 30 días

    rolling_mean = daily_trips.rolling(window=window).mean()
    rolling_std = daily_trips.rolling(window=window).std()

    ax1.plot(daily_trips.index, daily_trips.values, color='steelblue',
             alpha=0.5, label='Original', linewidth=0.8)
    ax1.plot(rolling_mean.index, rolling_mean.values, color='red',
             linewidth=2, label=f'Media móvil ({window} días)')
    ax1.plot(rolling_std.index, rolling_std.values, color='green',
             linewidth=2, label=f'Desv. Est. móvil ({window} días)')

    ax1.set_xlabel('Fecha')
    ax1.set_ylabel('Número de viajes')
    ax1.set_title('(a) Serie Original con Rolling Statistics', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)

    # Añadir texto interpretativo
    mean_trend = "creciente" if rolling_mean.iloc[-1] > rolling_mean.iloc[window] else "decreciente"
    std_trend = "variable" if rolling_std.std() > rolling_std.mean() * 0.3 else "estable"
    ax1.text(0.02, 0.02, f'Tendencia media: {mean_trend}\nVarianza: {std_trend}',
             transform=ax1.transAxes, fontsize=9, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    # (b) Descomposición de la serie temporal - ZOOM en estacionalidad
    ax2 = axes[0, 1]

    # Realizar descomposición (necesita serie sin NaN y con frecuencia)
    # Rellenar gaps si existen
    full_date_range = pd.date_range(start=daily_trips.index.min(),
                                     end=daily_trips.index.max(), freq='D')
    daily_trips_filled = daily_trips.reindex(full_date_range).interpolate(method='linear')

    try:
        decomposition = seasonal_decompose(daily_trips_filled, model='additive', period=7)

        # Zoom en 2 meses para visualizar estacionalidad semanal claramente
        # Seleccionar un período representativo (verano 2016 para buena actividad)
        zoom_start = pd.Timestamp('2016-06-01')
        zoom_end = pd.Timestamp('2016-07-31')

        # Filtrar datos para el período de zoom
        zoom_mask = (decomposition.seasonal.index >= zoom_start) & (decomposition.seasonal.index <= zoom_end)
        seasonal_zoom = decomposition.seasonal[zoom_mask]
        observed_zoom = daily_trips_filled[zoom_mask]

        # Graficar serie original y estacionalidad en el período de zoom
        ax2.plot(observed_zoom.index, observed_zoom.values,
                 color='steelblue', linewidth=1, alpha=0.7, label='Serie original')
        ax2.plot(seasonal_zoom.index, seasonal_zoom.values + observed_zoom.mean(),
                 color='green', linewidth=2, label='Componente estacional')

        # Marcar los lunes para resaltar el ciclo semanal
        mondays = seasonal_zoom.index[seasonal_zoom.index.dayofweek == 0]
        for monday in mondays:
            ax2.axvline(x=monday, color='orange', linestyle=':', alpha=0.5)

        ax2.set_xlabel('Fecha (Jun-Jul 2016)')
        ax2.set_ylabel('Número de viajes')
        ax2.set_title('(b) Estacionalidad Semanal (zoom 2 meses)', fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8)
        ax2.grid(True, alpha=0.3)

        # Añadir anotación explicativa
        ax2.text(0.02, 0.98, 'Líneas naranjas = Lunes\nCiclo semanal de 7 días',
                 transform=ax2.transAxes, fontsize=8, verticalalignment='top',
                 bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

        stationarity_stats['trend_strength'] = decomposition.trend.std() / daily_trips.std()
        stationarity_stats['seasonal_strength'] = decomposition.seasonal.std() / daily_trips.std()

        logger.info(f"\n[INFO] Descomposición de la serie:")
        logger.info(f"   Fuerza de tendencia: {stationarity_stats['trend_strength']:.3f}")
        logger.info(f"   Fuerza de estacionalidad: {stationarity_stats['seasonal_strength']:.3f}")

    except Exception as e:
        logger.warning(f"No se pudo realizar la descomposición: {e}")
        ax2.text(0.5, 0.5, 'Descomposición no disponible',
                 transform=ax2.transAxes, ha='center', va='center')

    # (c) Función de Autocorrelación (ACF)
    ax3 = axes[1, 0]
    plot_acf(daily_trips_filled.dropna(), ax=ax3, lags=60, alpha=0.05)
    ax3.set_title('(c) Función de Autocorrelación (ACF)', fontweight='bold')
    ax3.set_xlabel('Lags (días)')
    ax3.set_ylabel('Autocorrelación')

    # Añadir anotaciones para patrones
    ax3.axvline(x=7, color='orange', linestyle='--', alpha=0.7, label='Lag 7 (semanal)')
    ax3.axvline(x=30, color='purple', linestyle='--', alpha=0.7, label='Lag 30 (mensual)')
    ax3.legend(loc='upper right', fontsize=8)

    # (d) Función de Autocorrelación Parcial (PACF)
    ax4 = axes[1, 1]
    plot_pacf(daily_trips_filled.dropna(), ax=ax4, lags=60, alpha=0.05, method='ywm')
    ax4.set_title('(d) Función de Autocorrelación Parcial (PACF)', fontweight='bold')
    ax4.set_xlabel('Lags (días)')
    ax4.set_ylabel('Autocorrelación parcial')

    ax4.axvline(x=7, color='orange', linestyle='--', alpha=0.7, label='Lag 7 (semanal)')
    ax4.legend(loc='upper right', fontsize=8)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stationarity_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Gráfico guardado: stationarity_analysis.png")

    # =========================================================================
    # FIGURA 2: Análisis detallado para interpretación ADF
    # =========================================================================
    fig2, axes2 = plt.subplots(2, 2, figsize=(16, 12))
    fig2.suptitle('Diagnóstico de Estacionariedad para Test ADF',
                  fontweight='bold', fontsize=14)

    # (a) Serie diferenciada (primera diferencia)
    ax2a = axes2[0, 0]
    diff_series = daily_trips_filled.diff().dropna()

    ax2a.plot(diff_series.index, diff_series.values, color='steelblue',
              alpha=0.7, linewidth=0.8)
    ax2a.axhline(y=0, color='red', linestyle='--', linewidth=1)
    ax2a.axhline(y=diff_series.mean(), color='green', linestyle='-',
                 linewidth=2, label=f'Media: {diff_series.mean():.2f}')
    ax2a.fill_between(diff_series.index,
                      diff_series.mean() - 2*diff_series.std(),
                      diff_series.mean() + 2*diff_series.std(),
                      alpha=0.2, color='green', label='±2σ')

    ax2a.set_xlabel('Fecha')
    ax2a.set_ylabel('Δ Viajes (diferencia)')
    ax2a.set_title('(a) Serie Diferenciada (1ª diferencia)', fontweight='bold')
    ax2a.legend(loc='upper right')
    ax2a.grid(True, alpha=0.3)

    stationarity_stats['diff_mean'] = diff_series.mean()
    stationarity_stats['diff_std'] = diff_series.std()

    # (b) Distribución de la serie diferenciada
    ax2b = axes2[0, 1]
    ax2b.hist(diff_series.values, bins=50, color='steelblue',
              edgecolor='black', alpha=0.7, density=True)

    # Añadir curva normal teórica
    from scipy import stats
    x_range = np.linspace(diff_series.min(), diff_series.max(), 100)
    normal_curve = stats.norm.pdf(x_range, diff_series.mean(), diff_series.std())
    ax2b.plot(x_range, normal_curve, 'r-', linewidth=2, label='Normal teórica')

    ax2b.axvline(x=0, color='green', linestyle='--', linewidth=2, label='Cero')
    ax2b.set_xlabel('Δ Viajes')
    ax2b.set_ylabel('Densidad')
    ax2b.set_title('(b) Distribución de la Serie Diferenciada', fontweight='bold')
    ax2b.legend()
    ax2b.grid(True, alpha=0.3)

    # Test de normalidad (Jarque-Bera)
    jb_stat, jb_pvalue = stats.jarque_bera(diff_series.values)
    ax2b.text(0.02, 0.98, f'Jarque-Bera: {jb_stat:.2f}\np-value: {jb_pvalue:.4f}',
              transform=ax2b.transAxes, fontsize=9, verticalalignment='top',
              bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))

    stationarity_stats['jarque_bera_stat'] = jb_stat
    stationarity_stats['jarque_bera_pvalue'] = jb_pvalue

    # (c) ACF de la serie diferenciada
    ax2c = axes2[1, 0]
    plot_acf(diff_series, ax=ax2c, lags=40, alpha=0.05)
    ax2c.set_title('(c) ACF de Serie Diferenciada', fontweight='bold')
    ax2c.set_xlabel('Lags (días)')
    ax2c.set_ylabel('Autocorrelación')
    ax2c.axvline(x=7, color='orange', linestyle='--', alpha=0.7)

    # (d) Interpretación y guía para ADF
    ax2d = axes2[1, 1]
    ax2d.axis('off')

    interpretation_text = f"""
    INTERPRETACIÓN PARA TEST ADF (Augmented Dickey-Fuller)

    SERIE ORIGINAL:
    ─────────────────────────────────────────────────────
    • Media: {daily_trips.mean():.2f} viajes/día
    • Desv. Est.: {daily_trips.std():.2f}
    • Observaciones: {len(daily_trips)}

    INDICADORES DE NO ESTACIONARIEDAD:
    ─────────────────────────────────────────────────────
    • ACF decae lentamente → Serie no estacionaria
    • Picos en lag 7 → Estacionalidad semanal
    • Tendencia visible en rolling mean → Necesita diferenciación

    SERIE DIFERENCIADA (1ª diferencia):
    ─────────────────────────────────────────────────────
    • Media: {diff_series.mean():.2f} (≈0 indica estacionariedad)
    • Desv. Est.: {diff_series.std():.2f}
    • ACF decae rápidamente → Posible estacionariedad

    RECOMENDACIÓN PARA MODELADO:
    ─────────────────────────────────────────────────────
    • Si ADF p-value > 0.05 en serie original:
      La serie NO es estacionaria, aplicar diferenciación

    • Si ADF p-value ≤ 0.05 en serie diferenciada:
      La serie diferenciada ES estacionaria

    • Considerar: d=1 para ARIMA, estacionalidad s=7
    """

    ax2d.text(0.05, 0.95, interpretation_text, transform=ax2d.transAxes,
              fontsize=10, verticalalignment='top', fontfamily='monospace',
              bbox=dict(boxstyle='round', facecolor='lightgray', alpha=0.3))
    ax2d.set_title('(d) Guía de Interpretación', fontweight='bold')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'stationarity_adf_diagnostic.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Gráfico guardado: stationarity_adf_diagnostic.png")

    # =========================================================================
    # RESUMEN DE INDICADORES
    # =========================================================================
    logger.info("\n[INFO] INDICADORES DE ESTACIONARIEDAD:")
    logger.info("")
    logger.info("   Serie Original:")
    logger.info(f"      - Media: {daily_trips.mean():.2f}")
    logger.info(f"      - Desv. Est.: {daily_trips.std():.2f}")
    logger.info(f"      - ACF: Decaimiento lento sugiere no estacionariedad")
    logger.info("")
    logger.info("   Serie Diferenciada:")
    logger.info(f"      - Media: {diff_series.mean():.2f} (cercana a 0)")
    logger.info(f"      - Desv. Est.: {diff_series.std():.2f}")
    logger.info(f"      - ACF: Decaimiento rápido sugiere estacionariedad")
    logger.info("")
    logger.info("   Nota: Ejecutar test ADF en el script de modelado para")
    logger.info("         confirmar estadísticamente estas observaciones.")

    return stationarity_stats


def analyze_basic_stats(df: pd.DataFrame) -> dict:
    """
    Calcula estadisticas descriptivas basicas del dataset.

    Args:
        df: DataFrame con los datos de viajes.

    Returns:
        dict: Diccionario con metricas basicas del dataset.
    """
    logger.info("=" * 80)
    logger.info("ANÁLISIS ESTADÍSTICO BÁSICO")
    logger.info("=" * 80)

    stats = {
        'total_records': len(df),
        'date_range': f"{df['year'].min()}-{df['year'].max()}",
        'unique_stations_from': df['from_station_id'].nunique(),
        'unique_stations_to': df['to_station_id'].nunique(),
        'total_unique_stations': len(set(df['from_station_id'].unique()) | set(df['to_station_id'].unique())),
        'avg_trip_duration': df['tripduration'].mean(),
        'median_trip_duration': df['tripduration'].median(),
        'total_users': df['usertype'].value_counts().to_dict(),
    }

    logger.info(f"Registros totales: {stats['total_records']:,}")
    logger.info(f"Rango de fechas: {stats['date_range']}")
    logger.info(f"Estaciones unicas (origen): {stats['unique_stations_from']}")
    logger.info(f"Estaciones unicas (destino): {stats['unique_stations_to']}")
    logger.info(f"Estaciones unicas (total): {stats['total_unique_stations']}")
    logger.info(f"Duracion promedio de viaje: {stats['avg_trip_duration']:.2f} min")
    logger.info(f"Duracion mediana de viaje: {stats['median_trip_duration']:.2f} min")
    logger.info(f"Tipos de usuario: {stats['total_users']}")

    return stats


def analyze_temporal_patterns(df: pd.DataFrame) -> None:
    """
    Analiza patrones temporales de uso: horario, diario, mensual y anual.

    Args:
        df: DataFrame con los datos de viajes.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE PATRONES TEMPORALES")
    logger.info("=" * 80)

    # Viajes por hora
    hourly_trips = df.groupby('hour').size()
    logger.info(f"\n[INFO] Viajes por hora del dia:")
    logger.info(f"   Hora pico: {hourly_trips.idxmax()}h con {hourly_trips.max():,} viajes")
    logger.info(f"   Hora valle: {hourly_trips.idxmin()}h con {hourly_trips.min():,} viajes")

    # Crear gráfico de patrones horarios
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))

    # Viajes por hora
    hourly_trips.plot(kind='bar', ax=axes[0, 0], color='steelblue')
    axes[0, 0].set_title('Distribución de Viajes por Hora del Día', fontweight='bold')
    axes[0, 0].set_xlabel('Hora')
    axes[0, 0].set_ylabel('Número de Viajes')
    axes[0, 0].grid(True, alpha=0.3)

    # Viajes por día de la semana
    weekly_trips = df.groupby('day').size()
    weekly_trips.plot(kind='bar', ax=axes[0, 1], color='coral')
    axes[0, 1].set_title('Distribución de Viajes por Día de la Semana', fontweight='bold')
    axes[0, 1].set_xlabel('Día (0=Lun, 6=Dom)')
    axes[0, 1].set_ylabel('Número de Viajes')
    axes[0, 1].grid(True, alpha=0.3)

    # Viajes por mes
    monthly_trips = df.groupby('month').size()
    monthly_trips.plot(kind='bar', ax=axes[1, 0], color='green')
    axes[1, 0].set_title('Distribución de Viajes por Mes', fontweight='bold')
    axes[1, 0].set_xlabel('Mes')
    axes[1, 0].set_ylabel('Número de Viajes')
    axes[1, 0].grid(True, alpha=0.3)

    # Viajes por año
    yearly_trips = df.groupby('year').size()
    yearly_trips.plot(kind='bar', ax=axes[1, 1], color='purple')
    axes[1, 1].set_title('Distribución de Viajes por Año', fontweight='bold')
    axes[1, 1].set_xlabel('Año')
    axes[1, 1].set_ylabel('Número de Viajes')
    axes[1, 1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'temporal_patterns.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Grafico guardado: temporal_patterns.png")


def analyze_stations(df: pd.DataFrame) -> pd.DataFrame:
    """
    Analiza actividad y flujo neto de las estaciones de bicicletas.

    Args:
        df: DataFrame con los datos de viajes.

    Returns:
        pd.DataFrame: Estadisticas por estacion (salidas, llegadas, flujo neto).
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE ESTACIONES")
    logger.info("=" * 80)

    # Estadísticas por estación de origen
    station_stats_from = df.groupby(['from_station_id', 'from_station_name']).agg({
        'trip_id': 'count',
        'tripduration': 'mean',
        'latitude_start': 'first',
        'longitude_start': 'first',
    }).rename(columns={'trip_id': 'total_departures'})

    # Estadísticas por estación de destino
    station_stats_to = df.groupby(['to_station_id', 'to_station_name']).agg({
        'trip_id': 'count',
    }).rename(columns={'trip_id': 'total_arrivals'})

    # Combinar estadísticas
    station_stats = station_stats_from.copy()
    station_stats['total_arrivals'] = station_stats_to['total_arrivals']
    station_stats['net_flow'] = station_stats['total_departures'] - station_stats['total_arrivals']
    station_stats['total_activity'] = station_stats['total_departures'] + station_stats['total_arrivals']

    # Top 20 estaciones más activas
    top_20 = station_stats.nlargest(20, 'total_activity')

    logger.info(f"\n[INFO] Top 20 Estaciones mas Activas:")
    for idx, (station_id, row) in enumerate(top_20.iterrows(), 1):
        logger.info(f"   {idx}. {row.name[1]}: {row['total_activity']:,} viajes")

    # Crear visualización
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Top 20 por actividad total
    top_20['total_activity'].plot(kind='barh', ax=axes[0], color='steelblue')
    axes[0].set_title('Top 20 Estaciones por Actividad Total', fontweight='bold')
    axes[0].set_xlabel('Total de Viajes (Salidas + Llegadas)')
    axes[0].set_ylabel('Estación')
    axes[0].grid(True, alpha=0.3, axis='x')

    # Flujo neto (salidas - llegadas)
    top_20_flow = station_stats.nlargest(20, 'total_activity')['net_flow'].sort_values()
    colors = ['red' if x < 0 else 'green' for x in top_20_flow.values]
    top_20_flow.plot(kind='barh', ax=axes[1], color=colors, alpha=0.7)
    axes[1].set_title('Flujo Neto en Top 20 Estaciones (Salidas - Llegadas)', fontweight='bold')
    axes[1].set_xlabel('Flujo Neto (+ más salidas, - más llegadas)')
    axes[1].set_ylabel('Estación')
    axes[1].axvline(x=0, color='black', linestyle='--', linewidth=1)
    axes[1].grid(True, alpha=0.3, axis='x')

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'station_analysis.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Grafico guardado: station_analysis.png")

    return station_stats


def analyze_weather_impact(df: pd.DataFrame) -> None:
    """
    Analiza el impacto de las condiciones climaticas en el uso de bicicletas.

    Args:
        df: DataFrame con los datos de viajes.
    """
    logger.info("\n" + "=" * 80)
    logger.info("ANÁLISIS DE IMPACTO CLIMÁTICO")
    logger.info("=" * 80)

    # Viajes por evento climático
    weather_trips = df.groupby('events').size().sort_values(ascending=False)

    logger.info(f"\n[INFO] Viajes por condicion climatica:")
    for event, count in weather_trips.items():
        logger.info(f"   {event}: {count:,} viajes ({count/len(df)*100:.2f}%)")

    # Visualización
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Distribución por clima
    weather_trips.plot(kind='bar', ax=axes[0], color='skyblue')
    axes[0].set_title('Distribución de Viajes por Condición Climática', fontweight='bold')
    axes[0].set_xlabel('Condición Climática')
    axes[0].set_ylabel('Número de Viajes')
    axes[0].grid(True, alpha=0.3)
    axes[0].tick_params(axis='x', rotation=45)

    # Temperatura vs Viajes (convertir de Fahrenheit a Celsius)
    # Fórmula: C = (F - 32) * 5/9
    df['temperature_celsius'] = (df['temperature'] - 32) * 5/9
    temp_trips = df.groupby('temperature_celsius').size()
    axes[1].scatter(temp_trips.index, temp_trips.values, alpha=0.6, color='orange')
    axes[1].set_title('Relación entre Temperatura y Número de Viajes', fontweight='bold')
    axes[1].set_xlabel('Temperatura (°C)')
    axes[1].set_ylabel('Número de Viajes')
    axes[1].grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(FIGURES_DIR / 'weather_impact.png', dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"[OK] Grafico guardado: weather_impact.png")


def create_flow_map(df: pd.DataFrame, top_n_stations: int = 20) -> None:
    """
    Crea un mapa interactivo de flujos de movilidad entre estaciones.

    Genera un flow map que muestra:
    - Las top N estaciones por actividad total (mismas que station_analysis)
    - Lineas entre estaciones representando flujos (grosor proporcional al volumen)
    - Colores diferenciados para flujos de entrada/salida

    Args:
        df: DataFrame con los datos de viajes.
        top_n_stations: Numero de estaciones principales a mostrar (igual que station_analysis).
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREANDO FLOW MAP DE MOVILIDAD")
    logger.info("=" * 80)

    # Calcular estadísticas por estación (mismo criterio que analyze_stations)
    station_departures = df.groupby(['from_station_id', 'from_station_name',
                                      'latitude_start', 'longitude_start']).size()
    station_arrivals = df.groupby(['to_station_id', 'to_station_name',
                                    'latitude_end', 'longitude_end']).size()

    # Crear diccionario de actividad total por estación
    station_activity = {}
    for (sid, sname, lat, lon), count in station_departures.items():
        key = (sid, sname, lat, lon)
        station_activity[key] = station_activity.get(key, 0) + count
    for (sid, sname, lat, lon), count in station_arrivals.items():
        key = (sid, sname, lat, lon)
        station_activity[key] = station_activity.get(key, 0) + count

    # Seleccionar top N estaciones por actividad (mismas que station_analysis)
    sorted_stations = sorted(station_activity.items(), key=lambda x: x[1], reverse=True)
    top_stations = dict(sorted_stations[:top_n_stations])
    top_station_ids = {s[0] for s in top_stations.keys()}  # IDs de las top estaciones

    logger.info(f"[INFO] Mostrando top {top_n_stations} estaciones por actividad")

    # Calcular flujos solo entre las top estaciones
    flows = df.groupby([
        'from_station_id', 'from_station_name', 'latitude_start', 'longitude_start',
        'to_station_id', 'to_station_name', 'latitude_end', 'longitude_end'
    ]).size().reset_index(name='trip_count')

    # Filtrar: no circulares y solo entre top estaciones
    flows = flows[
        (flows['from_station_id'] != flows['to_station_id']) &
        (flows['from_station_id'].isin(top_station_ids)) &
        (flows['to_station_id'].isin(top_station_ids))
    ]

    logger.info(f"[INFO] Flujos entre top estaciones: {len(flows):,}")

    # Calcular centro del mapa (promedio de coordenadas)
    center_lat = df['latitude_start'].mean()
    center_lon = df['longitude_start'].mean()

    # Crear mapa base
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=12,
        tiles='cartodbpositron'
    )

    # Normalizar valores para visualización
    max_activity = max(top_stations.values())
    max_flow = flows['trip_count'].max()

    # Añadir líneas de flujo (dibujar primero para que queden debajo)
    logger.info("[INFO] Dibujando flujos entre estaciones...")
    for _, row in flows.iterrows():
        # Calcular grosor de línea proporcional al flujo
        weight = 1 + (row['trip_count'] / max_flow) * 8

        # Color basado en intensidad del flujo
        opacity = 0.3 + (row['trip_count'] / max_flow) * 0.5

        # Crear línea con gradiente de color
        folium.PolyLine(
            locations=[
                [row['latitude_start'], row['longitude_start']],
                [row['latitude_end'], row['longitude_end']]
            ],
            weight=weight,
            color='#e74c3c',
            opacity=opacity,
            popup=f"{row['from_station_name']} → {row['to_station_name']}<br>Viajes: {row['trip_count']:,}"
        ).add_to(m)

    # Añadir marcadores de estaciones
    logger.info("[INFO] Dibujando estaciones...")
    for (sid, sname, lat, lon), activity in top_stations.items():
        # Calcular radio proporcional a la actividad
        radius = 5 + (activity / max_activity) * 20

        # Calcular flujo neto para color
        departures = station_departures.get((sid, sname, lat, lon), 0)
        arrivals = station_arrivals.get((sid, sname, lat, lon), 0)
        net_flow = departures - arrivals

        # Color: verde si más salidas, rojo si más llegadas
        if net_flow > 0:
            color = '#27ae60'  # Verde - más salidas
            flow_text = f"+{net_flow:,} (más salidas)"
        else:
            color = '#e74c3c'  # Rojo - más llegadas
            flow_text = f"{net_flow:,} (más llegadas)"

        folium.CircleMarker(
            location=[lat, lon],
            radius=radius,
            color=color,
            fill=True,
            fillColor=color,
            fillOpacity=0.7,
            popup=folium.Popup(
                f"<b>{sname}</b><br>"
                f"ID: {sid}<br>"
                f"Actividad total: {activity:,}<br>"
                f"Salidas: {departures:,}<br>"
                f"Llegadas: {arrivals:,}<br>"
                f"Flujo neto: {flow_text}",
                max_width=300
            )
        ).add_to(m)

    # Añadir leyenda
    legend_html = '''
    <div style="position: fixed; bottom: 50px; left: 50px; z-index: 1000;
                background-color: white; padding: 10px; border-radius: 5px;
                border: 2px solid gray; font-size: 12px;">
        <b>Leyenda</b><br>
        <i style="background: #27ae60; width: 12px; height: 12px;
           display: inline-block; border-radius: 50%;"></i> Más salidas<br>
        <i style="background: #e74c3c; width: 12px; height: 12px;
           display: inline-block; border-radius: 50%;"></i> Más llegadas<br>
        <span style="color: #e74c3c;">―</span> Flujo de viajes<br>
        <small>Tamaño = actividad total</small>
    </div>
    '''
    m.get_root().html.add_child(folium.Element(legend_html))

    # Guardar mapa
    output_path = FIGURES_DIR / 'flow_map_mobility.html'
    m.save(str(output_path))

    logger.info(f"[OK] Flow map guardado: {output_path}")
    logger.info(f"[INFO] Abrir en navegador para visualizar interactivamente")


def create_availability_matrix(df: pd.DataFrame, sample_days: int = 30) -> pd.DataFrame:
    """
    Crea matriz de disponibilidad de bicicletas por estacion y dia.

    Args:
        df: DataFrame con los datos de viajes.
        sample_days: Numero de dias a considerar para la muestra.

    Returns:
        pd.DataFrame: Matriz de disponibilidad (filas=dias, columnas=estaciones).
    """
    logger.info("\n" + "=" * 80)
    logger.info("CREANDO MATRIZ DE DISPONIBILIDAD (MUESTRA)")
    logger.info("=" * 80)

    # Filtrar últimos N días para análisis rápido
    df['date'] = pd.to_datetime(df['starttime']).dt.date
    latest_date = pd.to_datetime(df['date'].max())
    cutoff_date = (latest_date - pd.Timedelta(days=sample_days)).date()
    sample_df = df[df['date'] >= cutoff_date]

    # Calcular salidas y llegadas por estación por día
    departures = sample_df.groupby(['date', 'from_station_id']).size().unstack(fill_value=0)
    arrivals = sample_df.groupby(['date', 'to_station_id']).size().unstack(fill_value=0)

    # Disponibilidad simplificada (llegadas - salidas)
    # Nota: En preprocesamiento se hará un cálculo más preciso
    availability = arrivals.sub(departures, fill_value=0)

    logger.info(f"[INFO] Matriz de disponibilidad creada:")
    logger.info(f"   Dias: {len(availability)}")
    logger.info(f"   Estaciones: {len(availability.columns)}")

    return availability

def main():
    """
    Funcion principal que ejecuta el pipeline completo de EDA.

    Ejecuta secuencialmente todos los analisis, organizados en:
    - Analisis general del dataset
    - Analisis de la variable TARGET (disponibilidad de bicicletas)
    - Analisis de variables EXOGENAS (temporales, climaticas, usuario)
    """
    start_time = time.time()

    try:
        logger.info("=" * 80)
        logger.info("INICIANDO ANALISIS EXPLORATORIO DE DATOS (EDA)")
        logger.info("=" * 80)

        # Cargar datos
        logger.info("\n[INFO] Cargando datos...")
        df = load_raw_data(DATA_FILE)

        # =====================================================================
        # SECCION 1: ANALISIS GENERAL DEL DATASET
        # =====================================================================
        logger.info("\n" + "#" * 80)
        logger.info("# SECCION 1: ANALISIS GENERAL DEL DATASET")
        logger.info("#" * 80)

        # 1.1 Estadisticas basicas
        analyze_basic_stats(df)

        # =====================================================================
        # SECCION 2: ANALISIS DE VARIABLE TARGET (DISPONIBILIDAD)
        # =====================================================================
        logger.info("\n" + "#" * 80)
        logger.info("# SECCION 2: ANALISIS DE VARIABLE TARGET (DISPONIBILIDAD DE BICICLETAS)")
        logger.info("#" * 80)

        # 2.1 Analisis de estaciones (salidas, llegadas, flujo neto)
        station_stats = analyze_stations(df)

        # 2.2 Flow map de movilidad entre estaciones
        create_flow_map(df, top_n_stations=20)

        # 2.3 Matriz de disponibilidad
        create_availability_matrix(df, sample_days=30)

        # =====================================================================
        # SECCION 3: ANALISIS DE VARIABLES EXOGENAS
        # =====================================================================
        logger.info("\n" + "#" * 80)
        logger.info("# SECCION 3: ANALISIS DE VARIABLES EXOGENAS")
        logger.info("#" * 80)

        # --- 3.1 Variables Temporales ---
        logger.info("\n" + "-" * 40)
        logger.info("3.1 VARIABLES TEMPORALES (hora, dia, mes, año)")
        logger.info("-" * 40)
        analyze_temporal_patterns(df)

        # --- 3.2 Variables Climaticas ---
        logger.info("\n" + "-" * 40)
        logger.info("3.2 VARIABLES CLIMATICAS (temperatura, eventos)")
        logger.info("-" * 40)
        analyze_weather_impact(df)

        # Distribucion de temperatura
        analyze_distributions(df)

        # --- 3.3 Variables de Usuario ---
        logger.info("\n" + "-" * 40)
        logger.info("3.3 VARIABLES DE USUARIO (tipo de usuario)")
        logger.info("-" * 40)
        analyze_user_comparison(df)

        # =====================================================================
        # SECCION 4: ANALISIS DE CORRELACIONES (TARGET vs EXOGENAS)
        # =====================================================================
        logger.info("\n" + "#" * 80)
        logger.info("# SECCION 4: CORRELACIONES ENTRE VARIABLES")
        logger.info("#" * 80)
        corr_matrix = analyze_correlations(df)

        # =====================================================================
        # SECCION 5: ANALISIS DE ESTACIONARIEDAD (preparación para ADF)
        # =====================================================================
        logger.info("\n" + "#" * 80)
        logger.info("# SECCION 5: ANÁLISIS DE ESTACIONARIEDAD (preparación para ADF)")
        logger.info("#" * 80)
        stationarity_stats = analyze_stationarity(df)

        # =====================================================================
        # GUARDAR RESULTADOS
        # =====================================================================
        station_stats_path = OUTPUTS_DIR / 'station_statistics.csv'
        station_stats.to_csv(station_stats_path)
        logger.info(f"[OK] Estadisticas de estaciones guardadas: {station_stats_path}")

        corr_matrix.to_csv(OUTPUTS_DIR / 'correlation_matrix.csv')
        logger.info(f"[OK] Matriz de correlacion guardada: correlation_matrix.csv")

        stationarity_df = pd.DataFrame([stationarity_stats])
        stationarity_df.to_csv(OUTPUTS_DIR / 'stationarity_statistics.csv', index=False)
        logger.info(f"[OK] Estadisticas de estacionariedad guardadas: stationarity_statistics.csv")

        # Resumen final
        elapsed_time = time.time() - start_time
        logger.info("\n" + "=" * 80)
        logger.info("ANALISIS EXPLORATORIO COMPLETADO")
        logger.info("=" * 80)
        logger.info(f"Tiempo de ejecucion: {elapsed_time:.2f} segundos")
        logger.info(f"Resultados guardados en: {OUTPUTS_DIR}")
        logger.info(f"Figuras guardadas en: {FIGURES_DIR}")

        logger.info("\n[INFO] RESUMEN DE GRAFICOS GENERADOS:")
        logger.info("\n   VARIABLE TARGET (Disponibilidad):")
        logger.info("   - station_analysis.png: Analisis de estaciones (salidas, llegadas, flujo neto)")
        logger.info("   - flow_map_mobility.html: Mapa interactivo de flujos de movilidad")

        logger.info("\n   VARIABLES EXOGENAS:")
        logger.info("   - temporal_patterns.png: Patrones temporales (hora, dia, mes, año)")
        logger.info("   - weather_impact.png: Impacto del clima")
        logger.info("   - distributions_key_variables.png: Distribucion de tripduration y temperature")
        logger.info("   - user_comparison.png: Patrones por tipo de usuario")

        logger.info("\n   CORRELACIONES:")
        logger.info("   - correlation_heatmap.png: Heatmap de correlaciones")

        logger.info("\n   ESTACIONARIEDAD (preparación para ADF):")
        logger.info("   - stationarity_analysis.png: Serie temporal, descomposición, ACF y PACF")
        logger.info("   - stationarity_adf_diagnostic.png: Serie diferenciada y guía para test ADF")

    except Exception as e:
        logger.error(f"Error durante la ejecucion del EDA: {e}")
        raise

if __name__ == "__main__":
    main()
