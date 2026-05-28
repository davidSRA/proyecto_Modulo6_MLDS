import sys
import os

# Agrega la carpeta scripts al path sin importar desde donde ejecutes
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_acquisition.main import obtener_datos
from preprocessing.main import preprocesar_datos
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
# ══════════════════════════════════════════════════════════════════════
# TÉCNICA 1: INDICADORES TÉCNICOS FINANCIEROS
# (MACD, Bandas de Bollinger, ATR, Oscilador Estocástico)
# ══════════════════════════════════════════════════════════════════════
def calcular_macd(close, span_fast=12, span_slow=26, span_signal=9):
    """
    MACD (Moving Average Convergence Divergence).
    Captura momentum y cambios de tendencia.
    - MACD_line:   diferencia entre EMA rápida y lenta
    - Signal_line: EMA del MACD (disparo de compra/venta)
    - MACD_hist:   diferencia MACD - Signal (fuerza del momentum)
    """
    ema_fast   = close.ewm(span=span_fast,   adjust=False).mean()
    ema_slow   = close.ewm(span=span_slow,   adjust=False).mean()
    macd_line  = ema_fast - ema_slow
    signal     = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist       = macd_line - signal
    return macd_line, signal, hist

def calcular_bollinger(close, window=20, num_std=2):
    """
    Bandas de Bollinger.
    Miden volatilidad relativa al precio actual.
    - BB_upper / BB_lower: límites superior e inferior
    - BB_width:  ancho normalizado (expansión = alta volatilidad)
    - BB_pct:    posición del precio dentro de las bandas [0,1]
                 >0.8 = sobrecompra,  <0.2 = sobreventa
    """
    sma        = close.rolling(window=window).mean()
    std        = close.rolling(window=window).std()
    upper      = sma + num_std * std
    lower      = sma - num_std * std
    width      = (upper - lower) / sma          # normalizado
    pct        = (close - lower) / (upper - lower)
    return width, pct

def calcular_atr(high, low, close, window=14):
    """
    ATR (Average True Range).
    Mide la volatilidad real del mercado considerando gaps.
    True Range = max(High-Low, |High-Close_prev|, |Low-Close_prev|)
    """
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low  - prev_close).abs()
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    # Normalizar por precio para comparabilidad
    atr_norm = atr / close
    return atr_norm

def calcular_estocastico(high, low, close, k_window=14, d_window=3):
    """
    Oscilador Estocástico %K y %D.
    Compara el precio de cierre con su rango en una ventana.
    %K > 80 = sobrecompra, %K < 20 = sobreventa.
    """
    lowest_low   = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_window).mean()
    return k, d
def calcular_indicadores(df):
    df = df.copy()
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calcular_macd(df["Close"])
    df["BB_Width"], df["BB_Pct"] = calcular_bollinger(df["Close"])
    df["ATR_Norm"] = calcular_atr(df["High"], df["Low"], df["Close"])
    df["Stoch_K"], df["Stoch_D"] = calcular_estocastico(df["High"], df["Low"], df["Close"])

    return df

# ══════════════════════════════════════════════════════════════════════
# TÉCNICA 2: FEATURES DE LAG (Series de tiempo estructuradas)
# Capturan dependencia temporal: "¿qué pasó hace N días?"
# ══════════════════════════════════════════════════════════════════════

def calcular_lags(df, columnas, lags=[1, 2, 3, 5]):
    """
    Crea features de lag para las columnas indicadas.
    Esencial en series de tiempo: el modelo ve el pasado explícitamente.
    lags=[1,2,3,5] → ayer, anteayer, hace 3 días, hace una semana
    """
    for col in columnas:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df
# ══════════════════════════════════════════════════════════════════════
# TÉCNICA 3: ESTADÍSTICAS ROLLING AVANZADAS
# Distribución local de los retornos en ventanas de tiempo
# ══════════════════════════════════════════════════════════════════════

def calcular_rolling_stats(log_return, windows=[5, 10]):
    """
    Para cada ventana calcula:
    - Media rolling    → tendencia de corto plazo
    - Std rolling      → volatilidad realizada (distinto a Volatility_10 del preprocesamiento)
    - Skewness rolling → asimetría de retornos (colas)
    - Momentum         → retorno acumulado en la ventana (Return_Ndays)
    """
    features = {}
    for w in windows:
        features[f"RetMean_{w}d"]  = log_return.rolling(w).mean()
        features[f"RetStd_{w}d"]   = log_return.rolling(w).std()
        features[f"RetSkew_{w}d"]  = log_return.rolling(w).skew()
    return pd.DataFrame(features, index=log_return.index)


# ══════════════════════════════════════════════════════════════════════
# Funcion para extraer features aplanadas (para XGBoost y MLP)
# ══════════════════════════════════════════════════════════════════════
def extraer_features_flat(df, windows=[5, 10],lags=[1,2,3,4,5]):
    df = df.copy()
    # ── Técnica 1: Indicadores técnicos ───────────────────────────────
    df = calcular_indicadores(df)
    # ── Técnica 2: Lags ───────────────────────────────────────────────
    df = calcular_lags(df, columnas=["LogReturn", "ATR_Norm"], lags=lags)
    # ── Técnica 3: Rolling stats ──────────────────────────────────────
    rolling_feats = calcular_rolling_stats(df["LogReturn"], windows=windows)
    df = pd.concat([df, rolling_feats], axis=1)
    cols_drop = ["Open", "High", "Low", "Volume","Close"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    df = df.dropna()
    return df
# ══════════════════════════════════════════════════════════════════════
# Funcion para extraer features para el LSTM
# ══════════════════════════════════════════════════════════════════════
def extraer_features_seq(df):
    df = df.copy()
    # ── Técnica 1: Indicadores técnicos ───────────────────────────────
    df = calcular_indicadores(df)
    cols_drop = ["Open", "High", "Low", "Volume","Close"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    df = df.dropna()
    return df
def crear_secuencias(X, y, timesteps=10):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i : i + timesteps])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)
    
# ══════════════════════════════════════════════════════════════════════
# ANÁLISIS Y VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

def analizar_features(df):
    """Genera visualizaciones clave de las nuevas features."""

    target_cols  = ["Target_1d", "Target_5d", "Target_Vol5d"]
    feature_cols = [c for c in df.columns if c not in target_cols + ["Date"]]

    # 1. Correlación con LogReturn
    corr_target = (
        df.corr()["LogReturn"]
          .drop("LogReturn")
          .sort_values(key=abs, ascending=False)
    )

    plt.figure(figsize=(10, 8))
    corr_target.plot(kind="barh", color=corr_target.map(lambda x: "steelblue" if x > 0 else "tomato"))
    plt.axvline(0, color="black", linewidth=0.8)
    plt.title("Correlación de cada feature con LogReturn")
    plt.xlabel("Correlación de Pearson")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "correlacion_features_target.png"))
    plt.close()
    #print("✅ Guardado: correlacion_features_target.png")

    # 2. Distribución de indicadores técnicos
    tech_cols = ["MACD_Hist", "BB_Width", "BB_Pct", "ATR_Norm", "Stoch_K"]
    fig, axes = plt.subplots(1, len(tech_cols), figsize=(16, 4))
    for ax, col in zip(axes, tech_cols):
        df[col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=10)
        ax.set_xlabel("")
    plt.suptitle("Distribución de Indicadores Técnicos", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "distribucion_indicadores.png"))
    plt.close()
    #print("✅ Guardado: distribucion_indicadores.png")

    # 3. Pairplot: indicadores técnicos vs Target_5d
    #    Muestra relación entre las features más importantes y el target principal
    pp_cols = ["MACD_Hist", "BB_Pct", "ATR_Norm", "Stoch_K", "Target_5d"]
    fig = sns.pairplot(
        df[pp_cols].dropna(),
        diag_kind="kde",                # densidad en la diagonal
        plot_kws={"alpha": 0.3, "s": 10},
        diag_kws={"fill": True}
    )
    fig.figure.suptitle("Pairplot: Indicadores Técnicos vs Target_5d", y=1.02)
    fig.savefig(os.path.join(BASE_DIR, "pairplot_tecnicos.png"), bbox_inches="tight")
    plt.close()
    #print("✅ Guardado: pairplot_tecnicos.png")


    # 4. Resumen
    print(f"\nTotal de features extraídas: {len(feature_cols)}")
    print("\n Features disponibles:")
    for col in feature_cols:
        print(f"  • {col}")

    return corr_target
if __name__ == "__main__":
    # 1. Datos crudos
    df_raw = obtener_datos()

    df_preprocesed1 = preprocesar_datos(df_raw)
    df_preprocessed2 = df_preprocesed1.copy()
    # 2. Extracción de características aplanadas (MLP y XGBoost)
    df_plain = extraer_features_flat(df_preprocesed1)

    # 3. Extraccion de caracteristicas con secuenciales (LSTM): 
    df_seq = extraer_features_seq(df_preprocessed2)

    # 4. Análisis
    corr = analizar_features(df_plain)

    print("\n Top 10 features más correlacionadas con LogReturn:")
    print(corr.head(10).to_string())



        



