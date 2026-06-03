import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ══════════════════════════════════════════════════════════════════════
# TÉCNICA 1: INDICADORES TÉCNICOS FINANCIEROS
# ══════════════════════════════════════════════════════════════════════

def calcular_macd(close, span_fast=12, span_slow=26, span_signal=9):
    ema_fast = close.ewm(span=span_fast, adjust=False).mean()
    ema_slow = close.ewm(span=span_slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal = macd_line.ewm(span=span_signal, adjust=False).mean()
    hist = macd_line - signal
    return macd_line, signal, hist


def calcular_bollinger(close, window=20, num_std=2):
    sma = close.rolling(window=window).mean()
    std = close.rolling(window=window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    width = (upper - lower) / sma
    pct = (close - lower) / (upper - lower)
    return width, pct


def calcular_atr(high, low, close, window=14):
    prev_close = close.shift(1)
    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.ewm(span=window, adjust=False).mean()
    return atr / close


def calcular_estocastico(high, low, close, k_window=14, d_window=3):
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = 100 * (close - lowest_low) / (highest_high - lowest_low)
    d = k.rolling(window=d_window).mean()
    return k, d


def calcular_indicadores(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["MACD"], df["MACD_Signal"], df["MACD_Hist"] = calcular_macd(df["Close"])
    df["BB_Width"], df["BB_Pct"] = calcular_bollinger(df["Close"])
    df["ATR_Norm"] = calcular_atr(df["High"], df["Low"], df["Close"])
    df["Stoch_K"], df["Stoch_D"] = calcular_estocastico(df["High"], df["Low"], df["Close"])
    return df


# ══════════════════════════════════════════════════════════════════════
# TÉCNICA 2: FEATURES DE LAG
# ══════════════════════════════════════════════════════════════════════

def calcular_lags(df: pd.DataFrame, columnas: list, lags: list = [1, 2, 3, 5]) -> pd.DataFrame:
    for col in columnas:
        for lag in lags:
            df[f"{col}_lag{lag}"] = df[col].shift(lag)
    return df


# ══════════════════════════════════════════════════════════════════════
# TÉCNICA 3: ESTADÍSTICAS ROLLING
# ══════════════════════════════════════════════════════════════════════

def calcular_rolling_stats(log_return: pd.Series, windows: list = [5, 10]) -> pd.DataFrame:
    features = {}
    for w in windows:
        features[f"RetMean_{w}d"] = log_return.rolling(w).mean()
        features[f"RetStd_{w}d"] = log_return.rolling(w).std()
        features[f"RetSkew_{w}d"] = log_return.rolling(w).skew()
    return pd.DataFrame(features, index=log_return.index)


# ══════════════════════════════════════════════════════════════════════
# EXTRACCIÓN DE FEATURES
# ══════════════════════════════════════════════════════════════════════

def extraer_features_flat(df: pd.DataFrame, windows: list = [5, 10], lags: list = [1, 2, 3, 4, 5]) -> pd.DataFrame:
    df = df.copy()
    df = calcular_indicadores(df)
    df = calcular_lags(df, columnas=["LogReturn", "ATR_Norm"], lags=lags)
    rolling_feats = calcular_rolling_stats(df["LogReturn"], windows=windows)
    df = pd.concat([df, rolling_feats], axis=1)
    cols_drop = ["Open", "High", "Low", "Volume", "Close"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    return df.dropna()


def extraer_features_seq(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = calcular_indicadores(df)
    cols_drop = ["Open", "High", "Low", "Volume", "Close"]
    df = df.drop(columns=[c for c in cols_drop if c in df.columns])
    return df.dropna()


def crear_secuencias(X: np.ndarray, y: np.ndarray, timesteps: int = 10):
    Xs, ys = [], []
    for i in range(len(X) - timesteps):
        Xs.append(X[i: i + timesteps])
        ys.append(y[i + timesteps])
    return np.array(Xs), np.array(ys)


# ══════════════════════════════════════════════════════════════════════
# ANÁLISIS Y VISUALIZACIÓN
# ══════════════════════════════════════════════════════════════════════

TARGET_COLS = ["Target_1d", "Target_5d", "Target_Vol5d"]


def analizar_features(df: pd.DataFrame):
    feature_cols = [c for c in df.columns if c not in TARGET_COLS + ["Date"]]

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

    tech_cols = ["MACD_Hist", "BB_Width", "BB_Pct", "ATR_Norm", "Stoch_K"]
    fig, axes = plt.subplots(1, len(tech_cols), figsize=(16, 4))
    for ax, col in zip(axes, tech_cols):
        df[col].hist(bins=40, ax=ax, color="steelblue", edgecolor="white")
        ax.set_title(col, fontsize=10)
    plt.suptitle("Distribución de Indicadores Técnicos", fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "distribucion_indicadores.png"))
    plt.close()

    pp_cols = ["MACD_Hist", "BB_Pct", "ATR_Norm", "Stoch_K", "Target_5d"]
    fig = sns.pairplot(
        df[pp_cols].dropna(),
        diag_kind="kde",
        plot_kws={"alpha": 0.3, "s": 10},
        diag_kws={"fill": True},
    )
    fig.figure.suptitle("Pairplot: Indicadores Técnicos vs Target_5d", y=1.02)
    fig.savefig(os.path.join(BASE_DIR, "pairplot_tecnicos.png"), bbox_inches="tight")
    plt.close()

    print(f"\nTotal de features extraídas: {len(feature_cols)}")
    print("\nFeatures disponibles:")
    for col in feature_cols:
        print(f"  • {col}")

    return corr_target


if __name__ == "__main__":
    from aaplpredict.data_acquisition.main import obtener_datos
    from aaplpredict.preprocessing.main import preprocesar_datos

    df_raw = obtener_datos()
    df_prep = preprocesar_datos(df_raw)
    df_plain = extraer_features_flat(df_prep)
    df_seq = extraer_features_seq(df_prep.copy())
    corr = analizar_features(df_plain)
    print("\nTop 10 features más correlacionadas con LogReturn:")
    print(corr.head(10).to_string())
