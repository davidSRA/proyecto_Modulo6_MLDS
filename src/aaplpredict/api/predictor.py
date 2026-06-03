import os
from datetime import date, timedelta
from importlib.resources import files
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf
import xgboost as xgb
import yfinance as yf

from aaplpredict.api.schemas import PredictionInput, PredictionOutput
from aaplpredict.preprocessing.main import preprocesar_datos
from aaplpredict.training.feature_extraction import (
    extraer_features_flat,
    extraer_features_seq,
)

TARGET_COLS = ["Target_1d", "Target_5d", "Target_Vol5d"]

# Días mínimos que necesitan los indicadores técnicos (MACD slow=26 + signal=9)
_MIN_INDICADORES = 35

# CSV instalado como package data — funciona igual en dev y producción
_PARAMS_CSV = files("aaplpredict.training").joinpath("mejores_params.csv")


def _find_models_dir() -> Path:
    """Encuentra la carpeta models/ sin requerir variables de entorno manuales."""
    # 1. Variable de entorno explícita (override manual si se necesita)
    if "MODELS_DIR" in os.environ:
        return Path(os.environ["MODELS_DIR"])
    # 2. Railway clona el repo en /app
    if Path("/app/models").exists():
        return Path("/app/models")
    # 3. Desarrollo local: carpeta models/ en la raíz del proyecto
    return Path(__file__).parents[3] / "models"


_MODELS_DIR = _find_models_dir()


# ─────────────────────────────────────────────
# Lectura de hiperparámetros desde el CSV
# ─────────────────────────────────────────────

def _leer_params(nombre: str) -> dict:
    df = pd.read_csv(_PARAMS_CSV)
    fila = df[df["modelo"] == nombre]
    if fila.empty:
        raise FileNotFoundError(
            f"No se encontró '{nombre}' en mejores_params.csv. "
            "Ejecuta primero scripts/run_training.py."
        )
    return {k: v for k, v in fila.iloc[0].items() if pd.notna(v) and k != "modelo"}


def _calcular_lookback(nombre: str, params: dict) -> int:
    """
    Días de historia necesarios para producir features válidas:
    - LSTM:        timesteps filas tras feature extraction + margen indicadores
    - XGBoost/MLP: 1 sola fila final, pero necesita lookback de rolling/lags
    """
    if nombre == "LSTM":
        return int(params["timesteps"]) + _MIN_INDICADORES + 5
    window_2 = int(params.get("window_2", 15))
    max_lag  = int(params.get("max_lag",  7))
    return max(_MIN_INDICADORES, window_2, max_lag) + 5


# ─────────────────────────────────────────────
# Carga de modelos desde archivos locales
# ─────────────────────────────────────────────

def _cargar_modelo(nombre: str):
    if nombre == "XGBoost":
        model = xgb.XGBRegressor()
        model.load_model(_MODELS_DIR / "xgboost_model.ubj")
        return model, None                                      # sin scaler

    scaler = joblib.load(_MODELS_DIR / f"{nombre.lower()}_scaler.pkl")
    model  = tf.keras.models.load_model(_MODELS_DIR / f"{nombre.lower()}_model.keras")
    return model, scaler


# ─────────────────────────────────────────────
# Descarga ligera: solo los últimos N días de AAPL
# ─────────────────────────────────────────────

def _obtener_datos_recientes(n_dias: int) -> pd.DataFrame:
    hoy    = date.today()
    inicio = hoy - timedelta(days=n_dias * 2)   # factor 2 para cubrir festivos/fines de semana
    df = yf.download("AAPL", start=str(inicio), end=str(hoy), progress=False)
    df.columns = df.columns.get_level_values(0)
    # reset_index() mueve el índice 'Date' a columna normal — evita la ambigüedad
    # de tener 'Date' como índice Y como columna al mismo tiempo
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").tail(n_dias).reset_index(drop=True)
    return df


# ─────────────────────────────────────────────
# Conversión log-return → precio
#
# Target_1d y Target_5d son retornos logarítmicos acumulados:
#   precio_Nd = precio_actual * exp(Target_Nd)
#
# Target_Vol5d es la desviación estándar de los log-returns diarios
# durante los próximos 5 días. Lo usamos como banda ±1σ alrededor
# de la predicción a 5 días (en espacio logarítmico, luego exponenciamos):
#   precio_5d_sup = precio_actual * exp(Target_5d + Target_Vol5d)
#   precio_5d_inf = precio_actual * exp(Target_5d - Target_Vol5d)
# No se necesita propagación adicional porque Vol5d ya es la incertidumbre
# del modelo sobre los retornos, no un error derivado de otro cálculo.
# ─────────────────────────────────────────────

def _logreturn_a_precio(precio_actual: float, target_1d: float,
                         target_5d: float, target_vol5d: float) -> dict:
    precio_1d  = precio_actual * np.exp(target_1d)
    precio_5d  = precio_actual * np.exp(target_5d)
    sup = precio_actual * np.exp(target_5d + target_vol5d)
    inf = precio_actual * np.exp(target_5d - target_vol5d)
    return {
        "precio_1d":        round(float(precio_1d),  2),
        "precio_5d":        round(float(precio_5d),  2),
        "intervalo_5d_sup": round(float(sup), 2),
        "intervalo_5d_inf": round(float(inf), 2),
    }


# ─────────────────────────────────────────────
# Predicción principal
# ─────────────────────────────────────────────

def predict(data: PredictionInput) -> PredictionOutput:
    nombre = data.modelo
    modelos_disponibles = ["XGBoost", "MLP", "LSTM"]
    if nombre not in modelos_disponibles:
        raise ValueError(f"Modelo '{nombre}' no disponible. Opciones: {modelos_disponibles}")

    params   = _leer_params(nombre)
    lookback = _calcular_lookback(nombre, params)

    df_raw       = _obtener_datos_recientes(lookback)
    precio_actual = float(df_raw["Close"].iloc[-1])   # último cierre antes de preprocesar
    df_prep      = preprocesar_datos(df_raw)

    if df_prep.empty:
        raise ValueError("No se obtuvieron datos suficientes para generar features.")

    modelo, scaler = _cargar_modelo(nombre)

    if nombre == "LSTM":
        timesteps = int(params["timesteps"])
        df_feat   = extraer_features_seq(df_prep)
        feat_cols = [c for c in df_feat.columns if c not in TARGET_COLS + ["Date"]]
        X         = df_feat[feat_cols].values

        if len(X) < timesteps:
            raise ValueError(
                f"Datos insuficientes: se tienen {len(X)} filas, se necesitan {timesteps}."
            )

        X_scaled = scaler.transform(X)
        X_seq    = X_scaled[-timesteps:].reshape(1, timesteps, X.shape[1])
        preds    = modelo.predict(X_seq, verbose=0)[0]

    else:
        windows = [int(params["window_1"]), int(params["window_2"])]
        lags    = list(range(1, int(params["max_lag"]) + 1))

        df_feat   = extraer_features_flat(df_prep, windows=windows, lags=lags)
        drop_cols = [c for c in TARGET_COLS + ["Date"] if c in df_feat.columns]
        X         = df_feat.drop(columns=drop_cols).values[-1:].reshape(1, -1)

        if scaler is not None:
            X = scaler.transform(X)

        preds = modelo.predict(X)[0]

    target_1d, target_5d, target_vol5d = float(preds[0]), float(preds[1]), float(preds[2])

    precios = _logreturn_a_precio(precio_actual, target_1d, target_5d, target_vol5d)

    return PredictionOutput(
        modelo_usado     = nombre,
        precio_actual    = round(precio_actual, 2),
        volatilidad_5d   = round(target_vol5d, 6),
        **precios,
    )
