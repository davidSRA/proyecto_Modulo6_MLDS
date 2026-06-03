import sys
import os
import pickle
import tempfile

from feature_extraction import extraer_features_flat, extraer_features_seq, crear_secuencias
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_acquisition.main import obtener_datos
from preprocessing.main import preprocesar_datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import mlflow
import mlflow.xgboost
import mlflow.keras

# Servidor MLflow — lanzar ANTES de ejecutar este script:
#   python -m mlflow server --backend-store-uri sqlite:///tracking.db \
#          --default-artifact-root file:mlruns -p 5000
mlflow.set_tracking_uri("http://localhost:5000")

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGET_COLS = ["Target_1d", "Target_5d", "Target_Vol5d"]
SPLIT_RATIO = 0.80

# Params que pertenecen al modelo (separan de los de feature engineering)
_XGB_PARAMS = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"}
_MLP_PARAMS = {"n_layers", "units", "dropout", "learning_rate", "batch_size"}
_LSTM_PARAMS = {"timesteps", "n_lstm_layers", "lstm_units", "dropout", "learning_rate", "batch_size"}

# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════

def _split_temporal(arr):
    n = int(len(arr) * SPLIT_RATIO)
    return arr[:n], arr[n:]


def _drop_non_features(df):
    drop_cols = [c for c in TARGET_COLS + ["Date"] if c in df.columns]
    return df.drop(columns=drop_cols).values, df[TARGET_COLS].values


def _metricas(y_true: np.ndarray, y_pred: np.ndarray, nombre: str) -> dict:
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    mse = mean_squared_error( y_true, y_pred, multioutput="uniform_average")
    r2  = r2_score(           y_true, y_pred, multioutput="uniform_average")
    print(f"\n{'─'*42}\n  {nombre}\n{'─'*42}")
    print(f"  MAE : {mae:.6f}")
    print(f"  MSE : {mse:.6f}")
    print(f"  R²  : {r2:.4f}")
    return {"modelo": nombre, "MAE": float(mae), "MSE": float(mse), "R2": float(r2)}

# ══════════════════════════════════════════════════════════════════════
# REENTRENAMIENTO
# ══════════════════════════════════════════════════════════════════════

def retrain_xgb(df_prep: pd.DataFrame, params: dict):
    windows = [int(params["window_1"]), int(params["window_2"])]
    lags    = list(range(1, int(params["max_lag"]) + 1))

    df   = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _drop_non_features(df)
    X_tr, X_te = _split_temporal(X)
    y_tr, y_te = _split_temporal(y)

    model_params = {
        k: (int(v) if k in ("n_estimators", "max_depth") else float(v))
        for k, v in params.items() if k in _XGB_PARAMS
    }
    model = xgb.XGBRegressor(**model_params, tree_method="hist", verbosity=0)
    model.fit(X_tr, y_tr)

    metricas = _metricas(y_te, model.predict(X_te), "XGBoost")
    return model, metricas, None          # None = sin scaler


def retrain_mlp(df_prep: pd.DataFrame, params: dict):
    windows = [int(params["window_1"]), int(params["window_2"])]
    lags    = list(range(1, int(params["max_lag"]) + 1))

    df   = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _drop_non_features(df)
    X_tr, X_te = _split_temporal(X)
    y_tr, y_te = _split_temporal(y)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr)
    X_te_sc  = scaler.transform(X_te)

    n_layers, units = int(params["n_layers"]), int(params["units"])
    dropout, lr     = float(params["dropout"]), float(params["learning_rate"])
    batch_size      = int(params["batch_size"])

    tf.keras.backend.clear_session()
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(X_tr_sc.shape[1],))]
        + [layer
           for _ in range(n_layers)
           for layer in (
               tf.keras.layers.Dense(units, activation="relu"),
               tf.keras.layers.Dropout(dropout),
           )]
        + [tf.keras.layers.Dense(len(TARGET_COLS))]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae")
    model.fit(
        X_tr_sc, y_tr,
        epochs=100, batch_size=batch_size,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )],
        verbose=0,
    )

    metricas = _metricas(y_te, model.predict(X_te_sc, verbose=0), "MLP")
    return model, metricas, scaler


def retrain_lstm(df_prep: pd.DataFrame, params: dict):
    timesteps  = int(params["timesteps"])
    n_layers   = int(params["n_lstm_layers"])
    units      = int(params["lstm_units"])
    dropout    = float(params["dropout"])
    lr         = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    df        = extraer_features_seq(df_prep)
    feat_cols = [c for c in df.columns if c not in TARGET_COLS + ["Date"]]
    X_flat    = df[feat_cols].values
    y_flat    = df[TARGET_COLS].values

    X_tr_raw, X_te_raw = _split_temporal(X_flat)
    y_tr_raw, y_te_raw = _split_temporal(y_flat)

    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_raw)
    X_te_sc  = scaler.transform(X_te_raw)

    X_tr_seq, y_tr_seq =   (X_tr_sc, y_tr_raw, timesteps)
    X_te_seq, y_te_seq = crear_secuencias(X_te_sc, y_te_raw, timesteps)

    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(timesteps, X_flat.shape[1])))
    for i in range(n_layers):
        model.add(tf.keras.layers.LSTM(
            units, return_sequences=(i < n_layers - 1), dropout=dropout
        ))
    model.add(tf.keras.layers.Dense(len(TARGET_COLS)))
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae")
    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=60, batch_size=batch_size,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )],
        verbose=0,
    )

    metricas = _metricas(y_te_seq, model.predict(X_te_seq, verbose=0), "LSTM")
    return model, metricas, scaler

# ══════════════════════════════════════════════════════════════════════
# MLFLOW
# ══════════════════════════════════════════════════════════════════════

def log_mlflow(nombre: str, params: dict, metricas: dict, modelo, scaler=None) -> None:
    mlflow.set_experiment(nombre)
    with mlflow.start_run(run_name=f"best_{nombre.lower()}"):
        mlflow.log_params(params)
        mlflow.log_metrics({k: v for k, v in metricas.items() if k != "modelo"})

        if nombre == "XGBoost":
            mlflow.xgboost.log_model(
                modelo, artifact_path="model",
                registered_model_name=f"{nombre}_predictor",
            )
        else:
            mlflow.keras.log_model(
                modelo, artifact_path="model",
                registered_model_name=f"{nombre}_predictor",
            )

        if scaler is not None:
            with tempfile.NamedTemporaryFile(suffix=".pkl", delete=False) as f:
                pickle.dump(scaler, f)
                tmp_path = f.name
            mlflow.log_artifact(tmp_path, artifact_path="scaler")
            os.unlink(tmp_path)

    print(f"  MLflow [{nombre}] registrado")

# ══════════════════════════════════════════════════════════════════════
# GRÁFICA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════

def grafica_comparativa(resultados: list[dict]) -> None:
    df      = pd.DataFrame(resultados)
    modelos = df["modelo"].tolist()
    colores = ["steelblue", "tomato", "seagreen"]

    metricas_cfg = [
        ("MAE", "MAE  (↓ mejor)",  False),
        ("MSE", "MSE  (↓ mejor)",  False),
        ("R2",  "R²   (↑ mejor)",  True),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, titulo, mayor_es_mejor) in zip(axes, metricas_cfg):
        valores   = df[col].tolist()
        mejor_idx = (
            valores.index(max(valores)) if mayor_es_mejor
            else valores.index(min(valores))
        )
        bars = ax.bar(modelos, valores, color=colores, edgecolor="white", width=0.5)
        bars[mejor_idx].set_edgecolor("gold")
        bars[mejor_idx].set_linewidth(2.5)

        rango = max(valores) - min(valores) if max(valores) != min(valores) else abs(max(valores))
        for bar, val in zip(bars, valores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + rango * 0.03,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            )

        y_min = min(0, min(valores)) * 1.15
        y_max = max(valores) * 1.25 if max(valores) > 0 else max(valores) * 0.75
        ax.set_ylim(y_min, y_max)
        ax.set_title(titulo, fontsize=11, pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)

    fig.suptitle(
        "Comparativa de modelos  —  conjunto de prueba (último 20 % temporal)",
        fontsize=12, y=1.03,
    )
    plt.tight_layout()
    ruta = os.path.join(BASE_DIR, "comparativa_modelos.png")
    plt.savefig(ruta, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"\nGráfica guardada en: {ruta}")

# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    csv_path = os.path.join(BASE_DIR, "mejores_params.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"No se encontró '{csv_path}'.\n"
            "Ejecuta primero main.py para generar los mejores hiperparámetros."
        )

    # Leer params (NaN = param que no aplica a ese modelo → se ignora)
    df_csv = pd.read_csv(csv_path)
    best = {
        row["modelo"]: {k: v for k, v in row.items()
                        if k not in ("modelo", "best_mae_cv") and pd.notna(v)}
        for _, row in df_csv.iterrows()
    }

    faltantes = {"XGBoost", "MLP", "LSTM"} - set(best)
    if faltantes:
        raise ValueError(f"Faltan modelos en el CSV: {faltantes}")

    print("Cargando datos...")
    df_raw  = obtener_datos()
    df_prep = preprocesar_datos(df_raw)
    n_test  = len(df_prep) - int(len(df_prep) * SPLIT_RATIO)
    print(f"Dataset: {len(df_prep)} filas  |  test: últimas {n_test} filas\n")

    retrain_fn = {
        "XGBoost": retrain_xgb,
        "MLP":     retrain_mlp,
        "LSTM":    retrain_lstm,
    }

    print("Reentrenando mejores modelos...")
    resultados = []
    for nombre in ["XGBoost", "MLP", "LSTM"]:
        print(f"\n[{nombre}]")
        modelo, metricas, scaler = retrain_fn[nombre](df_prep, best[nombre])
        log_mlflow(nombre, best[nombre], metricas, modelo, scaler)
        resultados.append(metricas)

    # Gráfica
    grafica_comparativa(resultados)

    # Resumen
    df_res  = pd.DataFrame(resultados).set_index("modelo")
    print(f"\n{'='*50}\nRESUMEN  (promedio sobre los 3 targets)\n{'='*50}")
    print(df_res.to_string(float_format=lambda x: f"{x:.6f}"))
    print(f"\n  Menor MAE → {df_res['MAE'].idxmin()}")
    print(f"  Mayor R²  → {df_res['R2'].idxmax()}")

    ruta_res = os.path.join(BASE_DIR, "resultados_evaluacion.csv")
    df_res.reset_index().to_csv(ruta_res, index=False)
    print(f"\nTabla guardada en: {ruta_res}")
