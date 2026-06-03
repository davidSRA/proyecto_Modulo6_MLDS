import sys
import os

from feature_extraction import extraer_features_flat, extraer_features_seq, crear_secuencias
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_acquisition.main import obtener_datos
from preprocessing.main import preprocesar_datos

import numpy as np
import pandas as pd
import optuna
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import TimeSeriesSplit

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGET_COLS = ["Target_1d", "Target_5d", "Target_Vol5d"]

# ══════════════════════════════════════════════════════════════════════
# UTILIDAD
# ══════════════════════════════════════════════════════════════════════

def _drop_non_features(df):
    drop_cols = [c for c in TARGET_COLS + ["Date"] if c in df.columns]
    return df.drop(columns=drop_cols).values, df[TARGET_COLS].values

# ══════════════════════════════════════════════════════════════════════
# OBJETIVOS OPTUNA  (métrica: MAE promedio en 5 folds temporales)
# ══════════════════════════════════════════════════════════════════════

def objective_xgb(trial, df_prep):
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "verbosity":   0,
    }
    windows = [trial.suggest_int("window_1", 3, 7), trial.suggest_int("window_2", 8, 15)]
    lags    = list(range(1, trial.suggest_int("max_lag", 3, 7) + 1))

    df   = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _drop_non_features(df)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBRegressor(**params)
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        scores.append(float(np.mean(np.abs(preds - y[val_idx]))))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))


def objective_mlp(trial, df_prep):
    n_layers   = trial.suggest_int("n_layers", 1, 3)
    units      = trial.suggest_int("units", 32, 256)
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])
    windows    = [trial.suggest_int("window_1", 3, 7), trial.suggest_int("window_2", 8, 15)]
    lags       = list(range(1, trial.suggest_int("max_lag", 3, 7) + 1))

    df   = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _drop_non_features(df)
    tscv = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

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
            epochs=50, batch_size=batch_size,
            validation_data=(X_val_sc, y_val),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )],
            verbose=0,
        )

        preds = model.predict(X_val_sc, verbose=0)
        scores.append(float(np.mean(np.abs(preds - y_val))))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))


def objective_lstm(trial, df_prep):
    timesteps  = trial.suggest_int("timesteps", 10, 30)
    n_layers   = trial.suggest_int("n_lstm_layers", 1, 2)
    units      = trial.suggest_int("lstm_units", 32, 128)
    dropout    = trial.suggest_float("dropout", 0.0, 0.3)
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    df        = extraer_features_seq(df_prep)
    feat_cols = [c for c in df.columns if c not in TARGET_COLS + ["Date"]]
    X_flat    = df[feat_cols].values
    y_flat    = df[TARGET_COLS].values
    n_feat    = X_flat.shape[1]

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_flat)):
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_flat[tr_idx])
        X_val_sc = scaler.transform(X_flat[val_idx])

        X_tr_seq,  y_tr_seq  = crear_secuencias(X_tr_sc,  y_flat[tr_idx],  timesteps)
        X_val_seq, y_val_seq = crear_secuencias(X_val_sc, y_flat[val_idx], timesteps)
        if len(X_val_seq) == 0:
            continue

        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(timesteps, n_feat)))
        for i in range(n_layers):
            model.add(tf.keras.layers.LSTM(
                units, return_sequences=(i < n_layers - 1), dropout=dropout
            ))
        model.add(tf.keras.layers.Dense(len(TARGET_COLS)))
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mae")
        model.fit(
            X_tr_seq, y_tr_seq,
            epochs=30, batch_size=batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=5, restore_best_weights=True
            )],
            verbose=0,
        )

        preds = model.predict(X_val_seq, verbose=0)
        scores.append(float(np.mean(np.abs(preds - y_val_seq))))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores)) if scores else float("inf")

# ══════════════════════════════════════════════════════════════════════
# MAIN – búsqueda de hiperparámetros
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    print("Cargando y preprocesando datos...")
    _df_raw  = obtener_datos()
    _df_prep = preprocesar_datos(_df_raw)
    print(f"Dataset listo: {len(_df_prep)} filas\n")

    cfg = [
        ("XGBoost", objective_xgb, 50),
        ("MLP",     objective_mlp, 40),
        ("LSTM",    objective_lstm, 30),
    ]

    rows = []
    for nombre, objective, n_trials in cfg:
        print(f"{'='*55}\nOptimizando {nombre} ({n_trials} trials)...\n{'='*55}")
        study = optuna.create_study(direction="minimize", pruner=pruner, study_name=nombre.lower())
        study.optimize(
            lambda trial, obj=objective: obj(trial, _df_prep),
            n_trials=n_trials,
            show_progress_bar=True,
        )
        print(f"{nombre:<10} → MAE CV: {study.best_value:.6f}")
        rows.append({"modelo": nombre, "best_mae_cv": study.best_value, **study.best_params})

    pd.DataFrame(rows).to_csv(os.path.join(BASE_DIR, "mejores_params.csv"), index=False)
    print(f"\nMejores hiperparámetros guardados en mejores_params.csv")
    print("Ejecuta entrenamiento_final.py para reentrenar, registrar en MLflow y generar la gráfica.")
