import sys
import os

# Agrega la carpeta scripts al path sin importar desde donde ejecutes
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

BASE_DIR   = os.path.dirname(os.path.abspath(__file__))
TARGET_COLS = ["Target_1d", "Target_5d", "Target_Vol5d"]

# ══════════════════════════════════════════════════════════════════════
# UTILIDAD
# ══════════════════════════════════════════════════════════════════════
def _drop_non_features(df):
    """Elimina targets y columna Date si existe; devuelve X, y como arrays."""
    drop_cols = [c for c in TARGET_COLS + ["Date"] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    y = df[TARGET_COLS].values
    return X, y

# ══════════════════════════════════════════════════════════════════════
# OBJETIVO 1 – XGBoost
# ══════════════════════════════════════════════════════════════════════
def objective_xgb(trial, df_prep):
    """
    Optimiza simultáneamente:
      • Hiperparámetros del modelo XGBoost
      • Hiperparámetros de feature engineering (ventanas rolling y lags)
    Métrica: MSE promedio en 5 folds temporales.
    """
    params = {
        "n_estimators":     trial.suggest_int("n_estimators", 100, 500),
        "max_depth":        trial.suggest_int("max_depth", 3, 8),
        "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
        "tree_method": "hist",
        "verbosity": 0,
    }
    windows = [
        trial.suggest_int("window_1", 3, 7),
        trial.suggest_int("window_2", 8, 15),
    ]
    lags = list(range(1, trial.suggest_int("max_lag", 3, 7) + 1))

    df = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _drop_non_features(df)

    tscv = TimeSeriesSplit(n_splits=5)
    scores = []
    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        model = xgb.XGBRegressor(**params)
        model.fit(X[tr_idx], y[tr_idx])
        preds = model.predict(X[val_idx])
        scores.append(np.mean((preds - y[val_idx]) ** 2))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))

# ══════════════════════════════════════════════════════════════════════
# OBJETIVO 2 – MLP (red densa)
# ══════════════════════════════════════════════════════════════════════
def objective_mlp(trial, df_prep):
    """
    Optimiza simultáneamente:
      • Arquitectura MLP: profundidad, ancho, dropout
      • Optimizador: learning rate, batch size
      • Feature engineering: mismas ventanas y lags que XGBoost
    Usa EarlyStopping para no desperdiciar cómputo en trials malos.
    """
    # ── Arquitectura ─────────────────────────────────────────────────
    n_layers   = trial.suggest_int("n_layers", 1, 3)
    units      = trial.suggest_int("units", 32, 256)
    dropout    = trial.suggest_float("dropout", 0.0, 0.4)
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # ── Feature engineering ──────────────────────────────────────────
    windows = [
        trial.suggest_int("window_1", 3, 7),
        trial.suggest_int("window_2", 8, 15),
    ]
    lags = list(range(1, trial.suggest_int("max_lag", 3, 7) + 1))

    df = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _drop_non_features(df)

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Escalar (fit solo en train para no filtrar info del futuro)
        scaler  = StandardScaler()
        X_tr_sc = scaler.fit_transform(X_tr)
        X_val_sc = scaler.transform(X_val)

        # Construir modelo
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
        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )

        model.fit(
            X_tr_sc, y_tr,
            epochs=50,
            batch_size=batch_size,
            validation_data=(X_val_sc, y_val),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        preds = model.predict(X_val_sc, verbose=0)
        scores.append(float(np.mean((preds - y_val) ** 2)))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores))

# ══════════════════════════════════════════════════════════════════════
# OBJETIVO 3 – LSTM (red recurrente)
# ══════════════════════════════════════════════════════════════════════
def objective_lstm(trial, df_prep):
    """
    Optimiza:
      • Longitud de la ventana temporal (timesteps)
      • Arquitectura LSTM: capas, unidades, dropout recurrente
      • Optimizador: learning rate, batch size
    Nota: extraer_features_seq NO incluye lags manuales ni rolling stats
    porque el propio LSTM aprende la dependencia temporal.
    Las secuencias se construyen DENTRO de cada fold para no filtrar
    información del futuro hacia el pasado.
    """
    # ── Hiperparámetros ───────────────────────────────────────────────
    timesteps  = trial.suggest_int("timesteps", 10, 30)
    n_layers   = trial.suggest_int("n_lstm_layers", 1, 2)
    units      = trial.suggest_int("lstm_units", 32, 128)
    dropout    = trial.suggest_float("dropout", 0.0, 0.3)
    lr         = trial.suggest_float("learning_rate", 1e-4, 1e-2, log=True)
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    # ── Preparar datos ────────────────────────────────────────────────
    df = extraer_features_seq(df_prep)
    feat_cols = [c for c in df.columns if c not in TARGET_COLS + ["Date"]]
    X_flat = df[feat_cols].values
    y_flat = df[TARGET_COLS].values
    n_features = X_flat.shape[1]

    tscv   = TimeSeriesSplit(n_splits=5)
    scores = []

    for fold, (tr_idx, val_idx) in enumerate(tscv.split(X_flat)):
        X_tr_raw, X_val_raw = X_flat[tr_idx], X_flat[val_idx]
        y_tr_raw, y_val_raw = y_flat[tr_idx], y_flat[val_idx]

        # Escalar (fit solo en train)
        scaler   = StandardScaler()
        X_tr_sc  = scaler.fit_transform(X_tr_raw)
        X_val_sc = scaler.transform(X_val_raw)

        # Crear secuencias dentro del fold (evita data leakage)
        X_tr_seq, y_tr_seq   = crear_secuencias(X_tr_sc,  y_tr_raw,  timesteps)
        X_val_seq, y_val_seq = crear_secuencias(X_val_sc, y_val_raw, timesteps)

        # Fold demasiado pequeño para generar secuencias
        if len(X_val_seq) == 0:
            continue

        # Construir LSTM
        tf.keras.backend.clear_session()
        model = tf.keras.Sequential()
        model.add(tf.keras.layers.Input(shape=(timesteps, n_features)))
        for i in range(n_layers):
            return_seq = (i < n_layers - 1)   # True en capas intermedias
            model.add(
                tf.keras.layers.LSTM(
                    units,
                    return_sequences=return_seq,
                    dropout=dropout,
                )
            )
        model.add(tf.keras.layers.Dense(len(TARGET_COLS)))

        model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
            loss="mse",
        )

        model.fit(
            X_tr_seq, y_tr_seq,
            epochs=30,
            batch_size=batch_size,
            validation_data=(X_val_seq, y_val_seq),
            callbacks=[
                tf.keras.callbacks.EarlyStopping(
                    monitor="val_loss", patience=5, restore_best_weights=True
                )
            ],
            verbose=0,
        )

        preds = model.predict(X_val_seq, verbose=0)
        scores.append(float(np.mean((preds - y_val_seq) ** 2)))

        trial.report(np.mean(scores), fold)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

    return float(np.mean(scores)) if scores else float("inf")

# ══════════════════════════════════════════════════════════════════════
# MAIN – Búsqueda de hiperparámetros
# ══════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    # Silenciar logs internos de Optuna; mostrar solo lo nuestro
    optuna.logging.set_verbosity(optuna.logging.WARNING)

    # ── Cargar datos UNA sola vez (evita I/O repetido en cada trial) ──
    print("Cargando y preprocesando datos...")
    _df_raw  = obtener_datos()
    _df_prep = preprocesar_datos(_df_raw)
    print(f"Dataset listo: {len(_df_prep)} filas, {_df_prep.shape[1]} columnas base\n")

    # Pruner compartido: descarta trials cuyo MSE parcial sea peor
    # que la mediana de los trials anteriores en el mismo paso
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=2)

    # ── XGBoost ──────────────────────────────────────────────────────
    print("=" * 55)
    print("Optimizando XGBoost  (50 trials)...")
    print("=" * 55)
    study_xgb = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="xgb",
    )
    study_xgb.optimize(
        lambda trial: objective_xgb(trial, _df_prep),
        n_trials=50,
        show_progress_bar=True,
    )
    print(f"XGB  → MSE: {study_xgb.best_value:.6f}")
    print(f"       Params: {study_xgb.best_params}\n")

    # ── MLP ──────────────────────────────────────────────────────────
    print("=" * 55)
    print("Optimizando MLP      (40 trials)...")
    print("=" * 55)
    study_mlp = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="mlp",
    )
    study_mlp.optimize(
        lambda trial: objective_mlp(trial, _df_prep),
        n_trials=40,
        show_progress_bar=True,
    )
    print(f"MLP  → MSE: {study_mlp.best_value:.6f}")
    print(f"       Params: {study_mlp.best_params}\n")

    # ── LSTM ─────────────────────────────────────────────────────────
    print("=" * 55)
    print("Optimizando LSTM     (30 trials)...")
    print("=" * 55)
    study_lstm = optuna.create_study(
        direction="minimize",
        pruner=pruner,
        study_name="lstm",
    )
    study_lstm.optimize(
        lambda trial: objective_lstm(trial, _df_prep),
        n_trials=30,
        show_progress_bar=True,
    )
    print(f"LSTM → MSE: {study_lstm.best_value:.6f}")
    print(f"       Params: {study_lstm.best_params}\n")

    # ── Resumen comparativo ───────────────────────────────────────────
    print("=" * 55)
    print("RESUMEN COMPARATIVO (MSE promedio – 5 folds)")
    print("=" * 55)
    resultados = {
        "XGBoost": study_xgb.best_value,
        "MLP":     study_mlp.best_value,
        "LSTM":    study_lstm.best_value,
    }
    for nombre, mse in sorted(resultados.items(), key=lambda x: x[1]):
        marca = " ← mejor" if nombre == min(resultados, key=resultados.get) else ""
        print(f"  {nombre:<10}  MSE = {mse:.6f}{marca}")

    # ── Guardar mejores params en CSV ─────────────────────────────────
    rows = []
    for nombre, study in [("XGBoost", study_xgb), ("MLP", study_mlp), ("LSTM", study_lstm)]:
        row = {"modelo": nombre, "best_mse": study.best_value}
        row.update(study.best_params)
        rows.append(row)
    pd.DataFrame(rows).to_csv(
        os.path.join(BASE_DIR, "mejores_params.csv"), index=False
    )
    print(f"\nParams guardados en: {os.path.join(BASE_DIR, 'mejores_params.csv')}")
