import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from feature_extraction import extraer_features_flat, extraer_features_seq, crear_secuencias
from data_acquisition.main import obtener_datos
from preprocessing.main import preprocesar_datos

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import xgboost as xgb
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
TARGET_COLS = ["Target_1d", "Target_5d", "Target_Vol5d"]
SPLIT_RATIO = 0.80   # 80 % train / 20 % test (orden temporal preservado)

# Qué params pertenecen al modelo y cuáles al feature engineering
_XGB_PARAMS  = {"n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree"}
_MLP_PARAMS  = {"n_layers", "units", "dropout", "learning_rate", "batch_size"}
_LSTM_PARAMS = {"timesteps", "n_lstm_layers", "lstm_units", "dropout", "learning_rate", "batch_size"}
_FEAT_PARAMS = {"window_1", "window_2", "max_lag"}

# ══════════════════════════════════════════════════════════════════════
# CARGA DE PARÁMETROS
# ══════════════════════════════════════════════════════════════════════

def cargar_mejores_params(csv_path: str) -> dict:
    """
    Lee mejores_params.csv generado por main.py.
    Devuelve:  { "XGBoost": {param: valor, ...}, "MLP": {...}, "LSTM": {...} }
    Los valores NaN (params que no aplican a ese modelo) son ignorados.
    """
    df = pd.read_csv(csv_path)
    resultado = {}
    for _, row in df.iterrows():
        modelo = row["modelo"]
        params = {
            k: v for k, v in row.items()
            if k not in ("modelo", "best_mse") and pd.notna(v)
        }
        resultado[modelo] = params
    return resultado


# ══════════════════════════════════════════════════════════════════════
# UTILIDADES
# ══════════════════════════════════════════════════════════════════════

def _split_features_targets(df):
    """Separa X e y eliminando columnas que no son features."""
    drop_cols = [c for c in TARGET_COLS + ["Date"] if c in df.columns]
    X = df.drop(columns=drop_cols).values
    y = df[TARGET_COLS].values
    return X, y


def _split_temporal(arr, ratio=SPLIT_RATIO):
    """Corta un array en (train, test) respetando el orden temporal."""
    n = int(len(arr) * ratio)
    return arr[:n], arr[n:]


def _metricas(y_true, y_pred, nombre: str) -> dict:
    """
    Calcula MSE, MAE y R² promediados sobre los 3 targets.
    Imprime un resumen y devuelve un dict con los valores.
    """
    mse = mean_squared_error(y_true, y_pred, multioutput="uniform_average")
    mae = mean_absolute_error(y_true, y_pred, multioutput="uniform_average")
    r2  = r2_score(y_true, y_pred,            multioutput="uniform_average")
    print(f"\n{'─' * 42}")
    print(f"  {nombre}")
    print(f"{'─' * 42}")
    print(f"  MSE : {mse:.6f}")
    print(f"  MAE : {mae:.6f}")
    print(f"  R²  : {r2:.4f}")
    return {"modelo": nombre, "MSE": mse, "MAE": mae, "R2": r2}


# ══════════════════════════════════════════════════════════════════════
# EVALUADORES
# ══════════════════════════════════════════════════════════════════════

def evaluar_xgboost(df_prep: pd.DataFrame, params: dict) -> dict:
    # ── Feature engineering ──────────────────────────────────────────
    windows = [int(params["window_1"]), int(params["window_2"])]
    lags    = list(range(1, int(params["max_lag"]) + 1))

    df = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _split_features_targets(df)
    X_tr, X_te = _split_temporal(X)
    y_tr, y_te = _split_temporal(y)

    # ── Modelo ───────────────────────────────────────────────────────
    model_params = {
        k: (int(v) if k in ("n_estimators", "max_depth") else float(v))
        for k, v in params.items()
        if k in _XGB_PARAMS
    }
    model = xgb.XGBRegressor(**model_params, tree_method="hist", verbosity=0)
    model.fit(X_tr, y_tr)

    return _metricas(y_te, model.predict(X_te), "XGBoost")


def evaluar_mlp(df_prep: pd.DataFrame, params: dict) -> dict:
    # ── Feature engineering ──────────────────────────────────────────
    windows = [int(params["window_1"]), int(params["window_2"])]
    lags    = list(range(1, int(params["max_lag"]) + 1))

    df = extraer_features_flat(df_prep, windows=windows, lags=lags)
    X, y = _split_features_targets(df)
    X_tr, X_te = _split_temporal(X)
    y_tr, y_te = _split_temporal(y)

    # ── Escalar (fit solo en train) ───────────────────────────────────
    scaler  = StandardScaler()
    X_tr_sc = scaler.fit_transform(X_tr)
    X_te_sc = scaler.transform(X_te)

    # ── Arquitectura ─────────────────────────────────────────────────
    n_layers   = int(params["n_layers"])
    units      = int(params["units"])
    dropout    = float(params["dropout"])
    lr         = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    tf.keras.backend.clear_session()
    model = tf.keras.Sequential(
        [tf.keras.layers.Input(shape=(X_tr_sc.shape[1],))]
        + [capa
           for _ in range(n_layers)
           for capa in (
               tf.keras.layers.Dense(units, activation="relu"),
               tf.keras.layers.Dropout(dropout),
           )]
        + [tf.keras.layers.Dense(len(TARGET_COLS))]
    )
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    model.fit(
        X_tr_sc, y_tr,
        epochs=100,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )],
        verbose=0,
    )

    return _metricas(y_te, model.predict(X_te_sc, verbose=0), "MLP")


def evaluar_lstm(df_prep: pd.DataFrame, params: dict) -> dict:
    # ── Hiperparámetros ───────────────────────────────────────────────
    timesteps  = int(params["timesteps"])
    n_layers   = int(params["n_lstm_layers"])
    units      = int(params["lstm_units"])
    dropout    = float(params["dropout"])
    lr         = float(params["learning_rate"])
    batch_size = int(params["batch_size"])

    # ── Features secuenciales ─────────────────────────────────────────
    df = extraer_features_seq(df_prep)
    feat_cols  = [c for c in df.columns if c not in TARGET_COLS + ["Date"]]
    X_flat     = df[feat_cols].values
    y_flat     = df[TARGET_COLS].values
    n_features = X_flat.shape[1]

    X_tr_raw, X_te_raw = _split_temporal(X_flat)
    y_tr_raw, y_te_raw = _split_temporal(y_flat)

    # ── Escalar (fit solo en train) ───────────────────────────────────
    scaler   = StandardScaler()
    X_tr_sc  = scaler.fit_transform(X_tr_raw)
    X_te_sc  = scaler.transform(X_te_raw)

    # ── Secuencias dentro del split (sin data leakage) ────────────────
    X_tr_seq, y_tr_seq = crear_secuencias(X_tr_sc, y_tr_raw, timesteps)
    X_te_seq, y_te_seq = crear_secuencias(X_te_sc, y_te_raw, timesteps)

    # ── Arquitectura ─────────────────────────────────────────────────
    tf.keras.backend.clear_session()
    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Input(shape=(timesteps, n_features)))
    for i in range(n_layers):
        model.add(tf.keras.layers.LSTM(
            units,
            return_sequences=(i < n_layers - 1),
            dropout=dropout,
        ))
    model.add(tf.keras.layers.Dense(len(TARGET_COLS)))

    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=lr), loss="mse")
    model.fit(
        X_tr_seq, y_tr_seq,
        epochs=60,
        batch_size=batch_size,
        validation_split=0.1,
        callbacks=[tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        )],
        verbose=0,
    )

    return _metricas(y_te_seq, model.predict(X_te_seq, verbose=0), "LSTM")


# ══════════════════════════════════════════════════════════════════════
# GRÁFICA COMPARATIVA
# ══════════════════════════════════════════════════════════════════════

def grafica_comparativa(resultados: list[dict]) -> None:
    """
    Genera una figura con 3 subgráficas (MSE, MAE, R²).
    La barra del mejor modelo en cada métrica se resalta con borde dorado.
    """
    df = pd.DataFrame(resultados)
    modelos = df["modelo"].tolist()
    colores = ["steelblue", "tomato", "seagreen"]

    metricas_cfg = [
        ("MSE", "MSE ",  False),   # False = menor es mejor
        ("MAE", "MAE ",  False),
        ("R2",  "R²  ",  True),    # True  = mayor es mejor
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 5))

    for ax, (col, titulo, mayor_es_mejor) in zip(axes, metricas_cfg):
        valores = df[col].tolist()
        mejor_idx = (
            valores.index(max(valores)) if mayor_es_mejor
            else valores.index(min(valores))
        )

        bars = ax.bar(modelos, valores, color=colores, edgecolor="white", width=0.5)

        # Resaltar el mejor con borde dorado
        bars[mejor_idx].set_edgecolor("gold")
        bars[mejor_idx].set_linewidth(2.5)

        # Etiquetas numéricas encima de cada barra
        for bar, val in zip(bars, valores):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + abs(max(valores) - min(valores)) * 0.03,
                f"{val:.4f}",
                ha="center", va="bottom", fontsize=9.5, fontweight="bold",
            )

        # Límites del eje Y: siempre incluir 0 como referencia
        y_min = min(0, min(valores)) * 1.15
        y_max = max(valores) * 1.25 if max(valores) > 0 else max(valores) * 0.75
        ax.set_ylim(y_min, y_max)

        ax.set_title(titulo, fontsize=11, pad=8)
        ax.grid(axis="y", linestyle="--", alpha=0.4)
        ax.spines[["top", "right"]].set_visible(False)
        ax.tick_params(axis="x", labelsize=10)

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

    print("Leyendo mejores hiperparámetros desde CSV...")
    best = cargar_mejores_params(csv_path)

    modelos_requeridos = {"XGBoost", "MLP", "LSTM"}
    faltantes = modelos_requeridos - set(best.keys())
    if faltantes:
        raise ValueError(
            f"Faltan modelos en el CSV: {faltantes}. "
            "Asegúrate de haber optimizado los 3 modelos en main.py."
        )

    print("Cargando datos...")
    df_raw  = obtener_datos()
    df_prep = preprocesar_datos(df_raw)
    print(f"Dataset: {len(df_prep)} filas  |  Split: "
          f"{int(len(df_prep)*SPLIT_RATIO)} train / "
          f"{len(df_prep) - int(len(df_prep)*SPLIT_RATIO)} test\n")

    print("Entrenando y evaluando con los mejores hiperparámetros...")
    resultados = [
        evaluar_xgboost(df_prep, best["XGBoost"]),
        evaluar_mlp    (df_prep, best["MLP"]),
        evaluar_lstm   (df_prep, best["LSTM"]),
    ]

    # ── Tabla resumen ─────────────────────────────────────────────────
    print(f"\n{'═' * 42}")
    print("  RESUMEN  (promedio sobre los 3 targets)")
    print(f"{'═' * 42}")
    df_res = pd.DataFrame(resultados).set_index("modelo")
    print(df_res.to_string(float_format=lambda x: f"{x:.6f}"))

    mejor_mse = df_res["MSE"].idxmin()
    mejor_r2  = df_res["R2"].idxmax()
    print(f"\n  Menor MSE → {mejor_mse}")
    print(f"  Mayor R²  → {mejor_r2}")

    # ── Gráfica ───────────────────────────────────────────────────────
    grafica_comparativa(resultados)

    # ── Guardar CSV de resultados ─────────────────────────────────────
    ruta_res = os.path.join(BASE_DIR, "resultados_evaluacion.csv")
    df_res.reset_index().to_csv(ruta_res, index=False)
    print(f"Tabla guardada en:  {ruta_res}")
