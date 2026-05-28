import sys
import os

# Agrega la carpeta scripts al path sin importar desde donde ejecutes
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_acquisition.main import obtener_datos
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def construir_targets(df):
    """
    Target_1d:    retorno acumulado mañana         → señal inmediata
    Target_5d:    retorno acumulado próxima semana  → horizonte principal
    Target_Vol5d: volatilidad de esos 5 días        → base del IC
    """
    # Retorno acumulado a 1 y 5 días
    for h in [1, 5]:
        df[f"Target_{h}d"] = sum(
            df["LogReturn"].shift(-i) for i in range(1, h + 1)
        )

    # Volatilidad realizada de los próximos 5 días
    df["Target_Vol5d"] = (
        pd.concat([df["LogReturn"].shift(-i) for i in range(1, 6)], axis=1)
        .std(axis=1)
    )
    return df
def preprocesar_datos(df):
    """
    Solo limpieza y transformaciones base.
    Conserva OHLCV para que feature_extraction pueda usarlos.
    """
    df = df.copy()

    # 1. Eliminar duplicados y ordenar por fecha
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()

    # 2. Transformaciones base (escala logarítmica)

    df["LogVolume"] = np.log(df["Volume"])
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    # 3. Variable objetivo: retorno del día SIGUIENTE
    #    El modelo predice hacia adelante, no el día actual
    df = construir_targets(df)

    # 4. Eliminar la última fila (Target = NaN por el shift)
    df = df.dropna()

    return df   

if __name__ == "__main__":
    df = obtener_datos()
    datos_preprocesados = preprocesar_datos(df)
    print(datos_preprocesados.tail())