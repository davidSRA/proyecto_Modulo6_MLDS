import sys
import os

# Agrega la carpeta scripts al path sin importar desde donde ejecutes
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from data_acquisition.main import obtener_datos
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
def preprocesar_datos(df):
    # Calculamos la media movil simple (SMA) para ventanas de 10
    df["SMA_10"] = df["Close"].rolling(window=10).mean()

    # Calculamos el retorno logaritmico
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))

    # Calculamos el indice de fuerza relativa (RSI)
    delta = df["Close"].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)

    n = 14
    avg_gain = gain.ewm(alpha=1/n, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/n, adjust=False).mean()
    rs = avg_gain / avg_loss
    df["RSI"] = 100 - (100 / (1 + rs))

    # Volatilidad
    df["Volatility_10"] = df["Close"].rolling(window=10).std()

    # Rango del precio
    df["Range"] = (df["High"] - df["Low"]) / df["Close"]

    # Valor logaritmico del volumen
    df["LogVolume"] = np.log(df["Volume"])

    # Gap Overnight
    df["Gap"] = np.log(df["Open"]) - np.log(df["Close"].shift(1))

    # Ratio
    df["Ratio"] = np.log(df["Close"] / df["SMA_10"])

    # Eliminamos columnas que no interesan
    df = df.drop(columns=["SMA_10", "Volume", "Open", "Close", "High", "Low"])
    df = df.dropna()

    cols = ['Ratio', 'Gap', 'LogVolume', 'LogReturn',
            'RSI', 'Volatility_10', 'Range']
    corr = df[cols].corr()

    # ── Matriz de correlación ──────────────────────────────────────────
    plt.figure(figsize=(10, 8))
    sns.heatmap(corr, annot=True, fmt=".2f", cmap='coolwarm')
    plt.title("Correlation Matrix")
    plt.tight_layout()
    plt.savefig(os.path.join(BASE_DIR, "correlacion.png"))
    plt.close()

    # ── Pairplot ───────────────────────────────────────────────────────
    fig = sns.pairplot(df[cols])
    fig.savefig(os.path.join(BASE_DIR, "pairplot.png"))
    plt.close()

    return df
if __name__ == "__main__":
    df = obtener_datos()
    datos_preprocesados = preprocesar_datos(df)
    print(datos_preprocesados.tail())