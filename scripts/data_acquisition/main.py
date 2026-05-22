import os
import kagglehub
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import yfinance as yf


def obtener_datos():
    # Descargamos el Dataset de Kaggle
    path = kagglehub.dataset_download("iamtanmayshukla/apple-inc-aapl-stock-data-1980-2024")
    df = pd.read_csv(f"{path}/aapl_us_2025.csv")
    df.index.name = "Index"
    df["Date"]=pd.to_datetime(df["Date"])

    # Importamos datos de Yahoo Finanace para complementar y actualizar el dataset:
    df_now = yf.download("AAPL", start="2025-01-18",end = "2026-04-03")
    df_now.columns = df_now.columns.get_level_values(0)
    df_now.index.name = 'Index'
    df_now.reset_index()
    df_now["Date"]=pd.to_datetime(df_now.index)

    #Unificamos los dos Datasets:
    df_total = pd.concat([df, df_now]).sort_values(by='Date')
    df_total.index = range(len(df_total))

    return df_total

    #print(df_total.tail())
if __name__ == "__main__":
    datos = obtener_datos()
    print(datos.head())

