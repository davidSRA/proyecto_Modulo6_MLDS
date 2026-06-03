import numpy as np
import pandas as pd

from aaplpredict.data_acquisition.main import obtener_datos


def construir_targets(df: pd.DataFrame) -> pd.DataFrame:
    for h in [1, 5]:
        df[f"Target_{h}d"] = sum(
            df["LogReturn"].shift(-i) for i in range(1, h + 1)
        )
    df["Target_Vol5d"] = (
        pd.concat([df["LogReturn"].shift(-i) for i in range(1, 6)], axis=1)
        .std(axis=1)
    )
    return df


def preprocesar_datos(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df = df[~df.index.duplicated(keep="first")]
    df = df.sort_index()
    df["LogVolume"] = np.log(df["Volume"])
    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df = construir_targets(df)
    df = df.dropna()
    return df


if __name__ == "__main__":
    df = obtener_datos()
    print(preprocesar_datos(df).tail())
