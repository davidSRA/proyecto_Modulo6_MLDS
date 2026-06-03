import os

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from fpdf import FPDF

from aaplpredict.data_acquisition.main import obtener_datos

BASE_DIR = os.path.dirname(os.path.abspath(__file__))


def _guardar_figura(fig, nombre: str) -> str:
    ruta = os.path.join(BASE_DIR, nombre)
    fig.savefig(ruta, bbox_inches="tight")
    plt.close(fig)
    return ruta


def exportar_datos(df) -> None:
    imagenes = []

    df["LogReturn"] = np.log(df["Close"] / df["Close"].shift(1))
    df = df.dropna()

    nulos = df.isnull().sum().to_string()
    filas_cero = (df == 0).sum().to_string()
    fechas_dup = df["Date"].duplicated().sum()
    invalid_rows = df[
        (df["Low"] > df["Open"]) |
        (df["Low"] > df["Close"]) |
        (df["High"] < df["Open"]) |
        (df["High"] < df["Close"])
    ]
    filas_invalidas = invalid_rows.to_string() if not invalid_rows.empty else "Ninguna"
    asimetria = df["LogReturn"].skew()
    curtosis = df["LogReturn"].kurt()

    fig, ax = plt.subplots(figsize=(8, 4))
    df["LogReturn"].hist(bins=100, ax=ax)
    ax.set_title("Distribución de la Variable Objetivo")
    ax.set_xlabel("LogReturn")
    ax.set_ylabel("Frecuencia")
    imagenes.append(_guardar_figura(fig, "hist_basico.png"))

    fig, ax = plt.subplots(figsize=(8, 4))
    sns.histplot(data=df, x="LogReturn", kde=True, legend=False, ax=ax)
    ax.set_title("Distribución de los retornos logarítmicos")
    ax.set_xlabel("LogReturn")
    ax.set_ylabel("Frecuencia")
    imagenes.append(_guardar_figura(fig, "hist_kde.png"))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.boxplot(
        df["LogReturn"],
        vert=True, patch_artist=True,
        boxprops=dict(facecolor="steelblue", alpha=0.6),
        medianprops=dict(color="red", linewidth=2),
        flierprops=dict(marker="o", markerfacecolor="none",
                        markeredgecolor="gray", markersize=4),
    )
    ax.set_title("Boxplot de LogReturn")
    ax.set_ylabel("LogReturn")
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    imagenes.append(_guardar_figura(fig, "boxplot.png"))

    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()

    def titulo(texto):
        pdf.set_font("Arial", "B", 13)
        pdf.set_fill_color(230, 230, 230)
        pdf.cell(0, 9, texto, ln=True, fill=True)
        pdf.ln(2)

    def cuerpo(texto):
        pdf.set_font("Courier", size=9)
        for linea in texto.splitlines():
            pdf.cell(0, 5, linea, ln=True)
        pdf.ln(4)

    pdf.set_font("Arial", "B", 16)
    pdf.cell(0, 12, "Reporte EDA", ln=True, align="C")
    pdf.ln(4)

    titulo("1. Valores nulos")
    cuerpo(nulos)
    titulo("2. Filas con valor cero")
    cuerpo(filas_cero)
    titulo("3. Fechas duplicadas")
    cuerpo(str(fechas_dup))
    titulo("4. Filas inválidas (OHLC inconsistente)")
    cuerpo(filas_invalidas)
    titulo("5. Estadísticas de LogReturn")
    cuerpo(f"Asimetría : {asimetria:.4f}\nCurtosis  : {curtosis:.4f}")

    for img, subtitulo in zip(imagenes, [
        "6. Histograma de LogReturn",
        "7. Histograma con KDE",
        "8. Boxplot de LogReturn",
    ]):
        pdf.add_page()
        titulo(subtitulo)
        pdf.image(img, x=15, w=180)

    ruta_pdf = os.path.join(BASE_DIR, "reporte_eda.pdf")
    pdf.output(ruta_pdf)
    print(f"Reporte guardado en: {ruta_pdf}")


if __name__ == "__main__":
    df = obtener_datos()
    exportar_datos(df)
