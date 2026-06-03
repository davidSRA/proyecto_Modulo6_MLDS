from aaplpredict.data_acquisition.main import obtener_datos
from aaplpredict.eda.main import exportar_datos

if __name__ == "__main__":
    df = obtener_datos()
    exportar_datos(df)
