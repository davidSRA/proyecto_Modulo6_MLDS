# Definición de los datos

## Origen de los datos

- [ ] Dataset histórico de acciones de Apple Inc. (AAPL) obtenido desde Kaggle
- [ ] Datos financieros actualizados obtenidos desde Yahoo Finance mediante la librería yfinance

## Especificación de los scripts para la carga de datos

- [ ] l siguiente script se utilizó para la carga y consolidación de los datos:
- [ ] # Descarga del dataset desde Kaggle path = kagglehub.dataset_download("iamtanmayshukla/apple-inc-aapl-stock-data-1980-2024")
- [ ] df = pd.read_csv(f"{path}/aapl_us_2025.csv")

## Referencias a rutas o bases de datos origen y destino

- [ ] path = kagglehub.dataset_download("iamtanmayshukla/apple-inc-aapl-stock-data-1980-2024")


### Rutas de origen de datos

- [ ] path = kagglehub.dataset_download("iamtanmayshukla/apple-inc-aapl-stock-data-1980-2024")
- [ ] f"{path}/aapl_us_2025.csv"
- [ ] Describir los procedimientos de transformación y limpieza de los datos.

### Base de datos de destino

- [ ] Data set final = df_total
- [ ] Campo	Tipo de dato : Date	datetime64
- [ ] Open	float64
- [ ] High	float64
- [ ] Low	float64
- [ ] Close	float64
- [ ] Adj Close	float64
- [ ] Volume	int64
- [ ] el proceso final consistió en:

- [ ] Integración de múltiples fuentes de datos.
- [ ] Estandarización de formatos.
- [ ] Consolidación en un único DataFrame (df_total).
- [ ] Preparación de los datos para análisis exploratorio y modelado financiero.
