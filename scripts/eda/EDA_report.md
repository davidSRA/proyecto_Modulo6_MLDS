# Analisis Exploratorio de datos

Con el fin de complementar y actualizar el conjunto de datos original, se descargó información financiera adicional de :contentReference[oaicite:0]{index=0} (AAPL) desde :contentReference[oaicite:1]{index=1} utilizando la librería `yfinance`, para el periodo comprendido entre el 18 de enero de 2025 y el 3 de abril de 2026.

Después de integrar la información, el conjunto de datos final contiene **10.473 registros**, correspondientes a **datos estructurados de series de tiempo financieras**, con un tamaño aproximado de **0.47 MB**.

Durante la revisión de calidad de los datos se identificó que:
- No existen valores faltantes en las variables originales (`Date`, `Open`, `High`, `Low`, `Close`, `Volume`).
- Se presenta un valor faltante en las variables `Return` y `LogReturn`, generado naturalmente por el cálculo del rezago temporal.
- Todas las variables presentan el tipo de dato esperado (`Timestamp` para fechas y `float` para variables numéricas).

Posteriormente, se calculó la variable objetivo **LogReturn**, definida como el retorno logarítmico diario del precio de cierre:

`LogReturn = log(Close_t / Close_(t-1))`

Una vez eliminados los valores nulos, se construyó un histograma de la variable objetivo para evaluar su distribución. Además, se calcularon las métricas de **asimetría (skewness)** y **curtosis (kurtosis)**, con el fin de analizar la forma de la distribución e identificar posibles desviaciones respecto a una distribución normal antes del entrenamiento del modelo.

## Visualización de la variable objetivo

### Boxplot
El diagrama de caja permite identificar la dispersión de los datos y posibles valores atípicos en la variable objetivo.

![Boxplot LogReturn](images/boxplot.png)

---

### Histograma básico
El histograma permite observar la distribución general de la variable `LogReturn`.

![Histograma básico](images/hist_basico.png)

---

### Histograma con curva KDE
El histograma con estimación de densidad (KDE) permite visualizar con mayor detalle la forma de la distribución y su concentración.

![Histograma KDE](images/hist_kde.png)
