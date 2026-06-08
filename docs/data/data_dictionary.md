# Diccionario de datos

## Base de datos 

Este dataset contiene información histórica de las acciones de **Apple Inc. (AAPL)**,
incluyendo precios de apertura, cierre, máximos, mínimos y volumen de transacciones.
Los datos provienen de **Kaggle** y fueron complementados con información actualizada
obtenida desde Yahoo Finance mediante la librería yfinance.

| Variable | Descripción | Tipo de dato | Rango/Valores posibles | Fuente de datos |
| --- | --- | --- | --- | --- |
| Date | Fecha de cotización de la acción. | datetime64[ns] | 1984-09-07 00:00:00 a 2025-01-17 00:00:00 | Kaggle / Yahoo Finance |
| Open | Precio de apertura de la acción en el mercado.	 | float64 | 0.05 a 258.19 | Kaggle / Yahoo Finance |
| High | Precio máximo alcanzado durante el día.	 | float64 | 0.06 a 260.10 | Kaggle / Yahoo Finance |
| Low | Precio mínimo alcanzado durante el día. | float64 | 0.05 a 257.63 | Kaggle / Yahoo Finance |
| Close | Precio de cierre de la acción.	 | float64 | 0.05 a 259.02 | Kaggle / Yahoo Finance |
| Adj Close | Precio de cierre ajustado por dividendos  | float64 | Rango/Valores posibles | Kaggle / Yahoo Finance |
| Volume | Cantidad de acciones negociadas durante el día. | 2838199.8879233 a 8788462909.8256 | Rango/Valores posibles | Kaggle / Yahoo Finance |


