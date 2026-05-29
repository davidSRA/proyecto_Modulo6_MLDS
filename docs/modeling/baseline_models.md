# Reporte del Modelo Baseline

<div align="justify">
El modelo base corresponde al primer enfoque desarrollado para la predicción de los log retornos retornos financieros utilizando distintas técnicas de machine learning relacioanadas al modelado y predicción de series de tiempo bursátiles o financieras. **El objetivo principal del modelo es establecer una línea base de desempeño para comparar posteriormente modelos más avanzados y configuraciones optimizadas**, lo anterior a traves del uso de diferentes librerias de Python para la optimización de hiperparametros como Optuna.

El pipeline implementado incluye:

1. Adquisición de datos financieros.
2. Preprocesamiento de precios y retornos.
3.Extracción de características técnicas y temporales.
4. Entrenamiento de modelos de mahcine learning aveces llamados de aprendizaje automático.
5. Evaluación del desempeño predictivo.

</div>


## Descripción del modelo

<div align="justify">
   
Los modelos considerados para el analisis de las series financieras, y que se consideran dentro de la linea base incluyen:

1. Redes neuronales MLP
2. XGBoost
3. Redes LSTM para secuencias temporales

</div>


## Variables de entrada
<div align="justify">

La principal variable de entrada es el precio de las acciones de APPLE.
Posteriormente, esta variable fue calculada los log retornos de acuerdo con la siguiente formula: 

$$LogReturn_t =
\ln\left(
\frac{P_t}{P_{t-1}}
\right)$$

Posteriormente, a partur de esta variable, se construyeron otras variables variables de entrada, las cuales fueron construidas mediante técnicas conocidas del analisis de series de datos temporales para el suavizado de las series como, los promedios moviles, y sus diferentes versiones. En general las medias moviles, permiten sustraer o eliminar las variabilidad de la serie, para volverla una serie estacional y eliminar su volatilidad. Los indicadores calculados a partir de los log retornos son los siguientes:.

**Variables derivadas de los log retornos**
1. MACD (Moving Average Convergence Divergence)
2. MACD
3. MACD_Signal
4. MACD_Hist

**Elementos que capturan momentum y cambios de tendencia mediante medias móviles exponenciales.**
1. Bandas de Bollinger
2. BB_Width
3. BB_Pct

**Miden volatilidad relativa y posición del precio dentro de las bandas.**

1. ATR (Average True Range)
2. ATR_Norm

**Mide volatilidad real considerando gaps del mercado.**

1. Oscilador Estocástico
2. Stoch_K
3. Stoch_D

</div>


## Variable objetivo
<div align="justify">
   
Dentro del modelo de Machine Learning utilizado, Las variables objetivo utilizadas son:

1. Target_1d
2. Target_5d
3. Target_Vol5d

Donde:

Target_1d: Es el retorno esperado a 1 día.
Target_5d: Es el retorno acumulado esperado a 5 días.
Target_Vol5d: Es la volatilidad futura esperada en 5 días.

El principal objetivo del modelo corresponde al modelamiento y predicción de la variable de los retornos: Target_1d y Target_5d.

</div>


## Evaluación del modelo
<div align="justify">
### Métricas de evaluación

Dentro del listado de las Métricas de evaluación mas utilizadas y teniendo en cuenta que los datos de este proyecto aplicado son de tipo continuo, las métricas consideradas para evaluar el desempeño del modelo son las siguienets:

1. RMSE (Root Mean Squared Error): Mide el error cuadrático promedio entre predicciones y valores reales.

2. MAE (Mean Absolute Error): Mide el error absoluto promedio.

3. $R^2$ (Coeficiente de determinación): Indica la proporción de variabilidad explicada por el modelo.

</div>


### Resultados de evaluación
<div align="justify">
   
Tabla que muestra los resultados de evaluación del modelo baseline, incluyendo las métricas de evaluación.



</div>


## Análisis de los resultados
<div align="justify">
   
Descripción de los resultados del modelo baseline, incluyendo fortalezas y debilidades del modelo.
</div>


## Conclusiones
<div align="justify">
   
Conclusiones generales sobre el rendimiento del modelo baseline y posibles áreas de mejora.
</div>


## Referencias
<div align="justify">
Lista de referencias utilizadas para construir el modelo baseline y evaluar su rendimiento.

Espero que te sea útil esta plantilla. Recuerda que puedes adaptarla a las necesidades específicas de tu proyecto.
</div>
