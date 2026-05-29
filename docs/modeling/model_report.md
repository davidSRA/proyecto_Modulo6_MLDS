# Reporte del Modelo Final

<div align="justify">
## Resumen Ejecutivo

El presente documento describe el desarrollo y evaluación del modelo final para predicción de variables financieras utilizando técnicas avanzadas de machine learning, deep learning y optimización bayesiana de hiperparámetros.

El sistema implementado compara tres enfoques principales:

XGBoost
Redes neuronales densas (MLP)
Redes recurrentes LSTM

La optimización de hiperparámetros fue realizada utilizando Optuna, permitiendo optimizar simultáneamente:

Arquitectura de modelos
Parámetros de entrenamiento
Técnicas de feature engineering
Configuración temporal de series financieras

La evaluación se realizó mediante validación cruzada temporal (TimeSeriesSplit) utilizando 5 folds para evitar filtración de información futura (data leakage).

La métrica principal utilizada fue el error cuadrático medio (MSE)
</div>

## Descripción del Problema
<div align="justify">

El objetivo principal del proyecto consiste en predecir el comportamiento futuro de variables financieras a partir de información histórica del mercado.

Los mercados financieros presentan características complejas como:

Alta volatilidad
Dependencia temporal
No estacionariedad
Ruido estadístico
Relaciones no lineales

Debido a estas propiedades, los modelos tradicionales presentan limitaciones importantes para capturar patrones predictivos robustos.

El problema abordado corresponde a un problema de regresión multisalida (multi-output regression), donde se busca estimar simultáneamente:

Retorno esperado a 1 día (Target_1d)
Retorno esperado a 5 días (Target_5d)
Volatilidad futura a 5 días (Target_Vol5d)

**Objetivos del modelo**
*Objetivo general*

Desarrollar un sistema predictivo robusto para modelar retornos financieros utilizando técnicas modernas de inteligencia artificial y optimización automática.

Objetivos específicos
Construir variables derivadas mediante feature engineering financiero.
Comparar modelos de árboles, redes densas y redes recurrentes.
Optimizar automáticamente hiperparámetros mediante Optuna.
Minimizar el error de predicción utilizando validación temporal.
</div>

## Descripción del Modelo
<div align="justify">

XGBoost

Modelo basado en gradient boosting sobre árboles de decisión.

Hiperparámetros optimizados
n_estimators
max_depth
learning_rate
subsample
colsample_bytree

Además:

ventanas rolling
número máximo de lags
MLP (Multi Layer Perceptron)

Red neuronal densa completamente conectada.

Componentes optimizados
Número de capas
Número de neuronas
Dropout
Learning rate
Batch size

Incluye:

Escalamiento mediante StandardScaler
EarlyStopping
Validación temporal
LSTM

Red neuronal recurrente especializada en series de tiempo.

Hiperparámetros optimizados
Número de timesteps
Número de capas LSTM
Unidades ocultas
Dropout
Learning rate
Batch size

Las secuencias temporales fueron construidas dinámicamente dentro de cada fold para evitar filtración temporal.

Optimización de hiperparámetros

La optimización fue realizada utilizando Optuna.

Estrategia utilizada
Algoritmo TPE

Tree-structured Parzen Estimator.

Permite realizar búsqueda bayesiana eficiente sobre espacios complejos de hiperparámetros.

Pruning

Se utilizó:

MedianPruner

Para eliminar trials con desempeño inferior durante el entrenamiento.

Esto permitió reducir significativamente el costo computacional.
</div>

## Evaluación del Modelo
<div align="justify">

Estrategia de validación
Se utilizó: **TimeSeriesSplit(n_splits=5)**

Esta metodología preserva la estructura temporal de los datos y evita utilizar información futura durante entrenamiento.

**Métricas de evaluación**
MSE (Mean Squared Error)

Métrica principal utilizada para optimización.
**$MSE=\frac{1}{n}\sum_{i=1}^{n}(y_i-\hat{y}_i)^2$**

Penaliza fuertemente errores grandes.

| Modelo | MSE Promedio | Desempeño |
|---|---|---| 
| MLP | 0.001491 | Mejor modelo | 
| LSTM | 0.001506 | Muy competitivo | 
| XGBoost | 0.001690 | Buen desempeño |

El modelo MLP obtuvo el menor valor de MSE, alcanzando un error promedio de: $$ MSE_{MLP} = 0.001491 $$ lo que indica una mayor capacidad para capturar relaciones no lineales entre las variables financieras construidas mediante feature engineering. El modelo LSTM presentó un desempeño muy cercano al MLP: $$ MSE_{LSTM} = 0.001506 $$ demostrando una adecuada capacidad para modelar dependencias temporales y patrones secuenciales dentro de las series financieras. Por su parte, XGBoost obtuvo: $$ MSE_{XGBoost} = 0.001690 $$


Los mejores hiperparámetros encontrados fueron almacenados automáticamente en:

mejores_params.csv

Este archivo contiene:

Modelo
Mejor MSE
Configuración óptima de hiperparámetros
</div>

## Conclusiones y Recomendaciones
<div align="justify">

En esta sección se presentarán las conclusiones y recomendaciones a partir de los resultados obtenidos. Se deben incluir los puntos fuertes y débiles del modelo, las limitaciones y los posibles escenarios de aplicación.
</div>


## Referencias
<div align="justify">

Chen, T., Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System.
Kingma, D., Ba, J. (2014). Adam: A Method for Stochastic Optimization.
Murphy, J. (1999). Technical Analysis of the Financial Markets.
Tsay, R. (2010). Analysis of Financial Time Series.
Chollet, F. (2021). Deep Learning with Python.
</div>

Optuna Developers. Optuna Documentation.
