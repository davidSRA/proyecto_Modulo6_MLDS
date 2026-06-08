# Informe de salida

## Resumen Ejecutivo

Este informe describe los resultados del proyecto de predicción de la acción de Apple Inc. (AAPL) mediante técnicas de Machine Learning y Deep Learning. El objetivo central fue modelar los log-retornos de la acción, proporcionando una herramienta cuantitativa robusta para apoyar decisiones de inversión. Siguiendo la metodología Team Data Science Project (TDSP), se desarrolló un pipeline completo que culminó en el despliegue de una API a través de Railway, logrando entregar predicciones de forma oportuna y confiable.

## Resultados del proyecto

- **Entregables y Logros:**
  - Se implementó un pipeline riguroso que abarca desde la extracción de datos financieros (Yahoo Finance) hasta la ingeniería de características (indicadores técnicos y transformaciones temporales).
  - Se configuró exitosamente la validación cruzada temporal (TimeSeriesSplit) para prevenir el *data leakage*, garantizando que el modelo respete la secuencia cronológica de los datos bursátiles.
  - Se desplegó la arquitectura completa mediante FastAPI, habilitando el consumo del modelo en producción con Railway.

- **Evaluación del Modelo Final:** - Se construyó una línea base comparando tres arquitecturas: MLP, redes recurrentes LSTM y XGBoost. 
  - **XGBoost** demostró la mayor capacidad de generalización y el mejor desempeño global ($MSE_{XGBoost} = 0.001690$), siendo el único capaz de capturar la estructura de los datos con un coeficiente de determinación ($R^2$) positivo.
  - La red neuronal **LSTM** ($MSE_{LSTM} = 0.001506$) y el perceptrón multicapa (**MLP**) mostraron desempeños métricos cercanos, pero exhibieron limitaciones ($R^2$ negativo) para explicar de forma completa la alta variabilidad y no estacionariedad de la serie.
- **Relevancia para el negocio:** Se entrega una herramienta de análisis algorítmico que sirve como *benchmark* frente al análisis técnico tradicional, reduciendo el riesgo en la toma de decisiones para inversionistas y cumpliendo los estrictos criterios de rigor exigidos.
- 
## Lecciones aprendidas

- **Desafíos con Datos Financieros:** La naturaleza altamente volátil y ruidosa del mercado (comportamientos de ruido blanco) representa un reto. Modelar retornos futuros, en lugar de predecir el precio directo, confirmó ser el enfoque matemáticamente más sólido.
- **Optimización y Modelamiento:** Las arquitecturas de Deep Learning (LSTM y MLP) requirieron procesos de optimización exhaustivos a través de *Optuna*. Se evidenció que en entornos de datos tabulares financieros con alta varianza, los modelos basados en árboles (XGBoost) pueden converger más rápido hacia patrones generales sin sobreajustarse al ruido del mercado.
- **Ingeniería de Características:** La adición de variables derivadas (Target a 1d y 5d, y volatilidad a 5d) y el mantenimiento limpio de la matriz OHLCV (Open, High, Low, Close, Volume) fueron determinantes para encontrar señales válidas en la fase de entrenamiento.

## Impacto del proyecto

- **Impacto en la Industria:** El proyecto establece un puente directo entre la academia y la industria *fintech*, proveyendo a inversionistas minoristas y experimentados de un modelo explicativo y de baja latencia para la consulta de expectativas de retorno de AAPL.
- **Oportunidades Futuras:** - La infraestructura actual fue diseñada modularmente, lo que permite escalar la predicción a otros activos financieros y *tickers* del mercado.
  - Se puede explorar la integración de datos alternativos (como análisis de sentimiento en noticias o redes sociales) que complementen el movimiento de volumen y precio.

## Conclusiones

- El modelo XGBoost se consolidó como la herramienta más eficiente para esta tarea específica, equilibrando la precisión con la capacidad de generalizar datos futuros invisibles.
- Se cumplió a cabalidad con la estandarización TDSP y se superaron los retos del despliegue en la nube, entregando un producto de software funcional con métricas confiables y reproducibles.
- Los modelos dejan una puerta abierta para ser re-entrenados con datasets de mayor volumen e incluir información de noticias financieras para eliminar el ruido y/o utilizarse en la predicción.

## Agradecimientos

- Especial reconocimiento a la labor conjunta del equipo técnico: a Santiago Rodriguez por la dirección estratégica del proyecto, a Brayan Arturo Camargo por el diseño y validación de los modelos como Científico de Datos, y Oscar alfonso en la estructuración de la la Ingeniería de Datos.
- Agradecimiento a los evaluadores de la UNAL por la retroalimentación y el estándar de rigor académico exigido a lo largo del proceso.
