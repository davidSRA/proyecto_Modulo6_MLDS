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
## Lecciones aprendidas

- Identificación de los principales desafíos y obstáculos encontrados durante el proyecto.
- Lecciones aprendidas en relación al manejo de los datos, el modelamiento y la implementación del modelo.
- Recomendaciones para futuros proyectos de machine learning.

## Impacto del proyecto

- **Impacto en la Industria:** El proyecto establece un puente directo entre la academia y la industria *fintech*, proveyendo a inversionistas minoristas y experimentados de un modelo explicativo y de baja latencia para la consulta de expectativas de retorno de AAPL.
- **Oportunidades Futuras:** - La infraestructura actual fue diseñada modularmente, lo que permite escalar la predicción a otros activos financieros y *tickers* del mercado.
  - Se puede explorar la integración de datos alternativos (como análisis de sentimiento en noticias o redes sociales) que complementen el movimiento de volumen y precio.

## Conclusiones

- Resumen de los resultados y principales logros del proyecto.
- Conclusiones finales y recomendaciones para futuros proyectos.

## Agradecimientos

- Agradecimientos al equipo de trabajo y a los colaboradores que hicieron posible este proyecto.
- Agradecimientos especiales a los patrocinadores y financiadores del proyecto.
