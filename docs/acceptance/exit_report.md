# Informe de salida

## Resumen Ejecutivo

Este informe describe los resultados del proyecto de predicción de la acción de Apple Inc. (AAPL) mediante técnicas de Machine Learning y Deep Learning. El objetivo central fue modelar los log-retornos de la acción, proporcionando una herramienta cuantitativa robusta para apoyar decisiones de inversión. Siguiendo la metodología Team Data Science Project (TDSP), se desarrolló un pipeline completo que culminó en el despliegue de una API a través de Railway, logrando entregar predicciones de forma oportuna y confiable.

## Resultados del proyecto

- **Entregables y Logros:**
  - Se implementó un pipeline riguroso que abarca desde la extracción de datos financieros (Yahoo Finance) hasta la ingeniería de características (indicadores técnicos y transformaciones temporales).
  - Se configuró exitosamente la validación cruzada temporal (TimeSeriesSplit) para prevenir el *data leakage*, garantizando que el modelo respete la secuencia cronológica de los datos bursátiles.
  - Se desplegó la arquitectura completa mediante FastAPI, habilitando el consumo del modelo en producción con Railway.

- Evaluación del modelo final y comparación con el modelo base.
- Descripción de los resultados y su relevancia para el negocio.

## Lecciones aprendidas

- Identificación de los principales desafíos y obstáculos encontrados durante el proyecto.
- Lecciones aprendidas en relación al manejo de los datos, el modelamiento y la implementación del modelo.
- Recomendaciones para futuros proyectos de machine learning.

## Impacto del proyecto

- Descripción del impacto del modelo en el negocio o en la industria.
- Identificación de las áreas de mejora y oportunidades de desarrollo futuras.

## Conclusiones

- Resumen de los resultados y principales logros del proyecto.
- Conclusiones finales y recomendaciones para futuros proyectos.

## Agradecimientos

- Agradecimientos al equipo de trabajo y a los colaboradores que hicieron posible este proyecto.
- Agradecimientos especiales a los patrocinadores y financiadores del proyecto.
