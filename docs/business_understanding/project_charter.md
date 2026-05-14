# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

[Predicción del precio de la acción de Apple mediante Machine Learning]

## Objetivo del Proyecto

[Predecir de manera asertiva el precio de la acción de apple usando diversas herramientas de aprendizaje de máquina]

## Alcance del Proyecto

### Incluye:

- [Descripción de los datos disponibles]
- [Descripción de los resultados esperados]
- [Criterios de éxito del proyecto]

### Excluye:

- [Descripción de lo que no está incluido en el proyecto]

## Metodología

El proyecto se enfoca en la acción AAPL y utiliza datos históricos diarios con precios de apertura, máximo, mínimo, cierre, volumen e indicadores derivados. El flujo metodológico es el siguiente:

1. Consolidar y preparar el conjunto de datos histórico.
2. Transformar la serie de precios en log-retornos diarios.
3. Construir variables derivadas como RSI, volatilidad, rango diario, volumen logarítmico, gap y ratio.
4. Crear ventanas temporales de 20 días para capturar dependencias recientes.
5. Entrenar y comparar modelos MLP, Random Forest y tres variantes LSTM.
6. Reconstruir precios a partir de los log-retornos predichos.
7. Usar el mejor modelo para pronosticar los próximos 20 días y estimar una banda de confianza.

## Cronograma

| Etapa | Duración Estimada | Fechas |
|------|---------|-------|
| Entendimiento del negocio y carga de datos | 2 semanas | del 1 de mayo al 15 de mayo |
| Preprocesamiento, análisis exploratorio | 4 semanas | del 16 de mayo al 15 de junio |
| Modelamiento y extracción de características | 4 semanas | del 16 de junio al 15 de julio |
| Despliegue | 2 semanas | del 16 de julio al 31 de julio |
| Evaluación y entrega final | 3 semanas | del 1 de agosto al 21 de agosto |

Hay que tener en cuenta que estas fechas son de ejemplo, estas deben ajustarse de acuerdo al proyecto.

## Equipo del Proyecto

- [Nombre y cargo del líder del proyecto]
- [Nombre y cargo de los miembros del equipo]

## Presupuesto

[Descripción del presupuesto asignado al proyecto]

## Stakeholders

- [Nombre y cargo de los stakeholders del proyecto]
- [Descripción de la relación con los stakeholders]
- [Expectativas de los stakeholders]

## Aprobaciones

- [Nombre y cargo del aprobador del proyecto]
- [Firma del aprobador]
- [Fecha de aprobación]
