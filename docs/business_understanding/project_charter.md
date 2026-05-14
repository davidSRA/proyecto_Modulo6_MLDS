# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

Predicción del precio de la acción de Apple Inc. (AAPL) mediante aprendizaje profundo
Prueba DSR

## Objetivo del Proyecto

El objetivo de este proyecto es construir y comparar modelos de aprendizaje automático y aprendizaje profundo para estimar el comportamiento diario de la acción de Apple Inc. (AAPL). En lugar de predecir directamente el precio, se modela el log-retorno diario, ya que esta variable reduce la dependencia de escala y es más adecuada para trabajar con series financieras.

## Alcance del Proyecto

### Incluye:

- Descripción de los datos: Los datos contienen la información histórica del precio de la acción de Apple, entre ellos precio de apertura, cierre, máximo y mínimo.
- Descripción de los resultados esperados: Un modelo de aprendizaje de máquina que permita predecir el movimiento del precio de la acción de manera eficiente.
- Criterios de éxito del proyecto: 1) Cumplir los estándares de metodología de Team Data Science Project para el desarrollo de proyectos de ML. 2) Que el modelo funcione de manera efectiva para la predicción del precio de la acción de apple.

### Excluye:

- No incluye datos diferentes al movimiento del precio y volumen de la acción tales como noticias o profundidad de mercado. 

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
| Entendimiento del negocio y carga de datos | 5 dias | del 9 de mayo al 14 de mayo |
| Preprocesamiento, análisis exploratorio | 5 dias | del 15 de mayo al 20 de mayo |
| Modelamiento y extracción de características | 5 dias | del 20 de mayo al 5 de junio |
| Despliegue | 1 semanas | del 5 de junio al 10 de junio |
| Evaluación y entrega final | 1 dia | del 10 junio 11 de junio |

## Equipo del Proyecto

- [Santiago Rodriguez Cargo: Director del proyecto]
- [Oscar Fabian Alfonso Cargo: Ingenero de datos]
- [Brayan Camargo Cargo: Cientifico de Datos]

## Presupuesto

[El proyecto tiene caracter academico, razon por la cual el costo del mismo es de $ 0 pesos COP. Sin embargo, cualquier costo relacionado con creditos, capacidad de procesamiento y/o almacenamiento, se entiende asumido por la UNAL]

## Stakeholders
Los beneficiarios directos son los inversionistas minoristas principiantes que aún no dominan las
bases del análisis bursátil y quieren respaldo cuantitativo. También puede interesar a inversores más
experimentados que busquen comparar su criterio con una predicción automática. El proyecto se
enfocará en quienes siguen la acción de Apple, pero el pipeline final podrá adaptarse a otras acciones.

- [Nombre y cargo de los stakeholders del proyecto]
- [Descripción de la relación con los stakeholders]
- [Expectativas de los stakeholders]

## Aprobaciones

- [Nombre y cargo del aprobador del proyecto]
- [Firma del aprobador]
- [Fecha de aprobación]
