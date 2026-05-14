# Project Charter - Entendimiento del Negocio

## Nombre del Proyecto

[Predicción del precio de la acción de Apple mediante Machine Learning]

## Objetivo del Proyecto
jajaja
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
