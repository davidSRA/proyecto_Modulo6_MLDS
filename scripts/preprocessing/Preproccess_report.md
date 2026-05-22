## Preprocesamiento y análisis de relaciones entre variables

Durante la etapa de preprocesamiento se realizaron transformaciones orientadas a mejorar la capacidad predictiva del modelo y a extraer información temporal relevante del comportamiento del activo financiero.

Inicialmente, se verificó la consistencia de las variables numéricas, validando tipos de datos, ausencia de duplicados y tratamiento de valores faltantes. Posteriormente, se calcularon nuevas variables derivadas a partir del precio de cierre, incluyendo **promedios móviles (Moving Averages)** de diferentes ventanas temporales, con el objetivo de capturar tendencias de corto, mediano y largo plazo en la serie.

Estas variables permiten suavizar la volatilidad diaria y facilitan la identificación de patrones persistentes en el comportamiento del mercado.

### Análisis de correlación

Como parte del análisis exploratorio posterior al preprocesamiento, se construyó una **matriz de correlación** entre la variable objetivo `LogReturn`, los precios históricos y los promedios móviles calculados.

Este análisis permitió:

- Identificar variables con mayor asociación lineal con la variable objetivo.
- Detectar posibles problemas de **multicolinealidad** entre variables derivadas.
- Evaluar redundancia entre predictores antes del entrenamiento.
- Priorizar variables con mayor valor explicativo.

Una alta correlación entre algunos promedios móviles era esperada debido a su naturaleza acumulativa; sin embargo, su análisis permitió decidir cuáles conservar dentro del modelo.

### Matriz de correlación

![Matriz de correlación](correlacion.png)

---

### Análisis gráfico multivariado (Pair Plot)

Posteriormente, se construyó un **pair plot** para analizar visualmente las relaciones entre la variable objetivo y las variables transformadas.

Esta herramienta permitió observar:

- Distribución marginal de cada variable.
- Posibles relaciones lineales o no lineales.
- Presencia de valores atípicos.
- Agrupamientos o patrones temporales relevantes.
- Nivel de dispersión entre predictores.

El pair plot complementa el análisis de correlación al ofrecer una visión visual más detallada de la interacción entre variables.

### Pair Plot

![Pair Plot](pairplot.png)

---

### Conclusiones del preprocesamiento

A partir de este proceso se concluye que:

- Las variables derivadas capturan adecuadamente la dinámica temporal del activo.
- No se identificaron problemas críticos de calidad de datos.
- Existen relaciones significativas entre algunas variables explicativas y la variable objetivo.
- El conjunto de datos quedó preparado para la fase de entrenamiento y validación del modelo predictivo.
