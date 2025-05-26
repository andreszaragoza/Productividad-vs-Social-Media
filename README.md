# Análisis de Redes Sociales y Productividad

![Imagen representativa del análisis de redes sociales y productividad](../image/Social-Media-Manager.webp)

Este repositorio contiene dos notebooks de Jupyter que implementan un análisis de datos sobre la relación entre el uso de redes sociales y la productividad laboral.

## Contenido

El repositorio consta de dos notebooks principales:

### 1. Etl.ipynb

Este notebook implementa el proceso de Extracción, Transformación y Carga (ETL) de los datos. Sus principales características son:

- **Importación de datos**: Carga el dataset `social_media_vs_productivity.csv` utilizando la biblioteca Polars.
- **Exploración inicial**: Realiza un análisis exploratorio de los datos, mostrando información básica como dimensiones, tipos de datos y valores nulos.
- **Limpieza de datos**: Identifica y maneja valores nulos en el conjunto de datos, reemplazándolos con la mediana para columnas numéricas.
- **Análisis de correlación**: Examina la relación entre el tiempo de uso de redes sociales y la productividad.
- **Análisis por plataforma**: Compara la productividad promedio según la plataforma de redes sociales preferida por los usuarios.
- **Visualización**: Incluye visualizaciones para representar la relación entre el tiempo en redes sociales y la productividad.

### 2. Procesamiento.ipynb

Este notebook implementa un pipeline de procesamiento más avanzado con técnicas de aprendizaje automático. Sus principales características son:

- **Pipeline ETL básico**: Implementa una función de pipeline ETL que realiza transformaciones de datos de manera automatizada.
- **Manejo de datos por tipo**: Procesa columnas numéricas y categóricas de forma diferenciada.
- **Características derivadas**: Crea nuevas variables como el "índice de distracción" y categoriza el uso de redes sociales.
- **Agregaciones**: Realiza análisis agregados por tipo de trabajo y categoría de uso de redes sociales.
- **Detección de anomalías**: Utiliza el algoritmo Isolation Forest para identificar patrones anómalos en los datos.
- **Segmentación de usuarios**: Implementa clustering con K-Means para segmentar usuarios en 4 grupos según sus características.
- **Análisis de segmentos**: Analiza las características de cada segmento de usuarios identificado.

## Tecnologías utilizadas

- **Polars**: Biblioteca de análisis de datos de alto rendimiento para manipulación de dataframes.
- **NumPy**: Biblioteca para computación numérica.
- **Matplotlib**: Biblioteca para visualización de datos.
- **Seaborn**: Biblioteca para visualización estadística.
- **Scikit-learn**: Biblioteca de aprendizaje automático para algoritmos de clustering y detección de anomalías.

## Hallazgos principales

- La correlación entre el tiempo en redes sociales y la productividad real es de -0.010072, lo que sugiere una relación negativa muy débil.
- TikTok muestra la mayor productividad promedio (4.976946) entre las plataformas de redes sociales analizadas.
- Se identificaron 4 segmentos de usuarios con diferentes patrones de uso de redes sociales y niveles de estrés.
- El análisis de anomalías permitió identificar casos atípicos en el comportamiento de los usuarios.

## Uso

Para ejecutar estos notebooks:

1. Asegúrese de tener instaladas todas las dependencias necesarias.
2. Los datos deben estar ubicados en la ruta `../data/raw/social_media_vs_productivity.csv`.
3. Ejecute primero `Etl.ipynb` para el procesamiento básico de los datos.
4. Ejecute `Procesamiento.ipynb` para el análisis avanzado y la segmentación de usuarios.