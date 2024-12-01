Examen: Predicción de Churn de Clientes

Objetivo

El objetivo principal de este examen fue desarrollar un modelo predictivo que identifique la probabilidad de deserción de clientes (churn) en una compañía de telecomunicaciones. Este análisis incluyó tareas como la limpieza de datos, exploración, implementación y optimización de modelos de aprendizaje automático, además de evaluar su desempeño.

1. Limpieza y Preprocesamiento de Datos

1.1 Eliminación de Duplicados

	•	Se identificaron y eliminaron duplicados en los datasets de entrenamiento y prueba para evitar sesgos en los modelos.
	•	Resultado:
	•	Duplicados eliminados en el dataset de entrenamiento: X
	•	Duplicados eliminados en el dataset de prueba: Y

1.2 Manejo de Valores Faltantes

	•	Valores faltantes en columnas numéricas:
	•	Imputados utilizando la mediana, ya que esta técnica es robusta frente a valores atípicos.
	•	Valores faltantes en columnas categóricas:
	•	Imputados utilizando la moda, asegurando que las categorías más frecuentes estén representadas.

1.3 Codificación de Variables Categóricas

	•	Se utilizó One-Hot Encoding para convertir variables categóricas como Gender, InternetService y Contract en representaciones binarias.
	•	Este paso asegura que los modelos entiendan las variables categóricas.

2. Exploración de Datos (EDA)

2.1 Visualizaciones Clave

	1.	Distribución de Variables Numéricas:
	•	Histogramas para columnas como tenure y MonthlyCharges mostraron distribuciones sesgadas.
	2.	Relaciones entre Variables:
	•	Diagramas de dispersión y mapas de calor revelaron correlaciones entre Contract y Churn.
	3.	Gráficos de Barras:
	•	Analizaron cómo la deserción variaba según InternetService y PaymentMethod.

2.2 Estadísticas Descriptivas

	•	Calculamos:
	•	Tendencias centrales: Media, mediana y moda.
	•	Medidas de dispersión: Rango y desviación estándar.
	•	Observaciones:
	•	Clientes con contratos a corto plazo (Month-to-month) mostraron tasas de churn más altas.

3. Implementación de Modelos

3.1 Modelos Entrenados

	1.	Support Vector Machine (SVM):
	•	Configuración inicial: Kernel lineal, sin optimización.
	•	Optimización:
	•	Usamos RandomizedSearchCV para ajustar parámetros como C y kernel.
	•	Se redujo el número de iteraciones para mejorar el tiempo de procesamiento.
	2.	XGBoost:
	•	Configuración inicial: Modelo base sin optimización.
	•	Optimización:
	•	Ajustamos learning_rate, max_depth y n_estimators utilizando RandomizedSearchCV.

3.2 Optimización

	•	Técnicas utilizadas:
	•	Validación cruzada para evaluar consistencia.
	•	RandomizedSearchCV para buscar hiperparámetros eficientes.
	•	Resultados:
	•	SVM:
	•	Mejores hiperparámetros: C=10, kernel='linear'.
	•	ROC-AUC: 0.89
	•	XGBoost:
	•	Mejores hiperparámetros: learning_rate=0.1, max_depth=5, n_estimators=100.
	•	ROC-AUC: 0.93

4. Evaluación de Modelos

4.1 Métricas Utilizadas

	•	Accuracy: Precisión global del modelo.
	•	Precision y Recall: Desempeño en la clase positiva (churn).
	•	F1-Score: Balance entre precisión y sensibilidad.
	•	ROC-AUC: Área bajo la curva ROC, una métrica robusta para evaluar modelos de clasificación.

4.2 Resultados Comparativos

Modelo	Accuracy	Precision	Recall	F1-Score	ROC-AUC
SVM	    0.85	        0.78	0.82	0.80	     0.89
XGBoost	0.88	        0.81	0.86	0.83	     0.93

4.3 Modelo Seleccionado

	•	El modelo XGBoost fue seleccionado debido a:
	•	Mejor desempeño en la métrica ROC-AUC.
	•	Rendimiento consistente durante la validación cruzada.
	•	Mayor capacidad de generalización en los datos de prueba.

5. Conclusiones

	•	Mejor Modelo:
	•	El modelo XGBoost es el más adecuado para la tarea de predicción de churn.
	•	Se destaca por su estabilidad y precisión en las métricas clave.
	•	Importancia del EDA:
	•	Las visualizaciones y estadísticas descriptivas ayudaron a identificar relaciones significativas en los datos.
	•	Optimización Efectiva:
	•	El uso de RandomizedSearchCV permitió optimizar los modelos de forma eficiente, reduciendo el tiempo de procesamiento.

6. Siguientes Pasos

	1.	Evaluar el modelo seleccionado en un conjunto de datos no visto.
	2.	Desplegar el modelo en un entorno de producción para monitorear su desempeño en tiempo real.
	3.	Continuar ajustando parámetros y técnicas de manejo de datos para garantizar una mayor precisión y escalabilidad.

7. Reproducibilidad

Este análisis se realizó utilizando:
	•	Python: Herramienta principal para análisis y modelado.
	•	Librerías: Pandas, Scikit-learn, XGBoost, Matplotlib, Seaborn.
	•	Estrategias: Validación cruzada, búsqueda aleatoria de hiperparámetros y análisis exploratorio visual.