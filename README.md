Markdown

# Telecom X - Parte 2: Análisis Predictivo de la Evasión de Clientes (Churn)

## 1. Propósito del Análisis
El objetivo principal de este proyecto es construir y evaluar modelos de **machine learning** para predecir la evasión de clientes (*churn*) en la empresa de telecomunicaciones **Telecom X**. A través de este análisis, buscamos identificar los factores más influyentes en la cancelación de servicios, lo que permitirá a la empresa desarrollar estrategias de retención proactivas y dirigidas a los clientes de alto riesgo.

---

## 2. Estructura del Proyecto
El proyecto está organizado de la siguiente manera para garantizar su claridad y reproducibilidad:
* `TelecomX_Data_Procesado.csv`: Este es el archivo de datos principal, resultado de la limpieza y el pre-procesamiento realizados en la Parte 1 del desafío. Contiene solo las columnas relevantes, con los datos corregidos y estandarizados.
* `notebook.ipynb`: El cuaderno principal que contiene todo el código Python, desde la carga de los datos hasta la construcción y evaluación de los modelos predictivos.
* `Visualizaciones`: Las visualizaciones generadas por el cuaderno se incluyen como imágenes en este `README` para ilustrar los hallazgos clave.

---

## 3. Proceso de Preparación de Datos y Modelización

### Clasificación y Procesamiento de Variables
Las variables del conjunto de datos se clasificaron en:
* **Variables numéricas**: `Tiempo_Contrato`, `Costo_Mensual`, `Gasto_Total` y `Cuentas_Diarias`.
* **Variables categóricas**: `Genero`, `Servicio_Internet`, `Tipo_Contrato`, `Metodo_Pago`, entre otras.

### Etapas de Normalización y Codificación
* **Codificación (Encoding)**: Se aplicó *One-Hot Encoding* a las variables categóricas utilizando `pd.get_dummies()`. Esta técnica transforma las categorías en columnas binarias, lo que las hace compatibles con los algoritmos de machine learning.
* **Balanceo de Clases**: Dado el desbalance de clases observado (73% de "no evasión" frente a 27% de "evasión"), se utilizó la técnica de *oversampling* **SMOTE** para generar datos sintéticos y equilibrar la proporción a un 50% para cada clase. Esto mejora la capacidad del modelo para predecir correctamente la clase minoritaria (la evasión).
* **Normalización**: Se utilizó `StandardScaler` para normalizar los datos antes de entrenar el modelo de **Regresión Logística**. Esta etapa es crucial, ya que los modelos basados en distancias (como KNN) o en optimización de parámetros (como la Regresión Logística) son sensibles a la escala de los datos.

### Separación de los Datos
El conjunto de datos balanceado se dividió en:
* **Conjunto de Entrenamiento (80%)**: Utilizado para entrenar los modelos.
* **Conjunto de Prueba (20%)**: Utilizado para evaluar el rendimiento de los modelos en datos no vistos.

---

## 4. Ejemplos de Gráficos e Insights Obtenidos

### Correlación de variables con la Evasión
El análisis de correlación reveló que los factores más influyentes en la evasión son:
* **Correlación positiva (aumenta la evasión)**: Tipo de contrato `Mensual`, Servicio de Internet `Fibra óptica`.
* **Correlación negativa (reduce la evasión)**: `Tiempo_Contrato` y `Tipo_Contrato_Dos años`.

### Análisis dirigido: Tiempo de Contrato y Evasión
Los *boxplots* muestran claramente que los clientes que evaden tienen un tiempo de contrato significativamente menor.

### Análisis dirigido: Gasto Total y Evasión
El gasto total es mayor en los clientes que se quedan, lo cual es de esperar, ya que estos clientes tienen más tiempo de contrato.

---

## 5. Instrucciones de Ejecución
Para ejecutar el cuaderno y replicar el análisis, asegúrate de tener las siguientes librerías instaladas:
```bash
pip install pandas numpy scikit-learn matplotlib seaborn imbalanced-learn
