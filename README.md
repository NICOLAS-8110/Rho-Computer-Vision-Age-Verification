<img width="1280" height="640" alt="WhatsApp Image 2026-11-17 at 3 00 46 PM (16)" src="https://github.com/user-attachments/assets/679bae9d-0dbb-4f9a-ab2b-dd5dc7f75bf1" />

# 👁️ Ρ Proyecto Rho | Estimación de Edad con Visión Artificial (Good Seed)

## 📋 Descripción
La cadena de supermercados **Good Seed** desea explorar el uso de técnicas de visión artificial para ayudar a cumplir la normativa relacionada con la venta de bebidas alcohólicas. El objetivo principal consiste en desarrollar un modelo capaz de estimar la edad de una persona a partir de una fotografía facial, permitiendo identificar clientes potenciales menores de edad y apoyar el proceso de verificación en los puntos de venta.

Para abordar esta problemática se implementó un modelo de Deep Learning basado en la arquitectura **ResNet50** preentrenada con pesos de ImageNet, utilizando técnicas de transferencia de aprendizaje (*Transfer Learning*) para mejorar el desempeño del sistema.

## 🎯 Objetivos
- ✅ Realizar un análisis exploratorio de datos (EDA) sobre las 7.591 fotografías faciales para evaluar la distribución de edades, calidad e iluminación.
- ✅ Implementar un flujo de carga y preprocesamiento eficiente mediante `ImageDataGenerator`, escalando píxeles (`1./255`) y redimensionando imágenes a `224 x 224` píxeles.
- ✅ Construir una red neuronal convolucional aplicando *Transfer Learning* con **ResNet50** (`include_top=False`), `GlobalAveragePooling2D` y una capa de salida continua `Dense(1)`.
- ✅ Entrenar y evaluar el modelo con aceleración por GPU utilizando MSE como pérdida y el Error Absoluto Medio (**MAE**) como métrica principal de negocio.

## 🛠️ Herramientas y Habilidades Utilizadas
* **Lenguaje:** Python
* **Librerías:** Pandas, NumPy, Matplotlib, Seaborn, TensorFlow y Keras
* **Arquitectura de Red:** ResNet50 (Transfer Learning con pesos de ImageNet)
* **Entorno:** Google Colab (entrenamiento con GPU)
* **Técnicas:** Visión por Computadora, Deep Learning, Transfer Learning, Generadores de Imágenes (`ImageDataGenerator`) y Regresión Continua.

## 📊 Fuente de Datos y Estructura
El conjunto de datos cuenta con **7.591 imágenes faciales** de personas con edades comprendidas entre **1 y 100 años**:
* **`labels.csv`**: Tabla con los nombres de archivo (`file_name`) y la edad real (`real_age`).
* **`final_files/`**: Carpeta con las fotografías en formato `.jpg`.

## ⚙️ Metodología
1. **Análisis Exploratorio de Datos (EDA):** Revisión de valores nulos, análisis de la concentración etaria (mayor densidad entre los 20 y 40 años) e inspección visual de muestras faciales.
2. **Preparación y Partición de Datos:** División en **80% entrenamiento** y **20% validación** mediante `ImageDataGenerator()`, aplicando reescalado de píxeles y ajuste de resolución a `224 x 224` píxeles.
3. **Arquitectura del Modelo:**
   * Base convolucional **ResNet50** preentrenada en ImageNet (`include_top=False`).
   * Capa de reducción espacial `GlobalAveragePooling2D`.
   * Capa de salida lineal `Dense(1)` para estimación de edad.
   * Compilación con optimizador **Adam**, función de pérdida **MSE** y métrica **MAE**.
4. **Entrenamiento:** Ejecución durante **20 épocas** utilizando procesamiento GPU.

## 💡 Resultados y Métricas

### 🏆 Mejor Desempeño Alcanzado (Época 18)
| Métrica | Valor |
| :--- | :---: |
| **Pérdida de Validación (MSE)** | 112.6679 |
| **Validación MAE** | **8.1314 años** |

### 📊 Métricas al Finalizar las 20 Épocas
| Métrica | Entrenamiento | Validación |
| :--- | :---: | :---: |
| **Pérdida (MSE)** | 15.9708 | 141.7143 |
| **MAE** | 3.0936 años | 9.4365 años |

## 📈 Principales Hallazgos
* **Capacidad de aprendizaje:** La arquitectura **ResNet50** logró extraer patrones faciales relevantes asociados al envejecimiento, alcanzando un **MAE mínimo de 8.13 años** en el conjunto de validación.
* **Desbalanceo etario:** La mayor concentración de registros entre los 20 y 40 años facilita la precisión en adultos jóvenes, mientras que los extremos etarios presentan mayor variabilidad.
* **Aplicación práctica en punto de venta:** Dado el margen de error promedio (~8 años), el modelo no debe reemplazar la verificación documental tradicional, pero resulta altamente valioso como **sistema de alerta temprana** para asistir al personal de caja en **Good Seed**.

## 📂 Archivos Principales
* `Proyecto_Rho_Estimacion_de_Edad.ipynb`
* `labels.csv`
* `final_files/`

## 🏁 Conclusión
Se desarrolló con éxito un sistema de visión artificial basado en **ResNet50** capaz de estimar la edad facial con un **MAE de 8.13 años**. La solución cumple el objetivo exploratorio de **Good Seed** como herramienta de apoyo preventivo en la venta de bebidas alcohólicas. Para futuras iteraciones se recomienda incorporar aumento de datos (*Data Augmentation*), parada temprana (*Early Stopping*) y descongelamiento parcial de capas (*Fine-Tuning*) para reducir aún más el error de generalización.
