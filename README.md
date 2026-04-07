# Clasificación de Cáncer Bucal con Deep Learning

Sistema de clasificación binaria de imágenes intraorales basado en **MobileNetV2** con Transfer Learning y Data Augmentation. Detecta la presencia de cáncer bucal a partir de fotografías clínicas con una precisión de **97.2%** en el conjunto de prueba.

**Demo en vivo:** [clasificacion-cancer-bucal.streamlit.app](https://clasificacion-cancer-bucal.streamlit.app/)

---

## Contexto

El cáncer bucal representa cerca del 3% de los cánceres diagnosticados a nivel mundial. La detección temprana incrementa la tasa de supervivencia a cinco años del 50% al 83%, pero el acceso limitado a especialistas en muchas regiones retrasa el diagnóstico. Este proyecto explora el uso de redes neuronales convolucionales como herramienta de apoyo al tamizaje clínico.

---

## Resultados del modelo seleccionado

El pipeline comparó seis configuraciones de entrenamiento. MobileNetV2 con Data Augmentation obtuvo el mejor rendimiento en todas las métricas:

| Modelo         | Dataset   | Accuracy | AUC-ROC | F1-Score |
|----------------|-----------|----------|---------|---------|
| CNN Scratch    | Sin Aug   | 88.3%    | 0.9601  | 0.8812  |
| CNN Scratch    | Con Aug   | 90.1%    | 0.9698  | 0.9023  |
| EfficientNetB0 | Sin Aug   | 93.4%    | 0.9845  | 0.9321  |
| EfficientNetB0 | Con Aug   | 95.8%    | 0.9914  | 0.9578  |
| MobileNetV2    | Sin Aug   | 94.1%    | 0.9901  | 0.9411  |
| **MobileNetV2**| **Con Aug**| **97.2%**| **0.9946**| **0.9733**|

---

## Arquitectura

- **Backbone:** MobileNetV2 preentrenado en ImageNet (feature extractor congelado en fase 1)
- **Clasificador:** `Dropout(0.3) → Linear(1280, 256) → ReLU → Dropout(0.2) → Linear(256, 1)`
- **Función de pérdida:** `BCEWithLogitsLoss` con `pos_weight` para compensar desbalance de clases
- **Optimizador:** Adam con `ReduceLROnPlateau`
- **Entrenamiento:** Fine-tuning en dos fases — primero el clasificador, luego las últimas capas del backbone
- **Hardware:** NVIDIA RTX 5070 Ti con CUDA 12.8 y Mixed Precision (AMP)

---

## Dataset

**Fuente:** [Oral Cancer Images for Classification](https://www.kaggle.com/datasets/muhammadatef/oral-cancer-images-for-classification) en Kaggle.

Las imágenes no están incluidas en el repositorio por razones de tamaño. Para reproducir los experimentos descarga el dataset desde Kaggle y colócalo en `dataset_original/`.

| Clase  | Etiqueta |
|--------|----------|
| Cancer | 1        |
| Normal | 0        |

**División:** 70% entrenamiento / 15% validación / 15% prueba

**Data Augmentation (10 técnicas):** rotación, flip horizontal/vertical, zoom, variación de brillo, contraste, saturación, ruido gaussiano, CLAHE, recorte aleatorio y escalado.

---

## Estructura del proyecto

```
.
├── notebooks/
│   ├── desarrollo_modelo.ipynb       # EDA y análisis exploratorio
│   ├── preprocesamiento.ipynb        # Pipeline de preprocesamiento y augmentation
│   └── ENTRENAMIENTO.ipynb           # Entrenamiento comparativo y Score-CAM
├── app/
│   ├── app.py                        # Aplicación Streamlit
│   └── requirements.txt             # Dependencias para despliegue
├── modelos/
│   └── mobilenet_con_aug/
│       └── best.pt                   # Pesos del modelo con mejor rendimiento
├── dataset_procesado/
│   ├── sin_augmentacion/manifiesto.csv
│   └── con_augmentacion/manifiesto.csv
└── resultados/
    └── gradcam/                      # Visualizaciones Score-CAM generadas
```

---

## Ejecución local

**Requisitos:** Python 3.10+

```bash
# Clonar el repositorio
git clone https://github.com/ErnestoSCL/clasificacion-cancer-bucal.git
cd clasificacion-cancer-bucal

# Crear entorno virtual e instalar dependencias
python -m venv venv
venv\Scripts\activate          # Windows
pip install -r app/requirements.txt

# Ejecutar la aplicación
streamlit run app/app.py
```

La app estará disponible en `http://localhost:8501`.

---

## Explicabilidad — Score-CAM

Para entender qué regiones de la imagen activan la predicción se implementó **Score-CAM** sobre la capa `features[13]` del backbone. A diferencia de Grad-CAM, Score-CAM no depende de gradientes, lo que produce mapas de calor más estables en modelos preentrenados con fine-tuning parcial.

Los mapas generados muestran que el modelo presta atención a las lesiones visibles (manchas blancas, eritroplasia, úlceras) y no a estructuras anatómicas irrelevantes como dientes o labios.

---

## Despliegue

La aplicación está publicada en **Streamlit Community Cloud** y se puede desplegar también en Hugging Face Spaces. El modelo corre en CPU, por lo que no requiere GPU en producción.

Para desplegar en Streamlit Cloud:

1. Hacer fork del repositorio
2. Ir a [share.streamlit.io](https://share.streamlit.io) y conectar el repositorio
3. Configurar: **Main file path** → `app/app.py`

---

## Aviso

Esta herramienta fue desarrollada con fines académicos e investigativos. No está certificada para uso diagnóstico clínico. Cualquier hallazgo debe ser confirmado por un profesional de salud bucodental calificado.

---

## Tecnologías

Python · PyTorch · Torchvision · Streamlit · Plotly · Pillow · NumPy · Matplotlib
