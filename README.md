# TPI Tumor Cerebro

TP de Procesamiento de Señales e Imágenes Biomédicas (16.63 - ITBA).
Pipeline de segmentación y caracterización de tumores cerebrales (meningioma vs glioma)
en imágenes de resonancia magnética (MRI).

## Estructura
- `01_caracterizacion_dataset.py` — análisis del dataset (conteo, histogramas)
- `02_preprocesamiento.py` — preprocesamiento (gaussiano, N4, ajuste de tono, punto negro)
- `03_segmentacion.py` — segmentación K-means vs Multi-Otsu + falsos positivos
- `04_regiones_concentricas.py` — regiones concéntricas (núcleo, borde, peritumoral)
- `05_comparacion_clases.py` — extracción de features GLCM y comparación entre clases
- `app.py` — aplicación web interactiva (Streamlit, "NeuroScan")
- `APP.bat` — lanzador de la aplicación

## Requisitos
Python 3.x con: numpy, scikit-image, SimpleITK, scikit-learn, pandas, matplotlib, streamlit, pillow

## Uso
Correr los scripts en orden (01 a 05). La aplicación se ejecuta con:
\`\`\`
streamlit run app.py
\`\`\`

## Dataset
Brain Tumor MRI Dataset (Kaggle). No incluido en el repositorio por tamaño.

## Integrantes
Alvarez Taboada, Gowland, Losinno
