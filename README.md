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
En caso de no funcionar se puede hacer doble click en el archivo lanzador y se abrirá en una ventana la interfaz.

## Datos
La carpeta `archive_prep/` contiene las 65 imágenes seleccionadas del 
Brain Tumor MRI Dataset (Kaggle), filtradas según los criterios del informe 
(plano axial, tumores hiperintensos; clases meningioma, glioma y sin tumor).

## Dataset
Brain Tumor MRI Dataset (Kaggle):
https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset
Las 65 imágenes utilizadas (incluidas en `archive_prep/`) son una selección 
propia de ese dataset según los criterios descritos en el informe.

## Integrantes
Alvarez Taboada, Gowland, Losinno
