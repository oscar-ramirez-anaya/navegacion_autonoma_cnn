# Modelos entrenados

Esta carpeta contiene los modelos generados por `gtsrb_cnn_colab.ipynb` /
`train_gtsrb_cnn.py`:

- `gtsrb_cnn.keras` — modelo Keras completo.
- `gtsrb_cnn.tflite` — modelo exportado a TensorFlow Lite (el que carga el
  controlador de Webots). Copiar tambien a `../../controllers/sign_detector/`.

Si los archivos no estan presentes, ejecuta el notebook de entrenamiento en
Google Colab (con GPU), descarga el `.tflite` y colocalo aqui.
