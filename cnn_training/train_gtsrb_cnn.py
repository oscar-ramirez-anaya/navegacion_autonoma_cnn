"""
===============================================================================
  Actividad 4.x — Entrenamiento de la CNN para GTSRB (MR4010.10)
===============================================================================

  Diseno, entrenamiento y exportacion de una Red Neuronal Convolucional (CNN)
  en Keras para clasificar las 43 clases del dataset GTSRB (German Traffic Sign
  Recognition Benchmark), con el objetivo de superar el 90% de exactitud en el
  conjunto de prueba oficial.

  Este script es el espejo reproducible del notebook `gtsrb_cnn_colab.ipynb`.
  La actividad pide entregar el notebook de Google Colab; este .py se versiona
  para poder re-entrenar de forma identica desde linea de comandos.

  Etapas:
      1. Carga y preprocesamiento del dataset GTSRB (32x32 RGB, normalizado).
      2. Definicion de la CNN (Conv + Pooling + Dropout + Fully-Connected).
      3. Entrenamiento con aumentacion de datos y callbacks.
      4. Evaluacion (exactitud, reporte de clasificacion, matriz de confusion).
      5. (Opcional) Fine-tuning de dominio con recortes reales del simulador
         Webots, declarado explicitamente como tecnica de adaptacion.
      6. Exportacion del modelo a Keras (.keras) y a TensorFlow Lite (.tflite).

  Uso (local con GPU o Colab):
      # Estructura esperada del dataset (descarga de Kaggle):
      #   DATA_DIR/Train/0../42/*.png        (imagenes de entrenamiento)
      #   DATA_DIR/Test/*.png + Test.csv     (conjunto de prueba oficial)
      #   DATA_DIR/Meta/...                  (metadatos, opcional)
      python train_gtsrb_cnn.py --data_dir ./GTSRB --epochs 25

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438
===============================================================================
"""

import os
import argparse
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt

try:
    import seaborn as sns
    _HAS_SNS = True
except ImportError:
    _HAS_SNS = False

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import classification_report, confusion_matrix

# -----------------------------------------------------------------------------
# Configuracion global
# -----------------------------------------------------------------------------
IMG_SIZE = 32          # las senales se redimensionan a 32x32 px (estandar GTSRB)
NUM_CLASSES = 43       # GTSRB tiene 43 clases de senales
SEED = 42

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(SCRIPT_DIR, "model")
SHOTS_DIR = os.path.join(SCRIPT_DIR, "..", "screenshots")

np.random.seed(SEED)
tf.random.set_seed(SEED)


# =============================================================================
# 1. CARGA Y PREPROCESAMIENTO DEL DATASET
# =============================================================================

def cargar_entrenamiento(data_dir):
    """
    Carga las imagenes de entrenamiento desde la estructura de carpetas de GTSRB:
        data_dir/Train/0/*.png ... data_dir/Train/42/*.png
    Cada subcarpeta es una clase (0..42). Todas las imagenes se redimensionan a
    32x32 RGB. Devuelve X (float32 normalizado [0,1]) y y (enteros de clase).
    """
    train_dir = os.path.join(data_dir, "Train")
    X, y = [], []
    for class_id in range(NUM_CLASSES):
        class_dir = os.path.join(train_dir, str(class_id))
        if not os.path.isdir(class_dir):
            continue
        for fname in os.listdir(class_dir):
            if not fname.lower().endswith((".png", ".ppm", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(class_dir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)        # OpenCV lee en BGR
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img)
            y.append(class_id)
    X = np.array(X, dtype=np.float32) / 255.0                 # normalizar a [0,1]
    y = np.array(y, dtype=np.int64)
    print(f"[DATA] Entrenamiento: {X.shape[0]} imagenes, {len(np.unique(y))} clases")
    return X, y


def cargar_prueba(data_dir):
    """
    Carga el conjunto de prueba oficial usando Test.csv, que contiene las columnas
    'ClassId' y 'Path' (ruta relativa a data_dir). Devuelve X_test, y_test.
    """
    csv_path = os.path.join(data_dir, "Test.csv")
    if not os.path.exists(csv_path):
        print("[DATA] Test.csv no encontrado; se omitira la evaluacion de prueba.")
        return None, None
    df = pd.read_csv(csv_path)
    X, y = [], []
    for _, row in df.iterrows():
        img = cv2.imread(os.path.join(data_dir, row["Path"]))
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
        X.append(img)
        y.append(int(row["ClassId"]))
    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)
    print(f"[DATA] Prueba: {X.shape[0]} imagenes")
    return X, y


# =============================================================================
# 2. ARQUITECTURA DE LA CNN
# =============================================================================

def construir_cnn():
    """
    CNN secuencial que ejercita las capas que pide la rubrica:
      - Convolution (Conv2D): extraen bordes, colores y formas de las senales.
      - Pooling (MaxPooling2D): reducen dimensionalidad y dan invarianza espacial.
      - Dropout: regularizacion que apaga neuronas al azar para evitar overfitting.
      - Fully-Connected (Dense): combinan las caracteristicas para clasificar.

    Dos bloques convolucionales (32 y 64 filtros) seguidos de un clasificador
    denso. La salida softmax produce la probabilidad de cada una de las 43 clases.
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),

        # --- Bloque convolucional 1 ---
        layers.Conv2D(32, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),                 # regularizacion tras el pooling

        # --- Bloque convolucional 2 ---
        layers.Conv2D(64, (3, 3), activation="relu", padding="same"),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.25),

        # --- Clasificador denso ---
        layers.Flatten(),
        layers.Dense(256, activation="relu"),
        layers.Dropout(0.5),                  # dropout fuerte antes de la salida
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",   # etiquetas enteras (no one-hot)
        metrics=["accuracy"],
    )
    return model


# =============================================================================
# 3. ENTRENAMIENTO
# =============================================================================

def entrenar(model, X, y, epochs, batch_size=128, val_split=0.2):
    """
    Entrena con aumentacion de datos. La aumentacion (rotaciones leves, zoom y
    desplazamientos) simula las variaciones de perspectiva y escala de las
    senales vistas desde un vehiculo, mejorando la generalizacion.
    NO se hace flip horizontal: invertiria el significado de muchas senales.
    """
    # Particion entrenamiento/validacion estratificada manual
    idx = np.arange(len(X))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    n_val = int(len(X) * val_split)
    val_idx, tr_idx = idx[:n_val], idx[n_val:]
    X_tr, y_tr = X[tr_idx], y[tr_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    datagen = ImageDataGenerator(
        rotation_range=12,        # +-12 grados
        zoom_range=0.15,          # +-15% de zoom
        width_shift_range=0.10,   # desplazamiento horizontal
        height_shift_range=0.10,  # desplazamiento vertical
        shear_range=0.10,
    )
    datagen.fit(X_tr)

    callbacks = [
        EarlyStopping(monitor="val_accuracy", patience=6, restore_best_weights=True),
        ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3, min_lr=1e-5),
    ]

    history = model.fit(
        datagen.flow(X_tr, y_tr, batch_size=batch_size),
        validation_data=(X_val, y_val),
        epochs=epochs,
        callbacks=callbacks,
        verbose=2,
    )
    return history


# =============================================================================
# 4. EVALUACION
# =============================================================================

def evaluar(model, X_test, y_test, history):
    os.makedirs(SHOTS_DIR, exist_ok=True)

    # --- Curvas de entrenamiento ---
    fig, ax = plt.subplots(1, 2, figsize=(11, 4))
    ax[0].plot(history.history["accuracy"], label="entrenamiento")
    ax[0].plot(history.history["val_accuracy"], label="validacion")
    ax[0].set_title("Exactitud"); ax[0].set_xlabel("epoca"); ax[0].legend()
    ax[1].plot(history.history["loss"], label="entrenamiento")
    ax[1].plot(history.history["val_loss"], label="validacion")
    ax[1].set_title("Perdida"); ax[1].set_xlabel("epoca"); ax[1].legend()
    fig.tight_layout()
    fig.savefig(os.path.join(SHOTS_DIR, "training_curves.png"), dpi=120)
    print(f"[EVAL] Curvas guardadas en screenshots/training_curves.png")

    if X_test is None:
        return None

    # --- Exactitud en prueba ---
    loss, acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"\n[EVAL] Exactitud en el conjunto de prueba GTSRB: {acc*100:.2f}%")
    objetivo = "CUMPLIDA (>90%)" if acc > 0.90 else "NO cumplida"
    print(f"[EVAL] Meta de la actividad: {objetivo}")

    # --- Reporte de clasificacion ---
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    print("\n[EVAL] Reporte de clasificacion (resumen):")
    print(classification_report(y_test, y_pred, digits=3, zero_division=0))

    # --- Matriz de confusion ---
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(12, 10))
    if _HAS_SNS:
        sns.heatmap(cm, cmap="viridis", square=True, cbar=True)
    else:
        plt.imshow(cm, cmap="viridis")
        plt.colorbar()
    plt.title(f"Matriz de confusion GTSRB (exactitud {acc*100:.2f}%)")
    plt.xlabel("Prediccion"); plt.ylabel("Real")
    plt.tight_layout()
    plt.savefig(os.path.join(SHOTS_DIR, "confusion_matrix.png"), dpi=120)
    print("[EVAL] Matriz de confusion guardada en screenshots/confusion_matrix.png")
    return acc


# =============================================================================
# 5. FINE-TUNING DE DOMINIO (recortes reales del simulador Webots)
# =============================================================================

def fine_tuning_webots(model, signs_dir, epochs=8):
    """
    Adaptacion de dominio. Las texturas de las senales de Webots (estilo EE.UU.)
    difieren de las senales alemanas de GTSRB, por lo que una CNN entrenada solo
    con GTSRB reconoce de forma confiable unicamente ALTO y CEDA EL PASO en el
    simulador. Esta etapa toma unos pocos recortes reales capturados en Webots
    (organizados como signs_dir/<class_id>/*.png) y reajusta el modelo a baja
    tasa de aprendizaje, con fuerte aumentacion, para reconocer las texturas del
    mundo sin olvidar GTSRB.

    Tecnica declarada explicitamente en el reporte (transfer learning /
    aumentacion de datos), conforme a la politica de la actividad.
    """
    if not signs_dir or not os.path.isdir(signs_dir):
        print("[FT] Carpeta de recortes de Webots no encontrada; se omite el fine-tuning.")
        return model

    X, y = [], []
    for class_id in range(NUM_CLASSES):
        cdir = os.path.join(signs_dir, str(class_id))
        if not os.path.isdir(cdir):
            continue
        for fname in os.listdir(cdir):
            if not fname.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            img = cv2.imread(os.path.join(cdir, fname))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
            X.append(img); y.append(class_id)
    if not X:
        print("[FT] No hay recortes etiquetados; se omite el fine-tuning.")
        return model

    X = np.array(X, dtype=np.float32) / 255.0
    y = np.array(y, dtype=np.int64)
    print(f"[FT] Recortes de Webots: {X.shape[0]} imagenes, {len(np.unique(y))} clases")

    datagen = ImageDataGenerator(
        rotation_range=15, zoom_range=0.20,
        width_shift_range=0.12, height_shift_range=0.12,
        brightness_range=(0.6, 1.4),
    )
    # Re-compilar con LR bajo para no destruir lo aprendido en GTSRB
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy", metrics=["accuracy"],
    )
    model.fit(datagen.flow(X, y, batch_size=16), epochs=epochs, verbose=2)
    print("[FT] Fine-tuning de dominio completado.")
    return model


# =============================================================================
# 6. EXPORTACION (Keras + TFLite) Y VERIFICACION DE PARIDAD
# =============================================================================

def exportar(model, X_ref=None):
    os.makedirs(MODEL_DIR, exist_ok=True)
    keras_path = os.path.join(MODEL_DIR, "gtsrb_cnn.keras")
    tflite_path = os.path.join(MODEL_DIR, "gtsrb_cnn.tflite")

    # --- Modelo Keras ---
    model.save(keras_path)
    print(f"[EXPORT] Modelo Keras guardado en {keras_path}")

    # --- Conversion a TensorFlow Lite (lo que carga el controlador) ---
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(tflite_path, "wb") as f:
        f.write(tflite_model)
    print(f"[EXPORT] Modelo TFLite guardado en {tflite_path} "
          f"({len(tflite_model)/1024:.0f} KB)")

    # --- Verificacion de paridad Keras vs TFLite ---
    if X_ref is not None and len(X_ref) > 0:
        interp = tf.lite.Interpreter(model_content=tflite_model)
        interp.allocate_tensors()
        inp = interp.get_input_details()[0]; out = interp.get_output_details()[0]
        muestras = X_ref[:20]
        keras_pred = np.argmax(model.predict(muestras, verbose=0), axis=1)
        tfl_pred = []
        for i in range(len(muestras)):
            interp.set_tensor(inp["index"], muestras[i:i+1].astype(np.float32))
            interp.invoke()
            tfl_pred.append(int(np.argmax(interp.get_tensor(out["index"])[0])))
        coincidencia = np.mean(keras_pred == np.array(tfl_pred)) * 100
        print(f"[EXPORT] Paridad Keras vs TFLite en {len(muestras)} muestras: "
              f"{coincidencia:.0f}% de coincidencia")
    return keras_path, tflite_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    ap = argparse.ArgumentParser(description="Entrenamiento CNN GTSRB")
    ap.add_argument("--data_dir", default="./GTSRB", help="Carpeta del dataset GTSRB")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch_size", type=int, default=128)
    ap.add_argument("--webots_signs", default="", help="Carpeta de recortes de Webots para fine-tuning")
    args = ap.parse_args()

    print("=" * 70)
    print("  Entrenamiento de la CNN para GTSRB (Actividad 4.x)")
    print("=" * 70)
    print(f"TensorFlow {tf.__version__} | GPU: {tf.config.list_physical_devices('GPU')}")

    # 1. Datos
    X, y = cargar_entrenamiento(args.data_dir)
    X_test, y_test = cargar_prueba(args.data_dir)

    # 2-3. Modelo + entrenamiento
    model = construir_cnn()
    model.summary()
    history = entrenar(model, X, y, epochs=args.epochs, batch_size=args.batch_size)

    # 4. Evaluacion
    evaluar(model, X_test, y_test, history)

    # 5. Fine-tuning de dominio (opcional)
    if args.webots_signs:
        model = fine_tuning_webots(model, args.webots_signs)
        if X_test is not None:
            _, acc = model.evaluate(X_test, y_test, verbose=0)
            print(f"[FT] Exactitud GTSRB tras fine-tuning: {acc*100:.2f}%")

    # 6. Exportacion
    exportar(model, X_ref=X_test if X_test is not None else X)
    print("\n[OK] Proceso completo. Copia cnn_training/model/gtsrb_cnn.tflite al "
          "controlador controllers/sign_detector/.")


if __name__ == "__main__":
    main()
