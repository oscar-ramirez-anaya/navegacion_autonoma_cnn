# Arquitectura — Detección de Señales con CNN (GTSRB) + Webots

Documento técnico de apoyo al reporte principal (`README.md`). Describe los dos
pipelines del proyecto: **entrenamiento** (offline, en Google Colab) e
**inferencia** (online, dentro del controlador de Webots).

---

## 1. Pipeline de entrenamiento (offline)

```
┌──────────────────────────────────────────────────────────────────────┐
│                     ENTRENAMIENTO (Google Colab / GPU)                 │
│                                                                        │
│  GTSRB (Kaggle)                                                        │
│   Train/0..42/   ──> Carga + RGB + resize 32x32 + /255                 │
│   Test.csv       ──────────────┐                                       │
│                                v                                       │
│                       ┌──────────────────┐                            │
│                       │   Aumentacion    │  rot/zoom/shift (sin flip)  │
│                       └────────┬─────────┘                            │
│                                v                                       │
│         ┌───────────────────────────────────────────┐                 │
│         │  CNN: Conv-Conv-Pool-Drop x2 -> Dense-Drop │                 │
│         │       -> Dense(43, softmax)                │                 │
│         └────────────────────┬──────────────────────┘                 │
│                              v                                         │
│        Evaluacion: accuracy prueba >90% + matriz de confusion          │
│                              v                                         │
│   (opcional) Fine-tuning de dominio con recortes reales de Webots      │
│                              v                                         │
│        Exportacion: gtsrb_cnn.keras  +  gtsrb_cnn.tflite               │
└──────────────────────────────────────────────────────────────────────┘
```

**Decisiones de diseño**
- **32×32 RGB:** tamaño estándar de GTSRB; el color es informativo (rojo/azul) y
  se conserva (a diferencia del SVM de la actividad previa, que usaba grises).
- **Dos bloques conv (32→64):** suficiente capacidad para 43 clases pequeñas;
  Dropout (0.25 y 0.5) controla el sobreajuste.
- **Sin flip horizontal:** invertiría el significado de señales direccionales.
- **`sparse_categorical_crossentropy`:** etiquetas enteras, sin one-hot.

---

## 2. Pipeline de inferencia (online, controlador Webots)

```
┌──────────────────────────────────────────────────────────────────────┐
│                  CONTROLADOR  sign_detector.py  (Webots)               │
│                                                                        │
│   Camara 256x128 (BGRA)                                                │
│        │  get_image()                                                  │
│        v                                                               │
│   ┌─────────────────────────────┐                                     │
│   │ Propuesta por color+forma   │  HSV: rojo + azul + amarillo        │
│   │ (mascaras -> contornos ->   │  -> morfologia -> bounding boxes    │
│   │  bounding boxes top-3)      │                                     │
│   └───────────────┬─────────────┘                                     │
│                   v  por cada recorte                                  │
│   ┌─────────────────────────────┐                                     │
│   │ Preprocesamiento            │  BGR->RGB, resize 32x32, /255       │
│   └───────────────┬─────────────┘                                     │
│                   v                                                    │
│   ┌─────────────────────────────┐                                     │
│   │ CNN TFLite (Interpreter)    │  softmax -> argmax + confianza      │
│   └───────────────┬─────────────┘                                     │
│                   v  confianza >= 0.60                                 │
│   ┌─────────────────────────────┐    ┌──────────────────────────┐     │
│   │ Mapeo clase GTSRB -> nombre │───>│ Overlay en Display        │     │
│   │ (sign_labels.py)            │    │ + consola + conteo (>=8)  │     │
│   └─────────────────────────────┘    └──────────────────────────┘     │
│                                                                        │
│   Teclado:  flechas = manejar | 'A' = guardar recorte | 'R' = recto    │
└──────────────────────────────────────────────────────────────────────┘
```

**Dispositivos del mundo (`city_2025b_lidar.wbt`, nodo `BmwX5`)**

| Dispositivo | Nombre        | Config                  | Uso                       |
|-------------|---------------|-------------------------|---------------------------|
| Cámara      | `camera`      | 256×128, FOV 1, BGRA    | Captura de la escena      |
| Pantalla    | `display_image` | 256×128 RGB           | Overlay de la detección   |
| Teclado     | —             | Keyboard()              | Manejo manual             |

---

## 3. Puente GTSRB ↔ Webots (mapeo de señales)

El mundo usa texturas estilo EE.UU.; GTSRB son señales alemanas. `sign_labels.py`
define `WEBOTS_SIGN_MAP`: cada una de las 16 señales físicas se asocia a la clase
GTSRB más cercana, que sirve como **ranura de clasificación** y como
**etiqueta-objetivo** del fine-tuning.

- Transfieren de forma **nativa** (sin fine-tuning): **ALTO** (14), **CEDA EL PASO** (13).
- El resto (límites de velocidad, precaución, orden) se reconocen tras el
  **fine-tuning de dominio** con recortes capturados en el simulador.

16 señales físicas → **14 clases GTSRB distintas**, superando la meta de ≥8 (50%).

---

## 4. Runtime de inferencia

El controlador no carga TensorFlow completo. Importa, en orden de preferencia:
`ai_edge_litert` → `tflite_runtime` → `tensorflow.lite`. El entorno Python 3.13
del simulador (`/Users/.../proyectos/env`) ya incluye `numpy` y `opencv-python`;
solo falta instalar uno de los runtimes TFLite (ver `controllers/sign_detector/requirements.txt`).
