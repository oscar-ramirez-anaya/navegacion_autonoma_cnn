"""
===============================================================================
  Actividad 4.x — Deteccion de Senales de Transito con CNN (MR4010.10)
  Controlador Webots: manejo manual con teclado + clasificacion CNN (GTSRB)
===============================================================================

  Descripcion:
      Controlador para Webots que permite conducir el vehiculo manualmente con
      el teclado y, en cada frame, detecta y clasifica las senales de transito
      visibles usando una Red Neuronal Convolucional (CNN) entrenada con el
      dataset GTSRB y exportada a formato TensorFlow Lite (.tflite).

  Pipeline de deteccion (en cada paso de simulacion):
      Camara (256x128 BGRA)
        -> Propuesta de region por color+forma (mascaras HSV rojo/azul/amarillo
           -> contornos -> bounding boxes candidatas)
        -> Recorte -> 32x32 RGB -> normalizacion [0,1]
        -> Inferencia CNN (TFLite) -> softmax -> argmax + confianza
        -> Mapeo clase GTSRB -> nombre legible (sign_labels.py)
        -> Overlay de la etiqueta sobre la pantalla de a bordo (Display)
        -> Registro en consola y conteo de senales distintas detectadas

  Controles de teclado:
      Flecha ARRIBA / ABAJO : acelerar / frenar (+-5 km/h)
      Flecha IZQ / DER      : girar el volante (+-0.05 rad, limite +-0.5)
      Tecla 'A'             : guardar el recorte de la senal detectada
                              (para la etapa de fine-tuning del notebook)
      Tecla 'R'             : reiniciar el volante a 0

  Este controlador reutiliza el esqueleto del controlador de la Actividad 3.1
  (autonomous_driver.py: get_image, init de dispositivos, bucle de teclado) y
  el patron de Display de la Actividad 2.1 (simple_controller_act_2_1).

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438

  Institucion:
      Instituto Tecnologico y de Estudios Superiores de Monterrey
      Maestria en Inteligencia Artificial

  Fecha: Junio 2026
===============================================================================
"""

import os
import sys
import time
import numpy as np
import cv2
import traceback

# --- Etiquetas y mapeo de senales (modulo local) ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sign_labels import (
    GTSRB_SIGN_NAMES, friendly_name,
    TOTAL_WORLD_SIGNS, MIN_SIGNS_REQUIRED,
)

# -----------------------------------------------------------------------------
# Carga del runtime de inferencia TFLite (con varios respaldos)
# -----------------------------------------------------------------------------
# Se intenta primero el runtime ligero (ai-edge-litert / tflite_runtime) y, si
# no esta disponible, se usa el interprete incluido en tensorflow.lite.
CNN_AVAILABLE = True
_Interpreter = None
try:
    from ai_edge_litert.interpreter import Interpreter as _Interpreter   # LiteRT moderno
except ImportError:
    try:
        from tflite_runtime.interpreter import Interpreter as _Interpreter  # tflite-runtime clasico
    except ImportError:
        try:
            from tensorflow.lite import Interpreter as _Interpreter         # fallback: TF completo
        except ImportError:
            print("[WARN] No se encontro runtime TFLite (ai-edge-litert / "
                  "tflite-runtime / tensorflow). La clasificacion CNN queda desactivada.")
            CNN_AVAILABLE = False

# --- Imports de Webots ---
from controller import Display, Keyboard
from vehicle import Driver


# ============================================================
# 1. CONSTANTES
# ============================================================

# Manejo manual
MAX_ANGLE = 0.5          # rad — angulo maximo del volante
MAX_SPEED = 120          # km/h — limite de velocidad manual
SPEED_INCR = 5           # km/h por pulsacion
ANGLE_INCR = 0.05        # rad por pulsacion
DEBOUNCE_TIME = 0.10     # s — anti-rebote del teclado

# CNN / inferencia
IMG_SIZE = 32            # debe coincidir con el entrenamiento (32x32 RGB)
CONF_THRESHOLD = 0.60    # confianza minima para aceptar una deteccion
INFER_EVERY = 2          # correr la CNN 1 de cada N frames (rendimiento)
TOP_K_ROIS = 3           # cuantas regiones candidatas clasificar por frame

# Filtros de las regiones candidatas (imagen 256x128 = 32768 px)
ROI_MIN_AREA = 80        # px^2 — descarta ruido pequeno
ROI_MAX_AREA = 18000     # px^2 — descarta fondos/edificios enormes
ROI_MIN_AR = 0.40        # relacion de aspecto minima (w/h)
ROI_MAX_AR = 2.50        # relacion de aspecto maxima (w/h)
ROI_PAD = 3              # px de margen alrededor del bounding box

DEBUG_EVERY = 30         # imprimir estado cada N pasos

# Rutas del modelo (relativas al controlador, con respaldo en cnn_training/model)
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..", "..")
MODEL_PATH_LOCAL = os.path.join(SCRIPT_DIR, "gtsrb_cnn.tflite")
MODEL_PATH_TRAIN = os.path.join(PROJECT_ROOT, "cnn_training", "model", "gtsrb_cnn.tflite")

# Carpeta donde se guardan los recortes capturados con la tecla 'A'
CAPTURE_DIR = os.path.join(PROJECT_ROOT, "cnn_training", "webots_signs", "raw")


# ============================================================
# 2. CAPTURA DE CAMARA
# ============================================================

def get_image(camera):
    """Extrae la imagen de la camara como matriz Numpy BGRA (alto x ancho x 4)."""
    raw = camera.getImage()
    if raw is None:
        return None
    return np.frombuffer(raw, np.uint8).reshape(
        (camera.getHeight(), camera.getWidth(), 4)
    )


# ============================================================
# 3. PROPUESTA DE REGIONES POR COLOR + FORMA
# ============================================================

def proponer_regiones(bgr):
    """
    Propone regiones candidatas que podrian contener una senal de transito,
    segmentando por color (rojo, azul y amarillo, que cubren las senales del
    mundo) y buscando contornos con tamano y forma plausibles.

    Las senales de transito son objetos cromaticos compactos; un umbral en el
    espacio HSV los aisla del fondo (cielo, asfalto, vegetacion) mejor que en BGR.

    Retorna una lista de bounding boxes (x, y, w, h) ordenadas por area (mayor
    primero), recortada a TOP_K_ROIS elementos.
    """
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)

    # Rojo: ocupa los dos extremos del circulo de matiz, por eso dos rangos.
    rojo1 = cv2.inRange(hsv, (0, 70, 50), (10, 255, 255))
    rojo2 = cv2.inRange(hsv, (170, 70, 50), (180, 255, 255))
    # Azul: senales de orden (mandato) circulares azules.
    azul = cv2.inRange(hsv, (100, 110, 40), (130, 255, 255))
    # Amarillo: senales de precaucion (diamantes amarillos estilo EE.UU.).
    amarillo = cv2.inRange(hsv, (20, 90, 90), (35, 255, 255))

    mask = cv2.bitwise_or(cv2.bitwise_or(rojo1, rojo2), cv2.bitwise_or(azul, amarillo))

    # Limpieza morfologica: cerrar huecos y eliminar motas.
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

    contornos, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    h_img, w_img = bgr.shape[:2]
    cajas = []
    for c in contornos:
        area = cv2.contourArea(c)
        if area < ROI_MIN_AREA or area > ROI_MAX_AREA:
            continue
        x, y, w, h = cv2.boundingRect(c)
        if h == 0:
            continue
        ar = w / float(h)
        if ar < ROI_MIN_AR or ar > ROI_MAX_AR:
            continue
        # Margen y recorte a los limites de la imagen
        x0 = max(0, x - ROI_PAD)
        y0 = max(0, y - ROI_PAD)
        x1 = min(w_img, x + w + ROI_PAD)
        y1 = min(h_img, y + h + ROI_PAD)
        cajas.append((x0, y0, x1 - x0, y1 - y0, area))

    # Ordenar por area (las senales mas cercanas/grandes primero) y truncar
    cajas.sort(key=lambda b: b[4], reverse=True)
    return [(x, y, w, h) for (x, y, w, h, _a) in cajas[:TOP_K_ROIS]]


# ============================================================
# 4. INFERENCIA CNN (TFLite)
# ============================================================

class CNNClassifier:
    """Envoltura ligera del interprete TFLite para clasificar recortes de senales."""

    def __init__(self, model_path):
        self.interpreter = _Interpreter(model_path=model_path)
        self.interpreter.allocate_tensors()
        self.inp = self.interpreter.get_input_details()[0]
        self.out = self.interpreter.get_output_details()[0]

    def predict(self, bgr_crop):
        """
        Recibe un recorte BGR, lo preprocesa igual que en el entrenamiento
        (BGR->RGB, resize 32x32, normalizacion [0,1]) y devuelve (clase, confianza).
        """
        rgb = cv2.cvtColor(bgr_crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (IMG_SIZE, IMG_SIZE))
        x = resized.astype(np.float32) / 255.0
        x = np.expand_dims(x, axis=0)            # forma (1, 32, 32, 3)
        self.interpreter.set_tensor(self.inp["index"], x)
        self.interpreter.invoke()
        probs = self.interpreter.get_tensor(self.out["index"])[0]
        class_id = int(np.argmax(probs))
        confidence = float(probs[class_id])
        return class_id, confidence


# ============================================================
# 5. DIBUJO Y PANTALLA DE A BORDO
# ============================================================

def dibujar_overlay(bgr, detecciones):
    """
    Dibuja sobre la imagen BGR los bounding boxes y etiquetas de las senales
    detectadas. `detecciones` es una lista de (box, class_id, conf).
    """
    vis = bgr.copy()
    for (x, y, w, h), class_id, conf in detecciones:
        cv2.rectangle(vis, (x, y), (x + w, y + h), (0, 255, 0), 1)
        etiqueta = f"{friendly_name(class_id)} {conf*100:.0f}%"
        # Fondo del texto para legibilidad
        ty = max(10, y - 4)
        cv2.putText(vis, etiqueta, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(vis, etiqueta, (x, ty), cv2.FONT_HERSHEY_SIMPLEX,
                    0.32, (0, 255, 0), 1, cv2.LINE_AA)
    return vis


def mostrar_en_display(display, bgr):
    """
    Envia la imagen BGR a la pantalla de a bordo de Webots (Display "display_image").
    Webots espera bytes en orden RGB, por lo que se convierte antes de enviar.
    """
    if display is None:
        return
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    image_ref = display.imageNew(
        rgb.tobytes(), Display.RGB, width=rgb.shape[1], height=rgb.shape[0]
    )
    display.imagePaste(image_ref, 0, 0, False)
    display.imageDelete(image_ref)


# ============================================================
# 6. MAIN — BUCLE PRINCIPAL DEL CONTROLADOR
# ============================================================

def main():
    # --- Cargar modelo CNN (TFLite) ---
    classifier = None
    if CNN_AVAILABLE:
        for path in [MODEL_PATH_LOCAL, MODEL_PATH_TRAIN]:
            if os.path.exists(path):
                try:
                    classifier = CNNClassifier(path)
                    print(f"[INFO] Modelo CNN (TFLite) cargado desde: {path}")
                    break
                except Exception as e:
                    print(f"[WARN] No se pudo cargar el modelo en {path}: {e}")
        if classifier is None:
            print("[WARN] Modelo gtsrb_cnn.tflite no encontrado.")
            print(f"[WARN] Buscado en: {MODEL_PATH_LOCAL}")
            print(f"[WARN]          y: {MODEL_PATH_TRAIN}")
            print("[WARN] Entrena el modelo en cnn_training/ y copia el .tflite aqui.")
            print("[WARN] Sin modelo, el vehiculo solo se conduce manualmente (sin deteccion).")

    # --- Inicializacion de Webots ---
    driver = Driver()
    timestep = int(driver.getBasicTimeStep())

    camera = driver.getDevice("camera")
    camera.enable(timestep)
    cam_w, cam_h = camera.getWidth(), camera.getHeight()
    print(f"[INFO] Camara {cam_w}x{cam_h}")

    # Pantalla de a bordo (puede no existir en algunos mundos)
    try:
        display_img = Display("display_image")
    except Exception:
        display_img = None
        print("[WARN] Display 'display_image' no disponible; solo salida por consola.")

    keyboard = Keyboard()
    keyboard.enable(timestep)

    # --- Variables de control ---
    speed = 0.0
    angle = 0.0
    last_press = {}
    step_count = 0
    senales_detectadas = set()    # clases GTSRB distintas vistas en el recorrido
    ultima_deteccion = []         # cache para el overlay entre frames de inferencia

    os.makedirs(CAPTURE_DIR, exist_ok=True)

    print("[INFO] Controles: flechas = manejar | 'A' = guardar recorte | 'R' = enderezar volante")
    print(f"[INFO] Meta de la actividad: detectar >= {MIN_SIGNS_REQUIRED} de {TOTAL_WORLD_SIGNS} senales")
    print("[INFO] Simulacion iniciada.")

    # --------------------------------------------------------
    # CICLO PRINCIPAL
    # --------------------------------------------------------
    while driver.step() != -1:
        try:
            step_count += 1

            # ---- 1. Leer camara ----
            bgra = get_image(camera)
            if bgra is None:
                continue
            bgr = bgra[:, :, :3].copy()   # descartar canal alfa

            # ---- 2. Deteccion + clasificacion (1 de cada INFER_EVERY frames) ----
            if classifier is not None and step_count % INFER_EVERY == 0:
                detecciones = []
                for (x, y, w, h) in proponer_regiones(bgr):
                    crop = bgr[y:y + h, x:x + w]
                    if crop.size == 0:
                        continue
                    class_id, conf = classifier.predict(crop)
                    if conf >= CONF_THRESHOLD:
                        detecciones.append(((x, y, w, h), class_id, conf))
                        if class_id not in senales_detectadas:
                            senales_detectadas.add(class_id)
                            print(f"╔══════════════════════════════════════════")
                            print(f"║ [DETECCION] Senal NUEVA: {friendly_name(class_id)}")
                            print(f"║  Clase GTSRB: {class_id} | confianza: {conf*100:.1f}%")
                            print(f"║  Cobertura: {len(senales_detectadas)} clases distintas "
                                  f"(meta >= {MIN_SIGNS_REQUIRED})")
                            print(f"╚══════════════════════════════════════════")
                ultima_deteccion = detecciones

            # ---- 3. Overlay + pantalla de a bordo ----
            vis = dibujar_overlay(bgr, ultima_deteccion)
            mostrar_en_display(display_img, vis)

            # ---- 4. Estado periodico en consola ----
            if step_count % DEBUG_EVERY == 0:
                activas = ", ".join(friendly_name(c) for _, c, _ in ultima_deteccion) or "ninguna"
                print(f"[ESTADO] v={speed:.0f} km/h | volante={angle:+.2f} rad | "
                      f"en vista: {activas} | total distintas: {len(senales_detectadas)}")

            # ---- 5. Teclado: manejo manual ----
            now = time.time()
            key = keyboard.getKey()
            if key != -1 and not (key in last_press and now - last_press[key] < DEBOUNCE_TIME):
                last_press[key] = now
                if key == keyboard.UP:
                    speed = min(MAX_SPEED, speed + SPEED_INCR)
                elif key == keyboard.DOWN:
                    speed = max(0.0, speed - SPEED_INCR)
                elif key == keyboard.RIGHT:
                    angle = min(MAX_ANGLE, angle + ANGLE_INCR)
                elif key == keyboard.LEFT:
                    angle = max(-MAX_ANGLE, angle - ANGLE_INCR)
                elif key == ord('R') or key == ord('r'):
                    angle = 0.0
                elif key == ord('A') or key == ord('a'):
                    # Guardar el mejor recorte detectado para el fine-tuning
                    ts = time.strftime("%Y%m%d_%H%M%S")
                    if ultima_deteccion:
                        (x, y, w, h), cid, _ = ultima_deteccion[0]
                        crop = bgr[y:y + h, x:x + w]
                        fn = os.path.join(CAPTURE_DIR, f"clase{cid:02d}_{ts}.png")
                        cv2.imwrite(fn, crop)
                        print(f"[CAPTURA] Recorte guardado: {fn} (sugerido clase {cid})")
                    else:
                        fn = os.path.join(CAPTURE_DIR, f"frame_{ts}.png")
                        cv2.imwrite(fn, bgr)
                        print(f"[CAPTURA] Frame completo guardado: {fn}")

            # ---- 6. Aplicar comandos al vehiculo ----
            driver.setSteeringAngle(angle)
            driver.setCruisingSpeed(speed)

        except Exception as e:
            print(f"[ERROR] {e}")
            traceback.print_exc()
            break

    # --- Resumen final ---
    print("\n==================== RESUMEN DEL RECORRIDO ====================")
    print(f"Senales (clases GTSRB) distintas detectadas: {len(senales_detectadas)}")
    for cid in sorted(senales_detectadas):
        print(f"   - [{cid}] {friendly_name(cid)}")
    meta = "CUMPLIDA" if len(senales_detectadas) >= MIN_SIGNS_REQUIRED else "NO cumplida"
    print(f"Meta (>= {MIN_SIGNS_REQUIRED} de {TOTAL_WORLD_SIGNS}): {meta}")
    print("==============================================================")


if __name__ == "__main__":
    main()
