"""
===============================================================================
  sign_labels.py — Etiquetas GTSRB y mapeo hacia las senales del mundo Webots
===============================================================================

  Este modulo centraliza dos cosas:

    1. GTSRB_SIGN_NAMES : el nombre legible (en espanol) de cada una de las 43
       clases del dataset German Traffic Sign Recognition Benchmark (GTSRB).
       El orden (0..42) es el estandar del dataset y DEBE coincidir con el orden
       de carpetas de entrenamiento usado por la CNN.

    2. WEBOTS_SIGN_MAP : la correspondencia entre las 16 senales fisicas que
       existen en el mundo `city_2025b_lidar.wbt` (texturas estilo EE.UU. de
       Webots) y la clase GTSRB mas cercana. Como las texturas de Webots NO son
       identicas a las senales alemanas, esta tabla cumple dos funciones:
         - documenta el mapeo aproximado (para el reporte), y
         - define la etiqueta-objetivo de cada recorte capturado en el simulador
           durante la etapa de fine-tuning (adaptacion de dominio) de la CNN.

  Nota academica: el mapeo es intencionalmente aproximado. STOP (14) y CEDA EL
  PASO (13) transfieren de forma nativa; el resto de las senales se reconocen de
  forma confiable solo despues del fine-tuning con recortes reales del simulador,
  por lo que la clase GTSRB asignada actua como "ranura" de la senal de Webots.

  Equipo:
      Antonio Olvera Donlucas          A01795617
      Carlos Monir Radovich Saad       A01797569
      Andres Roberto Osuna Gonzalez    A01796264
      Oscar Alberto Ramirez Anaya      A01795438
===============================================================================
"""

# -----------------------------------------------------------------------------
# 1. Nombres de las 43 clases GTSRB (orden estandar del dataset)
# -----------------------------------------------------------------------------
GTSRB_SIGN_NAMES = {
    0:  "Limite de velocidad (20 km/h)",
    1:  "Limite de velocidad (30 km/h)",
    2:  "Limite de velocidad (50 km/h)",
    3:  "Limite de velocidad (60 km/h)",
    4:  "Limite de velocidad (70 km/h)",
    5:  "Limite de velocidad (80 km/h)",
    6:  "Fin de limite de velocidad (80 km/h)",
    7:  "Limite de velocidad (100 km/h)",
    8:  "Limite de velocidad (120 km/h)",
    9:  "Prohibido rebasar",
    10: "Prohibido rebasar (vehiculos > 3.5 t)",
    11: "Preferencia en el siguiente cruce",
    12: "Via prioritaria",
    13: "Ceda el paso",
    14: "Alto",
    15: "Circulacion prohibida",
    16: "Prohibido vehiculos > 3.5 t",
    17: "Prohibido el paso (contramano)",
    18: "Precaucion general",
    19: "Curva peligrosa a la izquierda",
    20: "Curva peligrosa a la derecha",
    21: "Curva doble",
    22: "Camino con baches",
    23: "Camino resbaloso",
    24: "Estrechamiento a la derecha",
    25: "Obras en el camino",
    26: "Semaforo adelante",
    27: "Peatones",
    28: "Cruce de ninos",
    29: "Cruce de bicicletas",
    30: "Cuidado con hielo/nieve",
    31: "Cruce de animales",
    32: "Fin de todos los limites",
    33: "Girar a la derecha adelante",
    34: "Girar a la izquierda adelante",
    35: "Solo seguir de frente",
    36: "Seguir de frente o derecha",
    37: "Seguir de frente o izquierda",
    38: "Mantener la derecha",
    39: "Mantener la izquierda",
    40: "Rotonda obligatoria",
    41: "Fin de prohibicion de rebasar",
    42: "Fin de prohibicion de rebasar (> 3.5 t)",
}

NUM_CLASSES = 43

# -----------------------------------------------------------------------------
# 2. Mapeo de las 16 senales del mundo Webots -> clase GTSRB mas cercana
# -----------------------------------------------------------------------------
# Cada entrada describe una senal fisica del archivo `city_2025b_lidar.wbt`:
#   - node        : tipo de nodo PROTO en el mundo
#   - webots_name : el campo `name` del nodo (vacio "" para el primero de su tipo)
#   - texture     : textura signImage usada (o "default" del PROTO)
#   - gtsrb_class : id GTSRB asignado (ranura de clasificacion / fine-tuning)
#   - transfer    : "nativo" si la CNN de GTSRB la reconoce sin fine-tuning,
#                   "fine-tuning" si requiere los recortes del simulador.
WEBOTS_SIGN_MAP = [
    {"node": "StopSign",       "webots_name": "",                "texture": "stop",                     "gtsrb_class": 14, "transfer": "nativo"},
    {"node": "YieldSign",      "webots_name": "",                "texture": "yield",                    "gtsrb_class": 13, "transfer": "nativo"},
    {"node": "SpeedLimitSign", "webots_name": "",                "texture": "speed_limit_55",           "gtsrb_class": 2,  "transfer": "fine-tuning"},
    {"node": "SpeedLimitSign", "webots_name": "speed limit(1)",  "texture": "speed_limit_55",           "gtsrb_class": 2,  "transfer": "fine-tuning"},
    {"node": "SpeedLimitSign", "webots_name": "speed limit(2)",  "texture": "speed_limit_65",           "gtsrb_class": 3,  "transfer": "fine-tuning"},
    {"node": "SpeedLimitSign", "webots_name": "speed limit(3)",  "texture": "speed_limit_65",           "gtsrb_class": 3,  "transfer": "fine-tuning"},
    {"node": "SpeedLimitSign", "webots_name": "speed limit(4)",  "texture": "one_way_sign_left",        "gtsrb_class": 39, "transfer": "fine-tuning"},
    {"node": "CautionSign",    "webots_name": "",                "texture": "turn_left",                "gtsrb_class": 19, "transfer": "fine-tuning"},
    {"node": "CautionSign",    "webots_name": "caution sign(1)", "texture": "default",                  "gtsrb_class": 18, "transfer": "fine-tuning"},
    {"node": "CautionSign",    "webots_name": "caution sign(2)", "texture": "bump",                     "gtsrb_class": 22, "transfer": "fine-tuning"},
    {"node": "CautionSign",    "webots_name": "caution sign(3)", "texture": "cross_roads",              "gtsrb_class": 11, "transfer": "fine-tuning"},
    {"node": "CautionSign",    "webots_name": "caution sign(4)", "texture": "turn_right",               "gtsrb_class": 20, "transfer": "fine-tuning"},
    {"node": "OrderSign",      "webots_name": "",                "texture": "default",                  "gtsrb_class": 35, "transfer": "fine-tuning"},
    {"node": "OrderSign",      "webots_name": "order sign(1)",   "texture": "default",                  "gtsrb_class": 38, "transfer": "fine-tuning"},
    {"node": "OrderSign",      "webots_name": "order sign(2)",   "texture": "no_right_turn",            "gtsrb_class": 9,  "transfer": "fine-tuning"},
    {"node": "OrderSign",      "webots_name": "order sign(3)",   "texture": "no_pedestrian_crossing",   "gtsrb_class": 27, "transfer": "fine-tuning"},
]

# Conjunto de clases GTSRB que esperamos ver en el mundo (para conteo de cobertura).
WEBOTS_EXPECTED_CLASSES = sorted({s["gtsrb_class"] for s in WEBOTS_SIGN_MAP})

# Total de senales fisicas en el mundo y meta minima de la actividad (>= 50%).
TOTAL_WORLD_SIGNS = len(WEBOTS_SIGN_MAP)          # 16
MIN_SIGNS_REQUIRED = (TOTAL_WORLD_SIGNS + 1) // 2  # 8


def friendly_name(class_id):
    """Devuelve el nombre legible de una clase GTSRB (o un texto de respaldo)."""
    return GTSRB_SIGN_NAMES.get(int(class_id), f"Clase {class_id}")


if __name__ == "__main__":
    # Pequena verificacion manual del mapeo
    print(f"Clases GTSRB definidas: {len(GTSRB_SIGN_NAMES)} (esperado 43)")
    print(f"Senales en el mundo: {TOTAL_WORLD_SIGNS}  | meta minima: {MIN_SIGNS_REQUIRED}")
    print(f"Clases GTSRB distintas esperadas en el mundo: {WEBOTS_EXPECTED_CLASSES}")
    print("\nMapeo senal Webots -> clase GTSRB:")
    for s in WEBOTS_SIGN_MAP:
        nm = s["webots_name"] or "(principal)"
        print(f"  {s['node']:<14} {nm:<16} {s['texture']:<22} -> "
              f"{s['gtsrb_class']:>2}  {friendly_name(s['gtsrb_class'])}  [{s['transfer']}]")
