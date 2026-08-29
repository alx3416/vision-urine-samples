# sample_colors.py
# Extracción de color de las tiras reactivas en imágenes de MUESTRA (PNG).
# Escena: tira vertical sobre soporte claro (izquierda), separador verde y
# tubo con la muestra (derecha, se descartan por crop).
#
# La tira está invertida respecto a la etiqueta: el primer cuadro (arriba)
# es GLU y el último (abajo) es LEU. El CSV se escribe en orden de etiqueta
# (LEU -> GLU) para que cada bloque de 10 filas coincida con ref_colores.csv.
#
# Salida consolidada:
#   output/samples/sample_colors.csv      (todas las imágenes de la carpeta)
#   output/samples/detection_<archivo>.png (diagnóstico por imagen)

import os
import csv
import glob

import cv2

import utils

# ============ PARÁMETROS ============
DATA_DIR = os.path.join("data", "101D3300-PNG")
OUTPUT_DIR = os.path.join("output", "samples")
CSV_NAME = "sample_colors.csv"

CROP_FRAC = (0.10, 0.42, 0.05, 0.95)   # x0, x1, y0, y1 (tercio izquierdo)
S_MIN = 55                              # umbral de saturación de la tira
N_CUADROS = 10
RADIO_FRAC = 0.34                       # radio del ROI = frac * ancho de tira

# Orden FÍSICO de la tira (arriba -> abajo), índice 0..9.
CLASES_FISICO = ["GLU", "BIL", "KET", "SG", "BLO",
                 "pH", "PRO", "URO", "NIT", "LEU"]
# Orden de ETIQUETA para el CSV (como en ref_colores.csv).
CLASES_ETIQUETA = ["LEU", "NIT", "URO", "PRO", "pH",
                   "BLO", "SG", "KET", "BIL", "GLU"]
# ====================================


def procesar_imagen(path):
    """Devuelve (dict clase -> centro (x,y) en original, r, dict clase -> color).

    Si no se pudo localizar/segmentar la tira, devuelve (None, None, None).
    """
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if img_bgr is None:
        raise FileNotFoundError(f"No se pudo leer: {path}")

    crop, off_crop = utils.crop_grueso(img_bgr, CROP_FRAC)
    m = utils.mascara_tira(crop, S_MIN)
    rect = utils.rect_tira(m, crop.shape[0])
    if rect is None:
        return None, None, None

    banda, M, off_banda, ancho = utils.enderezar_tira(crop, rect)
    seg = utils.segmentar_cuadros(banda, S_MIN, N_CUADROS, RADIO_FRAC)
    if seg is None:
        return None, None, None
    centros_banda, r = seg

    # mapear centros a la imagen original para medir color y dibujar
    centros_orig = utils.mapear_a_original(centros_banda, M, off_banda, off_crop)

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    centros = {}
    colores = {}
    for idx, (cx, cy) in enumerate(centros_orig):
        clase = CLASES_FISICO[idx]
        centros[clase] = (cx, cy)
        try:
            colores[clase] = utils.color_mediano(img_rgb, cx, cy, r)
        except RuntimeError:
            colores[clase] = None  # círculo fuera de imagen -> sin color
    return centros, r, colores


def dibujar_diagnostico(path, centros, r, ruta_png):
    """Dibuja los círculos detectados (con color medible) sobre la imagen
    original y guarda el PNG. Un cuadro sin círculo = no detectado."""
    img_bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    for clase, (cx, cy) in centros.items():
        h, w = img_bgr.shape[:2]
        if not (0 <= cx < w and 0 <= cy < h):
            continue
        cv2.circle(img_bgr, (cx, cy), r, (0, 0, 255), 4, cv2.LINE_AA)
        cv2.putText(img_bgr, clase, (cx - r, cy - r - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 0, 0), 3, cv2.LINE_AA)
    os.makedirs(os.path.dirname(ruta_png) or ".", exist_ok=True)
    cv2.imwrite(ruta_png, img_bgr)


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_csv = os.path.join(OUTPUT_DIR, CSV_NAME)

    campos = ["imagen", "clase", "R", "G", "B", "H", "S", "V",
              "L", "a", "b", "cx", "cy", "n_pixeles"]

    rutas = sorted(glob.glob(os.path.join(DATA_DIR, "*.png")))
    if not rutas:
        raise FileNotFoundError(f"No hay PNG en {DATA_DIR}")

    n_ok, n_fail = 0, 0
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()

        for path in rutas:
            nombre = os.path.basename(path)
            try:
                centros, r, colores = procesar_imagen(path)
            except Exception as e:
                print(f"[ERROR] {nombre}: {e}")
                centros = None

            if centros is None:
                # tira no localizada: 10 filas vacías + no hay PNG útil
                print(f"[FALLO] {nombre}: tira no detectada")
                n_fail += 1
                for clase in CLASES_ETIQUETA:
                    fila = {c: "" for c in campos}
                    fila["imagen"], fila["clase"] = nombre, clase
                    wr.writerow(fila)
                continue

            # escribir en orden de etiqueta (LEU -> GLU)
            for clase in CLASES_ETIQUETA:
                col = colores.get(clase)
                cx, cy = centros.get(clase, ("", ""))
                if col is None:
                    fila = {c: "" for c in campos}
                    fila["imagen"], fila["clase"] = nombre, clase
                    fila["cx"], fila["cy"] = cx, cy
                else:
                    fila = {
                        "imagen": nombre, "clase": clase,
                        "R": col["R"], "G": col["G"], "B": col["B"],
                        "H": col["H"], "S": col["S"], "V": col["V"],
                        "L": col["L"], "a": col["a"], "b": col["b"],
                        "cx": cx, "cy": cy, "n_pixeles": col["n"],
                    }
                wr.writerow(fila)

            ruta_png = os.path.join(OUTPUT_DIR, f"detection_{nombre}")
            dibujar_diagnostico(path, centros, r, ruta_png)
            n_ok += 1
            print(f"[OK] {nombre}")

    print(f"\nProcesadas: {n_ok} OK, {n_fail} fallidas")
    print(f"CSV: {ruta_csv}")


if __name__ == "__main__":
    main()
