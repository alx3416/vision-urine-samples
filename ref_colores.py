# ref_colores.py
# Extracción de color de la tira reactiva de REFERENCIA (JPG de la etiqueta).
# Salida: output/reference/ref_colores.csv y output/reference/ref_deteccion.png
#
# La imagen de referencia se procesa en su orientación natural (LEU arriba,
# GLU abajo). La inversión de orden aplica solo a las tiras de muestra.

import os
import csv
import numpy as np

import utils

# ============ PARÁMETROS ============
REF_PATH = r"mission-reagent-parameter-pic-647x1024.jpg"
OUTPUT_DIR = os.path.join("output", "reference")
CSV_NAME = "ref_colores.csv"
PNG_NAME = "ref_deteccion.png"

REF_X0, REF_X1 = 0.09, 0.20
REF_Y0, REF_Y1 = 0.005, 0.86
RADIO_FRAC = 0.30

SEP_L_MIN = 238
SEP_CHROMA_MAX = 7
ALTO_MIN = 15

# (clase, número de cuadros de color en la etiqueta), en orden natural.
FILAS = [
    ("LEU", 5), ("NIT", 2), ("URO", 6), ("PRO", 6), ("pH", 7),
    ("BLO", 7), ("SG", 7), ("KET", 6), ("BIL", 4), ("GLU", 6),
]
# ====================================


def detectar_columna0(img_rgb, x0=REF_X0, x1=REF_X1, y0=REF_Y0, y1=REF_Y1):
    """Detecta los centros verticales (una fila por clase) en la primera
    columna de cuadros."""
    import cv2
    h, w = img_rgb.shape[:2]
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    roi = img_rgb[ya:yb, xa:xb]
    rh, rw = roi.shape[:2]

    lab = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2LAB).astype(float)
    band = lab[:, int(0.3 * rw):int(0.7 * rw), :].mean(axis=1)
    segs = utils.segmentos_1d(band, rh, SEP_L_MIN, SEP_CHROMA_MAX, ALTO_MIN)

    centros_y = [ya + (a + b) // 2 for (a, b) in segs]
    col0_x = xa + rw // 2
    return centros_y, col0_x, rw, (xa, ya, xb, yb)


def medir_paso_horizontal(img_rgb, y_fila):
    """Mide el paso horizontal (px entre columnas) usando la fila con más
    cuadros."""
    import cv2
    h, w = img_rgb.shape[:2]
    banda = img_rgb[y_fila - 15:y_fila + 15, :]
    lab = cv2.cvtColor(cv2.cvtColor(banda, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2LAB).astype(float).mean(axis=0)
    segs = utils.segmentos_1d(lab, w, SEP_L_MIN, SEP_CHROMA_MAX, ALTO_MIN)
    centros_x = [(a + b) // 2 for (a, b) in segs]

    col0_esperado = int(0.145 * w)
    centros_x = [c for c in centros_x if c > col0_esperado - 30]
    if len(centros_x) < 2:
        raise RuntimeError("No se pudo medir el paso horizontal.")

    paso = int(round(np.mean(np.diff(centros_x))))
    return paso, centros_x[0]


def construir_cuadricula(img_rgb):
    centros_y, col0_x, rw, bbox = detectar_columna0(img_rgb)
    if len(centros_y) != len(FILAS):
        print(f"AVISO: {len(centros_y)} filas detectadas, "
              f"se esperaban {len(FILAS)}. Revisa umbrales.")

    idx_max = max(range(len(FILAS)), key=lambda i: FILAS[i][1])
    paso_x, x_inicial = medir_paso_horizontal(img_rgb, centros_y[idx_max])
    col0_x = x_inicial

    cuadricula = {}
    for (clase, total), cy in zip(FILAS, centros_y):
        for idx in range(total):
            cx = col0_x + idx * paso_x
            cuadricula[(clase, idx)] = (cx, cy)

    r = int(RADIO_FRAC * rw)
    return cuadricula, r, bbox, paso_x


def exportar_csv(img_rgb, cuadricula, r, ruta_csv):
    campos = ["clase", "indice", "R", "G", "B", "H", "S", "V",
              "L", "a", "b", "cx", "cy", "n_pixeles"]

    filas_orden = []
    for (clase, _t) in FILAS:
        idxs = sorted(i for (c, i) in cuadricula if c == clase)
        for i in idxs:
            filas_orden.append((clase, i))

    os.makedirs(os.path.dirname(ruta_csv) or ".", exist_ok=True)
    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for (clase, idx) in filas_orden:
            cx, cy = cuadricula[(clase, idx)]
            col = utils.color_mediano(img_rgb, cx, cy, r)
            wr.writerow({
                "clase": clase, "indice": idx,
                "R": col["R"], "G": col["G"], "B": col["B"],
                "H": col["H"], "S": col["S"], "V": col["V"],
                "L": col["L"], "a": col["a"], "b": col["b"],
                "cx": cx, "cy": cy, "n_pixeles": col["n"],
            })
    return len(filas_orden)


def main():
    ref = utils.leer_jpg(REF_PATH)
    cuadricula, r, bbox, paso_x = construir_cuadricula(ref)
    print(f"Total de cuadros: {len(cuadricula)} · paso horizontal: {paso_x}px "
          f"· radio: {r}px")

    ruta_csv = os.path.join(OUTPUT_DIR, CSV_NAME)
    ruta_png = os.path.join(OUTPUT_DIR, PNG_NAME)

    n = exportar_csv(ref, cuadricula, r, ruta_csv)
    print(f"CSV guardado: {ruta_csv} ({n} filas)")

    utils.guardar_deteccion_png(ref, cuadricula, r, ruta_png, bbox=bbox)
    print(f"PNG guardado: {ruta_png}")


if __name__ == "__main__":
    main()
