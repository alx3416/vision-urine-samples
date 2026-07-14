# ref_grid.py
# Detección de la cuadrícula completa de cuadros en la tira de REFERENCIA (JPG)
# y exportación de color (mediana RGB -> HSV, CIELAB) a output/ref_colores.csv
import os
import csv
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============ PARÁMETROS ============
REF_PATH = r"mission-reagent-parameter-pic-647x1024.jpg"
OUTPUT_DIR = "output"
CSV_NAME = "ref_colores.csv"

REF_X0, REF_X1 = 0.09, 0.20
REF_Y0, REF_Y1 = 0.005, 0.86

RADIO_FRAC = 0.30
SEP_L_MIN = 238
SEP_CHROMA_MAX = 7
ALTO_MIN = 15

FILAS = [
    ("LEU", 5), ("NIT", 2), ("URO", 6), ("PRO", 6), ("pH", 7),
    ("BLO", 7), ("SG", 7), ("KET", 6), ("BIL", 4), ("GLU", 6),
]
# ====================================


def leer_jpg(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se encontró: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _segmentos_1d(perfil_lab, longitud, sep_L, sep_chroma, minimo):
    L = perfil_lab[:, 0]
    A = perfil_lab[:, 1] - 128
    B = perfil_lab[:, 2] - 128
    chroma = np.convolve(np.sqrt(A ** 2 + B ** 2), np.ones(3) / 3, mode="same")
    es_sep = (L > sep_L) & (chroma < sep_chroma)
    segs, i = [], 0
    while i < longitud:
        if not es_sep[i]:
            j = i
            while j < longitud and not es_sep[j]:
                j += 1
            if j - i > minimo:
                segs.append((i, j))
            i = j
        else:
            i += 1
    return segs


def detectar_columna0(img_rgb, x0=REF_X0, x1=REF_X1, y0=REF_Y0, y1=REF_Y1):
    h, w = img_rgb.shape[:2]
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    roi = img_rgb[ya:yb, xa:xb]
    rh, rw = roi.shape[:2]
    lab = cv2.cvtColor(cv2.cvtColor(roi, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2LAB).astype(float)
    band = lab[:, int(0.3 * rw):int(0.7 * rw), :].mean(axis=1)
    segs = _segmentos_1d(band, rh, SEP_L_MIN, SEP_CHROMA_MAX, ALTO_MIN)
    centros_y = [ya + (a + b) // 2 for (a, b) in segs]
    col0_x = xa + rw // 2
    return centros_y, col0_x, rw, (xa, ya, xb, yb)


def medir_paso_horizontal(img_rgb, y_fila):
    h, w = img_rgb.shape[:2]
    banda = img_rgb[y_fila - 15:y_fila + 15, :]
    lab = cv2.cvtColor(cv2.cvtColor(banda, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2LAB).astype(float).mean(axis=0)
    segs = _segmentos_1d(lab, w, SEP_L_MIN, SEP_CHROMA_MAX, ALTO_MIN)
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


def color_mediano(img_rgb, cx, cy, r):
    """Mediana RGB de los píxeles dentro del círculo, y sus conversiones."""
    h, w = img_rgb.shape[:2]
    # ventana acotada al bounding box del círculo
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)
    parche = img_rgb[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]
    mascara = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    pix = parche[mascara]                       # (N, 3) RGB
    n = len(pix)
    if n == 0:
        raise RuntimeError(f"Círculo vacío en ({cx},{cy}).")

    R, G, B = [int(round(np.median(pix[:, c]))) for c in range(3)]

    # convertir el color único (mediana) a HSV y LAB con OpenCV
    px_rgb = np.uint8([[[R, G, B]]])
    px_bgr = cv2.cvtColor(px_rgb, cv2.COLOR_RGB2BGR)
    H, S, V = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV)[0, 0].tolist()
    Lab = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2LAB)[0, 0].tolist()
    L, a, b = Lab
    return dict(R=R, G=G, B=B, H=H, S=S, V=V, L=L, a=a, b=b, n=n)


def exportar_csv(img_rgb, cuadricula, r, ruta_csv):
    campos = ["clase", "indice", "R", "G", "B", "H", "S", "V",
              "L", "a", "b", "cx", "cy", "n_pixeles"]
    filas_orden = []
    for (clase, _t) in FILAS:
        idxs = sorted(i for (c, i) in cuadricula if c == clase)
        for i in idxs:
            filas_orden.append((clase, i))

    with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=campos)
        wr.writeheader()
        for (clase, idx) in filas_orden:
            cx, cy = cuadricula[(clase, idx)]
            col = color_mediano(img_rgb, cx, cy, r)
            wr.writerow({
                "clase": clase, "indice": idx,
                "R": col["R"], "G": col["G"], "B": col["B"],
                "H": col["H"], "S": col["S"], "V": col["V"],
                "L": col["L"], "a": col["a"], "b": col["b"],
                "cx": cx, "cy": cy, "n_pixeles": col["n"],
            })
    return len(filas_orden)


def main():
    ref = leer_jpg(REF_PATH)
    cuadricula, r, bbox, paso_x = construir_cuadricula(ref)
    print(f"Total de cuadros: {len(cuadricula)} · paso horizontal: {paso_x}px "
          f"· radio: {r}px")

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    ruta_csv = os.path.join(OUTPUT_DIR, CSV_NAME)
    n = exportar_csv(ref, cuadricula, r, ruta_csv)
    print(f"CSV guardado: {ruta_csv} ({n} filas)")

    fig, ax = plt.subplots(1, 1, figsize=(9, 12))
    ax.imshow(ref)
    xa, ya, xb, yb = bbox
    ax.add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya,
                               fill=False, edgecolor="lime",
                               linewidth=1.5, linestyle="--"))
    for (clase, idx), (cx, cy) in cuadricula.items():
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                edgecolor="red", linewidth=1.5))
        ax.text(cx, cy, str(idx), color="blue", fontsize=6,
                ha="center", va="center")
    for (clase, _t) in FILAS:
        cx0, cy0 = cuadricula[(clase, 0)]
        ax.text(bbox[0] - 45, cy0, clase, color="black", fontsize=8,
                ha="right", va="center", fontweight="bold")
    ax.set_title("Referencia · cuadrícula CLASE,índice")
    ax.axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()