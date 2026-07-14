# roi_circulos.py
import os
import random
import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============ PARÁMETROS DE AJUSTE ============
NEF_DIR = r"E:\data\FOTOS ORINAS\101D3300"
REF_PATH = r"mission-reagent-parameter-pic-647x1024.jpg"

REGION_FRAC = 0.40      # fracción izquierda de la imagen NEF donde buscar la tira
N_CUADROS = 10          # número de cuadros/colores por tira
RADIO_FRAC = 0.30       # radio del círculo como fracción del ancho de la ROI
SAT_THR = 45            # umbral de saturación para segmentar la tira
SAT_THR_CUADRO = 50     # umbral para detectar el borde del primer cuadro
# ==============================================


# ---------------- Lectura ----------------
def leer_nef(path):
    with rawpy.imread(path) as raw:
        return raw.postprocess(use_camera_wb=True, output_bps=8,
                               no_auto_bright=False)


def leer_jpg(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se encontró: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def nef_aleatorio(directorio):
    nefs = [f for f in os.listdir(directorio) if f.lower().endswith(".nef")]
    if not nefs:
        raise FileNotFoundError(f"No hay .NEF en {directorio}")
    elegido = random.choice(nefs)
    return os.path.join(directorio, elegido), elegido


# ---------------- ROI referencia (fija) ----------------
def roi_referencia(img_ref, x0=0.09, x1=0.20, y0=0.005, y1=0.86):
    h, w = img_ref.shape[:2]
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    return img_ref[ya:yb, xa:xb], (xa, ya, xb, yb)


# ---------------- ROI tira NEF ----------------
def roi_tira_nef(img_rgb, region_frac=REGION_FRAC, sat_thr=SAT_THR,
                 min_aspect=3.0, area_frac_min=0.005,
                 margin_x=0.25, margin_y=0.02):
    H, W = img_rgb.shape[:2]
    tw = int(region_frac * W)
    region = img_rgb[:, :tw]

    hsv = cv2.cvtColor(cv2.cvtColor(region, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]
    mask = (S > sat_thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 120)), iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No se detectó la tira.")

    best = None
    min_area = area_frac_min * region.shape[0] * region.shape[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        if cv2.contourArea(c) < min_area:
            continue
        if h / max(w, 1) >= min_aspect and (best is None or
                                            cv2.contourArea(c) > best[-1]):
            best = (x, y, w, h, cv2.contourArea(c))
    if best is None:
        raise RuntimeError("No se encontró un objeto vertical tipo tira.")

    x, y, w, h, _ = best
    mx, my = int(margin_x * w), int(margin_y * h)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(tw, x + w + mx), min(H, y + h + my)
    return img_rgb[y1:y2, x1:x2], (x1, y1, x2, y2), tw


# ---------------- Centros: NEF (anclar en 1er cuadro + replicar) ----------------
def centros_tira_nef(roi_rgb, n=N_CUADROS, sat_thr=SAT_THR_CUADRO):
    """
    Ancla en el primer cuadro (el más contrastante con el fondo),
    mide el paso al segundo cuadro y lo replica hacia abajo para los n cuadros.
    Robusto ante los últimos cuadros pálidos que se confunden con el fondo.
    """
    rh, rw = roi_rgb.shape[:2]
    hsv = cv2.cvtColor(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    # perfil de saturación por fila (banda central para evitar bordes)
    sat_prof = hsv[:, int(0.3 * rw):int(0.7 * rw), 1].mean(axis=1)

    rows = np.where(sat_prof > sat_thr)[0]
    if len(rows) == 0:
        raise RuntimeError("No se detectó ningún cuadro coloreado.")
    top = rows[0]                                    # inicio 1er cuadro
    below = np.where(sat_prof[top:] < sat_thr)[0]
    blk = below[0] if len(below) else rh // n        # alto del 1er cuadro
    after = np.where(sat_prof[top + blk:] > sat_thr)[0]
    top2 = top + blk + (after[0] if len(after) else 0)  # inicio 2do cuadro
    paso = (top2 - top) if top2 > top else rh / n

    c0 = top + blk / 2.0                             # centro del 1er cuadro
    cx = rw // 2
    centros = [(cx, int(round(c0 + i * paso))) for i in range(n)]
    return centros


# ---------------- Centros: referencia (dividir rango de color) ----------------
def centros_tira_ref(roi_rgb, n=N_CUADROS, sat_thr=SAT_THR_CUADRO):
    """
    En la referencia los cuadros están limpios y equiespaciados: se detecta
    el rango vertical con color y se divide en n segmentos iguales.
    """
    rh, rw = roi_rgb.shape[:2]
    hsv = cv2.cvtColor(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    sat_prof = hsv[:, int(0.3 * rw):int(0.7 * rw), 1].mean(axis=1)
    rows = np.where(sat_prof > sat_thr)[0]
    top, bottom = (rows[0], rows[-1]) if len(rows) else (0, rh - 1)

    paso = (bottom - top) / n
    cx = rw // 2
    centros = [(cx, int(round(top + (i + 0.5) * paso))) for i in range(n)]
    return centros


def dibujar_circulos(ax, centros, radio, color):
    for (cx, cy) in centros:
        ax.add_patch(plt.Circle((cx, cy), radio, fill=False,
                                 edgecolor=color, linewidth=2))


# ---------------- Preview ----------------
def main():
    nef_path, nombre = nef_aleatorio(NEF_DIR)
    muestra = leer_nef(nef_path)
    ref = leer_jpg(REF_PATH)

    roi_ref, bbox_ref = roi_referencia(ref)
    roi_nef, bbox_nef, tw = roi_tira_nef(muestra)

    centros_nef = centros_tira_nef(roi_nef)
    centros_ref = centros_tira_ref(roi_ref)

    r_nef = int(RADIO_FRAC * roi_nef.shape[1])
    r_ref = int(RADIO_FRAC * roi_ref.shape[1])

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # Arriba izq: NEF completa con límite del tercio/40% y bbox
    ax[0, 0].imshow(muestra)
    ax[0, 0].axvline(tw, color="cyan", linestyle="--", linewidth=1.2)
    xa, ya, xb, yb = bbox_nef
    ax[0, 0].add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya,
                                     fill=False, edgecolor="lime", linewidth=2))
    ax[0, 0].set_title(f"Muestra NEF: {nombre}")
    ax[0, 0].axis("off")

    # Arriba der: referencia con bbox
    xa, ya, xb, yb = bbox_ref
    ax[0, 1].imshow(ref)
    ax[0, 1].add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya,
                                     fill=False, edgecolor="red", linewidth=2))
    ax[0, 1].set_title("Referencia JPG")
    ax[0, 1].axis("off")

    # Abajo izq: ROI NEF con 10 círculos
    ax[1, 0].imshow(roi_nef)
    dibujar_circulos(ax[1, 0], centros_nef, r_nef, "red")
    ax[1, 0].set_title(f"ROI tira NEF · {N_CUADROS} cuadros")
    ax[1, 0].axis("off")

    # Abajo der: ROI referencia con 10 círculos
    ax[1, 1].imshow(roi_ref)
    dibujar_circulos(ax[1, 1], centros_ref, r_ref, "red")
    ax[1, 1].set_title(f"ROI referencia · {N_CUADROS} cuadros")
    ax[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()