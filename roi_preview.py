# roi_preview.py
import os
import random
import rawpy
import cv2
import numpy as np
import matplotlib.pyplot as plt

NEF_DIR = r"E:\data\FOTOS ORINAS\101D3300"
REF_PATH = r"mission-reagent-parameter-pic-647x1024.jpg"


# ---------------- Lectura ----------------
def leer_nef(path):
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=8,
                              no_auto_bright=False)
    return rgb


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


# ---------------- ROI referencia (fija por proporciones) ----------------
def roi_referencia(img_ref, x0=0.09, x1=0.20, y0=0.005, y1=0.86):
    h, w = img_ref.shape[:2]
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    bbox = (xa, ya, xb, yb)
    return img_ref[ya:yb, xa:xb], bbox


# ---------------- ROI tira NEF (solo tercio izquierdo) ----------------
def roi_tira_nef(img_rgb, region_frac=1/3, sat_thr=45,
                 min_aspect=3.0, area_frac_min=0.005,
                 margin_x=0.25, margin_y=0.02):
    """
    Detecta la tira reactiva dentro del tercio izquierdo de la imagen.
    La tira son cuadros de colores (saturados) sobre soporte blanco.
    """
    H, W = img_rgb.shape[:2]
    tw = int(region_frac * W)
    region = img_rgb[:, :tw]

    bgr = cv2.cvtColor(region, cv2.COLOR_RGB2BGR)
    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    S = hsv[:, :, 1]

    mask = (S > sat_thr).astype(np.uint8) * 255
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), iterations=1)
    # cierre vertical fuerte: une los cuadros en una sola columna continua
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (9, 120)), iterations=2)

    cnts, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                               cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        raise RuntimeError("No se detectó la tira en el tercio izquierdo.")

    best = None
    min_area = area_frac_min * region.shape[0] * region.shape[1]
    for c in cnts:
        x, y, w, h = cv2.boundingRect(c)
        area = cv2.contourArea(c)
        if area < min_area:
            continue
        aspect = h / max(w, 1)
        if aspect >= min_aspect and (best is None or area > best[-1]):
            best = (x, y, w, h, area)

    if best is None:
        raise RuntimeError("No se encontró un objeto vertical tipo tira.")

    x, y, w, h, _ = best
    mx, my = int(margin_x * w), int(margin_y * h)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2, y2 = min(tw, x + w + mx), min(H, y + h + my)

    bbox = (x1, y1, x2, y2)             # coords en la imagen completa
    return img_rgb[y1:y2, x1:x2], bbox, tw


# ---------------- Preview ----------------
def main():
    nef_path, nombre = nef_aleatorio(NEF_DIR)
    muestra = leer_nef(nef_path)
    ref = leer_jpg(REF_PATH)

    roi_ref, bbox_ref = roi_referencia(ref)
    roi_nef, bbox_nef, tw = roi_tira_nef(muestra)

    fig, ax = plt.subplots(2, 2, figsize=(13, 10))

    # Arriba izq: NEF completa con límite del tercio y bbox de la tira
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

    # Abajo: recortes
    ax[1, 0].imshow(roi_nef)
    ax[1, 0].set_title("ROI tira reactiva (NEF)")
    ax[1, 0].axis("off")

    ax[1, 1].imshow(roi_ref)
    ax[1, 1].set_title("ROI referencia")
    ax[1, 1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()