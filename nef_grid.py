# nef_grid.py
# Detección de los 10 cuadros en tiras reactivas de imágenes NEF y exportación
# de color (mediana RGB -> HSV, CIELAB) a output/nef_colores.csv.
# Modo lote (por defecto) o una imagen aleatoria. Guarda recortes ROI en PNG/SVG.
# Reutiliza utilidades de ref_grid.py
import os
import csv
import random
import rawpy
import cv2
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

from ref_grid import color_mediano   # misma lógica de muestreo/conversión

# ============ PARÁMETROS ============
NEF_DIR = r"E:\data\FOTOS ORINAS\101D3300"
OUTPUT_DIR = "output"
ROI_DIR = os.path.join(OUTPUT_DIR, "roi")
CSV_NAME = "nef_colores.csv"

PROCESAR_TODO = True    # True = todo el lote; False = una imagen aleatoria

REGION_FRAC = 0.40      # fracción izquierda donde buscar la tira
N_CUADROS = 10          # 1 cuadro por clase
RADIO_FRAC = 0.30       # radio como fracción del ANCHO del cuadro detectado
SAT_THR = 45            # umbral para segmentar la tira dentro de la región
SAT_THR_CUADRO = 50     # umbral para bordes de cuadros (perfil de saturación)

CLASES = ["LEU", "NIT", "URO", "PRO", "pH", "BLO", "SG", "KET", "BIL", "GLU"]
# ====================================


def leer_nef(path):
    with rawpy.imread(path) as raw:
        return raw.postprocess(use_camera_wb=True, output_bps=8,
                               no_auto_bright=False)


def listar_nef(directorio):
    nefs = sorted(f for f in os.listdir(directorio)
                  if f.lower().endswith(".nef"))
    if not nefs:
        raise FileNotFoundError(f"No hay .NEF en {directorio}")
    return nefs


def roi_tira_nef(img_rgb, region_frac=REGION_FRAC, sat_thr=SAT_THR,
                 min_aspect=3.0, area_frac_min=0.005,
                 margin_x=0.25, margin_y=0.02):
    """Detecta la ROI de la tira en el 40% izquierdo de la imagen."""
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


def detectar_cuadros_nef(roi_rgb, n=N_CUADROS, sat_thr=SAT_THR_CUADRO):
    """
    Ancla en el cuadro superior (buen contraste en NEF), mide su alto/ancho y
    el paso vertical al 2º cuadro, y replica el paso para los n cuadros.
    Devuelve: centros (cx, cy) en coords de la ROI, y ancho del cuadro superior.
    """
    rh, rw = roi_rgb.shape[:2]
    hsv = cv2.cvtColor(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    sat = hsv[:, int(0.3 * rw):int(0.7 * rw), 1].mean(axis=1)

    rows = np.where(sat > sat_thr)[0]
    if len(rows) == 0:
        raise RuntimeError("No se detectó ningún cuadro coloreado en la NEF.")
    top = rows[0]
    below = np.where(sat[top:] < sat_thr)[0]
    blk = below[0] if len(below) else rh // n
    after = np.where(sat[top + blk:] > sat_thr)[0]
    top2 = top + blk + (after[0] if len(after) else 0)
    paso = (top2 - top) if top2 > top else rh / n

    banda = hsv[top:top + blk, :, 1]
    col_sat = banda.mean(axis=0)
    cols = np.where(col_sat > sat_thr)[0]
    if len(cols) >= 2:
        ancho_cuadro = cols[-1] - cols[0]
        cx = (cols[0] + cols[-1]) // 2
    else:
        ancho_cuadro = rw
        cx = rw // 2

    c0 = top + blk / 2.0
    centros = [(cx, int(round(c0 + i * paso))) for i in range(n)]
    return centros, int(ancho_cuadro)


def guardar_roi_figura(roi_rgb, centros_roi, r, nombre_base):
    """Guarda el recorte ROI con círculos y etiquetas en PNG y SVG."""
    rh, rw = roi_rgb.shape[:2]
    fig, ax = plt.subplots(figsize=(rw / 100 * 2, rh / 100 * 2))
    ax.imshow(roi_rgb)
    for clase, (cx, cy) in zip(CLASES, centros_roi):
        ax.add_patch(plt.Circle((cx, cy), r, fill=False,
                                edgecolor="red", linewidth=2))
        ax.text(cx + r + 5, cy, clase, color="red",
                va="center", fontsize=8)
    ax.axis("off")
    plt.tight_layout()
    png = os.path.join(ROI_DIR, f"{nombre_base}_ROI.png")
    svg = os.path.join(ROI_DIR, f"{nombre_base}_ROI.svg")
    fig.savefig(png, dpi=100, bbox_inches="tight")
    fig.savefig(svg, bbox_inches="tight")
    plt.close(fig)
    return png, svg


def procesar_imagen(nef_path, nombre, writer):
    """Procesa una NEF: detecta, escribe filas al CSV, guarda ROI PNG/SVG."""
    img = leer_nef(nef_path)
    roi, bbox, tw = roi_tira_nef(img)
    centros_roi, ancho_cuadro = detectar_cuadros_nef(roi)
    r = int(RADIO_FRAC * ancho_cuadro)

    x0, y0 = bbox[0], bbox[1]
    centros_abs = [(cx + x0, cy + y0) for (cx, cy) in centros_roi]

    for clase, (cx, cy) in zip(CLASES, centros_abs):
        col = color_mediano(img, cx, cy, r)
        writer.writerow({
            "imagen": nombre, "clase": clase,
            "R": col["R"], "G": col["G"], "B": col["B"],
            "H": col["H"], "S": col["S"], "V": col["V"],
            "L": col["L"], "a": col["a"], "b": col["b"],
            "cx": cx, "cy": cy, "n_pixeles": col["n"],
        })

    nombre_base = os.path.splitext(nombre)[0]
    guardar_roi_figura(roi, centros_roi, r, nombre_base)
    return r, ancho_cuadro


CAMPOS = ["imagen", "clase", "R", "G", "B", "H", "S", "V",
          "L", "a", "b", "cx", "cy", "n_pixeles"]


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROI_DIR, exist_ok=True)
    ruta_csv = os.path.join(OUTPUT_DIR, CSV_NAME)

    if PROCESAR_TODO:
        matplotlib.use("Agg")   # sin ventanas en modo lote
        nefs = listar_nef(NEF_DIR)
        print(f"Procesando {len(nefs)} imágenes...")
        # reescribir CSV desde cero
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CAMPOS)
            wr.writeheader()
            for k, nombre in enumerate(nefs, 1):
                path = os.path.join(NEF_DIR, nombre)
                try:
                    r, ancho = procesar_imagen(path, nombre, wr)
                    print(f"  [{k}/{len(nefs)}] {nombre} · r={r} ancho={ancho}")
                except Exception as e:
                    print(f"  [{k}/{len(nefs)}] {nombre} · ERROR: {e}")
        print(f"CSV: {ruta_csv} · ROIs en: {ROI_DIR}")
    else:
        # una imagen aleatoria, con preview interactivo
        nombre = random.choice(listar_nef(NEF_DIR))
        path = os.path.join(NEF_DIR, nombre)
        print(f"Imagen: {nombre}")
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CAMPOS)
            wr.writeheader()
            r, ancho = procesar_imagen(path, nombre, wr)

        img = leer_nef(path)
        roi, bbox, tw = roi_tira_nef(img)
        centros_roi, _ = detectar_cuadros_nef(roi)
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(img)
        ax[0].axvline(tw, color="cyan", linestyle="--", linewidth=1)
        xa, ya, xb, yb = bbox
        ax[0].add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya,
                                      fill=False, edgecolor="lime", linewidth=2))
        ax[0].set_title(f"NEF: {nombre}")
        ax[0].axis("off")
        ax[1].imshow(roi)
        for clase, (cx, cy) in zip(CLASES, centros_roi):
            ax[1].add_patch(plt.Circle((cx, cy), r, fill=False,
                                       edgecolor="red", linewidth=2))
            ax[1].text(cx + r + 5, cy, clase, color="red",
                       va="center", fontsize=8)
        ax[1].set_title(f"ROI tira · {N_CUADROS} cuadros")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
        print(f"CSV: {ruta_csv} · ROIs en: {ROI_DIR}")


if __name__ == "__main__":
    main()