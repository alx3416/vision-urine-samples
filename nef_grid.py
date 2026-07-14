# nef_grid.py
# Detección de los 10 cuadros en tiras reactivas de imágenes NEF y exportación
# de color (mediana RGB -> HSV, CIELAB) a output/nef_colores.csv.
# Modo lote (por defecto) o una imagen aleatoria. Guarda recortes ROI en PNG/SVG.
# Correcciones: deskew (inclinación), paso por autocorrelación (robusto ante
# cuadros pálidos finales) y radio fijo constante en píxeles.
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
RADIO_PX = 60           # radio FIJO de los círculos, en píxeles (igual en todas)
SAT_THR = 45            # umbral para segmentar la tira dentro de la región
SAT_THR_CUADRO = 50     # umbral para el primer cuadro (ancla)
SUAVIZADO = 15          # ventana de suavizado del perfil de saturación

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


def detectar_tira(img_rgb, region_frac=REGION_FRAC, sat_thr=SAT_THR,
                  min_aspect=3.0, area_frac_min=0.005):
    """
    Detecta la tira en el 40% izquierdo. Devuelve la región, el bounding box
    del contorno y la desviación angular respecto a la vertical.
    """
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
            best = (x, y, w, h, cv2.contourArea(c), c)
    if best is None:
        raise RuntimeError("No se encontró un objeto vertical tipo tira.")
    x, y, w, h, _, c = best

    rect = cv2.minAreaRect(c)
    (rw_r, rh_r), ang = rect[1], rect[2]
    desv = ang if rw_r < rh_r else (ang + 90 if ang < 0 else ang - 90)
    if desv > 45:
        desv -= 90
    if desv < -45:
        desv += 90
    return region, (x, y, w, h), desv


def enderezar(region, bbox, desv, margin_x=0.25, margin_y=0.03):
    """Rota la región para dejar la tira vertical y recorta la ROI con márgenes."""
    x, y, w, h = bbox
    cx0, cy0 = x + w / 2.0, y + h / 2.0
    M = cv2.getRotationMatrix2D((cx0, cy0), desv, 1.0)
    rot = cv2.warpAffine(region, M, (region.shape[1], region.shape[0]),
                         flags=cv2.INTER_LINEAR, borderValue=(255, 255, 255))
    mx, my = int(margin_x * w), int(margin_y * h)
    x1, y1 = max(0, x - mx), max(0, y - my)
    x2 = min(region.shape[1], x + w + mx)
    y2 = min(region.shape[0], y + h + my)
    return rot[y1:y2, x1:x2]


def _paso_por_autocorrelacion(perfil, rh, n=N_CUADROS):
    """
    Estima el paso (periodo) entre cuadros por autocorrelación del perfil de
    saturación. Robusto: no depende de detectar los cuadros pálidos finales.
    Busca el pico de autocorrelación en un rango plausible alrededor de rh/n.
    """
    x = perfil - perfil.mean()
    ac = np.correlate(x, x, mode="full")[len(x) - 1:]
    lo, hi = int(rh / (n + 4)), int(rh / (n - 2))
    lo = max(lo, 1)
    hi = min(hi, len(ac) - 1)
    if hi <= lo:
        return rh // n
    return lo + int(np.argmax(ac[lo:hi]))


def detectar_cuadros_nef(roi_rgb, n=N_CUADROS, sat_thr=SAT_THR_CUADRO,
                         k=SUAVIZADO):
    """
    Ancla en el primer cuadro (siempre contrastante) y replica el paso estimado
    por autocorrelación. Devuelve los centros (cx, cy) en coords de la ROI.
    """
    rh, rw = roi_rgb.shape[:2]
    hsv = cv2.cvtColor(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2HSV)
    sat = hsv[:, int(0.3 * rw):int(0.7 * rw), 1].mean(axis=1).astype(float)
    sat_s = np.convolve(sat, np.ones(k) / k, mode="same")

    rows = np.where(sat_s > sat_thr)[0]
    if len(rows) == 0:
        raise RuntimeError("No se detectó ningún cuadro coloreado en la NEF.")
    top = rows[0]                                   # borde sup. 1er cuadro

    paso = _paso_por_autocorrelacion(sat_s, rh, n)
    c0 = top + paso / 2.0                            # centro del 1er cuadro

    # centro horizontal robusto: mediana de centros de color por fila
    colmask = hsv[:, :, 1] > sat_thr
    cxs = []
    lim = int(min(rh, top + paso * n))
    for yy in range(int(top), lim, max(1, paso // 3)):
        cc = np.where(colmask[yy])[0]
        if len(cc) >= 2:
            cxs.append((cc[0] + cc[-1]) // 2)
    cx = int(np.median(cxs)) if cxs else rw // 2

    centros = [(cx, int(round(c0 + i * paso))) for i in range(n)]
    return centros, paso


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


CAMPOS = ["imagen", "clase", "R", "G", "B", "H", "S", "V",
          "L", "a", "b", "cx", "cy", "n_pixeles"]


def procesar_imagen(nef_path, nombre, writer, r=RADIO_PX):
    """Procesa una NEF: deskew, detecta, escribe CSV, guarda ROI PNG/SVG."""
    img = leer_nef(nef_path)
    region, bbox, desv = detectar_tira(img)
    roi = enderezar(region, bbox, desv)
    centros_roi, paso = detectar_cuadros_nef(roi)

    for clase, (cx, cy) in zip(CLASES, centros_roi):
        col = color_mediano(roi, cx, cy, r)
        writer.writerow({
            "imagen": nombre, "clase": clase,
            "R": col["R"], "G": col["G"], "B": col["B"],
            "H": col["H"], "S": col["S"], "V": col["V"],
            "L": col["L"], "a": col["a"], "b": col["b"],
            "cx": cx, "cy": cy, "n_pixeles": col["n"],
        })

    nombre_base = os.path.splitext(nombre)[0]
    guardar_roi_figura(roi, centros_roi, r, nombre_base)
    return paso, desv


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    os.makedirs(ROI_DIR, exist_ok=True)
    ruta_csv = os.path.join(OUTPUT_DIR, CSV_NAME)

    if PROCESAR_TODO:
        matplotlib.use("Agg")
        nefs = listar_nef(NEF_DIR)
        print(f"Procesando {len(nefs)} imágenes...")
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CAMPOS)
            wr.writeheader()
            for k, nombre in enumerate(nefs, 1):
                path = os.path.join(NEF_DIR, nombre)
                try:
                    paso, desv = procesar_imagen(path, nombre, wr)
                    print(f"  [{k}/{len(nefs)}] {nombre} · paso={paso} "
                          f"desv={desv:+.2f}° r={RADIO_PX}")
                except Exception as e:
                    print(f"  [{k}/{len(nefs)}] {nombre} · ERROR: {e}")
        print(f"CSV: {ruta_csv} · ROIs en: {ROI_DIR}")
    else:
        nombre = random.choice(listar_nef(NEF_DIR))
        path = os.path.join(NEF_DIR, nombre)
        print(f"Imagen: {nombre}")
        with open(ruta_csv, "w", newline="", encoding="utf-8") as f:
            wr = csv.DictWriter(f, fieldnames=CAMPOS)
            wr.writeheader()
            paso, desv = procesar_imagen(path, nombre, wr)
        print(f"paso {paso}px · desviación {desv:+.2f}° · radio {RADIO_PX}px")

        img = leer_nef(path)
        region, bbox, d2 = detectar_tira(img)
        roi = enderezar(region, bbox, d2)
        centros_roi, _ = detectar_cuadros_nef(roi)
        fig, ax = plt.subplots(1, 2, figsize=(12, 8))
        ax[0].imshow(img)
        ax[0].set_title(f"NEF: {nombre}")
        ax[0].axis("off")
        ax[1].imshow(roi)
        for clase, (cx, cy) in zip(CLASES, centros_roi):
            ax[1].add_patch(plt.Circle((cx, cy), RADIO_PX, fill=False,
                                       edgecolor="red", linewidth=2))
            ax[1].text(cx + RADIO_PX + 5, cy, clase, color="red",
                       va="center", fontsize=8)
        ax[1].set_title(f"ROI enderezada · {N_CUADROS} cuadros")
        ax[1].axis("off")
        plt.tight_layout()
        plt.show()
        print(f"CSV: {ruta_csv} · ROIs en: {ROI_DIR}")


if __name__ == "__main__":
    main()