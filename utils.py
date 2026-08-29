# utils.py
# Funciones reusables entre los scripts de extracción de color de tiras reactivas.

import os
import cv2
import numpy as np


# ---------- Lectura de imágenes ----------

def leer_jpg(path):
    """Lee una imagen (JPG/PNG) en disco y la devuelve en RGB uint8."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se encontró: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def leer_nef(path):
    """Decodifica un archivo NEF (RAW Nikon) a RGB uint8.

    Se conserva por compatibilidad; el flujo actual usa solo PNG/JPG.
    Requiere el paquete rawpy.
    """
    import rawpy
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,
            output_bps=8,
            no_auto_bright=False,
        )
    return rgb


# ---------- Segmentación 1D por separadores (blanco acromático) ----------

def segmentos_1d(perfil_lab, longitud, sep_L, sep_chroma, minimo):
    """Devuelve segmentos (inicio, fin) de un perfil LAB 1D.

    Un separador es un pixel claro y poco cromático (fondo blanco entre
    cuadros). Los segmentos son tramos consecutivos de NO-separador más
    largos que `minimo`.
    """
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


# ---------- Color mediano en una región circular ----------

def color_mediano(img_rgb, cx, cy, r):
    """Mediana RGB de los píxeles dentro del círculo (cx, cy, r) y sus
    conversiones a HSV y CIELAB (OpenCV)."""
    h, w = img_rgb.shape[:2]
    x1, x2 = max(0, cx - r), min(w, cx + r + 1)
    y1, y2 = max(0, cy - r), min(h, cy + r + 1)
    parche = img_rgb[y1:y2, x1:x2]

    yy, xx = np.ogrid[y1:y2, x1:x2]
    mascara = (xx - cx) ** 2 + (yy - cy) ** 2 <= r ** 2
    pix = parche[mascara]  # (N, 3) RGB
    n = len(pix)
    if n == 0:
        raise RuntimeError(f"Círculo vacío en ({cx},{cy}).")

    R, G, B = [int(round(np.median(pix[:, c]))) for c in range(3)]

    px_rgb = np.uint8([[[R, G, B]]])
    px_bgr = cv2.cvtColor(px_rgb, cv2.COLOR_RGB2BGR)
    H, S, V = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2HSV)[0, 0].tolist()
    L, a, b = cv2.cvtColor(px_bgr, cv2.COLOR_BGR2LAB)[0, 0].tolist()

    return dict(R=R, G=G, B=B, H=H, S=S, V=V, L=L, a=a, b=b, n=n)


# ---------- Dibujo y guardado del resultado de detección ----------

def guardar_deteccion_png(img_rgb, cuadricula, r, ruta_png,
                          bbox=None, etiquetas_fila=True):
    """Dibuja con OpenCV los círculos de detección sobre la imagen y guarda
    un PNG para referencia/documentación del proceso.

    - `cuadricula`: dict {(clase, indice): (cx, cy)}
    - `bbox`: (xa, ya, xb, yb) opcional, se dibuja como rectángulo guía.
    - `etiquetas_fila`: si True, escribe la clase junto al círculo índice 0.
    """
    os.makedirs(os.path.dirname(ruta_png) or ".", exist_ok=True)
    vis = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR).copy()

    if bbox is not None:
        xa, ya, xb, yb = bbox
        cv2.rectangle(vis, (xa, ya), (xb, yb), (0, 255, 0), 1, cv2.LINE_AA)

    for (clase, idx), (cx, cy) in cuadricula.items():
        cv2.circle(vis, (cx, cy), r, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(vis, str(idx), (cx - 4, cy + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 0, 0), 1, cv2.LINE_AA)

    if etiquetas_fila:
        clases_vistas = {}
        for (clase, idx), (cx, cy) in cuadricula.items():
            if idx == 0:
                clases_vistas[clase] = (cx, cy)
        x_texto = bbox[0] if bbox is not None else 5
        for clase, (cx, cy) in clases_vistas.items():
            cv2.putText(vis, clase, (max(0, x_texto - 45), cy + 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    cv2.imwrite(ruta_png, vis)
    return ruta_png
