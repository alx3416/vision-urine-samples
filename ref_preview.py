# ref_preview.py
# Exploración y preview de la tira reactiva de REFERENCIA (JPG).
# Detecta los 10 cuadros individualmente y dibuja un círculo concéntrico en cada uno.
import cv2
import numpy as np
import matplotlib.pyplot as plt

# ============ PARÁMETROS ============
REF_PATH = r"mission-reagent-parameter-pic-647x1024.jpg"

# Recorte fijo de la columna de la tira (fracciones del ancho/alto de la imagen)
REF_X0, REF_X1 = 0.09, 0.20
REF_Y0, REF_Y1 = 0.005, 0.86

N_CUADROS = 10          # número de cuadros esperados
RADIO_FRAC = 0.30       # radio del círculo como fracción del ancho de la ROI
SEP_L_MIN = 238         # L (LAB) mínimo para considerar una fila "separador blanco"
SEP_CHROMA_MAX = 7      # croma máximo para considerarla separador (cuadros pálidos > esto)
ALTO_MIN = 15           # alto mínimo de un bloque válido (px)
# ====================================


def leer_jpg(path):
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se encontró: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def roi_referencia(img_ref, x0=REF_X0, x1=REF_X1, y0=REF_Y0, y1=REF_Y1):
    """Recorta la columna de la tira de referencia por proporciones fijas."""
    h, w = img_ref.shape[:2]
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    return img_ref[ya:yb, xa:xb], (xa, ya, xb, yb)


def detectar_cuadros(roi_rgb, sep_L=SEP_L_MIN, sep_chroma=SEP_CHROMA_MAX,
                     alto_min=ALTO_MIN):
    """
    Detecta cada cuadro como un segmento vertical de filas que NO son
    separador blanco. Un separador = fila muy clara (L alto) y casi sin
    croma; así los cuadros pálidos (croma bajo pero no nulo) no se pierden.
    Devuelve la lista de centros (cx, cy).
    """
    rh, rw = roi_rgb.shape[:2]
    lab = cv2.cvtColor(cv2.cvtColor(roi_rgb, cv2.COLOR_RGB2BGR),
                       cv2.COLOR_BGR2LAB).astype(float)
    band = lab[:, int(0.3 * rw):int(0.7 * rw), :].mean(axis=1)
    L = band[:, 0]
    A = band[:, 1] - 128
    B = band[:, 2] - 128
    chroma = np.sqrt(A ** 2 + B ** 2)
    chroma = np.convolve(chroma, np.ones(3) / 3, mode="same")

    es_sep = (L > sep_L) & (chroma < sep_chroma)

    segmentos = []
    i = 0
    while i < rh:
        if not es_sep[i]:
            j = i
            while j < rh and not es_sep[j]:
                j += 1
            if j - i > alto_min:
                segmentos.append((i, j))
            i = j
        else:
            i += 1

    cx = rw // 2
    centros = [(cx, (a + b) // 2) for (a, b) in segmentos]
    return centros, segmentos


def main():
    ref = leer_jpg(REF_PATH)
    roi_ref, bbox_ref = roi_referencia(ref)
    centros, segmentos = detectar_cuadros(roi_ref)

    if len(centros) != N_CUADROS:
        print(f"AVISO: se detectaron {len(centros)} cuadros "
              f"(se esperaban {N_CUADROS}). Ajusta los umbrales SEP_*.")

    r = int(RADIO_FRAC * roi_ref.shape[1])

    fig, ax = plt.subplots(1, 2, figsize=(10, 8))

    # Izquierda: imagen completa con bbox de la tira
    xa, ya, xb, yb = bbox_ref
    ax[0].imshow(ref)
    ax[0].add_patch(plt.Rectangle((xa, ya), xb - xa, yb - ya,
                                  fill=False, edgecolor="red", linewidth=2))
    ax[0].set_title("Referencia JPG")
    ax[0].axis("off")

    # Derecha: ROI con un círculo concéntrico por cuadro detectado
    ax[1].imshow(roi_ref)
    for i, (cx, cy) in enumerate(centros):
        ax[1].add_patch(plt.Circle((cx, cy), r, fill=False,
                                    edgecolor="red", linewidth=2))
        ax[1].text(cx + r + 3, cy, str(i + 1), color="red",
                   va="center", fontsize=9)
    ax[1].set_title(f"ROI referencia · {len(centros)} cuadros detectados")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()