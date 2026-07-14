# paso1_lectura.py
import os
import random
import rawpy
import cv2
import matplotlib.pyplot as plt

NEF_DIR = r"E:\data\FOTOS ORINAS\101D3300"
REF_PATH = r"mission-reagent-parameter-pic-647x1024.jpg"


def leer_nef(path):
    """Decodifica un archivo NEF (RAW Nikon) a RGB uint8."""
    with rawpy.imread(path) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,      # balance de blancos de la cámara (color fiel)
            output_bps=8,
            no_auto_bright=False,
        )
    return rgb  # ya está en RGB


def leer_jpg(path):
    """Lee la referencia JPG y la devuelve en RGB."""
    bgr = cv2.imread(path, cv2.IMREAD_COLOR)
    if bgr is None:
        raise FileNotFoundError(f"No se encontró: {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def nef_aleatorio(directorio):
    nefs = [f for f in os.listdir(directorio) if f.lower().endswith(".nef")]
    if not nefs:
        raise FileNotFoundError(f"No hay archivos .NEF en {directorio}")
    elegido = random.choice(nefs)
    return os.path.join(directorio, elegido), elegido


def main():
    nef_path, nef_name = nef_aleatorio(NEF_DIR)
    img_muestra = leer_nef(nef_path)
    img_ref = leer_jpg(REF_PATH)

    fig, axes = plt.subplots(1, 2, figsize=(14, 8))
    axes[0].imshow(img_muestra)
    axes[0].set_title(f"Muestra NEF: {nef_name}")
    axes[0].axis("off")

    axes[1].imshow(img_ref)
    axes[1].set_title("Referencia (JPG)")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()