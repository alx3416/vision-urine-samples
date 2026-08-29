#!/usr/bin/env python3
"""Convierte todas las imágenes NEF (RAW Nikon) de un directorio a PNG."""

from pathlib import Path

import rawpy
import imageio.v3 as iio

# --- Configuración ---
PATH_ENTRADA = r"data\101D3300"      # directorio con los archivos NEF
PATH_SALIDA = r"data\101D3300-PNG"     # directorio de salida
BITS = 8                            # 8 o 16
SOBRESCRIBIR = False               # True para sobrescribir PNG existentes


def leer_nef(path, bps=8):
    """Decodifica un archivo NEF (RAW Nikon) a RGB uint8 o uint16."""
    with rawpy.imread(str(path)) as raw:
        rgb = raw.postprocess(
            use_camera_wb=True,      # balance de blancos de la cámara (color fiel)
            output_bps=bps,
            no_auto_bright=False,
        )
    return rgb


def convertir():
    entrada = Path(PATH_ENTRADA)
    salida = Path(PATH_SALIDA)
    salida.mkdir(parents=True, exist_ok=True)

    nefs = sorted(p for p in entrada.iterdir() if p.suffix.lower() == ".nef")
    if not nefs:
        print(f"No se encontraron archivos NEF en: {entrada}")
        return

    for nef in nefs:
        destino = salida / f"{nef.stem}.png"
        if destino.exists() and not SOBRESCRIBIR:
            print(f"Omitido (ya existe): {destino.name}")
            continue
        try:
            rgb = leer_nef(nef, bps=BITS)
            iio.imwrite(destino, rgb)
            print(f"OK: {nef.name} -> {destino.name} ({BITS} bits)")
        except Exception as e:
            print(f"ERROR: {nef.name}: {e}")


if __name__ == "__main__":
    convertir()