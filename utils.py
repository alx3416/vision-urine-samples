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


# =====================================================================
#  Localización y segmentación de la tira reactiva en imágenes de MUESTRA
#  (escena real: tira vertical sobre soporte claro, sin separadores
#   blancos impresos entre cuadros).
# =====================================================================

def crop_grueso(img_bgr, frac):
    """Recorta la región de interés (tercio izquierdo) descartando el
    separador verde y el tubo. `frac` = (x0, x1, y0, y1) en fracciones."""
    h, w = img_bgr.shape[:2]
    x0, x1, y0, y1 = frac
    xa, xb = int(x0 * w), int(x1 * w)
    ya, yb = int(y0 * h), int(y1 * h)
    return img_bgr[ya:yb, xa:xb], (xa, ya)


def mascara_tira(crop_bgr, s_min):
    """Máscara binaria de la banda de la tira por saturación + cierre
    morfológico vertical (une cuadros, rellena los claros intermedios)."""
    S = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    m = (S > s_min).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (9, 201)))
    m = cv2.morphologyEx(m, cv2.MORPH_OPEN,
                         cv2.getStructuringElement(cv2.MORPH_RECT, (11, 11)))
    return m


def rect_tira(mascara, alto_crop):
    """minAreaRect de la componente conexa 'alta y angosta' (la tira)."""
    num, lab, st, _ = cv2.connectedComponentsWithStats(mascara, 8)
    best, best_score = None, -1
    for i in range(1, num):
        x, y, ww, hh, area = st[i]
        if hh < 0.25 * alto_crop or hh / max(ww, 1) < 3:
            continue
        score = hh * (hh / max(ww, 1))
        if score > best_score:
            best_score, best = score, i
    if best is None:
        return None
    comp = (lab == best).astype(np.uint8)
    cnts, _ = cv2.findContours(comp, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return cv2.minAreaRect(max(cnts, key=cv2.contourArea))


def _normaliza_rect(rect):
    """Asegura lado largo vertical y ángulo en (-45, 45]."""
    (cx, cy), (w, h), a = rect
    if w > h:
        w, h = h, w
        a += 90
    while a > 45:
        a -= 90
    while a < -45:
        a += 90
    return (cx, cy), (w, h), a


def enderezar_tira(crop_bgr, rect):
    """Rota el crop para dejar la tira vertical y recorta la banda.

    Devuelve (banda_bgr, M, offset_banda, ancho_rect) donde M es la matriz
    de rotación afín y offset_banda=(bx0, by0) es la esquina de la banda en
    la imagen rotada, para poder mapear coordenadas de vuelta al original.
    """
    (cx, cy), (w, h), a = _normaliza_rect(rect)
    M = cv2.getRotationMatrix2D((cx, cy), a, 1.0)
    ch, cw = crop_bgr.shape[:2]
    rot = cv2.warpAffine(crop_bgr, M, (cw, ch), borderValue=(255, 255, 255))
    pad = int(0.45 * w)
    # Padding vertical: el minAreaRect solo cubre la parte saturada; los
    # cuadros finales (NIT/LEU) suelen ser casi blancos y quedan fuera del
    # rect. Se extiende la banda ~35% arriba y abajo para no perderlos.
    pad_y = int(0.35 * h)
    bx0 = max(0, int(cx - w / 2) - pad)
    bx1 = min(cw, int(cx + w / 2) + pad)
    by0 = max(0, int(cy - h / 2) - pad_y)
    by1 = min(ch, int(cy + h / 2) + pad_y)
    banda = rot[by0:by1, bx0:bx1]
    return banda, M, (bx0, by0), w


def _eje_saturacion(S, y0, y1, s_min, grado=2):
    """Polinomio x(y) del eje central de la tira (centroide-x ponderado por
    saturación en la franja [y0, y1])."""
    bh, bw = S.shape
    ys, xs = [], []
    for y in range(max(0, y0), min(bh, y1)):
        row = S[y].astype(float)
        if row.max() < s_min:
            continue
        wgt = np.clip(row - s_min, 0, None)
        if wgt.sum() < 1:
            continue
        ys.append(y)
        xs.append((np.arange(bw) * wgt).sum() / wgt.sum())
    if len(ys) < 20:
        return None
    ys = np.array(ys, float)
    xs = np.array(xs, float)
    p = np.polyfit(ys, xs, grado)
    res = np.abs(xs - np.polyval(p, ys))
    keep = res < 2.5 * np.median(res) + 1
    if keep.sum() > grado + 2:
        p = np.polyfit(ys[keep], xs[keep], grado)
    return p


def _perfil_saturacion(S, coef, ancho):
    """Perfil vertical de saturación media en una ventana centrada en el eje."""
    bh, bw = S.shape
    half = int(0.30 * ancho)
    prof = np.zeros(bh)
    for y in range(bh):
        xc = int(np.polyval(coef, y))
        a = max(0, xc - half)
        b = min(bw, xc + half + 1)
        if b > a:
            prof[y] = S[y, a:b].mean()
    return np.convolve(prof, np.ones(9) / 9, mode="same")


def _paso_autocorr(prof, y0, y1, n):
    """Paso periódico entre cuadros por autocorrelación del perfil."""
    seg = prof[y0:y1] - prof[y0:y1].mean()
    ac = np.correlate(seg, seg, mode="full")[len(seg) - 1:]
    largo = y1 - y0
    lo = max(int(largo / n * 0.7), 5)
    hi = int(largo / n * 1.4)
    if hi <= lo or hi >= len(ac):
        return largo / n
    return lo + int(np.argmax(ac[lo:hi]))


def _rejilla_cuadros(prof, y0, y1, paso, n):
    """Ubica los n cuadros con paso fijo, anclando el cuadro 0 al PRIMER
    pico de saturación (primer cuadro real, aunque sea pálido).

    Anclar por el primer pico —no por la máxima energía global— evita que
    la rejilla se desplace hacia los cuadros más saturados del centro de la
    tira y deje fuera el primer cuadro (GLU), que a veces es pálido. El paso
    es físicamente constante, así que basta un ancla fiable arriba para
    proyectar los 10.
    """
    from scipy.signal import find_peaks

    # Picos de saturación = centros candidatos de cuadro. Se pide una
    # prominencia mínima ALTA para el ancla: el tramo blanco de la tira bajo
    # la pinza y los reflejos generan picos débiles (prominencia baja) que no
    # son cuadros; anclar en ellos corre la rejilla 1-2 posiciones y pierde
    # el primer cuadro real (GLU).
    prom_ancla = max(20, prof.max() * 0.30)
    pk_fuertes, _ = find_peaks(prof, distance=int(paso * 0.6),
                               prominence=prom_ancla)
    pk_fuertes = [p for p in pk_fuertes
                  if y0 - paso * 0.5 <= p <= y1 + paso * 0.5]

    # Picos con prominencia baja: solo para el refinado de fase (no ancla).
    pk, _ = find_peaks(prof, distance=int(paso * 0.6),
                       prominence=max(12, prof.max() * 0.10))
    pk = [p for p in pk if y0 - paso * 0.5 <= p <= y1 + paso * 0.5]

    # El paso viene por autocorrelación (físicamente constante ~190px); NO
    # se recalcula desde los picos, que pueden incluir espurios y comprimir
    # la rejilla.
    paso_f = float(paso)
    half = paso_f / 2.0

    # Ancla = primer pico FUERTE (primer cuadro real). Si hay un cuadro
    # pálido legítimo justo antes (a un paso, con saturación comparable a la
    # de los cuadros reales), se retrocede para incluirlo; el tramo
    # blanco/ruido bajo la pinza tiene saturación bastante menor, así que no
    # arrastra el ancla.
    if len(pk_fuertes) >= 1:
        b0 = float(pk_fuertes[0])
        # nivel de cuadro = mediana de saturación en los picos fuertes; un
        # cuadro pálido real está cerca de este nivel, el tramo blanco no.
        nivel = float(np.median([prof[p] for p in pk_fuertes]))
        umb_cuadro = 0.55 * nivel
        while b0 - paso_f >= y0:
            yp = int(round(b0 - paso_f))
            a = max(0, yp - int(paso_f * 0.2))
            b = min(len(prof), yp + int(paso_f * 0.2))
            if b > a and prof[a:b].mean() > umb_cuadro:
                b0 -= paso_f          # sí hay cuadro antes: incluirlo
            else:
                break                 # tramo blanco/ruido: parar
    elif len(pk) >= 1:
        b0 = float(pk[0])
    else:
        b0 = float(y0) + half

    # Refinar la fase alineando la rejilla a TODOS los picos detectados por
    # mínimos cuadrados de fase (sin desplazar de cuadro): para cada pico se
    # toma el índice de cuadro más cercano y se ajusta el offset común. Esto
    # respeta el ancla superior y no se sesga hacia los cuadros más
    # saturados del centro (que un criterio de energía sí favorecería).
    if len(pk) >= 3:
        offs = []
        for p in pk:
            i = round((p - b0) / paso_f)      # índice de cuadro para el pico
            if 0 <= i < n:
                offs.append(p - (b0 + paso_f * i))
        if offs:
            b0 = b0 + float(np.median(offs))

    centros = [int(round(b0 + paso_f * i)) for i in range(n)]
    bordes = [int(round(b0 - half + paso_f * i)) for i in range(n + 1)]
    return centros, bordes


def _recentrar_x(S, xc0, yc, r, win_frac=0.9):
    """Ajuste fino horizontal: xc que maximiza saturación media del disco."""
    bh, bw = S.shape
    win = int(r * win_frac)
    y0 = max(0, yc - r)
    y1 = min(bh, yc + r + 1)
    mejor, best = -1, xc0
    for xc in range(max(r, xc0 - win), min(bw - r, xc0 + win) + 1, 2):
        x0 = max(0, xc - r)
        x1 = min(bw, xc + r + 1)
        val = S[y0:y1, x0:x1].mean()
        if val > mejor:
            mejor, best = val, xc
    return best


def segmentar_cuadros(banda_bgr, s_min, n, radio_frac=0.34,
                      ancho_min=110, ancho_max=200):
    """Segmenta los n cuadros de la tira enderezada.

    Devuelve (centros, r) con centros = lista de (x, y) en coordenadas de la
    banda, en orden físico (arriba->abajo), o None si no se pudo segmentar.

    `ancho_min`/`ancho_max` acotan el ancho estimado de la tira (px) a un
    rango físico: en escenas de fondo cálido la máscara de saturación se
    desborda e infla el ancho (y con él el radio), lo que descuadra el
    arranque de la rejilla; el clamp lo evita.
    """
    S = cv2.cvtColor(banda_bgr, cv2.COLOR_BGR2HSV)[:, :, 1]
    bh, bw = S.shape

    # Ancho de la tira: mediana con umbral MÁS ALTO que el de la máscara,
    # para medir el núcleo saturado y no el fondo que la máscara sí incluye.
    s_ancho = s_min + 15
    anchos = []
    for row in S:
        on = np.where(row > s_ancho)[0]
        if len(on) > 3:
            anchos.append(on[-1] - on[0])
    ancho = float(np.median(anchos)) if anchos else (ancho_min + ancho_max) / 2
    ancho = float(np.clip(ancho, ancho_min, ancho_max))

    coef0 = _eje_saturacion(S, 0, bh, s_min)
    if coef0 is None:
        return None
    prof = _perfil_saturacion(S, coef0, ancho)
    umb = max(28, prof.max() * 0.28)
    idx = np.where(prof > umb)[0]
    if len(idx) < 10:
        return None
    y0, y1 = int(idx[0]), int(idx[-1])

    coef = _eje_saturacion(S, y0, y1, s_min)
    if coef is None:
        coef = coef0
    prof = _perfil_saturacion(S, coef, ancho)

    paso = _paso_autocorr(prof, y0, y1, n)
    centros_y, _bordes = _rejilla_cuadros(prof, y0, y1, paso, n)

    r = int(radio_frac * ancho)
    centros = []
    for yc in centros_y:
        yc = min(max(yc, 0), bh - 1)
        xc0 = int(np.polyval(coef, yc))
        xc = _recentrar_x(S, xc0, yc, r)
        centros.append((xc, yc))
    return centros, r


def mapear_a_original(centros_banda, M, offset_banda, offset_crop):
    """Mapea centros de la banda enderezada a coordenadas de la imagen
    ORIGINAL, deshaciendo: recorte de banda -> rotación -> crop grueso."""
    bx0, by0 = offset_banda
    cx0, cy0 = offset_crop
    Minv = cv2.invertAffineTransform(M)
    pts = []
    for (x, y) in centros_banda:
        # banda -> imagen rotada (coords del crop)
        xr, yr = x + bx0, y + by0
        # rotada -> crop original (deshacer rotación)
        xo = Minv[0, 0] * xr + Minv[0, 1] * yr + Minv[0, 2]
        yo = Minv[1, 0] * xr + Minv[1, 1] * yr + Minv[1, 2]
        # crop -> imagen original (deshacer crop grueso)
        pts.append((int(round(xo + cx0)), int(round(yo + cy0))))
    return pts
