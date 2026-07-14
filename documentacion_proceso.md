# Extracción de color en tiras reactivas de urianálisis

## Documentación técnica del proceso

Este documento resume el pipeline desarrollado para la extracción semiautomática y estable de color desde tiras reactivas de urianálisis, tanto de la carta de referencia (formato JPG) como de las capturas fotográficas de las muestras (formato RAW Nikon, `.NEF`). El objetivo final del proyecto es asistir al especialista en la comparación entre el color observado en cada almohadilla reactiva y los colores patrón de la carta de referencia. La presente etapa cubre la lectura de imágenes, la detección de las regiones de interés (ROI), la localización de cada almohadilla y la extracción robusta de su color representativo en tres espacios de color.

---

## 1. Lectura de imágenes

### 1.1 Referencia JPG

La carta de referencia es una imagen fija de dimensiones $647 \times 1024$ píxeles. Se lee en formato BGR y se convierte al espacio RGB para su procesamiento uniforme con el resto del pipeline.

### 1.2 Muestras NEF (RAW)

Los archivos `.NEF` son imágenes RAW del sensor de la cámara, sin procesamiento de color aplicado. Se decodifican mediante *demosaicing* con balance de blancos de la propia cámara, lo que preserva la fidelidad cromática necesaria para el análisis. Formalmente, el sensor entrega un mosaico de Bayer $M(x,y)$ del que se reconstruye la imagen tricromática:

$$
I_{\text{RGB}}(x,y) = \Phi_{\text{demosaic}}\big(M(x,y),\, \mathbf{w}_{\text{cam}}\big)
$$

donde $\mathbf{w}_{\text{cam}} = (w_R, w_G, w_B)$ es el vector de balance de blancos de la cámara y $\Phi_{\text{demosaic}}$ el operador de interpolación cromática. La salida se cuantiza a 8 bits por canal, de modo que cada componente queda acotada en $[0, 255]$.

---

## 2. Detección de la región de interés (ROI)

### 2.1 Referencia: recorte fijo

Dado que la carta de referencia es invariante, la columna patrón (columna $0$) se recorta mediante fracciones fijas del ancho $W$ y alto $H$ de la imagen:

$$
\text{ROI}_{\text{ref}} = I\big[\,y_0 H : y_1 H,\; x_0 W : x_1 W\,\big]
$$

con $(x_0, x_1) = (0.09,\, 0.20)$ y $(y_0, y_1) = (0.005,\, 0.86)$.

### 2.2 Muestra NEF: detección por saturación

La tira reactiva se localiza dentro del $40\%$ izquierdo de la imagen (parámetro `REGION_FRAC`), región controlada donde la tira aparece de forma consistente. Esto evita procesar la barra verde separadora y el resto de la escena.

Sobre esa región se calcula la máscara binaria a partir del canal de saturación $S$ del espacio HSV:

$$
B(x,y) =
\begin{cases}
1 & \text{si } S(x,y) > \tau_S \\
0 & \text{en otro caso}
\end{cases}
\qquad \tau_S = 45
$$

La tira, compuesta por almohadillas cromáticas, contrasta con el soporte blanco de baja saturación. La máscara se depura con operaciones morfológicas de apertura y cierre:

$$
B' = \big( (B \circ K_{\text{o}}) \bullet K_{\text{c}} \big)
$$

donde $\circ$ y $\bullet$ denotan apertura y cierre morfológico, $K_{\text{o}}$ es un elemento estructurante cuadrado ($7 \times 7$) que elimina ruido, y $K_{\text{c}}$ es un elemento rectangular vertical ($9 \times 120$) que fusiona las almohadillas separadas por espacios blancos en una sola columna continua.

Entre los contornos resultantes se selecciona la tira como el de mayor área que satisface una relación de aspecto vertical:

$$
\text{tira} = \arg\max_{c \in \mathcal{C}} \; \text{área}(c)
\quad \text{sujeto a} \quad \frac{h_c}{w_c} \ge 3
$$

Finalmente se añade un margen proporcional $(m_x, m_y) = (0.25\, w,\ 0.02\, h)$ alrededor del *bounding box* para asegurar la inclusión completa de la tira.

---

## 3. Localización de las almohadillas (cuadros)

### 3.1 Referencia: detección por bloques

En la carta de referencia las almohadillas están limpias y separadas por espacios blancos. Se calcula un perfil de color por fila promediando la banda central de la ROI en el espacio CIELAB:

$$
\bar{\mathbf{p}}(y) = \frac{1}{|\mathcal{X}|}\sum_{x \in \mathcal{X}} \text{Lab}(x,y),
\qquad \mathcal{X} = \big[\,0.3\,w_r,\ 0.7\,w_r\,\big]
$$

Se define el **croma** de cada fila a partir de las componentes cromáticas $a^\*, b^\*$ (centradas en $128$ en la convención de 8 bits de OpenCV):

$$
C(y) = \sqrt{\big(a(y) - 128\big)^2 + \big(b(y) - 128\big)^2}
$$

Una fila se clasifica como **separador blanco** cuando es simultáneamente muy luminosa y prácticamente acromática:

$$
\text{sep}(y) = \big[\,L(y) > 238\,\big] \;\wedge\; \big[\,C(y) < 7\,\big]
$$

Este doble criterio es clave: distingue las almohadillas pálidas (croma bajo pero no nulo, $C \approx 11$) de los espacios blancos reales ($C \approx 0$), evitando fusionar o perder cuadros claros. Los segmentos verticales continuos de filas *no separadoras* con longitud mayor a un mínimo definen cada almohadilla, y su centro es:

$$
c_i^y = \left\lfloor \frac{y_i^{\text{ini}} + y_i^{\text{fin}}}{2} \right\rfloor
$$

### 3.2 Referencia: cuadrícula 2D

La carta contiene múltiples columnas por fila, con nombres y número de almohadillas específicos por clase:

| Clase | Almohadillas | Clase | Almohadillas |
|-------|:---:|-------|:---:|
| LEU | 5 | BLO | 7 |
| NIT | 2 | SG | 7 |
| URO | 6 | KET | 6 |
| PRO | 6 | BIL | 4 |
| pH | 7 | GLU | 6 |

El espaciado horizontal es uniforme. El paso horizontal $\Delta x$ se mide automáticamente detectando las almohadillas a lo largo de la fila con mayor número de columnas (pH o BLO) y promediando las distancias entre centros consecutivos:

$$
\Delta x = \frac{1}{N-1} \sum_{j=1}^{N-1} \left( c_{j}^x - c_{j-1}^x \right)
$$

El centro de la almohadilla en la fila de clase $k$ e índice de columna $j$ se obtiene por replicación:

$$
\big(c_{k,j}^x,\ c_{k,j}^y\big) = \big(x_{\text{ini}} + j\,\Delta x,\ \; c_k^y\big),
\qquad j = 0, 1, \dots, n_k - 1
$$

Cada almohadilla se etiqueta como `CLASE,índice` (p. ej. `LEU,0`, `pH,6`).

### 3.3 Muestra NEF: anclaje y replicación vertical

En las capturas NEF, las dos últimas almohadillas (BIL, GLU) presentan colores muy pálidos que se confunden con el soporte blanco, lo que impide detectarlas directamente por bloques. La estrategia robusta consiste en anclar en la **primera almohadilla** —que siempre presenta buen contraste (típicamente LEU, de tono cian)— y replicar el paso vertical.

Se calcula el perfil de saturación por fila de la ROI:

$$
\bar{S}(y) = \frac{1}{|\mathcal{X}|}\sum_{x \in \mathcal{X}} S(x,y),
\qquad \mathcal{X} = \big[\,0.3\,w_r,\ 0.7\,w_r\,\big]
$$

El borde superior de la primera almohadilla es la primera fila que supera el umbral $\tau_C = 50$:

$$
y_{\text{top}} = \min \{\, y : \bar{S}(y) > \tau_C \,\}
$$

Sea $\ell$ el alto de la primera almohadilla (extensión hasta que $\bar{S}$ vuelve a caer bajo el umbral) y $y_{\text{top2}}$ el inicio de la segunda. El paso vertical es:

$$
\Delta y = y_{\text{top2}} - y_{\text{top}}
$$

y los centros de las $N = 10$ almohadillas se generan por replicación desde el centro de la primera:

$$
c_i^y = \left\lfloor \Big( y_{\text{top}} + \tfrac{\ell}{2} \Big) + i\,\Delta y \right\rceil,
\qquad i = 0, 1, \dots, 9
$$

Las almohadillas se etiquetan con las 10 clases en orden vertical: LEU, NIT, URO, PRO, pH, BLO, SG, KET, BIL, GLU.

---

## 4. Definición del área de muestreo circular

Sobre cada almohadilla se inscribe un círculo concéntrico para restringir el muestreo al color principal, evitando bordes y transiciones. El radio se determina de forma proporcional al ancho de la almohadilla detectada:

$$
r = \big\lfloor \rho \cdot w_{\text{cuadro}} \big\rfloor, \qquad \rho = 0.30
$$

En la referencia, $w_{\text{cuadro}}$ corresponde al ancho de la ROI de la columna $0$; en las muestras NEF, al ancho horizontal de color de la primera almohadilla. El conjunto de píxeles muestreados $\Omega$ es el disco de radio $r$ centrado en $(c^x, c^y)$:

$$
\Omega = \big\{ (x,y) : (x - c^x)^2 + (y - c^y)^2 \le r^2 \big\}
$$

El uso de una máscara circular real (no el *bounding box*) garantiza que la extracción de color se realice estrictamente dentro de la almohadilla.

---

## 5. Extracción de color representativo

### 5.1 Mediana RGB

El color representativo de cada almohadilla se obtiene mediante la **mediana** de cada canal sobre los píxeles del disco $\Omega$. La mediana es preferible a la media por su robustez ante valores atípicos (brillos especulares, sombras, ruido del sensor):

$$
\tilde{c}_k = \operatorname{median}\big\{\, I_k(x,y) : (x,y) \in \Omega \,\big\},
\qquad k \in \{R, G, B\}
$$

resultando en el color único $\tilde{\mathbf{c}} = (\tilde{c}_R,\ \tilde{c}_G,\ \tilde{c}_B)$.

### 5.2 Conversión a HSV y CIELAB

El color mediano se convierte a los espacios HSV y CIELAB mediante OpenCV, operando sobre el único píxel representativo $\tilde{\mathbf{c}}$ (no sobre medianas independientes por espacio, lo que sería inconsistente):

$$
\tilde{\mathbf{c}}_{\text{HSV}} = \Phi_{\text{RGB}\to\text{HSV}}(\tilde{\mathbf{c}}),
\qquad
\tilde{\mathbf{c}}_{\text{Lab}} = \Phi_{\text{RGB}\to\text{Lab}}(\tilde{\mathbf{c}})
$$

**Rangos en la convención de 8 bits de OpenCV.** Todos los valores quedan acotados en $[0, 255]$, con la particularidad del matiz:

$$
H \in [0, 179], \quad S, V \in [0, 255]
$$
$$
L, a, b \in [0, 255]
$$

El matiz $H$ se escala a $[0,179]$ para caber en 8 bits (medio grado por unidad). Los valores CIELAB en esta convención difieren de los rangos teóricos ($L^\* \in [0,100]$, $a^\*, b^\* \in [-128, 127]$); esta diferencia de escala se contemplará en etapas posteriores de comparación.

---

## 6. Formato de salida

### 6.1 Referencia (`output/ref_colores.csv`)

Una fila por almohadilla de la cuadrícula completa (56 en total):

```
clase, indice, R, G, B, H, S, V, L, a, b, cx, cy, n_pixeles
```

### 6.2 Muestras NEF (`output/nef_colores.csv`)

Una fila por almohadilla (10 por imagen), con identificación del archivo de origen:

```
imagen, clase, R, G, B, H, S, V, L, a, b, cx, cy, n_pixeles
```

El procesamiento en lote reescribe el CSV desde cero en cada corrida para evitar duplicados. Adicionalmente, por cada muestra se exporta el recorte de la ROI con los círculos y etiquetas superpuestos en formato PNG y SVG (`output/roi/<filename>_ROI.png` y `.svg`).

---

## 7. Resumen del pipeline

$$
\underbrace{I_{\text{RAW}}}_{\text{NEF}}
\;\xrightarrow{\ \Phi_{\text{demosaic}}\ }\;
I_{\text{RGB}}
\;\xrightarrow{\ \text{ROI}\ }\;
\text{tira}
\;\xrightarrow{\ \text{anclaje} + \Delta y\ }\;
\{(c_i^x, c_i^y)\}_{i=0}^{9}
\;\xrightarrow{\ \Omega,\ \operatorname{median}\ }\;
\tilde{\mathbf{c}}
\;\xrightarrow{\ \Phi\ }\;
(\text{RGB}, \text{HSV}, \text{Lab})
$$

El pipeline entrega, de forma estable y reproducible para todas las imágenes del lote, el color representativo de cada almohadilla reactiva en tres espacios de color, listo para la etapa siguiente de comparación cuantitativa contra los patrones de la carta de referencia.
