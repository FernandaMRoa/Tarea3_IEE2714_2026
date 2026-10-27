################ Pregunta 1 #######################
# Eliminacion de ruido periodico mediante filtrado notch
#
# Se implementa la cadena completa de filtrado en frecuencia:
#   imagen -> padding configurable -> FFT 2D (centrada) -> H(u,v) notch reject
#          -> G = F*H (espectro complejo) -> IFFT 2D -> recorte al tamano original
# con dos familias de notch reject (ideal y Gaussiano), y se estudia el ancho del notch y el padding.
#
# rawpy se usa solo para leer el archivo RAW. No se usa ninguna funcion externa que detecte
# componentes periodicas ni que construya los filtros notch.
#
# La imagen debe estar en 'imagenes/P1.CR2'.
# Las figuras generadas se guardan en 'figuras/p1/' y los datos en 'resultados/p1_resultados.json'

import os
import json
import time
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle, Circle
from matplotlib.colors import LogNorm
from scipy import fft as sfft
from scipy.ndimage import uniform_filter          # solo para promediar en ventanas (metricas)
from skimage.transform import downscale_local_mean   # solo para mostrar la imagen reducida
import rawpy                                       # solo para leer el RAW

# Carpetas de entrada y salida
RUTA_IMG = os.path.join('imagenes', 'P1.CR2')
CARPETA_FIG = os.path.join('figuras', 'p1')
CARPETA_RES = 'resultados'
os.makedirs(CARPETA_FIG, exist_ok=True)
os.makedirs(CARPETA_RES, exist_ok=True)

MOSTRAR = False     # True para ver las figuras en pantalla ademas de guardarlas

# Notches: lista de centros (fy, fx) en ciclos/pixel. El simetrico (-fy, -fx) se agrega solo.
NOTCHES = [(0.0, 0.25)]

# Padding: modo ('ninguno', 'ceros', 'simetrico', 'reflejo', 'replicar') y pixeles por borde
PADDING_MODO = 'simetrico'
PADDING_TAM = 128

# Anchos explorados en ciclos/pixel (ideal: radio D0, Gaussiano: sigma)
ANCHOS = [0.0005, 0.001, 0.002, 0.004, 0.008]

# Configuraciones de padding que se comparan en el punto 3
PADDINGS = [('ninguno', 0),
            ('ceros', 32), ('ceros', 128), ('ceros', 512),
            ('simetrico', 32), ('simetrico', 128), ('simetrico', 512),
            ('reflejo', 128), ('replicar', 128)]

# Configuracion final (punto 4)
TIPO_FINAL = 'gaussiano'
ANCHO_FINAL = 0.002

# Escalas comunes para las imagenes diferencia
LIM_DIF = 0.04       # diferencia ruidosa - filtrada
LIM_NOPER = 0.005    # parte no periodica de la diferencia

# Regiones de analisis (fila_ini, fila_fin, col_ini, col_fin) y recortes ampliados (fila, col, lado)
REG_CIELO = (120, 800, 200, 1200)         # cielo oscuro, casi homogeneo
REG_NEBULOSA = (1120, 1840, 1520, 2800)   # nebulosa, con mucho detalle
RECORTES = {'cielo': (400, 600, 64), 'nebulosa': (1480, 1590, 96),
            'borde izq.': (2200, 0, 64), 'borde sup.': (0, 2400, 64)}

# Ventana del espectro que se muestra alrededor del notch (0, +0.25)
ZOOM_FY = (-0.02, 0.02)
ZOOM_FX = (0.23, 0.27)

resultados = {}     # aqui se guardan todos los numeros que se usan en el informe


def guardar(fig, nombre):
    "Guarda la figura en figuras/p1/ y la muestra solo si MOSTRAR = True."
    ruta = os.path.join(CARPETA_FIG, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    if MOSTRAR:
        plt.show()
    plt.close(fig)
    print('  figura guardada ->', ruta)


################## 1. Lectura de la imagen ##################
#
# El RAW se revela con rawpy (debayer + balance de blancos de la camara) y se pasa a luminancia,
# Y = 0.2125 R + 0.7154 G + 0.0721 B, en float32 con rango [0, 1].

def cargar_cr2(ruta):
    "Lee el RAW y lo devuelve en escala de grises [0, 1]"
    with rawpy.imread(ruta) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
    rgb = rgb.astype(np.float32) / 65535.0
    gris = 0.2125 * rgb[:, :, 0] + 0.7154 * rgb[:, :, 1] + 0.0721 * rgb[:, :, 2]
    return gris.astype(np.float32)


img = cargar_cr2(RUTA_IMG)
M, N = img.shape
print(f'Imagen {RUTA_IMG}: tamaño {M}x{N}, media {img.mean():.3f}')
resultados['imagen'] = {'forma': [M, N], 'media': float(img.mean())}


################## 2. Padding, coordenadas de frecuencia y filtros notch ##################
#
# Padding: se agregan p pixeles en cada uno de los 4 bordes con np.pad y despues de la IFFT se
# recorta [p:p+M, p:p+N].
# Las frecuencias se trabajan en CICLOS/PIXEL (fftshift(fftfreq(P))), asi un notch representa la
# misma frecuencia fisica para cualquier tamaño de padding (con mas padding solo cambia la grilla).
#   ideal:     H = 0 si D_k <= D0 para algun notch k, 1 en el resto
#   gaussiano: H = prod_k (1 - exp(-D_k^2 / (2 sigma^2)))
# D_k es la distancia de cada pixel del espectro al centro del notch k.

MODOS_NUMPY = {'ceros': 'constant',      # fuera de la imagen vale 0
               'simetrico': 'symmetric', # espejo que repite el borde: ... c b a | a b c ...
               'reflejo': 'reflect',     # espejo sin repetir el borde: ... d c b | a b c ...
               'replicar': 'edge'}       # se copia el pixel del borde: ... a a a | a b c ...


def extender(imagen, modo, p):
    "Agrega p pixeles en cada borde"
    if modo == 'ninguno' or p == 0:
        return imagen.copy()
    return np.pad(imagen, p, mode=MODOS_NUMPY[modo])


def recortar(img_ext, modo, p, forma):
    "Recupera la zona original sacando p pixeles de cada borde"
    if modo == 'ninguno' or p == 0:
        return img_ext[:forma[0], :forma[1]]
    return img_ext[p:p + forma[0], p:p + forma[1]]


def coordenadas_frecuencia(forma):
    "Frecuencias (fy, fx) en ciclos/pixel del espectro centrado, como columna y fila"
    fy = np.fft.fftshift(np.fft.fftfreq(forma[0])).astype(np.float32)
    fx = np.fft.fftshift(np.fft.fftfreq(forma[1])).astype(np.float32)
    return fy[:, None], fx[None, :]


def pares_simetricos(notches):
    "Por cada centro (fy, fx) agrega su simetrico (-fy, -fx)"
    centros = []
    for fy0, fx0 in notches:
        centros.append((fy0, fx0))
        centros.append((-fy0, -fx0))
    return centros


def filtro_notch(forma, notches, ancho, tipo='gaussiano'):
    "Funcion de transferencia notch reject H(u,v) sobre el espectro centrado"
    FY, FX = coordenadas_frecuencia(forma)
    H = np.ones(forma, dtype=np.float32)
    for fy0, fx0 in pares_simetricos(notches):
        D = np.sqrt((FY - fy0) ** 2 + (FX - fx0) ** 2)   # distancia al centro del notch
        if tipo == 'ideal':
            H[D <= ancho] = 0.0
        else:
            H *= 1.0 - np.exp(-D ** 2 / (2.0 * ancho ** 2))
    return H


def filtrar_notch(imagen, notches, ancho, tipo='gaussiano', padding_modo=PADDING_MODO,
                  padding_tam=PADDING_TAM, H=None, espectros=False):
    """Cadena completa: padding -> FFT -> H -> F*H -> IFFT -> recorte.
    Si espectros=True tambien devuelve F, H, G y la parte imaginaria de la IFFT."""
    ext = extender(imagen, padding_modo, padding_tam)
    F = sfft.fftshift(sfft.fft2(ext, workers=-1))       # espectro complejo centrado
    if H is None:
        H = filtro_notch(ext.shape, notches, ancho, tipo)
    G = F * H                                           # se filtra el espectro complejo
    g = sfft.ifft2(sfft.ifftshift(G), workers=-1)       # pasar al espacio
    filtrada = recortar(np.real(g), padding_modo, padding_tam, imagen.shape)
    if not espectros:
        return filtrada
    imag = recortar(np.imag(g), padding_modo, padding_tam, imagen.shape)
    return filtrada, {'F': F, 'H': H, 'G': G, 'imag': imag, 'forma_ext': ext.shape}


################## 3. Metricas ##################
#
# - Ruido residual: amplitud de la componente de 0.25 ciclos/px que queda (proyeccion sobre
#   exp(-j 2pi 0.25 x)). Para un patron Acos(2 pi 0.25 x + phi) devuelve A.
#
# - Perdida de detalle: a la diferencia d = ruidosa - filtrada se le resta, en bloques de 16 filas, 
#   el patron promedio de periodo 4. Lo que queda (parte no periodica) es lo que el filtro quito y no
#   calza con el patron. Su RMS en la nebulosa mide la informacion perdida.
#
# - Artefactos: la misma medida en el cielo homogeneo (en esa zona no hay estructura que perder).

def region(imagen, reg):
    f0, f1, c0, c1 = reg
    return imagen[f0:f1, c0:c1]


def recorte(imagen, nombre):
    f, c, l = RECORTES[nombre]
    return imagen[f:f + l, c:c + l]


def amplitud_patron(imagen, fx=0.25):
    "Amplitud de la componente de columnas de frecuencia fx"
    x = np.arange(imagen.shape[1])
    return float(2.0 * np.abs(np.mean(imagen * np.exp(-2j * np.pi * fx * x)[None, :])))


def parte_no_periodica(d, periodo=4, bloque=16):
    "d menos el patron promedio de las 4 fases (columnas j, j+4, ...) de cada bloque de filas"
    resto = np.zeros_like(d)
    for f0 in range(0, d.shape[0], bloque):
        b = d[f0:f0 + bloque]
        patron = np.array([b[:, j::periodo].mean() for j in range(periodo)])
        patron = patron - patron.mean()                 # el patron no lleva el nivel medio
        patron = np.tile(patron, b.shape[1] // periodo + 1)[:b.shape[1]]
        resto[f0:f0 + bloque] = b - patron[None, :]
    return resto


def rms_no_periodico(d):
    "RMS de la parte no periodica (sin su media, que es nivel y no detalle)"
    r = parte_no_periodica(d)
    return float(np.sqrt(np.mean((r - r.mean()) ** 2)))


def metricas(ruidosa, filtrada):
    "Ruido residual (en distintas zonas), perdida de detalle y artefactos"
    d = ruidosa - filtrada
    n = ruidosa.shape[1]
    return {'amp_residual_cielo': amplitud_patron(region(filtrada, REG_CIELO)),
            'amp_residual_nebulosa': amplitud_patron(region(filtrada, REG_NEBULOSA)),
            'amp_residual_borde_izq': amplitud_patron(filtrada[:, :16]),
            'amp_residual_borde_der': amplitud_patron(filtrada[:, n - 16:]),
            'amp_residual_borde_sup': amplitud_patron(filtrada[:16, 64:n - 64]),
            'perdida_detalle_nebulosa': rms_no_periodico(region(d, REG_NEBULOSA)),
            'artefactos_cielo': rms_no_periodico(region(d, REG_CIELO))}


def mapa_amplitud(imagen, fx=0.25, ventana=32, k=8):
    "Amplitud local del patron: se demodula, se promedia en ventanas y se reduce k veces"
    x = np.arange(imagen.shape[1])
    z = imagen * np.exp(-2j * np.pi * fx * x)[None, :]
    re = uniform_filter(z.real.astype(np.float32), ventana)
    im = uniform_filter(z.imag.astype(np.float32), ventana)
    return downscale_local_mean(2.0 * np.sqrt(re ** 2 + im ** 2), (k, k))


def amplitud_vs_distancia(imagen, eje, paso=8, largo=400):
    "Amplitud del patron en franjas a distancia creciente del borde izquierdo ('x') o superior ('y')"
    dist, amp = [], []
    for k in range(0, largo, paso):
        if eje == 'x':
            franja = imagen[:, k:k + 4 * paso]        # 32 columnas (periodos completos)
        else:
            franja = imagen[k:k + paso, 64:imagen.shape[1] - 64]
        dist.append(k)
        amp.append(amplitud_patron(franja))
    return np.array(dist), np.array(amp)


# Funciones auxiliares para las figuras
def escala_recorte(imagen, nombre):
    "Escala de grises fija para un recorte (percentiles 2 y 98 de la imagen ruidosa)"
    r = recorte(imagen, nombre)
    return float(np.percentile(r, 2)), float(np.percentile(r, 98))


def log_espectro(F):
    return np.log1p(np.abs(F))


def reducir_max(A, k):
    "Reduce A en bloques k x k tomando el maximo (asi los peaks del espectro no desaparecen)"
    P, Q = (A.shape[0] // k) * k, (A.shape[1] // k) * k
    return A[:P, :Q].reshape(P // k, k, Q // k, k).max(axis=(1, 3))


def ventana_espectro(F, forma):
    "Recorte del espectro centrado dentro de ZOOM_FY x ZOOM_FX (ciclos/pixel)"
    fy = np.fft.fftshift(np.fft.fftfreq(forma[0]))
    fx = np.fft.fftshift(np.fft.fftfreq(forma[1]))
    i0, i1 = np.searchsorted(fy, ZOOM_FY[0]), np.searchsorted(fy, ZOOM_FY[1])
    j0, j1 = np.searchsorted(fx, ZOOM_FX[0]), np.searchsorted(fx, ZOOM_FX[1])
    return F[i0:i1, j0:j1], [fx[j0], fx[j1 - 1], fy[i1 - 1], fy[i0]]


def mostrar(a, imagen, titulo, vmin=0, vmax=1, cmap='gray'):
    h = a.imshow(imagen, cmap=cmap, vmin=vmin, vmax=vmax, interpolation='nearest')
    a.set_title(titulo, fontsize=9)
    a.axis('off')
    return h


def mostrar_espectro(a, L, extent, titulo, vmin=None, vmax=None):
    a.imshow(L, cmap='gray', vmin=vmin, vmax=vmax, extent=extent, aspect='auto')
    a.set_title(titulo, fontsize=9)
    a.set_xlabel('fx [ciclos/px]', fontsize=8)
    a.set_ylabel('fy [ciclos/px]', fontsize=8)
    a.tick_params(labelsize=7)


# Figura: imagen original y recortes ampliados
fig = plt.figure(figsize=(15, 7.2))
gs = fig.add_gridspec(2, 4)
a = fig.add_subplot(gs[:, :2])
a.imshow(downscale_local_mean(img, (4, 4)), cmap='gray', vmin=0, vmax=1)
colores_rec = {'cielo': 'tab:cyan', 'nebulosa': 'tab:orange', 'borde izq.': 'tab:green', 'borde sup.': 'tab:pink'}
for nombre, (f, c, l) in RECORTES.items():
    a.add_patch(Rectangle((c / 4, f / 4), l / 4, l / 4, fill=False, ec=colores_rec[nombre], lw=1.8))
    a.text(c / 4 + l / 4 + 4, f / 4 + 10, nombre, color=colores_rec[nombre], fontsize=8)
for reg, col, txt in [(REG_CIELO, 'w', 'región cielo'), (REG_NEBULOSA, 'y', 'región nebulosa')]:
    a.add_patch(Rectangle((reg[2] / 4, reg[0] / 4), (reg[3] - reg[2]) / 4, (reg[1] - reg[0]) / 4,
                          fill=False, ec=col, ls='--', lw=1))
    a.text(reg[2] / 4, reg[1] / 4 + 18, txt, color=col, fontsize=8)
a.set_title(f'Imagen P1 en grises ({M}×{N}), mostrada reducida ×4')
a.axis('off')
for k, nombre in enumerate(RECORTES):
    v = escala_recorte(img, nombre)
    mostrar(fig.add_subplot(gs[k // 2, 2 + k % 2]), recorte(img, nombre),
            f'Recorte {nombre} ({RECORTES[nombre][2]} px)', *v)
guardar(fig, '00_imagen_original.png')


################## 4. Punto 1 ##################
#
# Se busca en el espectro sin padding los peaks mas fuertes fuera del centro y se compara con la
# evidencia espacial: el perfil de columnas de una zona oscura (cielo) muestra un patron que se
# repite cada 4 columnas, o sea fx = 1/4 = 0.25 ciclos/px.

F0 = sfft.fftshift(sfft.fft2(img, workers=-1))
L0 = log_espectro(F0)
fy_vec = np.fft.fftshift(np.fft.fftfreq(M))
fx_vec = np.fft.fftshift(np.fft.fftfreq(N))

zona = region(img, REG_CIELO)
perfil = zona.mean(axis=0)
patron4 = np.array([zona[:, j::4].mean() for j in range(4)])
patron4 = patron4 - patron4.mean()
resultados['patron_periodo4_cielo'] = patron4.tolist()

R = L0.copy()
R[M // 2 - 40:M // 2 + 41, N // 2 - 40:N // 2 + 41] = 0      # se tapa el centro
peaks = []
print('Peaks mas fuertes fuera del centro (fy, fx, log(1+|F|)):')
for i in np.argsort(R.ravel())[::-1][:6]:
    fila, col = divmod(int(i), N)
    peaks.append({'fy': float(fy_vec[fila]), 'fx': float(fx_vec[col]), 'log_mag': float(L0[fila, col])})
    print(f'  ({fy_vec[fila]:.4f}, {fx_vec[col]:.4f})  {L0[fila, col]:.2f}')
print(f'Mediana de log(1+|F|) = {np.median(L0):.2f}')
print('Patron de periodo 4 en el cielo [x1e-3]:', np.round(patron4 * 1000, 2))
resultados['peaks_espectro'] = peaks
resultados['mediana_log_espectro'] = float(np.median(L0))
del R

fig = plt.figure(figsize=(15, 9))
gs = fig.add_gridspec(2, 3, height_ratios=[1.15, 1], hspace=0.38, wspace=0.28)
a = fig.add_subplot(gs[0, :2])
a.imshow(reducir_max(L0, 4), cmap='gray', extent=[-0.5, 0.5, 0.5, -0.5], aspect='auto')
for fy0, fx0 in pares_simetricos(NOTCHES):
    a.add_patch(Circle((fx0, fy0), 0.02, fill=False, ec='r', lw=1.5))
a.set_xlabel('fx [ciclos/px]')
a.set_ylabel('fy [ciclos/px]')
a.set_title('log(1+|F|) sin padding (reducido ×4 con máximo por bloque). Círculos: notches elegidos')
W, ext = ventana_espectro(L0, img.shape)
mostrar_espectro(fig.add_subplot(gs[0, 2]), W, ext, 'Zoom alrededor de (fy, fx) = (0, +0.25)')
a = fig.add_subplot(gs[1, :2])
a.plot(fx_vec, L0[M // 2, :], 'k-', lw=0.6, label='fila fy = 0')
a.plot(fx_vec, L0[M // 2 + 30, :], '--', color='tab:blue', lw=0.6, alpha=0.8, label=f'fila fy = {30 / M:.4f} (referencia)')
a.axvline(0.25, color='r', ls=':', lw=1)
a.axvline(-0.25, color='r', ls=':', lw=1)
a.set_xlabel('fx [ciclos/px]')
a.set_ylabel('log(1+|F|)')
a.set_title('Corte horizontal del espectro: peaks aislados en fx = ±0.25 (líneas rojas)')
a.legend(fontsize=8, loc='upper right')
a.grid(alpha=0.3)
a = fig.add_subplot(gs[1, 2])
a.plot(np.arange(40), (perfil[:40] - perfil.mean()) * 1000, 'k.-', lw=1)
a.set_xlabel('columna')
a.set_ylabel('promedio de la columna - media [×10⁻³]')
a.set_title(f'Perfil de columnas en la región de cielo\npatrón de periodo 4: {np.round(patron4 * 1000, 1)} ×10⁻³')
a.grid(alpha=0.3)
guardar(fig, '01_espectro.png')

# Funciones de transferencia de ambas familias (zoom en el notch +0.25)
forma_ext = (M + 2 * PADDING_TAM, N + 2 * PADDING_TAM)
fx_e = np.fft.fftshift(np.fft.fftfreq(forma_ext[1]))
fig, ax = plt.subplots(2, 4, figsize=(15, 7))
for i, tipo in enumerate(['ideal', 'gaussiano']):
    p = 'D0' if tipo == 'ideal' else 'σ'
    for j, (w, estilo) in enumerate(zip([0.001, 0.002, 0.004], ['-', '--', ':'])):
        H = filtro_notch(forma_ext, NOTCHES, w, tipo)
        Hz, ext = ventana_espectro(H, forma_ext)
        mostrar_espectro(ax[i, j], Hz, ext, f'H {tipo}, {p} = {w} c/px', 0, 1)
        ax[i, 3].plot(fx_e, H[forma_ext[0] // 2, :], estilo, lw=1.5, label=f'{p} = {w}')
    ax[i, 3].set_xlim(ZOOM_FX)
    ax[i, 3].set_xlabel('fx [ciclos/px]')
    ax[i, 3].set_ylabel('H(fy = 0, fx)')
    ax[i, 3].set_title(f'Corte de H {tipo} en fy = 0')
    ax[i, 3].legend(fontsize=8)
    ax[i, 3].grid(alpha=0.3)
fig.suptitle(f'Funciones de transferencia notch (zoom en el notch +0.25; padding {PADDING_MODO} {PADDING_TAM} px)')
plt.tight_layout()
guardar(fig, '02_mascaras.png')


################## 5. Punto 2 ##################
#
# Para cada ancho se filtra la imagen completa (padding simetrico 128 px) y se calculan las tres
# metricas. Para los casos principales se muestran recortes, el espectro antes/despues y la
# diferencia con escala comun.

_, esp0 = filtrar_notch(img, NOTCHES, 0.001, 'gaussiano', espectros=True)
F_pad = esp0['F']
L_antes, ext_z = ventana_espectro(log_espectro(F_pad), forma_ext)
v_esp = (float(np.percentile(L_antes, 1)), float(L_antes.max()))   # escala comun de espectros
del esp0

resultados['barrido_anchos'] = {'ideal': [], 'gaussiano': []}
guardados = {}
print('Familia    | ancho  | residual cielo | perdida detalle | artefactos')
for tipo in ['ideal', 'gaussiano']:
    for w in ANCHOS:
        H = filtro_notch(forma_ext, NOTCHES, w, tipo)
        filt = filtrar_notch(img, NOTCHES, w, tipo, H=H)
        m = metricas(img, filt)
        m['ancho'] = w
        resultados['barrido_anchos'][tipo].append(m)
        Ld, _ = ventana_espectro(log_espectro(F_pad * H), forma_ext)
        dif_neb = recorte(img - filt, 'nebulosa')
        guardados[(tipo, w)] = (recorte(filt, 'cielo').copy(), recorte(filt, 'nebulosa').copy(), Ld,
                                dif_neb.copy(), parte_no_periodica(dif_neb))
        print(f'{tipo:10s} | {w:<6} | {m["amp_residual_cielo"]:.5f}        | '
              f'{m["perdida_detalle_nebulosa"]:.5f}         | {m["artefactos_cielo"]:.5f}')
amp0_cielo = amplitud_patron(region(img, REG_CIELO))
amp0_neb = amplitud_patron(region(img, REG_NEBULOSA))
print(f'Amplitud del patron sin filtrar: cielo {amp0_cielo:.5f}, nebulosa {amp0_neb:.5f}')
resultados['amp_patron_original'] = {'cielo': amp0_cielo, 'nebulosa': amp0_neb}

v_cielo = escala_recorte(img, 'cielo')
v_neb = escala_recorte(img, 'nebulosa')
anchos_fig = [0.0005, 0.002, 0.008]
for tipo, nombre_fig in [('ideal', '03a_anchos_ideal.png'), ('gaussiano', '03b_anchos_gaussiano.png')]:
    p = 'D0' if tipo == 'ideal' else 'σ'
    fig, ax = plt.subplots(len(anchos_fig) + 1, 5, figsize=(16, 14), gridspec_kw={'hspace': 0.42, 'wspace': 0.3})
    mostrar(ax[0, 0], recorte(img, 'cielo'), 'Ruidosa: recorte cielo', *v_cielo)
    mostrar(ax[0, 1], recorte(img, 'nebulosa'), 'Ruidosa: recorte nebulosa', *v_neb)
    mostrar_espectro(ax[0, 2], L_antes, ext_z, 'Espectro ANTES (zoom en +0.25)', *v_esp)
    ax[0, 3].axis('off')
    ax[0, 4].axis('off')
    ax[0, 3].text(0.0, 0.5, f'Familia: {tipo}\nPadding: {PADDING_MODO} {PADDING_TAM} px\n'
                  f'Notches: (fy, fx) = ±{NOTCHES[0]}\nDiferencia = ruidosa - filtrada (escala común ±{LIM_DIF})\n'
                  f'Parte no periódica = diferencia sin el patrón\nde periodo 4 (escala común ±{LIM_NOPER})',
                  fontsize=10.5, va='center')
    for i, w in enumerate(anchos_fig):
        f_cielo, f_neb, Ld, d_neb, np_neb = guardados[(tipo, w)]
        mostrar(ax[i + 1, 0], f_cielo, f'Filtrada cielo ({p} = {w})', *v_cielo)
        mostrar(ax[i + 1, 1], f_neb, f'Filtrada nebulosa ({p} = {w})', *v_neb)
        mostrar_espectro(ax[i + 1, 2], Ld, ext_z, f'Espectro DESPUÉS ({p} = {w})', *v_esp)
        mostrar(ax[i + 1, 3], d_neb, f'Diferencia nebulosa ({p} = {w})', -LIM_DIF, LIM_DIF, 'seismic')
        im = mostrar(ax[i + 1, 4], np_neb, f'Parte no periódica ({p} = {w})', -LIM_NOPER, LIM_NOPER, 'seismic')
    fig.colorbar(im, ax=ax[1:, 4].ravel().tolist(), fraction=0.04, pad=0.03, label=f'parte no periódica (±{LIM_NOPER})')
    fig.suptitle(f'Notch {tipo}: efecto del ancho', fontsize=12, y=0.93)
    guardar(fig, nombre_fig)
del guardados

# Metricas en funcion del ancho 
fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
estilos = {'ideal': dict(ls='--', marker='s', color='tab:red'),
           'gaussiano': dict(ls='-', marker='o', color='tab:blue')}
claves = [('amp_residual_cielo', 'Ruido residual: amplitud del patrón (cielo)'),
          ('perdida_detalle_nebulosa', 'Pérdida de detalle: RMS no periódico (nebulosa)'),
          ('artefactos_cielo', 'Artefactos: RMS no periódico (cielo)')]
for a, (clave, titulo) in zip(ax, claves):
    for tipo in ['ideal', 'gaussiano']:
        a.plot(ANCHOS, [m[clave] for m in resultados['barrido_anchos'][tipo]], lw=1.6, ms=6, label=tipo, **estilos[tipo])
    a.set_xscale('log')
    a.set_xlabel('ancho del notch (D0 o σ) [ciclos/px]')
    a.set_title(titulo)
    a.grid(alpha=0.3, which='both')
ax[0].axhline(amp0_cielo, color='k', ls=':', lw=1, label='sin filtrar')
ax[0].set_yscale('log')
for a in ax:
    a.legend(fontsize=8)
plt.tight_layout()
guardar(fig, '04_metricas_anchos.png')


################## 6. Punto 3 ##################
#
# Con el notch final (Gaussiano sigma = 0.002) se compara el filtrado sin padding con extension por
# ceros, simetrica, reflejo y replicar, con varios tamaños. Como el interior queda igual en todos
# los casos, los bordes se analizan con: la amplitud residual en franjas pegadas a cada borde, un
# mapa de la amplitud local del patron y recortes ampliados.

resultados['padding'] = []
recortes_pad, mapas_pad, curvas_pad = {}, {}, {}
print('Padding        | borde izq | borde der | borde sup | cielo   | tiempo')
for modo, p in PADDINGS:
    t0 = time.perf_counter()
    filt = filtrar_notch(img, NOTCHES, ANCHO_FINAL, TIPO_FINAL, padding_modo=modo, padding_tam=p)
    dt = time.perf_counter() - t0
    m = metricas(img, filt)
    m.update({'modo': modo, 'tam': p, 'tiempo_s': dt})
    resultados['padding'].append(m)
    recortes_pad[(modo, p)] = {n: recorte(filt, n).copy() for n in ['borde izq.', 'borde sup.']}
    mapas_pad[(modo, p)] = mapa_amplitud(filt)
    curvas_pad[(modo, p)] = (amplitud_vs_distancia(filt, 'x'), amplitud_vs_distancia(filt, 'y'))
    print(f'{modo + " " + str(p):14s} | {m["amp_residual_borde_izq"]:.5f}   | {m["amp_residual_borde_der"]:.5f}   | '
          f'{m["amp_residual_borde_sup"]:.5f}   | {m["amp_residual_cielo"]:.5f} | {dt:.2f} s')

# Mapas de amplitud residual del patron sobre toda la imagen
casos_mapa = [('ninguno', 0), ('ceros', 128), ('replicar', 128),
              ('simetrico', 32), ('simetrico', 128), ('simetrico', 512), ('reflejo', 128)]
fig, ax = plt.subplots(2, 4, figsize=(17, 6.2))
norma = LogNorm(vmin=1e-4, vmax=2.5e-2)
im = ax[0, 0].imshow(mapa_amplitud(img), cmap='magma', norm=norma)
ax[0, 0].set_title('Ruidosa (sin filtrar)')
for a, (modo, p) in zip(ax.ravel()[1:], casos_mapa):
    nom = 'sin padding' if modo == 'ninguno' else f'{modo} {p} px'
    a.imshow(mapas_pad[(modo, p)], cmap='magma', norm=norma)
    a.set_title(f'Filtrada, {nom}')
for a in ax.ravel():
    a.axis('off')
fig.colorbar(im, ax=ax.ravel().tolist(), fraction=0.02, pad=0.01, label='amplitud local del patrón (ventana 32 px)')
fig.suptitle(f'Mapa de ruido periódico residual con distintos padding (notch {TIPO_FINAL} σ = {ANCHO_FINAL} c/px)')
guardar(fig, '05a_padding_mapas.png')

# Amplitud residual en funcion de la distancia al borde
fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
estilos_pad = {('ninguno', 0): dict(color='k', ls='-', lw=2),
               ('ceros', 32): dict(color='tab:red', ls='--', marker='x', ms=4),
               ('ceros', 128): dict(color='tab:red', ls='-.', marker='+', ms=5),
               ('ceros', 512): dict(color='tab:red', ls=':', lw=2),
               ('simetrico', 32): dict(color='tab:blue', ls='--', marker='o', ms=3),
               ('simetrico', 128): dict(color='tab:blue', ls='-', lw=1.6),
               ('simetrico', 512): dict(color='tab:blue', ls=':', lw=2),
               ('reflejo', 128): dict(color='tab:green', ls='-.', marker='^', ms=4),
               ('replicar', 128): dict(color='tab:purple', ls='--', marker='v', ms=4)}
for (modo, p), ((dx, ax_), (dy, ay)) in curvas_pad.items():
    nom = 'sin padding' if modo == 'ninguno' else f'{modo} {p}'
    ax[0].plot(dx, ax_, label=nom, **estilos_pad[(modo, p)])
    ax[1].plot(dy, ay, label=nom, **estilos_pad[(modo, p)])
for a, t in zip(ax, ['distancia al borde IZQUIERDO [px]', 'distancia al borde SUPERIOR [px]']):
    a.axhline(amp0_cielo, color='0.5', ls=':', lw=1)
    a.set_yscale('log')
    a.set_xlabel(t)
    a.set_ylabel('amplitud residual del patrón')
    a.grid(alpha=0.3, which='both')
ax[0].set_title('Ruido residual cerca del borde izquierdo (gris: amplitud sin filtrar)')
ax[1].set_title('Ruido residual cerca del borde superior')
ax[1].legend(ncol=2, fontsize=7)
plt.tight_layout()
guardar(fig, '05b_padding_perfiles.png')

# Recortes ampliados pegados a los bordes
casos = [('ninguno', 0), ('ceros', 128), ('simetrico', 32), ('simetrico', 128), ('simetrico', 512), ('reflejo', 128)]
fig, ax = plt.subplots(2, len(casos) + 1, figsize=(17, 5.6))
for i, (nombre, txt) in enumerate([('borde izq.', 'borde izquierdo'), ('borde sup.', 'borde superior')]):
    v = escala_recorte(img, nombre)
    mostrar(ax[i, 0], recorte(img, nombre), f'Ruidosa\n({txt})', *v)
    for j, (modo, p) in enumerate(casos):
        nom = 'sin padding' if modo == 'ninguno' else f'{modo} {p} px'
        mostrar(ax[i, j + 1], recortes_pad[(modo, p)][nombre], f'{nom}\n({txt})', *v)
fig.suptitle(f'Recortes de 64 px pegados al borde (notch {TIPO_FINAL} σ = {ANCHO_FINAL}); misma escala de grises por fila')
plt.tight_layout()
fig.subplots_adjust(hspace=0.45)
guardar(fig, '05c_padding_recortes.png')
del recortes_pad, mapas_pad


################## 7. Punto 4 ##################
#
# Gaussiano sigma = 0.002 c/px en (0, +-0.25) con padding simetrico de 128 px. Se revisa ademas
# que la parte imaginaria de la IFFT sea ~0 (el filtro conserva la simetria del espectro).

filt, esp = filtrar_notch(img, NOTCHES, ANCHO_FINAL, TIPO_FINAL, espectros=True)
m = metricas(img, filt)
m['max_abs_imag'] = float(np.abs(esp['imag']).max())
resultados['final'] = {'tipo': TIPO_FINAL, 'ancho': ANCHO_FINAL, 'notches': [list(n) for n in NOTCHES],
                       'padding_modo': PADDING_MODO, 'padding_tam': PADDING_TAM, **m}
print('Configuracion final:', resultados['final'])

fig = plt.figure(figsize=(16, 11))
gs = fig.add_gridspec(3, 4, height_ratios=[1.25, 1, 1])
mostrar(fig.add_subplot(gs[0, :2]), downscale_local_mean(img, (4, 4)), 'Ruidosa (reducida ×4)')
mostrar(fig.add_subplot(gs[0, 2:]), downscale_local_mean(filt, (4, 4)),
        f'Filtrada: notch {TIPO_FINAL} σ = {ANCHO_FINAL} c/px, padding {PADDING_MODO} {PADDING_TAM} px')
for j, nombre in enumerate(['cielo', 'nebulosa']):
    v = escala_recorte(img, nombre)
    mostrar(fig.add_subplot(gs[1, 2 * j]), recorte(img, nombre), f'Ruidosa: {nombre}', *v)
    mostrar(fig.add_subplot(gs[1, 2 * j + 1]), recorte(filt, nombre), f'Filtrada: {nombre}', *v)
Lz, ext = ventana_espectro(log_espectro(esp['F']), esp['forma_ext'])
mostrar_espectro(fig.add_subplot(gs[2, 0]), Lz, ext, 'Espectro antes (zoom +0.25)', *v_esp)
Lz, ext = ventana_espectro(log_espectro(esp['G']), esp['forma_ext'])
mostrar_espectro(fig.add_subplot(gs[2, 1]), Lz, ext, 'Espectro después (zoom +0.25)', *v_esp)
d_neb = recorte(img - filt, 'nebulosa')
a = fig.add_subplot(gs[2, 2])
im = mostrar(a, d_neb, f'Diferencia nebulosa (±{LIM_DIF})', -LIM_DIF, LIM_DIF, 'seismic')
fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
a = fig.add_subplot(gs[2, 3])
im = mostrar(a, parte_no_periodica(d_neb), f'Parte no periódica de la diferencia (±{LIM_NOPER})', -LIM_NOPER, LIM_NOPER, 'seismic')
fig.colorbar(im, ax=a, fraction=0.046, pad=0.03)
plt.tight_layout()
guardar(fig, '06_final.png')


############# 8. Preguntas guiadas: valor de una frecuencia antes y despues #######################
#
# Se toma el bin mas cercano al centro del notch y uno a una distancia ~ sigma, y se muestra F, H y
# G = F*H. Tambien se revisa que el valor en (-fy, -fx) sea el conjugado (simetria hermitica).

P, Q = esp['forma_ext']
fy_e = np.fft.fftshift(np.fft.fftfreq(P))
fx_e = np.fft.fftshift(np.fft.fftfreq(Q))
fila = int(np.argmin(np.abs(fy_e - 0.0)))
col_c = int(np.argmin(np.abs(fx_e - 0.25)))                    # centro del notch
col_s = int(np.argmin(np.abs(fx_e - (0.25 + ANCHO_FINAL))))    # a una distancia ~ sigma
trazas = []
for col in [col_c, col_s]:
    Fv, Hv, Gv = esp['F'][fila, col], esp['H'][fila, col], esp['G'][fila, col]
    trazas.append({'fila': fila, 'col': col, 'fy': float(fy_e[fila]), 'fx': float(fx_e[col]),
                   'distancia_al_notch': float(abs(fx_e[col] - 0.25)),
                   'F_real': float(Fv.real), 'F_imag': float(Fv.imag), 'H': float(Hv),
                   'G_real': float(Gv.real), 'G_imag': float(Gv.imag)})
    print(f'(fy, fx) = ({fy_e[fila]:.5f}, {fx_e[col]:.5f}): F = {Fv:.2f}, H = {Hv:.4f}, G = {Gv:.2f}')
fila_s = P - fila if P % 2 == 0 else P - 1 - fila
col_cs = Q - col_c if Q % 2 == 0 else Q - 1 - col_c
Fm = esp['F'][fila_s, col_cs]
trazas.append({'simetrico_fy': float(fy_e[fila_s]), 'simetrico_fx': float(fx_e[col_cs]),
               'F_real': float(Fm.real), 'F_imag': float(Fm.imag), 'H': float(esp['H'][fila_s, col_cs])})
print(f'Simetrico (fy, fx) = ({fy_e[fila_s]:.5f}, {fx_e[col_cs]:.5f}): F = {Fm:.2f}')
resultados['traza_frecuencia'] = trazas
del esp


############# 9. Preguntas guiadas: eliminar solo uno de los notches del par #######################
#
# Se construye H solo con el notch en +0.25. H deja de ser simetrica, el espectro filtrado deja de
# ser hermitiano y la IFFT queda con parte imaginaria. La parte real conserva la mitad del patron.

ext_img = extender(img, PADDING_MODO, PADDING_TAM)
FY, FX = coordenadas_frecuencia(ext_img.shape)
D = np.sqrt((FY - NOTCHES[0][0]) ** 2 + (FX - NOTCHES[0][1]) ** 2)
H_uno = (1.0 - np.exp(-D ** 2 / (2.0 * ANCHO_FINAL ** 2))).astype(np.float32)   # solo +0.25
del D, FY, FX, ext_img
real_uno, esp_uno = filtrar_notch(img, None, None, H=H_uno, espectros=True)
imag_uno = esp_uno['imag']
resultados['un_solo_notch'] = {'amp_residual_cielo_real': amplitud_patron(region(real_uno, REG_CIELO)),
                               'amp_original_cielo': amp0_cielo,
                               'max_abs_imag': float(np.abs(imag_uno).max()),
                               'rms_imag_cielo': float(np.sqrt(np.mean(region(imag_uno, REG_CIELO) ** 2)))}
print('Un solo notch:', resultados['un_solo_notch'])

# Para que se vea el efecto se comparan los perfiles de columnas (promedio de cada columna en la
# region de cielo, asi queda solo el patron)
# y las diferencias ruidosa - filtrada en un recorte pequeño (32 px), todo con la misma escala.
def perfil_columnas(imagen, n_col=24):
    zona_ = region(imagen, REG_CIELO)
    p_ = zona_.mean(axis=0)
    return p_[:n_col] - p_[:n_col].mean()


col = np.arange(24)
fig = plt.figure(figsize=(17, 4.6))
gs = fig.add_gridspec(1, 5, width_ratios=[2.2, 1, 1, 1, 0.08], wspace=0.25)
a = fig.add_subplot(gs[0, 0])
a.plot(col, perfil_columnas(img) * 1000, 'k-o', ms=4, lw=1.6, label=f'ruidosa (amplitud {amp0_cielo * 1000:.1f})')
a.plot(col, perfil_columnas(filt) * 1000, '-', color='tab:green', marker='s', ms=4, lw=1.4,
       label=f'par simétrico (amplitud {amplitud_patron(region(filt, REG_CIELO)) * 1000:.2f})')
a.plot(col, perfil_columnas(real_uno) * 1000, '--', color='tab:blue', marker='^', ms=4, lw=1.6,
       label=f'un notch, parte real (amplitud {amplitud_patron(region(real_uno, REG_CIELO)) * 1000:.1f})')
a.plot(col, perfil_columnas(imag_uno) * 1000, ':', color='tab:red', marker='v', ms=4, lw=1.8,
       label=f'un notch, parte imaginaria (amplitud {amplitud_patron(region(imag_uno, REG_CIELO)) * 1000:.1f})')
a.axhline(0, color='0.5', lw=0.8)
a.set_xlabel('columna')
a.set_ylabel('promedio de la columna - media [×10⁻³]')
a.set_title('Perfil de columnas en la región de cielo')
a.legend(fontsize=8, loc='lower right')
a.grid(alpha=0.3)
f0, c0, l = RECORTES['cielo'][0], RECORTES['cielo'][1], 32
lim = 0.025
paneles = [(img - filt, 'Ruidosa - filtrada\n(par simétrico)'),
           (img - real_uno, 'Ruidosa - parte real\n(un notch: se quita la mitad)'),
           (imag_uno, 'Parte imaginaria\n(un notch: no nula)')]
for k, (dd, tit) in enumerate(paneles):
    im = mostrar(fig.add_subplot(gs[0, k + 1]), dd[f0:f0 + l, c0:c0 + l], tit, -lim, lim, 'seismic')
fig.colorbar(im, cax=fig.add_subplot(gs[0, 4]), label=f'escala común ±{lim}')
fig.suptitle(f'Efecto de eliminar solo uno de los notches del par (gaussiano σ = {ANCHO_FINAL})', y=1.03)
guardar(fig, '07_un_solo_notch.png')


# Guardado de todas las metricas
with open(os.path.join(CARPETA_RES, 'p1_resultados.json'), 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print('Fin. Los datos obtenidos se encuentran en resultados/p1_resultados.json')
