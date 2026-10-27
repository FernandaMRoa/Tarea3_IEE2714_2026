########################## Bonus - Seleccion adaptativa del filtro notch ###############################
#
# Se buscan automaticamente los parametros del notch Gaussiano de la Pregunta 1 que minimizan la
# varianza local media en regiones aproximadamente homogeneas de la imagen (cielo):
#   1) se busca el centro del notch fx0 con un ancho fijo pequeño
#   2) con el centro encontrado se barre el ancho sigma en escala logaritmica
# Despues se analiza por que minimizar solo la varianza lleva a filtrar de mas, y se agrega un
# criterio simple: elegir sigma en la "zona plana" de la funcion objetivo (donde J casi no cambia).
#
# Este codigo vuelve a leer la imagen y usa el mismo padding y el mismo notch Gaussiano que el
# codigo de la pregunta 1 (p1_filtro_notch.py).
# La imagen debe estar en 'imagenes/P1.CR2'.
# Las figuras generadas se guardan en `figuras/bonus/` y métricas en `resultados/bonus_resultados.json`.

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import fft as sfft
from scipy.ndimage import uniform_filter          # promedio movil para la varianza local
from skimage.transform import downscale_local_mean   # solo para mostrar la imagen reducida
import rawpy                                       # solo para leer el RAW

RUTA_IMG = os.path.join('imagenes', 'P1.CR2')
CARPETA_FIG = os.path.join('figuras', 'bonus')
CARPETA_RES = 'resultados'
os.makedirs(CARPETA_FIG, exist_ok=True)
os.makedirs(CARPETA_RES, exist_ok=True)

MOSTRAR = False
PADDING_TAM = 128                 # padding simetrico, igual que en la pregunta 1
FY0 = 0.0                         # el patron es de columnas: toda su energia esta en fy = 0
FX0_INICIAL = 0.25                # centro esperado (periodo 4 px)

# Regiones homogeneas (cielo sin la nebulosa): (fila_ini, fila_fin, col_ini, col_fin)
REGIONES_HOMOGENEAS = [(120, 800, 200, 1200), (2300, 2800, 2800, 3800), (150, 550, 3000, 4000)]
VENTANA_VARIANZA = 7              # ventana de la varianza local (px)
REG_NEBULOSA = (1120, 1840, 1520, 2800)   # region con detalle (no se usa en la funcion objetivo)
RECORTES = {'cielo': (400, 600, 64), 'nebulosa': (1480, 1590, 96)}

SIGMA_CENTRO = 0.0005                                  # paso 1: sigma fijo
CENTROS = FX0_INICIAL + np.linspace(-0.002, 0.002, 17)
SIGMAS = np.logspace(-4, -1.5, 26)                     # paso 2: anchos probados

resultados = {'regiones_homogeneas': REGIONES_HOMOGENEAS, 'ventana_varianza': VENTANA_VARIANZA,
              'padding': ['simetrico', PADDING_TAM]}


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_FIG, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    if MOSTRAR:
        plt.show()
    plt.close(fig)
    print('  figura guardada ->', ruta)


# 1. Lo mismo que en la Pregunta 1 (lectura, notch Gaussiano y metricas)

def cargar_cr2(ruta):
    with rawpy.imread(ruta) as raw:
        rgb = raw.postprocess(use_camera_wb=True, output_bps=16)
    rgb = rgb.astype(np.float32) / 65535.0
    return (0.2125 * rgb[:, :, 0] + 0.7154 * rgb[:, :, 1] + 0.0721 * rgb[:, :, 2]).astype(np.float32)


def filtro_notch_gaussiano(forma, fy0, fx0, sigma):
    "H = (1 - exp(-D1^2/2s^2)) (1 - exp(-D2^2/2s^2)) para el notch y su simetrico"
    fy = np.fft.fftshift(np.fft.fftfreq(forma[0])).astype(np.float32)[:, None]
    fx = np.fft.fftshift(np.fft.fftfreq(forma[1])).astype(np.float32)[None, :]
    H = np.ones(forma, dtype=np.float32)
    for s in (1, -1):
        D2 = (fy - s * fy0) ** 2 + (fx - s * fx0) ** 2
        H *= 1.0 - np.exp(-D2 / (2.0 * sigma ** 2))
    return H


def amplitud_patron(imagen, fx=0.25):
    x = np.arange(imagen.shape[1])
    return float(2.0 * np.abs(np.mean(imagen * np.exp(-2j * np.pi * fx * x)[None, :])))


def parte_no_periodica(d, periodo=4, bloque=16):
    resto = np.zeros_like(d)
    for f0 in range(0, d.shape[0], bloque):
        b = d[f0:f0 + bloque]
        patron = np.array([b[:, j::periodo].mean() for j in range(periodo)])
        patron = np.tile(patron - patron.mean(), b.shape[1] // periodo + 1)[:b.shape[1]]
        resto[f0:f0 + bloque] = b - patron[None, :]
    return resto


def perdida_detalle(ruidosa, filtrada):
    f0, f1, c0, c1 = REG_NEBULOSA
    r = parte_no_periodica((ruidosa - filtrada)[f0:f1, c0:c1])
    return float(np.sqrt(np.mean((r - r.mean()) ** 2)))


def recorte(imagen, nombre):
    f, c, l = RECORTES[nombre]
    return imagen[f:f + l, c:c + l]


img = cargar_cr2(RUTA_IMG)
M, N = img.shape

# La FFT de la imagen extendida se calcula una sola vez. En cada evaluacion solo cambia H
ext = np.pad(img, PADDING_TAM, mode='symmetric')
F_ext = sfft.fftshift(sfft.fft2(ext, workers=-1))
forma_ext = ext.shape
del ext


def aplicar_notch(fx0, sigma):
    "Filtra con el notch Gaussiano en (FY0, fx0) y su simetrico, y recorta al tamaño original"
    H = filtro_notch_gaussiano(forma_ext, FY0, fx0, sigma)
    g = np.real(sfft.ifft2(sfft.ifftshift(F_ext * H), workers=-1))
    return g[PADDING_TAM:PADDING_TAM + M, PADDING_TAM:PADDING_TAM + N]


# 2. Funcion objetivo: varianza local media en las regiones homogeneas
#
# var_local = E[x^2] - E[x]^2, con E[.] = promedio movil de 7x7. Se promedia dentro de cada region
# y despues entre las tres regiones.

def varianza_local_media(imagen):
    vals = []
    for f0, f1, c0, c1 in REGIONES_HOMOGENEAS:
        a = imagen[f0:f1, c0:c1].astype(np.float64)
        m = uniform_filter(a, VENTANA_VARIANZA)
        m2 = uniform_filter(a * a, VENTANA_VARIANZA)
        vals.append(np.mean(m2 - m * m))
    return float(np.mean(vals))


J0 = varianza_local_media(img)
resultados['J_sin_filtrar'] = J0
print(f'J sin filtrar = {J0:.6e}')


# 3. Busqueda del centro (paso 1) y del ancho (paso 2)

J_c = np.array([varianza_local_media(aplicar_notch(fx0, SIGMA_CENTRO)) for fx0 in CENTROS])
fx_opt = float(CENTROS[np.argmin(J_c)])
print(f'Centro optimo fx0 = {fx_opt:.5f}')
resultados['busqueda_centro'] = {'sigma_fijo': SIGMA_CENTRO, 'centros': CENTROS.tolist(),
                                 'J': J_c.tolist(), 'fx0_optimo': fx_opt}

J_s, det_s, amp_s = [], [], []
print('sigma    | J            | perdida detalle | residual cielo')
for s in SIGMAS:
    f = aplicar_notch(fx_opt, s)
    J_s.append(varianza_local_media(f))
    det_s.append(perdida_detalle(img, f))
    f0, f1, c0, c1 = REGIONES_HOMOGENEAS[0]
    amp_s.append(amplitud_patron(f[f0:f1, c0:c1]))
    print(f'{s:.5f}  | {J_s[-1]:.6e} | {det_s[-1]:.5f}         | {amp_s[-1]:.5f}')
J_s, det_s, amp_s = np.array(J_s), np.array(det_s), np.array(amp_s)


# 4. Criterio adicional: centro de la zona plana
#
# Minimizar solo J siempre lleva al sigma mas grande del rango: al crecer sigma, H baja en todas
# las frecuencias y se quita energia sea o no ruido periodico. Por eso se elige el sigma donde la
# pendiente |dJ/d log10(sigma)| es minima (diferencias centradas): a la izquierda J baja porque se
# elimina el patron y a la derecha vuelve a bajar porque se empieza a quitar ruido aleatorio y detalle.

logs = np.log10(SIGMAS)
pendiente = np.full(len(J_s), np.nan)
for k in range(1, len(J_s) - 1):
    pendiente[k] = abs(J_s[k + 1] - J_s[k - 1]) / (logs[k + 1] - logs[k - 1])
k_mes = int(np.nanargmin(pendiente[1:-1])) + 1
k_min = int(np.argmin(J_s))
s_mes, s_min = float(SIGMAS[k_mes]), float(SIGMAS[k_min])
print(f'Solo varianza: sigma = {s_min:.5f} | criterio de zona plana: sigma = {s_mes:.5f}')
resultados['busqueda_ancho'] = {'sigmas': SIGMAS.tolist(), 'J': J_s.tolist(),
                                'perdida_detalle_nebulosa': det_s.tolist(), 'amp_residual_cielo': amp_s.tolist(),
                                'pendiente_abs': [None if np.isnan(v) else float(v) for v in pendiente],
                                'sigma_argmin_J': s_min, 'sigma_zona_plana': s_mes}

fig, ax = plt.subplots(1, 3, figsize=(17, 4.6))
ax[0].plot(CENTROS, J_c, 'k-o', ms=4, lw=1.3, label=f'J(fx0), σ fijo = {SIGMA_CENTRO}')
ax[0].axhline(J0, color='0.5', ls=':', lw=1, label='sin filtrar')
ax[0].axvline(fx_opt, color='tab:red', ls='--', lw=1.2, label=f'mínimo: fx0 = {fx_opt:.4f}')
ax[0].set_xlabel('centro del notch fx0 [ciclos/px]  (fy0 = 0)')
ax[0].set_ylabel('varianza local media J')
ax[0].set_title('Paso 1: búsqueda del centro')
ax[0].legend(fontsize=8)
ax[0].grid(alpha=0.3)
ax[1].plot(SIGMAS, J_s, 'k-o', ms=4, lw=1.3, label='J(σ)')
ax[1].axvline(s_min, color='tab:red', ls='--', lw=1.4, label=f'solo varianza: σ = {s_min:.4f}')
ax[1].axvline(s_mes, color='tab:blue', ls='-.', lw=1.4, label=f'criterio zona plana: σ = {s_mes:.5f}')
ax[1].set_xscale('log')
ax[1].set_xlabel('ancho del notch σ [ciclos/px]')
ax[1].set_ylabel('varianza local media J')
ax[1].set_title(f'Paso 2: J vs σ (fx0 = {fx_opt:.4f}); J sin filtrar = {J0:.3e}')
ax[1].legend(loc='lower left', fontsize=8)
ax[1].grid(alpha=0.3, which='both')
ax[2].plot(SIGMAS, pendiente, color='tab:blue', ls='-', marker='s', ms=4, label='|dJ / d log10 σ|')
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_xlabel('ancho del notch σ [ciclos/px]')
ax[2].set_ylabel('pendiente de J', color='tab:blue')
ax2 = ax[2].twinx()
ax2.plot(SIGMAS, det_s * 1000, color='tab:orange', ls='--', marker='^', ms=4, label='pérdida de detalle (nebulosa)')
ax2.plot(SIGMAS, amp_s * 1000, color='tab:green', ls=':', marker='o', ms=3, label='amplitud residual del patrón (cielo)')
ax2.set_ylabel('[×10⁻³]')
ax2.set_yscale('log')
ax[2].axvline(s_mes, color='tab:blue', ls='-.', lw=1.2)
ax[2].axvline(s_min, color='tab:red', ls='--', lw=1.2)
lineas = ax[2].get_lines()[:1] + ax2.get_lines()
ax[2].legend(lineas, [l.get_label() for l in lineas], loc='upper center', fontsize=7.5)
ax[2].set_title('Criterio adicional y costo de filtrar de más')
ax[2].grid(alpha=0.3, which='both')
plt.tight_layout()
guardar(fig, '01_funcion_objetivo.png')


# 5. Resultado con ambos criterios

f_mes = aplicar_notch(fx_opt, s_mes)
f_min = aplicar_notch(fx_opt, s_min)
casos = [('Ruidosa', img), (f'Criterio zona plana σ = {s_mes:.5f}', f_mes), (f'Solo varianza σ = {s_min:.4f}', f_min)]
fig, ax = plt.subplots(3, 4, figsize=(16, 12.5), gridspec_kw={'hspace': 0.3})
for i, (tit, im_) in enumerate(casos):
    for j, nombre in enumerate(['cielo', 'nebulosa']):
        r = recorte(img, nombre)
        ax[i, j].imshow(recorte(im_, nombre), cmap='gray', vmin=np.percentile(r, 2), vmax=np.percentile(r, 98),
                        interpolation='nearest')
        ax[i, j].set_title(f'{tit}\n({nombre})')
    if i == 0:
        ax[0, 2].imshow(downscale_local_mean(img, (4, 4)), cmap='gray', vmin=0, vmax=1)
        for f0, f1, c0, c1 in REGIONES_HOMOGENEAS:
            ax[0, 2].add_patch(Rectangle((c0 / 4, f0 / 4), (c1 - c0) / 4, (f1 - f0) / 4, fill=False, ec='tab:cyan', lw=1.5))
        f0, f1, c0, c1 = REG_NEBULOSA
        ax[0, 2].add_patch(Rectangle((c0 / 4, f0 / 4), (c1 - c0) / 4, (f1 - f0) / 4, fill=False, ec='y', ls='--', lw=1.2))
        ax[0, 2].set_title('Regiones homogéneas usadas en J (celeste)\ny región de detalle (amarillo)')
        ax[0, 3].text(0.0, 0.5, f'J sin filtrar: {J0:.4e}\n'
                      f'Criterio zona plana:  σ = {s_mes:.5f}, J = {J_s[k_mes]:.4e}\n'
                      f'Solo J:   σ = {s_min:.4f}, J = {J_s[k_min]:.4e}\n\n'
                      f'Pérdida de detalle (RMS no periódico):\n'
                      f'   zona plana {det_s[k_mes]:.5f}  vs  solo J {det_s[k_min]:.5f}\n'
                      f'Amplitud residual del patrón (cielo):\n'
                      f'   zona plana {amp_s[k_mes]:.5f}  vs  solo J {amp_s[k_min]:.5f}',
                      fontsize=10.5, va='center', family='monospace')
    else:
        d = recorte(img - im_, 'nebulosa')
        ax[i, 2].imshow(d, cmap='seismic', vmin=-0.04, vmax=0.04, interpolation='nearest')
        ax[i, 2].set_title('Diferencia nebulosa (±0.04)')
        h = ax[i, 3].imshow(parte_no_periodica(d), cmap='seismic', vmin=-0.006, vmax=0.006, interpolation='nearest')
        ax[i, 3].set_title('Parte no periódica de la diferencia (±0.006)')
        fig.colorbar(h, ax=ax[i, 3], fraction=0.046, pad=0.03)
for a in ax.ravel():
    a.axis('off')
fig.suptitle('Resultado de la selección automática del notch gaussiano', fontsize=12, y=0.93)
guardar(fig, '02_resultado.png')


with open(os.path.join(CARPETA_RES, 'bonus_resultados.json'), 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print('Fin. Los datos obtenidos se encuentran en resultados/bonus_resultados.json')
