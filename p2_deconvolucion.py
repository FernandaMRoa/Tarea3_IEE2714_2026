################################## Pregunta 2 ######################################
# Deconvolucion de Wiener y Tikhonov iterativo
#
# Se degrada la imagen ideal x (cameraman.jpg) con la PSF h (psf.png) y ruido Gaussiano, y se
# restaura con dos metodos que usan el mismo modelo de convolucion (circular, con la FFT):
#   - Parte A: Wiener simplificado       X^ = H* Y / (|H|^2 + K)
#   - Parte B: Tikhonov por descenso de gradiente
#              x^{k+1} = x^k - tau [ H^T (H x^k - y) + lambda x^k ]
# No se usa ninguna funcion externa de deconvolucion, restauracion ni optimizacion.
#
# Las imagenes deben estar en 'imagenes/cameraman.jpg' e 'imagenes/psf.png'.
# Las figuras generadas se guardan en 'figuras/p2/' y los datos en 'resultados/p2_resultados.json'

import os
import json
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from scipy import fft as sfft
from scipy.ndimage import rotate, zoom   # solo para construir las PSF perturbadas del punto 2
from skimage import io

RUTA_X = os.path.join('imagenes', 'cameraman.jpg')
RUTA_PSF = os.path.join('imagenes', 'psf.png')
CARPETA_FIG = os.path.join('figuras', 'p2')
CARPETA_RES = 'resultados'
os.makedirs(CARPETA_FIG, exist_ok=True)
os.makedirs(CARPETA_RES, exist_ok=True)

MOSTRAR = False
SEMILLA = 0
NIVELES_RUIDO = [0.005, 0.02, 0.05]     # desviacion estandar del ruido (imagen en [0, 1])
SIGMA_TIKHONOV = 0.02                   # nivel de ruido de los puntos 2, 3 y 4
KS = np.logspace(-6, 1, 141)            # barrido logaritmico de K (20 valores por decada)

ANGULO_ROTACION = 30.0                  # perturbaciones de la PSF (punto 2)
FACTOR_ESCALA = 1.2
DESPLAZAMIENTO = (1, 1)

TAUS = [1.0, 1.9, 2.05]                 # pasos del descenso (limite de estabilidad 2/(max|H|^2 + lambda))
MAX_ITER = 3000
TOLERANCIAS = [1e-2, 1e-3, 1e-4, 1e-5, 1e-6]

RECORTE = (40, 90, 80)                  # recorte ampliado (fila, col, lado): cara y camara

resultados = {}


def guardar(fig, nombre):
    ruta = os.path.join(CARPETA_FIG, nombre)
    fig.savefig(ruta, dpi=150, bbox_inches='tight')
    if MOSTRAR:
        plt.show()
    plt.close(fig)
    print('  figura guardada ->', ruta)


def rmse(referencia, estimada):
    return float(np.sqrt(np.mean((estimada - referencia) ** 2)))


def recorte(imagen):
    f, c, l = RECORTE
    return imagen[f:f + l, c:c + l]


def mostrar(a, imagen, titulo, marco=False):
    "Imagen en escala [0, 1] comun (se recorta a [0, 1] solo para mostrarla)"
    a.imshow(np.clip(imagen, 0, 1), cmap='gray', vmin=0, vmax=1, interpolation='nearest')
    a.set_title(titulo, fontsize=9)
    a.axis('off')
    if marco:
        f, c, l = RECORTE
        a.add_patch(Rectangle((c, f), l, l, fill=False, ec='tab:red', lw=1.2))


# 1. Imagen ideal, PSF y modelo de degradacion
#
# x se lleva a [0, 1] y la PSF se normaliza para que sume 1. La convencion de bordes es circular
# (periodica) en toda la pregunta: con ella la DFT diagonaliza exactamente a H, y la solucion
# cerrada y la iterativa resuelven el mismo problema.

x = io.imread(RUTA_X).astype(np.float64)
if x.ndim == 3:
    x = x[:, :, :3].mean(axis=2)          # los 3 canales del jpg son iguales
x = x / 255.0
h = io.imread(RUTA_PSF).astype(np.float64)
if h.ndim == 3:
    h = h[:, :, 0]
h = h / h.sum()
M, N = x.shape


def psf2otf(h, forma, centrar=True):
    """OTF H(u,v) de la PSF h para imagenes de tamaño 'forma':
    1) se rellena con ceros hasta el tamaño de la imagen (h queda en la esquina)
    2) se desplaza circularmente para que el CENTRO de h (kh//2, kw//2) quede en el pixel (0, 0),
       que es el origen que asume la FFT
    3) FFT 2D (rfft2: la imagen es real y basta la mitad del espectro)"""
    kh, kw = h.shape
    hp = np.zeros(forma)
    hp[:kh, :kw] = h                                              # ajuste de tamaño
    if centrar:
        hp = np.roll(hp, (-(kh // 2), -(kw // 2)), axis=(0, 1))   # centro -> (0, 0)
    return sfft.rfft2(hp)


def aplicar_H(img, H):
    "H x = h * x (convolucion circular): multiplicar por H en frecuencia"
    return sfft.irfft2(sfft.rfft2(img) * H, s=img.shape)


def aplicar_HT(img, H):
    "H^T x = correlacion con h = convolucion con h(-n): multiplicar por conj(H)"
    return sfft.irfft2(sfft.rfft2(img) * np.conj(H), s=img.shape)


def degradar(img, H, sigma, semilla=SEMILLA):
    "y = h * x + n, con el mismo modelo circular que se usa para restaurar"
    rng = np.random.default_rng(semilla)
    return aplicar_H(img, H) + sigma * rng.standard_normal(img.shape)


def otf_completa(H, forma):
    "De la rfft2 (mitad del espectro) al espectro completo centrado (solo para mostrar)"
    return np.fft.fftshift(np.fft.fft2(sfft.irfft2(H, s=forma)))


H = psf2otf(h, x.shape)
asim = np.abs(h - h[::-1, ::-1]).max() / h.max()
print(f'Imagen {M}x{N}, PSF {h.shape}, |H| min = {np.abs(H).min():.2e}, max = {np.abs(H).max():.4f}')
print(f'Fraccion de frecuencias con |H|^2 < 1e-3: {np.mean(np.abs(H) ** 2 < 1e-3):.3f}')
print(f'Asimetria de la PSF max|h(n) - h(-n)| = {100 * asim:.2f} % del maximo')
resultados['datos'] = {'forma': [M, N], 'psf_forma': list(h.shape), 'H_min': float(np.abs(H).min()),
                       'H_max': float(np.abs(H).max()), 'asimetria_psf_rel': float(asim),
                       'max_imag_H': float(np.abs(np.imag(H)).max()),
                       'frac_H2_menor_1e-3': float(np.mean(np.abs(H) ** 2 < 1e-3))}

ys = {s: degradar(x, H, s) for s in NIVELES_RUIDO}
resultados['rmse_degradada'] = {str(s): rmse(x, ys[s]) for s in NIVELES_RUIDO}

fig, ax = plt.subplots(1, 3 + len(NIVELES_RUIDO), figsize=(19, 4.2))
mostrar(ax[0], x, f'Imagen ideal x ({M}×{N})', marco=True)
im = ax[1].imshow(h, cmap='inferno', interpolation='nearest')
ax[1].plot(h.shape[1] // 2, h.shape[0] // 2, 'c+', ms=10)
ax[1].set_title(f'PSF h ({h.shape[0]}×{h.shape[1]}, suma 1)', fontsize=9)
ax[1].axis('off')
plt.colorbar(im, ax=ax[1], fraction=0.046, pad=0.03)
im = ax[2].imshow(np.log10(np.abs(otf_completa(H, x.shape)) + 1e-12), cmap='viridis', vmin=-6, vmax=0)
ax[2].set_title('log10 |H(u,v)| (centrada)', fontsize=9)
ax[2].axis('off')
plt.colorbar(im, ax=ax[2], fraction=0.046, pad=0.03)
for a, s in zip(ax[3:], NIVELES_RUIDO):
    mostrar(a, ys[s], f'Degradada y, σn = {s}\nRMSE = {rmse(x, ys[s]):.4f}')
plt.tight_layout()
guardar(fig, '00_datos.png')


############### 2. Parte A: filtro de Wiener y barrido de K (punto 1) ##########################
#
# Para cada nivel de ruido se barre K en escala logaritmica, se busca el K que minimiza el RMSE y
# se comparan las reconstrucciones con K bajo (K*/100), optimo (K*) y alto (100 K*).

def wiener(y, H, K):
    "X^ = H* Y / (|H|^2 + K). Con K = 0 es la inversion directa Y / H"
    Y = sfft.rfft2(y)
    den = np.abs(H) ** 2 + K
    den = np.where(den == 0, np.finfo(float).tiny, den)     # por si |H| = 0 exacto
    return sfft.irfft2(np.conj(H) * Y / den, s=y.shape)


def barrido_K(y, H, Ks=KS):
    return np.array([rmse(x, wiener(y, H, K)) for K in Ks])


curvas, K_opt = {}, {}
resultados['wiener'] = {}
print('sigma_n | RMSE degradada | K optimo | RMSE Wiener | RMSE inversion directa')
for s in NIVELES_RUIDO:
    curvas[s] = barrido_K(ys[s], H)
    i = int(np.argmin(curvas[s]))
    K_opt[s] = float(KS[i])
    r_inv = rmse(x, wiener(ys[s], H, 0.0))
    resultados['wiener'][str(s)] = {'K_opt': K_opt[s], 'rmse_opt': float(curvas[s][i]),
                                    'rmse_degradada': rmse(x, ys[s]), 'rmse_inversion_directa': r_inv,
                                    'K_opt_sobre_sigma2': K_opt[s] / s ** 2}
    print(f'{s:7} | {rmse(x, ys[s]):.4f}         | {K_opt[s]:.2e} | {curvas[s][i]:.4f}      | {r_inv:.1f}')

fig, ax = plt.subplots(1, 2, figsize=(15, 4.8), gridspec_kw={'width_ratios': [1.4, 1]})
estilos = [dict(color='tab:blue', ls='-', marker='o', markevery=10, ms=4),
           dict(color='tab:orange', ls='--', marker='s', markevery=10, ms=4),
           dict(color='tab:green', ls='-.', marker='^', markevery=10, ms=4)]
for s, est in zip(NIVELES_RUIDO, estilos):
    i = int(np.argmin(curvas[s]))
    ax[0].plot(KS, curvas[s], lw=1.6, label=f'σn = {s}', **est)
    ax[0].plot(KS[i], curvas[s][i], '*', color=est['color'], ms=15, mec='k', zorder=5)
    ax[0].annotate(f'K* = {KS[i]:.2e}\nRMSE = {curvas[s][i]:.4f}', (KS[i], curvas[s][i]),
                   textcoords='offset points', xytext=(-10, -38), fontsize=7.5, color=est['color'])
    ax[0].axhline(rmse(x, ys[s]), color=est['color'], ls=':', lw=1)
ax[0].set_xscale('log')
ax[0].set_ylim(0.04, 0.16)
ax[0].set_xlabel('K (escala log)')
ax[0].set_ylabel('RMSE respecto de x')
ax[0].set_title('Wiener: RMSE vs K (★ óptimo; punteado: RMSE de la degradada)')
ax[0].grid(alpha=0.3, which='both')
ax[0].legend()
corte = np.abs(otf_completa(H, x.shape))[M // 2, :]
ax[1].semilogy(np.arange(N) - N // 2, corte ** 2, 'k-', lw=1.8, label='|H(0,v)|²')
for s, est in zip(NIVELES_RUIDO, estilos):
    ax[1].axhline(K_opt[s], color=est['color'], ls=est['ls'], lw=1.3, label=f'K* (σn = {s})')
ax[1].set_ylim(1e-10, 2)
ax[1].set_xlabel('frecuencia v (fila u = 0)')
ax[1].set_title('Donde |H|² < K, Wiener deja de invertir')
ax[1].grid(alpha=0.3, which='both')
ax[1].legend(fontsize=7.5)
plt.tight_layout()
guardar(fig, '01_wiener_rmse_vs_K.png')

fig, ax = plt.subplots(2 * len(NIVELES_RUIDO), 4, figsize=(13, 20), gridspec_kw={'hspace': 0.28})
for i, s in enumerate(NIVELES_RUIDO):
    y = ys[s]
    casos = [('Degradada', y), (f'K = K*/100 = {K_opt[s] / 100:.1e}', wiener(y, H, K_opt[s] / 100)),
             (f'K = K* = {K_opt[s]:.1e}', wiener(y, H, K_opt[s])),
             (f'K = 100 K* = {100 * K_opt[s]:.1e}', wiener(y, H, 100 * K_opt[s]))]
    for j, (tit, im_) in enumerate(casos):
        mostrar(ax[2 * i, j], im_, f'σn = {s} · {tit}\nRMSE = {rmse(x, im_):.4f}', marco=(j == 0))
        mostrar(ax[2 * i + 1, j], recorte(im_), 'recorte ampliado')
fig.suptitle('Wiener con K bajo, óptimo y alto (escala de grises [0, 1] común)', fontsize=12, y=0.925)
guardar(fig, '02_wiener_reconstrucciones.png')


############### 3. Wiener con una PSF ligeramente incorrecta (punto 2) ##########################
#
# Con sigma_n = 0.02 se restaura la misma imagen degradada (hecha con la PSF correcta) usando
# PSF perturbadas: rotada, mas ancha, desplazada 1 px y sin el roll de centrado (error de 7 px).

s = SIGMA_TIKHONOV
y = ys[s]
h_rot = np.clip(rotate(h, ANGULO_ROTACION, reshape=False, order=1), 0, None)
h_rot /= h_rot.sum()
h_esc = zoom(h, FACTOR_ESCALA, order=1)
a0 = (h_esc.shape[0] - h.shape[0]) // 2
h_esc = np.clip(h_esc[a0:a0 + h.shape[0], a0:a0 + h.shape[1]], 0, None)
h_esc /= h_esc.sum()
h_des = np.roll(h, DESPLAZAMIENTO, axis=(0, 1))
psfs = {'correcta': (h, psf2otf(h, x.shape)),
        f'rotada {ANGULO_ROTACION:.0f}°': (h_rot, psf2otf(h_rot, x.shape)),
        f'escalada ×{FACTOR_ESCALA}': (h_esc, psf2otf(h_esc, x.shape)),
        f'desplazada {DESPLAZAMIENTO} px': (h_des, psf2otf(h_des, x.shape)),
        'sin centrar (sin roll)': (h, psf2otf(h, x.shape, centrar=False))}
resultados['psf_incorrecta'] = {}
curvas_p, recs = {}, {}
print('PSF usada                | ||dh||/||h|| | K*       | RMSE (K* propio) | RMSE (K* correcta)')
for nombre, (hh, HH) in psfs.items():
    cur = barrido_K(y, HH)
    i = int(np.argmin(cur))
    curvas_p[nombre] = cur
    recs[nombre] = wiener(y, HH, KS[i])
    rec_kc = wiener(y, HH, K_opt[s])
    err = float(np.linalg.norm(hh - h) / np.linalg.norm(h))
    resultados['psf_incorrecta'][nombre] = {'K_opt': float(KS[i]), 'rmse_opt': float(cur[i]),
                                            'rmse_con_K_de_la_correcta': rmse(x, rec_kc), 'error_rel_psf': err}
    print(f'{nombre:24s} | {err:.3f}        | {KS[i]:.2e} | {cur[i]:.4f}           | {rmse(x, rec_kc):.4f}')

fig = plt.figure(figsize=(17, 10.8))
gs = fig.add_gridspec(3, 5, height_ratios=[0.8, 1, 1], hspace=0.32)
c = h.shape[0] // 2
for j, nombre in enumerate(psfs):
    a = fig.add_subplot(gs[0, j])
    a.imshow(psfs[nombre][0], cmap='inferno', interpolation='nearest', vmin=0, vmax=h.max())
    a.plot(c, c, 'c+', ms=10)
    a.set_title(f'PSF {nombre}', fontsize=9)
    a.axis('off')
    info = resultados['psf_incorrecta'][nombre]
    mostrar(fig.add_subplot(gs[1, j]), recs[nombre],
            f'Wiener con PSF {nombre}\nK* = {info["K_opt"]:.1e}, RMSE = {info["rmse_opt"]:.4f}', marco=True)
    mostrar(fig.add_subplot(gs[2, j]), recorte(recs[nombre]), 'recorte ampliado')
fig.suptitle(f'Wiener con PSF correcta y perturbada (σn = {s}); cada una con su K óptimo; '
             f'+ celeste: centro esperado de la PSF', fontsize=12, y=0.95)
guardar(fig, '03a_psf_incorrecta.png')

fig, ax = plt.subplots(figsize=(9, 4.8))
est_p = [dict(color='k', ls='-', lw=2.2), dict(color='tab:red', ls='--', lw=1.6),
         dict(color='tab:purple', ls='-.', lw=1.6), dict(color='tab:green', ls=':', lw=2.2),
         dict(color='tab:brown', ls=(0, (5, 2, 1, 2)), lw=1.4)]
for nombre, est in zip(psfs, est_p):
    i = int(np.argmin(curvas_p[nombre]))
    ax.plot(KS, curvas_p[nombre], label=f'PSF {nombre}', **est)
    ax.plot(KS[i], curvas_p[nombre][i], '*', color=est['color'], ms=13, mec='k', zorder=5)
ax.axhline(rmse(x, y), color='0.5', ls=':', lw=1, label='degradada')
ax.set_xscale('log')
ax.set_ylim(0.05, 0.2)
ax.set_xlabel('K')
ax.set_ylabel('RMSE')
ax.set_title(f'RMSE vs K con PSF correcta y perturbada (σn = {s})')
ax.grid(alpha=0.3, which='both')
ax.legend(fontsize=7.5)
plt.tight_layout()
guardar(fig, '03b_psf_incorrecta_curvas.png')


############### 4. Parte B: Tikhonov por descenso de gradiente (punto 3) ##########################
#
#   J(x) = 1/2 ||h*x - y||^2 + lambda/2 ||x||^2
#   gradiente = H^T (H x - y) + lambda x      (termino de fidelidad + termino de regularizacion)
# Se parte desde x^0 = y. El criterio de detencion es el cambio relativo
# ||x^{k+1} - x^k|| / ||x^k|| < tol, y ademas se corta si el costo diverge (> 1e12).
# Como max|H|^2 = 1, el descenso es estable solo si tau < 2 / (1 + lambda).

def costo_tikhonov(img, y, H, lam):
    r = aplicar_H(img, H) - y
    return 0.5 * np.sum(r ** 2) + 0.5 * lam * np.sum(img ** 2)


def tikhonov_gd(y, H, lam, tau, max_iter=MAX_ITER, tol=None, x_ref=None, guardar_mejor=False):
    "Descenso de gradiente. Registra por iteracion el costo, el RMSE y el cambio relativo"
    xk = y.copy()
    hist = {'costo': [], 'rmse': [], 'cambio': []}
    mejor = {'rmse': np.inf, 'iter': 0, 'x': xk.copy()}
    k_det = max_iter
    for k in range(max_iter):
        r = aplicar_H(xk, H) - y                         # residuo H x^k - y
        hist['costo'].append(0.5 * np.sum(r ** 2) + 0.5 * lam * np.sum(xk ** 2))
        if x_ref is not None:
            e = rmse(x_ref, xk)
            hist['rmse'].append(e)
            if guardar_mejor and e < mejor['rmse']:
                mejor = {'rmse': e, 'iter': k, 'x': xk.copy()}
        grad = aplicar_HT(r, H) + lam * xk               # fidelidad + regularizacion
        x_nuevo = xk - tau * grad
        cambio = np.linalg.norm(x_nuevo - xk) / max(np.linalg.norm(xk), 1e-12)
        hist['cambio'].append(float(cambio))
        xk = x_nuevo
        if not np.isfinite(hist['costo'][-1]) or hist['costo'][-1] > 1e12:
            k_det = k + 1                                # diverge: se corta
            break
        if tol is not None and cambio < tol:             # decision de detenerse
            k_det = k + 1
            break
    hist = {c_: np.array(v) for c_, v in hist.items()}
    hist['iter_detencion'] = k_det
    hist['mejor'] = mejor
    return xk, hist


K = K_opt[s]
lambdas = [('λ = 0', 0.0), ('λ = K*/10', K / 10), ('λ = K*', K), ('λ = 10 K*', 10 * K)]
resultados['tikhonov'] = {'K_opt': K, 'runs': []}
historias = {}
print('lambda     | tau  | tau limite | iters | RMSE final | RMSE min (k)')
for nom_l, lam in lambdas:
    for tau in TAUS:
        xk, hist = tikhonov_gd(y, H, lam, tau, MAX_ITER, x_ref=x, guardar_mejor=True)
        historias[(lam, tau)] = hist
        lim = 2.0 / (np.abs(H).max() ** 2 + lam)
        info = {'lambda': lam, 'tau': tau, 'tau_limite': lim, 'iteraciones': int(hist['iter_detencion']),
                'costo_final': float(hist['costo'][-1]), 'rmse_final': float(hist['rmse'][-1]),
                'rmse_min': float(hist['mejor']['rmse']), 'iter_rmse_min': int(hist['mejor']['iter']),
                'diverge': bool(hist['iter_detencion'] < MAX_ITER)}
        if lam > 0 and not info['diverge']:
            xc = wiener(y, H, lam)
            info['dif_rel_con_cerrada'] = float(np.linalg.norm(xk - xc) / np.linalg.norm(xc))
        resultados['tikhonov']['runs'].append(info)
        final = 'diverge' if info['diverge'] else f'{info["rmse_final"]:.4f}'
        print(f'{nom_l:10s} | {tau:<4} | {lim:.3f}      | {info["iteraciones"]:5d} | {final:10s} | '
              f'{info["rmse_min"]:.4f} ({info["iter_rmse_min"]})')

fig, ax = plt.subplots(2, len(lambdas), figsize=(18, 8.4))
est_tau = {1.0: dict(color='tab:blue', ls='-', lw=1.8), 1.9: dict(color='tab:orange', ls='--', lw=1.8),
           2.05: dict(color='tab:red', ls=':', lw=2.2)}
for j, (nom_l, lam) in enumerate(lambdas):
    for tau in TAUS:
        hist = historias[(lam, tau)]
        it = np.arange(1, len(hist['costo']) + 1)
        ax[0, j].plot(it, hist['costo'], label=f'τ = {tau}', **est_tau[tau])
        ax[1, j].plot(it, hist['rmse'], label=f'τ = {tau}', **est_tau[tau])
    if lam > 0:
        xc = wiener(y, H, lam)
        ax[0, j].axhline(costo_tikhonov(xc, y, H, lam), color='k', ls='-.', lw=1, label='costo sol. cerrada')
        ax[1, j].axhline(rmse(x, xc), color='k', ls='-.', lw=1, label='RMSE sol. cerrada')
    ax[1, j].axhline(resultados['wiener'][str(s)]['rmse_opt'], color='0.5', ls=':', lw=1, label='Wiener K*')
    ax[0, j].set_title(f'{nom_l} = {lam:.2e}   (τ límite = {2.0 / (1 + lam):.3f})')
    for i in range(2):
        ax[i, j].set_xscale('log')
        ax[i, j].set_yscale('log')
        ax[i, j].grid(alpha=0.3, which='both')
        ax[i, j].set_xlabel('iteración k')
        ax[i, j].legend(fontsize=7)
    ax[0, j].set_ylim(top=1e4)
    ax[1, j].set_ylim(0.06, 0.3)
    ax[0, j].set_ylabel('costo J(x^k)')
    ax[1, j].set_ylabel('RMSE(x^k)')
fig.suptitle(f'Descenso de gradiente para Tikhonov (σn = {s}, x⁰ = y, {MAX_ITER} iteraciones máx.)', fontsize=12)
plt.tight_layout()
guardar(fig, '04_tikhonov_convergencia.png')

# Efecto del criterio de detencion (lambda = K*, tau = 1.0 y 1.9)
xc = wiener(y, H, K)
resultados['detencion'] = []
print('Criterio de detencion (lambda = K*):')
print('tau | tol   | iters | RMSE    | ||x - x_cerrada|| / ||x_cerrada||')
for tau in [1.0, 1.9]:
    for tol in TOLERANCIAS:
        xk, hist = tikhonov_gd(y, H, K, tau, max_iter=20000, tol=tol, x_ref=x)
        d = {'tau': tau, 'tol': tol, 'iteraciones': int(hist['iter_detencion']), 'rmse': rmse(x, xk),
             'dif_rel_con_cerrada': float(np.linalg.norm(xk - xc) / np.linalg.norm(xc)),
             'prediccion_tol_sobre_tau_lambda': tol / (tau * K)}
        resultados['detencion'].append(d)
        print(f'{tau} | {tol:.0e} | {d["iteraciones"]:5d} | {d["rmse"]:.5f} | {d["dif_rel_con_cerrada"]:.2e}')

fig, ax = plt.subplots(1, 2, figsize=(15, 4.6))
for tau in [1.0, 1.9]:
    est = est_tau[tau]
    cambio = historias[(K, tau)]['cambio']
    ax[0].plot(np.arange(1, len(cambio) + 1), cambio, label=f'τ = {tau}', **est)
    datos = [r for r in resultados['detencion'] if r['tau'] == tau]
    ax[1].plot([r['iteraciones'] for r in datos], [r['dif_rel_con_cerrada'] for r in datos],
               marker='o' if tau == 1.0 else 's', label=f'τ = {tau}', **est)
    for r in datos:
        ax[1].annotate(f'tol={r["tol"]:.0e}', (r['iteraciones'], r['dif_rel_con_cerrada']),
                       textcoords='offset points', xytext=(5, 3), fontsize=7, color=est['color'])
for tol, estilo in zip(TOLERANCIAS, ['-', '--', '-.', ':', (0, (1, 1))]):
    ax[0].axhline(tol, color='0.4', ls=estilo, lw=0.9)
    ax[0].text(1.2, tol * 1.15, f'tol = {tol:.0e}', fontsize=7, color='0.3')
for a in ax:
    a.set_xscale('log')
    a.set_yscale('log')
    a.grid(alpha=0.3, which='both')
    a.legend()
ax[0].set_xlabel('iteración k')
ax[0].set_ylabel('||x^{k+1} - x^k|| / ||x^k||')
ax[0].set_title(f'Cambio relativo por iteración (λ = K* = {K:.2e})')
ax[1].set_xlabel('iteración en que se detiene')
ax[1].set_ylabel('||x_det - x_cerrada|| / ||x_cerrada||')
ax[1].set_title('Distancia a la solución cerrada al detenerse')
plt.tight_layout()
guardar(fig, '05_criterio_detencion.png')


############### 5. Solucion iterativa vs solucion cerrada (punto 4) ##########################
#
# lambda = K*: descenso con tau = 1.9 y tol = 1e-8, comparado con Wiener/Tikhonov cerrado.
# lambda = 0: la "solucion iterativa optima" es el iterado de menor RMSE. La cerrada es Y/H.

tau = 1.9
x_gd_K, h_K = tikhonov_gd(y, H, K, tau, max_iter=20000, tol=1e-8, x_ref=x, guardar_mejor=True)
x_cer_K = wiener(y, H, K)
x_gd_0, h_0 = tikhonov_gd(y, H, 0.0, tau, max_iter=MAX_ITER, x_ref=x, guardar_mejor=True)
x_gd_0_mejor, k0 = h_0['mejor']['x'], int(h_0['mejor']['iter'])
x_cer_0 = wiener(y, H, 0.0)
resultados['comparacion'] = {
    'lambda_K': {'lambda': K, 'tau': tau, 'iteraciones': int(h_K['iter_detencion']),
                 'rmse_iterativa_final': rmse(x, x_gd_K), 'rmse_iterativa_mejor': float(h_K['mejor']['rmse']),
                 'iter_mejor': int(h_K['mejor']['iter']), 'rmse_cerrada': rmse(x, x_cer_K),
                 'max_abs_dif': float(np.abs(x_gd_K - x_cer_K).max()),
                 'dif_rel': float(np.linalg.norm(x_gd_K - x_cer_K) / np.linalg.norm(x_cer_K))},
    'lambda_0': {'lambda': 0.0, 'tau': tau, 'iteraciones': MAX_ITER,
                 'rmse_iterativa_mejor': float(h_0['mejor']['rmse']), 'iter_mejor': k0,
                 'rmse_iterativa_final': rmse(x, x_gd_0), 'rmse_cerrada': rmse(x, x_cer_0),
                 'max_abs_cerrada': float(np.abs(x_cer_0).max())}}
print('Comparacion iterativo vs cerrado:', resultados['comparacion'])

fig = plt.figure(figsize=(18, 7.4))
gs = fig.add_gridspec(2, 6, hspace=0.25)
casos = [('Ideal x', x), (f'Degradada (σn = {s})', y),
         (f'Wiener / Tikhonov cerrado\nK = λ = {K:.2e}', x_cer_K),
         (f'Tikhonov iterativo λ = K\n({h_K["iter_detencion"]} it., τ = {tau})', x_gd_K),
         (f'Iterativo λ = 0, mejor iterado\n(k = {k0})', x_gd_0_mejor),
         ('Cerrado λ = 0 (Y/H)\nmostrado recortado a [0, 1]', x_cer_0)]
for j, (tit, im_) in enumerate(casos):
    mostrar(fig.add_subplot(gs[0, j]), im_, f'{tit}\nRMSE = {rmse(x, im_):.4g}', marco=True)
    mostrar(fig.add_subplot(gs[1, j]), recorte(im_), 'recorte ampliado')
fig.suptitle('Mejores reconstrucciones de ambos métodos (escala [0, 1] común)', fontsize=12)
guardar(fig, '06_comparacion_final.png')

fig, ax = plt.subplots(1, 3, figsize=(16, 4.4))
dif = x_gd_K - x_cer_K
lim = max(np.abs(dif).max(), 1e-12)
im = ax[0].imshow(dif, cmap='seismic', vmin=-lim, vmax=lim)
ax[0].set_title(f'Iterativo - cerrado (λ = K)\nmáx |dif| = {np.abs(dif).max():.1e}')
ax[0].axis('off')
plt.colorbar(im, ax=ax[0], fraction=0.046, pad=0.03)
it = np.arange(1, len(h_K['rmse']) + 1)
ax[1].plot(it, h_K['rmse'], color='tab:blue', ls='-', lw=1.8, label='iterativo λ = K')
ax[1].axhline(rmse(x, x_cer_K), color='k', ls='--', lw=1.2, label='cerrado λ = K')
ax[1].plot(np.arange(1, len(h_0['rmse']) + 1), h_0['rmse'], color='tab:red', ls='-.', lw=1.8, label='iterativo λ = 0')
ax[1].plot(k0 + 1, h_0['mejor']['rmse'], 'r*', ms=13, mec='k', label='mejor iterado λ = 0')
ax[1].set_xscale('log')
ax[1].set_xlabel('iteración k')
ax[1].set_ylabel('RMSE')
ax[1].set_ylim(0.06, 0.14)
ax[1].set_title(f'RMSE por iteración (τ = {tau})')
ax[1].grid(alpha=0.3, which='both')
ax[1].legend()
ax[2].plot(it, h_K['costo'] - costo_tikhonov(x_cer_K, y, H, K), color='tab:blue', ls='-', lw=1.8)
ax[2].set_xscale('log')
ax[2].set_yscale('log')
ax[2].set_xlabel('iteración k')
ax[2].set_ylabel('J(x^k) - J(x_cerrada)')
ax[2].set_title('Brecha de costo respecto de la solución cerrada (λ = K)')
ax[2].grid(alpha=0.3, which='both')
plt.tight_layout()
guardar(fig, '07_iterativo_vs_cerrado.png')


############# 6. Preguntas guiadas: implementacion de H^T #######################
#
# Prueba del adjunto <H u, v> = <u, H^T v> con u, v aleatorios. Ademas, con una PSF no simetrica
# (movimiento hacia un solo lado) se compara el descenso con H^T correcto y usando H en su lugar.

rng = np.random.default_rng(1)
u_, v_ = rng.standard_normal(x.shape), rng.standard_normal(x.shape)
izq = float(np.sum(aplicar_H(u_, H) * v_))
der = float(np.sum(u_ * aplicar_HT(v_, H)))
der_mal = float(np.sum(u_ * aplicar_H(v_, H)))           # usando H en vez de H^T

h_as = np.zeros((15, 15))
h_as[7, 7:15] = 1.0                                     # movimiento de 8 px hacia la derecha
h_as /= h_as.sum()
H_as = psf2otf(h_as, x.shape)
y_as = degradar(x, H_as, s)
x_bien, hb = tikhonov_gd(y_as, H_as, K, 1.9, max_iter=1500, x_ref=x)
xm = y_as.copy()                                        # version incorrecta: H en vez de H^T
rm = []
for k in range(1500):
    g = aplicar_H(aplicar_H(xm, H_as) - y_as, H_as) + K * xm
    xm = xm - 1.9 * g
    rm.append(rmse(x, xm))
    if not np.isfinite(rm[-1]) or rm[-1] > 1e6:
        break
x_cer_as = wiener(y_as, H_as, K)
resultados['adjunto'] = {'<Hu,v>': izq, '<u,H^T v>': der, '<u,H v>': der_mal,
                         'psf_real_dif_rel_HT_vs_H': abs(der - der_mal) / abs(der),
                         'asim_rmse_HT_correcta': rmse(x, x_bien), 'asim_rmse_cerrada': rmse(x, x_cer_as),
                         'asim_rmse_con_H_en_vez_de_HT': float(rm[-1]), 'asim_iter_con_H': len(rm)}
print('Prueba del adjunto y PSF no simetrica:', resultados['adjunto'])

fig, ax = plt.subplots(1, 4, figsize=(17, 4.2))
ax[0].imshow(h_as, cmap='inferno', interpolation='nearest')
ax[0].plot(7, 7, 'c+', ms=10)
ax[0].set_title('PSF no simétrica (movimiento\nhacia un lado, 8 px)', fontsize=9)
ax[0].axis('off')
mostrar(ax[1], x_bien, f'Gradiente con H^T = conj(H)\nRMSE = {rmse(x, x_bien):.4f}')
mostrar(ax[2], xm, f'Gradiente usando H en vez de H^T\nRMSE = {rm[-1]:.4g}')
ax[3].plot(np.arange(1, len(hb['rmse']) + 1), hb['rmse'], 'b-', lw=1.8, label='con H^T (correcto)')
ax[3].plot(np.arange(1, len(rm) + 1), rm, 'r--', lw=1.8, label='con H (incorrecto)')
ax[3].axhline(rmse(x, x_cer_as), color='k', ls=':', lw=1.2, label='solución cerrada')
ax[3].set_xscale('log')
ax[3].set_yscale('log')
ax[3].set_xlabel('iteración k')
ax[3].set_ylabel('RMSE')
ax[3].set_title(f'λ = {K:.2e}, τ = 1.9')
ax[3].grid(alpha=0.3, which='both')
ax[3].legend()
plt.tight_layout()
guardar(fig, '08_HT_psf_no_simetrica.png')


# Guardado de todas las metricas
with open(os.path.join(CARPETA_RES, 'p2_resultados.json'), 'w', encoding='utf-8') as f:
    json.dump(resultados, f, indent=2, ensure_ascii=False)
print('Fin. Los datos obtenidos se encuentran en resultados/p2_resultados.json')
