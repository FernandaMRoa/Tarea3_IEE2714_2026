# Tarea 3 — IEE2714 Fundamentos de Procesamiento de Imágenes (2° 2026)

**Filtrado en frecuencia y deconvolución regularizada**
María Fernanda Muñoz — Pontificia Universidad Católica de Chile

Este repositorio contiene el código completo de la Tarea 3. Cada pregunta está resuelta en un
script independiente de Python que genera todas las figuras y los valores numéricos usados en el
informe.

## Estructura

```
Tarea3_IEE2714_2026/
├── p1_filtro_notch.py           Pregunta 1: eliminación de ruido periódico con filtros notch
├── p2_deconvolucion.py          Pregunta 2: Wiener y Tikhonov por descenso de gradiente
├── bonus_notch_adaptativo.py    Bonus (P1): selección automática de los parámetros del notch
├── requirements.txt             dependencias
├── imagenes/                    imágenes de entrada
│   ├── P1.CR2                   imagen RAW con ruido periódico (Pregunta 1 y Bonus)
│   ├── cameraman.jpg            imagen ideal x (Pregunta 2)
│   └── psf.png                  PSF h de 15x15 (Pregunta 2)
├── figuras/                     figuras generadas (p1/, p2/, bonus/)
└── resultados/                  métricas generadas (p1_resultados.json, p2_resultados.json, bonus_resultados.json)
```

## Instalación

Se probó con Python 3.10 o superior.

```bash
pip install -r requirements.txt
```

`rawpy` se usa solo para leer el archivo RAW `P1.CR2`. La FFT/IFFT es la de `numpy`/`scipy`; no se
usan funciones externas de filtros notch, deconvolución de Wiener, Tikhonov, restauración ni
optimizadores.

## Ejecución y reproducción de las figuras

Los scripts usan rutas relativas, así que deben ejecutarse **desde la carpeta raíz del repositorio**:

```bash
python p1_filtro_notch.py
python p2_deconvolucion.py
python bonus_notch_adaptativo.py
```

Cada script:
- guarda sus figuras en `figuras/p1/`, `figuras/p2/` o `figuras/bonus/` (no se muestran en pantalla;
  para verlas, cambiar `MOSTRAR = True` al inicio del script),
- guarda todos los números citados en el informe en `resultados/*.json`,
- imprime en la consola las tablas principales.

Los resultados son reproducibles: la Pregunta 1 no tiene componentes aleatorias y en la Pregunta 2
el ruido se genera con semilla fija (`SEMILLA = 0`). Tiempo aproximado: P1 ~30 s, P2 ~1 min,
Bonus ~30 s.

## Pregunta 1 — Filtrado notch (`p1_filtro_notch.py`)

La imagen `P1.CR2` (nebulosa de Orión) se lee con `rawpy` y se pasa a escala de grises. Tiene un
patrón de franjas verticales que se repite cada 4 columnas, que aparece en el espectro como dos
picos en (fy, fx) = (0, ±0.25) ciclos/píxel.

Cadena implementada: padding configurable → FFT 2D centrada → función de transferencia notch reject
→ multiplicación del espectro complejo → IFFT → recorte al tamaño original.

Parámetros configurables al inicio del script:

| Parámetro | Descripción | Valor final |
|---|---|---|
| `NOTCHES` | centros (fy, fx) en ciclos/píxel; el simétrico se agrega solo | `[(0.0, 0.25)]` |
| `PADDING_MODO` | `'ninguno'`, `'ceros'`, `'simetrico'`, `'reflejo'`, `'replicar'` | `'simetrico'` |
| `PADDING_TAM` | píxeles agregados en cada borde | `128` |
| `TIPO_FINAL`, `ANCHO_FINAL` | familia (`'ideal'` o `'gaussiano'`) y ancho en ciclos/píxel | `'gaussiano'`, `0.002` |

| Figura | Contenido |
|---|---|
| `00_imagen_original.png` | imagen y recortes ampliados |
| `01_espectro.png` | espectro de magnitud e identificación del ruido (punto 1) |
| `02_mascaras.png` | funciones de transferencia ideal y Gaussiana |
| `03a_anchos_ideal.png`, `03b_anchos_gaussiano.png`, `04_metricas_anchos.png` | barrido de anchos (punto 2) |
| `05a_padding_mapas.png`, `05b_padding_perfiles.png`, `05c_padding_recortes.png` | comparación de padding (punto 3) |
| `06_final.png` | configuración final (punto 4) |
| `07_un_solo_notch.png` | pregunta guiada: eliminar solo un notch del par |

## Pregunta 2 — Deconvolución (`p2_deconvolucion.py`)

Se degrada `cameraman.jpg` con la PSF `psf.png` y ruido Gaussiano (σn = 0.005, 0.02 y 0.05). En
toda la pregunta se usa el mismo modelo de **convolución circular** implementado con la FFT.

- **Parte A:** Wiener simplificado `X = H* Y / (|H|² + K)`, con barrido logarítmico de K.
- **Parte B:** Tikhonov por descenso de gradiente `x ← x − τ [Hᵀ(Hx − y) + λx]`, con λ, τ, número
  máximo de iteraciones y tolerancia configurables (criterio: cambio relativo entre iteraciones).

| Figura | Contenido |
|---|---|
| `00_datos.png` | imagen ideal, PSF, \|H\| y observaciones degradadas |
| `01_wiener_rmse_vs_K.png`, `02_wiener_reconstrucciones.png` | Wiener: RMSE vs K (punto 1) |
| `03a_psf_incorrecta.png`, `03b_psf_incorrecta_curvas.png` | PSF ligeramente incorrecta (punto 2) |
| `04_tikhonov_convergencia.png`, `05_criterio_detencion.png` | Tikhonov iterativo: λ, τ y criterio de detención (punto 3) |
| `06_comparacion_final.png`, `07_iterativo_vs_cerrado.png` | solución iterativa vs cerrada (punto 4) |
| `08_HT_psf_no_simetrica.png` | pregunta guiada: implementación de Hᵀ |

## Bonus — Selección adaptativa del notch (`bonus_notch_adaptativo.py`)

Busca automáticamente el centro y el ancho del notch Gaussiano que minimizan la varianza local media
en tres regiones homogéneas de cielo. Como minimizar solo la varianza lleva al ancho más grande del
rango (filtrado excesivo), se agrega un criterio simple: elegir σ en la meseta de la función
objetivo (mínima pendiente de J vs log σ).

| Figura | Contenido |
|---|---|
| `01_funcion_objetivo.png` | función objetivo, parámetros seleccionados y costo de filtrar de más |
| `02_resultado.png` | resultado con ambos criterios |
