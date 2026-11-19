# Unidad 2 — GPU naive (memoria global)

**Informe compilado:** [`reporte_unidad2.pdf`](reporte_unidad2.pdf) — PDF de la
entrega (formato de la plantilla del curso). La fuente LaTeX vive en el
repositorio `infome-template`, rama `unidad2` (tags `unidad2-final`).
Este README es la guía de los datos de `results/unidad2/`.

Kernel CUDA naive (un hilo por elemento de `C`, bloques de 16×16, acceso directo
a memoria global), medición con `cudaEvent` del kernel y del total
(host→device + kernel + device→host), validación contra `cpu_ikj` (tolerancia
`1e-4`) y línea base de CPU re-medida en el mismo servidor.

## Archivos en `results/unidad2/`

| Archivo | Contenido | Sección del informe |
|---|---|---|
| `baseline/cpu_raw_<run>.csv` | una fila por ejecución de `cpu_baseline` | entorno y metodología |
| `baseline/cpu_summary_<run>.csv` | mediana/σ por implementación y N (T_CPU) | resultados, speedup |
| `raw_<run>.csv` | una fila por ejecución de `bench_gpu` (warm-up incluido) | datos crudos |
| `summary_<run>.csv` | por (impl, N): mín, mediana, media, σ, CV, GFLOPS y speedups | tabla de resultados |
| `validation_<run>.csv` | error relativo de Frobenius y máximo absoluto vs `cpu_ikj` | validación |
| `env/gpu_<run>_before.csv`, `env/gpu_<run>_after.csv` | relojes, temperatura, potencia, throttling, ocupación de la GPU | entorno de prueba |
| `env/build_<run>.txt` | versión de nvcc y datos del dispositivo | entorno de prueba |
| `tablas/tabla_*.{tex,md}` | tablas listas para `\input{}` | resultados |
| `figuras/*.png` | tiempo, GFLOPS y speedup frente a N | resultados y gráficos |
| `runs.csv`, `all_summary.csv` | índice de corridas y consolidado de resúmenes | trazabilidad |

## Cómo reproducir

```bash
make cpu gpu
./build/cpu_baseline --warmup 1 --reps 5 --csv results/unidad2/baseline/cpu_raw_<run>.csv 1024 2048 4096
python3 scripts/summarize.py --input results/unidad2/baseline/cpu_raw_<run>.csv --mode cpu \
    --out results/unidad2/baseline/cpu_summary_<run>.csv
./build/bench_gpu --unit 2 --warmup 1 --reps 5 --seed 42 --run-id <run> \
    --validation-file results/unidad2/validation_<run>.csv 1024 2048 4096 > results/unidad2/raw_<run>.csv
python3 scripts/summarize.py --input results/unidad2/raw_<run>.csv --mode gpu \
    --baseline results/unidad2/baseline/cpu_summary_<run>.csv --run-id <run> \
    --out results/unidad2/summary_<run>.csv
python3 scripts/make_tables.py --unit 2 --summary results/unidad2/all_summary.csv \
    --validation results/unidad2/validation_<run>.csv --out results/unidad2/tablas
python3 scripts/make_figures.py --unit 2 --summary results/unidad2/all_summary.csv --out results/unidad2/figuras
```

Todo lo anterior lo ejecuta `./run_all.sh --unit 2` (o `scripts/run_u2.sh --stage all`).

## Definiciones

- `t_kernel_s`: tiempo de cómputo puro del kernel, medido con `cudaEvent` alrededor
  del lanzamiento (sin transferencias).
- `t_total_s`: host→device (A y B) + kernel + device→host (C).
- `GFLOPS = 2·N³ / (t_kernel_mediana · 10⁹)`.
- `speedup_kernel = T_CPU / t_kernel_mediana` y `speedup_total = T_CPU / t_total_mediana`,
  con `T_CPU = t_median_s` de `cpu_ikj` medido en este mismo servidor.
- Validación: `rel_error = ‖C − C_ref‖_F / ‖C_ref‖_F` en `double`, con
  `C_ref` calculada por `cpu_ikj`; se acepta si `rel_error ≤ 1e-4`.

## Resultados medidos (servidor del curso)

Mediana de 5 repeticiones, `T_CPU` = `cpu_ikj` medido en el mismo servidor.

| N | `t_kernel` (s) | `t_total` (s) | GFLOPS | `S_kernel` | `S_total` | PCIe |
|---|---|---|---|---|---|---|
| 1024 | 0.023950 | 0.026685 | 89.66 | 6.74× | 6.05× | 10.2 % |
| 2048 | 0.046263 | 0.057315 | 371.35 | 29.54× | 23.84× | 19.3 % |
| 4096 | 0.397015 | 0.441634 | 346.18 | 43.25× | 38.88× | 10.1 % |

El rendimiento **no es monótono**: 89.66 GFLOPS en 1024 frente a 371.35 en 2048.
La hipótesis (pendiente de confirmar con el perfilador) es el patrón *strided*
sobre la columna de `B`: el salto de 4 KiB concentra las líneas en pocos
conjuntos de caché. Las tres medidas son estables (CV de 0.69 %, 5.53 % y
0.04 %), así que no es ruido. El ancho de banda efectivo de PCIe es 4.5--4.6
GB/s (memoria paginable, sin solape).

Validación: error relativo 0.00 en los tres tamaños (la GPU acumula en el mismo
orden y con FMA, igual que `cpu_ikj`).

## Limitaciones

- Servidor compartido: los relojes de la GPU no se fijan salvo con `--use-sudo`;
  la dispersión de cada medición queda reportada en `summary_<run>.csv` (σ y CV).
- El costo de transferencia se mide por producto (una copia de A y B y una de C);
  no se solapa con el cómputo (no hay streams ni pinned memory).
- La comparación con `cpu_ikj` es la del servidor; los valores de la Unidad 1
  corresponden a otra máquina y no se reutilizan como denominador.

## Código propio vs referencia

`gpu_naive` está escrito desde cero (`src/gpu/matmul_gpu.cu`) y es el código
evaluable de esta unidad. En la Unidad 3 se añade `gpu_cublas` únicamente como
referencia de rendimiento (techo), igual que OpenBLAS en la Unidad 1.