# Unidad 3 — GPU tiled (memoria compartida) y análisis

Kernel con memoria compartida (`TILE = 32`, bloques de 32×32 hilos, un elemento
de `C` por hilo), su variante con **padding** (`As[32][33]`, `Bs[32][33]`) para
eliminar los conflictos de banco, y `cublasSgemm` como techo de referencia.
Se mide igual que en la Unidad 2 (kernel con `cudaEvent` y total con
transferencias) y se valida contra `cpu_ikj` con tolerancia `1e-4`.

## Archivos en `results/unidad3/`

| Archivo | Contenido | Sección del informe |
|---|---|---|
| `baseline/cpu_summary_*.csv` | T_CPU del servidor (reutilizado de la Unidad 2 si existe) | speedup |
| `raw_<run>.csv` | una fila por ejecución (4 implementaciones × 3 N × repeticiones) | datos crudos |
| `summary_<run>.csv` | estadísticos, GFLOPS y speedups por (impl, N) | tabla de resultados |
| `validation_<run>.csv` | errores frente a `cpu_ikj` | validación |
| `env/gpu_<run>_{before,after}.csv`, `env/build_<run>.txt` | estado de la GPU y toolchain | entorno de prueba |
| `tablas/tabla_tiempos`, `tabla_speedup`, `tabla_validacion` | comunes con la Unidad 2 | resultados |
| `tablas/tabla_padding.{tex,md}` | `gpu_tiled` vs `gpu_tiled_pad` y Δ % | bank conflicts |
| `tablas/tabla_comparativa.{tex,md}` | naive / tiled / tiled_pad / cuBLAS y % de cuBLAS | análisis comparativo |
| `figuras/gflops_vs_n.png`, `speedup_vs_n.png`, `tiempo_vs_n.png` | las tres implementaciones propias | resultados y gráficos |
| `figuras/padding_vs_n.png` | barras de la ablación de padding | análisis de memoria |
| `ncu/ncu_<impl>_<n>.txt` | perfil de Nsight Compute (opcional, sin privilegios no se genera) | conflictos de banco |

## Cómo reproducir

```bash
make gpu
./build/bench_gpu --unit 3 --warmup 1 --reps 5 --seed 42 --run-id <run> \
    --validation-file results/unidad3/validation_<run>.csv 1024 2048 4096 > results/unidad3/raw_<run>.csv
python3 scripts/summarize.py --input results/unidad3/raw_<run>.csv --mode gpu \
    --baseline results/unidad3/baseline/cpu_summary_<run>.csv --run-id <run> \
    --out results/unidad3/summary_<run>.csv
python3 scripts/make_tables.py --unit 3 --summary results/unidad3/all_summary.csv \
    --validation results/unidad3/validation_<run>.csv --out results/unidad3/tablas
python3 scripts/make_figures.py --unit 3 --summary results/unidad3/all_summary.csv --out results/unidad3/figuras
bash scripts/ncu_profile.sh --sizes 1024 2048 4096 --out results/unidad3/ncu
```

Todo lo anterior lo ejecuta `./run_all.sh --unit 3` (o `scripts/run_u3.sh --stage all`).

## Definiciones

Mismas fórmulas que `docs/unidad2/README.md`. Además:

- `% de cuBLAS = GFLOPS(gpu_tiled_pad) / GFLOPS(gpu_cublas) × 100`.
- `Δ padding (%) = 100 · (GFLOPS_pad − GFLOPS_tiled) / GFLOPS_tiled`.

## Análisis sugerido

Con tiles de lado 32 y stride 32, la lectura `As[ty][k]` cae siempre en el banco
`k` para los 32 hilos de una columna: conflicto de 32 vías. Con stride 33 el
banco pasa a `(ty + k) % 32`, distinto por hilo. La tabla `tabla_padding` y la
figura `padding_vs_n` cuantifican ese efecto; la distancia naive → tiled → cuBLAS
muestra el resto del camino (coalescencia, ocupación, jerarquía de memoria).

## Limitaciones

- Servidor compartido: relojes solo con `--use-sudo`; dispersión reportada (σ, CV).
- `ncu` requiere permisos de contadores; si no los hay, la ablación empírica del
  padding sustituye a las métricas del perfilador.
- No se usan streams, `pinned memory` ni solapamiento de transferencias: el
  `t_total_s` mide la suma secuencial de copias y kernel.
