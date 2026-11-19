# Multiplicación de matrices de gran escala: CPU vs GPU con CUDA

Proyecto del curso **Arquitecturas de Alto Rendimiento (IF557ZIN)**.
Autor: David Fernando Prieto Cardoso.

Objetivo: implementar y comparar `C = A × B` (FP32, N = 1024, 2048 y 4096) en
CPU secuencial (Unidad 1), GPU naive en memoria global (Unidad 2) y GPU con
memoria compartida / tiling (Unidad 3), midiendo tiempo de cómputo, tiempo
total, speedup y GFLOPS.

Estado: **las tres unidades están implementadas y medidas.**

| Unidad | Código | Resultados | Informe |
|---|---|---|---|
| 1 — CPU secuencial | `src/cpu/` | `results/unidad1/` | `docs/unidad1/reporte_unidad1.pdf` |
| 2 — GPU naive | `src/gpu/matmul_gpu.cu` | `results/unidad2/` | `docs/unidad2/reporte_unidad2.pdf` |
| 3 — GPU tiling + cuBLAS | `src/gpu/matmul_gpu.cu` | `results/unidad3/` | `docs/unidad3/reporte_unidad3.pdf` |

Cada `docs/unidadN/` contiene el **informe compilado** (PDF, formato de la
plantilla del curso) y la guía de los datos de esa unidad; la fuente LaTeX vive
en el repositorio `infome-template`, una rama por unidad.

---

## Requisitos

- Linux, `g++` con C++17 (o `clang++`) y GNU `make`.
- **GPU NVIDIA + toolkit CUDA** para las Unidades 2 y 3: `nvcc` >= 12.0 con
  soporte de `sm_60` (`-arch=sm_60` es el objetivo del servidor del curso) y
  `libcublas`. Si `nvcc` no está en el `PATH`, los scripts prueban
  `/usr/local/cuda-12.2`, `/usr/local/cuda`, `/usr/local/cuda-12` y `/opt/cuda`.
- Rust (`cargo` + `rustc`) **solo** para el arnés de la Unidad 1 (`harness/`).
- Opcional (solo para `make bench-ref`): OpenBLAS con interfaz CBLAS.

> El proyecto **compila y se ejecuta sin Rust**: `build/cpu_baseline` cubre la
> Unidad 1 y `build/bench_gpu` cubre las Unidades 2 y 3.

---

## Uso rápido

### En el servidor con GPU (flujo completo, un comando)

```bash
chmod +x run_all.sh
./run_all.sh --unit all        # verifica entorno, compila, mide, agrega, tablas y figuras
```

`run_all.sh` deja los resultados en `results/unidad2/` y `results/unidad3/`,
escribe logs en `logs/`, y al terminar **imprime los comandos git** (nunca los
ejecuta). Opciones: `--check` (solo entorno), `--setup-only`, `--unit 2|3|all`,
`--skip-cpu`, `--use-sudo` (fija relojes de la GPU y los revierte), `--clean-sandbox`,
`--dry-run` (imprime las etapas sin ejecutar nada).

### Compilación y medición manual

```bash
make cpu gpu                      # biblioteca C++ + cpu_baseline + bench_gpu
./build/cpu_baseline 1024         # tabla de la línea base en CPU
./build/bench_gpu --help          # opciones del benchmark GPU

make test                         # tests de correctitud (Rust, vía FFI, Unidad 1)
make bench                        # benchmark CPU con el arnés -> results/unidad1/
make bench-ref                    # (opcional) añade OpenBLAS como techo de CPU
make env / make asm               # volcado de entorno / ensamblador y vectorización
make clean                        # borra build/ y harness/target/ (los resultados NO)
make help                         # todos los objetivos y variables
```

Ejemplos del binario GPU:

```bash
./build/bench_gpu --unit 2 --sizes 1024 2048 4096 --reps 5 \
    --validation-file results/unidad2/validation_manual.csv > results/unidad2/raw_manual.csv
./build/bench_gpu --unit 3 --impls gpu_tiled,gpu_tiled_pad --sizes 4096 --reps 3
./build/bench_gpu --device-info
```

---

## Qué se mide

| Implementación | Unidad | Tipo | Descripción |
|---|---|---|---|
| `cpu_ijk` | 1 | propia | orden i-j-k, referencia de correctitud (lenta) |
| `cpu_ikj` | 1 | propia | orden i-k-j, **línea base oficial** de la CPU |
| `cpu_openblas_1t` / `_mt` | 1 | referencia | `cblas_sgemm` a 1 hilo / multihilo (techo de CPU) |
| `gpu_naive` | 2 | propia | un hilo por elemento, solo memoria global (bloques 16×16) |
| `gpu_tiled` | 3 | propia | tiles de 32×32 en memoria compartida con `__syncthreads()` |
| `gpu_tiled_pad` | 3 | propia | igual, con *padding* (stride 33) para la ablación de bancos |
| `gpu_cublas` | 3 | referencia | `cublasSgemm` como techo de la GPU |

Metodología común: semilla fija 42 (SplitMix64, A = stream 0, B = stream 1),
1 calentamiento no medido + 5 repeticiones medidas (3 para N = 4096 en la CPU),
estadísticos sobre la **mediana**, `GFLOPS = 2·N³ / (t·10⁹)`. Validación
numérica contra la referencia con error relativo de Frobenius y tolerancia
`1e-4` (nunca comparación exacta).

Tiempos:

- **CPU:** `std::chrono::steady_clock` dentro de C++, solo el triple bucle.
- **GPU:** dos eventos CUDA por medición. `t_kernel_s` = solo el kernel
  (cómputo puro); `t_total_s` = copias Host↔Device + kernel.
- **Speedup:** `S_kernel = T_CPU / t_kernel` y `S_total = T_CPU / t_total`, con
  `T_CPU` medido **en el mismo servidor** (los tiempos de la Unidad 1 son de
  otra máquina y no se usan como denominador).

---

## Resultados (resumen)

Servidor del curso: 2× Xeon Gold 6134 · Quadro GP100 (16 GiB, `sm_60`) ·
nvcc 12.2 · PCIe Gen3 ×16. Detalle en `results/unidad*/env/`.

| N | CPU `ikj` | `gpu_naive` | `gpu_tiled` | `gpu_cublas` | S_kernel tiled |
|---|---|---|---|---|---|
| 1024 | 13.31 | 89.66 | 1649.43 | 6047.48 | 123.9× |
| 2048 | 12.57 | 371.35 | 2042.84 | 8740.69 | 162.5× |
| 4096 | 8.00 | 348.10 | 1964.18 | 8850.38 | 245.4× |

(GFLOPS, mediana de 5 repeticiones.) El kernel con tiling alcanza el 21–27 % del
techo de `cuBLAS` y reduce el tiempo de cómputo entre 5.6× y 18.5× respecto del
naive. El *padding* **no** mejora el rendimiento (+0.2 %, −1.4 %, −2.9 %)
porque con bloques de 32×32 el warp lee el tile de `A` con dirección uniforme
(*broadcast*): ver `results/unidad3/tablas/tabla_padding.tex` y
`docs/unidad3/README.md`. Con el kernel rápido, las transferencias por PCIe
pasan a pesar hasta el 65 % del tiempo total.

---

## Estructura

```
include/            API C documentada: matmul_cpu.h, matmul_gpu.h, cuda_util.h, matmul_ref.h
src/cpu/            ijk/ikj (propio), envoltorio OpenBLAS (referencia), binario CPU
src/gpu/            kernels CUDA (naive, tiled, tiled_pad), cuBLAS (referencia), bench_gpu
harness/            arnés Rust "matbench" de la Unidad 1 (orquesta, mide, valida, exporta)
scripts/            guard.sh, check_env, setup_sandbox, gpu_env_dump, run_u2, run_u3,
                    ncu_profile, summarize.py, make_tables.py, make_figures.py,
                    collect_env.sh (volcado manual de la Unidad 1)
run_all.sh          flujo completo del servidor (etapas reanudables + logs + resumen de fallo)
results/unidad1|2|3 Resultados versionados (CSV, tablas .tex/.md y figuras PNG)
docs/unidad1|2|3    Informe compilado (PDF) y guía de resultados de cada unidad
plan/               Contrato y plan de implementación por tareas (U0, U2, U3)
```

### Resultados por unidad

| Unidad | Contenido de `results/unidadN/` |
|---|---|
| 1 | `raw/summary/validation/telemetry_<run>.csv`, `env_<run>.json`, `runs.csv`, `all_*.csv` |
| 2 | `raw/summary/validation_<run>.csv`, `baseline/cpu_*.csv`, `env/gpu_*_{before,after}.csv`, `env/hardware_*.txt`, `tablas/*.{tex,md}`, `figuras/*.png`, `runs.csv`, `all_summary.csv` |
| 3 | igual que la 2, más `tablas/tabla_padding.*`, `tabla_comparativa.*`, `figuras/padding_vs_n.png` y `ncu/` (perfil opcional) |

Los datos crudos son suficientes para recalcular todo: `scripts/summarize.py`
agrega los `raw_*.csv`, `scripts/make_tables.py` genera las tablas y
`scripts/make_figures.py` las figuras.

### Reproducibilidad y sandbox

`run_all.sh` y los scripts de `scripts/`:

- calculan `REPO_ROOT` y **abortan con código 2** si no está bajo
  `/data/FernandoPrieto` (raíz del servidor del curso);
- redirigen `TMPDIR`, `XDG_CACHE_HOME` y `XDG_CONFIG_HOME` a `.sandbox/`, de modo
  que **no escriben nada fuera del repositorio**;
- son idempotentes: los marcadores de `.sandbox/state/` hacen que una segunda
  corrida imprima `OK (ya presente)` y no repita etapas;
- dejan en `logs/FALLO_<ts>.md` un resumen corto y autosuficiente si algo falla
  (etapa, comando, código de salida y últimas 40 líneas).

Códigos de salida: `0` ok · `1` error de etapa · `2` raíz inválida ·
`3` validación numérica fallida · `4` entorno insuficiente.

---

## Código propio vs bibliotecas de referencia

La multiplicación de matrices (GEMM) está resuelta de forma óptima por
bibliotecas existentes (OpenBLAS, MKL, cuBLAS, CUTLASS). **Aun así, el
algoritmo se implementa desde cero**, porque **la implementación es
precisamente lo que evalúa la consigna**. Copiar kernels o algoritmos de
repositorios ajenos incumpliría la consigna y podría considerarse plagio.

| ✅ Se implementa desde cero (evaluable) | ♻️ Se reutiliza (permitido) |
|---|---|
| CPU i-j-k e i-k-j (`src/cpu/matmul_cpu.cpp`) | OpenBLAS `sgemm` como techo y validación (Unidad 1) |
| Kernel naive (`src/gpu/matmul_gpu.cu`) | cuBLAS `Sgemm` como techo de GPU (Unidad 3) |
| Kernel tiled y variante con padding | Crates de Rust del arnés (`clap`, `csv`, `serde`) |
| Metodología de medición y análisis | Material educativo: se estudia y se cita (`docs/referencias.md`) |

Las bibliotecas externas solo se **llaman** por su API pública, en módulos
separados y marcados como «referencia». El proyecto funciona completo sin
OpenBLAS (`OPENBLAS=0` por defecto).

---

## Entregas

| Tag (repo) | Rama (informes) | Contenido |
|---|---|---|
| `unidad1` | `unidad1` | línea base en CPU + informe (`docs/unidad1/reporte_unidad1.pdf`) |
| `unidad2` | `unidad2` | kernel naive + mediciones + informe (`docs/unidad2/reporte_unidad2.pdf`) |
| `unidad3` | `unidad3` | tiling, padding y cuBLAS + informe (`docs/unidad3/reporte_unidad3.pdf`) |

Los informes viven en el repositorio `infome-template`, una rama por unidad.
