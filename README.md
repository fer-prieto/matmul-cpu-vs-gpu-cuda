# Multiplicación de matrices de gran escala: CPU vs GPU con CUDA

Proyecto del curso **Arquitecturas de Alto Rendimiento (IF557ZIN)**.
Autor: David Fernando Prieto Cardoso.

Objetivo: implementar y comparar `C = A × B` (FP32, N = 1024, 2048 y 4096) en
CPU secuencial (Unidad 1), GPU naive (Unidad 2) y GPU con memoria compartida
tiling (Unidad 3).

Estado: **Unidad 1 — línea base en CPU** completada. Las carpetas `src/gpu/` y
el registro de kernels CUDA se añadirán en las Unidades 2 y 3 sin tocar lo ya
hecho.

---

## Requisitos

- Linux, `g++` con C++17 (o `clang++`) y GNU `make`.
- Rust (`cargo` + `rustc`) para el arnés de benchmark `harness/`.
- Opcional (solo para `make bench-ref`): OpenBLAS con interfaz CBLAS
  (`cblas.h` + `libopenblas`). En Artix/Arch: `sudo pacman -S openblas`.

> El proyecto **compila y se ejecuta sin Rust**: el binario C++ autónomo
> `build/cpu_baseline` cubre por sí solo el entregable de la Unidad 1.

---

## Uso rápido

```bash
make                  # biblioteca C++ + binario autónomo (no necesita Rust)
./build/cpu_baseline  # tabla para N = 1024, 2048 y 4096 (o: ./build/cpu_baseline 512)

make test             # tests de correctitud (Rust, vía FFI)
make bench            # benchmark oficial con el arnés  -> results/unidad1/
make bench-ref        # (opcional) añade OpenBLAS 1 hilo y multihilo como techo
make env              # volcado manual del entorno de prueba
make asm              # ensamblador + informe del vectorizador de GCC
make clean            # borra build/ y harness/target/ (los resultados NO)
make help             # todos los objetivos y variables
```

`make bench` y `make bench-ref` aceptan opciones extra con `ARGS="..."`:

```bash
make bench ARGS="--sizes 256,512 --label prueba --out results/pruebas"
make bench ARGS="--include-slow --label oficial --notes 'AC conectado'"
```

Opciones principales del arnés (`matbench --help` para todas):
`--sizes`, `--impls`, `--reps`, `--seed`, `--include-slow`, `--no-validate`,
`--pin-cpu`, `--cooldown-s`, `--sample-ms`, `--label`, `--notes`, `--out`.

---

## Qué mide y cómo

- `cpu_ijk` (propia, **lenta**): orden i-j-k. Referencia de correctitud y de
  rendimiento (el speedup `ikj/ijk` muestra el efecto de la caché).
- `cpu_ikj` (propia): orden i-k-j, bucle interno contiguo y vectorizable.
  **Línea base oficial** de la Unidad 1.
- `cpu_openblas_1t` / `cpu_openblas_mt` (referencia, opcional): `cblas_sgemm`.
  Techo práctico de la CPU; `pct_of_openblas` se calcula contra la variante de
  1 hilo para que la comparación con el código secuencial sea justa.

Metodología: semilla fija (42), 1 warm-up + repeticiones (5 por defecto, 3 en
N=4096), estadísticos sobre la **mediana**, `GFLOPS = 2·N³ / (t_mediana·10⁹)`,
tiempo medido con `std::chrono::steady_clock` dentro de C++ (solo el triple
bucle / la llamada a `cblas_sgemm`). Validación cruzada contra la referencia
(`cpu_openblas_1t` si está compilado, si no `cpu_ijk`) más verificación de
Freivalds, tolerancia `1e-4`.

## Resultados

Cada corrida escribe en `results/unidad1/` (por defecto):

| Archivo | Contenido |
|---|---|
| `raw_<run>.csv` | Una fila por **ejecución** (calentamientos incluidos): tiempo, checksum, frecuencia, temperatura, throttling |
| `summary_<run>.csv` | Una fila por impl × N con estadísticos, GFLOPS, speedup y % de OpenBLAS |
| `validation_<run>.csv` | Errores relativos/máximos y Freivalds por impl × N |
| `env_<run>.json` | Entorno de prueba completo (CPU, cachés, SO, flags, OpenBLAS, telemetría) |
| `table_<run>.md`, `log_<run>.txt`, `telemetry_<run>.csv` | Tabla, log y muestras de telemetría |
| `runs.csv`, `all_raw.csv`, `all_summary.csv`, `all_validation.csv` | Consolidados de todas las corridas |

**Toda la data queda visible** en estos CSV/JSON para el análisis estadístico
posterior de los informes. Los resultados oficiales se **versionan en git**.

Para reproducir: preparar la máquina (corriente conectada, gobernador
`performance` si es posible, sin aplicaciones pesadas), `make test`, `make
bench` (o `make bench-ref`), y `git add results/unidad1 && git commit`.

---

## Código propio vs bibliotecas de referencia

La multiplicación de matrices (GEMM) está resuelta de forma óptima por
bibliotecas existentes (OpenBLAS, MKL, cuBLAS, CUTLASS). **Aun así, el
algoritmo se implementa desde cero**, porque **la implementación es
precisamente lo que evalúa la consigna**. Copiar kernels o algoritmos de
repositorios ajenos incumpliría la consigna y podría considerarse plagio.

| ✅ Se implementa desde cero (evaluable) | ♻️ Se reutiliza (permitido) |
|---|---|
| CPU i-j-k e i-k-j (`include/matmul_cpu.h`, `src/cpu/matmul_cpu.cpp`) | OpenBLAS `sgemm` como techo y validación (`include/matmul_ref.h`, `src/cpu/matmul_openblas.cpp`) |
| Kernel GPU naive (Unidad 2) | cuBLAS como techo (Unidad 3) |
| Kernel GPU tiled / shared memory (Unidad 3) | Crates de Rust para el arnés (`clap`, `csv`, `serde`) |
| Metodología de medición y análisis | Material educativo: se estudia y se cita (ver `docs/referencias.md`) |

Las bibliotecas externas solo se **llaman** por su API pública, en módulos
separados y marcados como "referencia", compilados únicamente con
`OPENBLAS=1`. El proyecto funciona completo sin ellas.

---

## Estructura

```
include/            API C documentada (propia + referencia)
src/cpu/            C++: ijk/ikj (propio), envoltorio OpenBLAS (referencia), binario
harness/            arnés Rust "matbench" (orquesta, mide, valida, exporta)
scripts/            collect_env.sh (volcado manual del entorno)
results/unidad1/    CSV/JSON de las corridas oficiales (versionados)
docs/               referencias.md y fuente/PDF del reporte por unidad
```
