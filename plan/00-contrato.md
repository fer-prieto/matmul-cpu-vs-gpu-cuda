# Contrato — Unidades 2 y 3: C = A×B en GPU (CUDA)

Decisiones Q1–Q10 **APROBADAS** por el usuario en Fase 2. Estado: pendiente la aprobación de los archivos existentes modificados (§2).

## 1. Decisiones (resumen vinculante)

| # | Decisión |
|---|---|
| Q1 | Binario C++/CUDA autónomo `build/bench_gpu` con CSV por stdout. **Sin Rust en el servidor**. |
| Q2 | Re-medir CPU en el servidor (T_CPU = mediana de `cpu_ikj`). Reportar `t_kernel_s` (cudaEvent, solo kernel) y `t_total_s` (H2D+kernel+D2H). GFLOPS con `t_kernel_s`. Dos speedups: `speedup_kernel`, `speedup_total`. |
| Q3 | Naive: 1 hilo por `C[i][j]`, bloque 16×16, grid `ceil(N/16)²`. Tiled: TILE=32, bloque 32×32, 1 elemento/hilo, `__shared__ As[32][32]`, `Bs[32][32]`. |
| Q4 | Variantes `gpu_tiled` (sin padding) y `gpu_tiled_pad` (As y Bs con stride 33). ncu solo si corre sin privilegios. |
| Q5 | `cublasSgemm` como `kind=referencia` (techo) en U3 + validación cruzada. |
| Q6 | Validación vs `cpu_ikj` del servidor, error rel. Frobenius ≤ 1e-4 (en f64), abortar etapa si falla. |
| Q7 | Sin sudo por defecto (nvidia-smi antes/después de cada corrida, reps ≥5, dispersión reportada). `--use-sudo` fija relojes y revierte en `trap EXIT`. |
| Q8 | Sin nvcc/internet necesarios: abortar limpio con resumen autosuficiente en `logs/`. Sin improvisar. |
| Q9 | Figuras adaptativas: matplotlib del sistema → venv `.sandbox/venv` (red) → si no, run_all genera CSV/.tex y deja comando local exacto. |
| Q10 | Versionar `results/unidad2` y `results/unidad3` completos y sanitizados; `logs/` gitignored. El plan da solo firmas/estructuras/índices, sin pseudocódigo de kernels. |

## 2. Archivos existentes que se modifican (aprobación requerida)

| Ruta | Cambio exacto | Tarea |
|---|---|---|
| `Makefile` | Añadir al final sección GPU: variables `NVCC ?= nvcc`, `CUDA_ARCH ?= 60`, `NVCCFLAGS := -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -Xcompiler "-O3 -Wall -Wextra"`, reglas para `src/gpu/*.cu`, target `gpu` (depende de `$(LIB)`), entrada en `help`. Nada más cambia: `all`, targets CPU y reglas existentes intactos. | U2-T04 |
| `.gitignore` | Añadir 2 líneas al final: `/.sandbox/` y `/logs/`. La línea local `contexto/` (sin commit) NO se toca. | U0-T06 |

Todo lo demás es archivo nuevo. `results/unidad1`, `src/cpu`, `include/matmul_cpu.h`, `harness`, `contexto`, `docs/unidad1` quedan intactos.

## 3. Árbol de archivos nuevos

```
AGENTS.md  plan/00-contrato.md  plan/INDICE.md  plan/U0..U3/T*.md
include/cuda_util.h      src/gpu/cuda_util.cu      (U2-T01)
include/matmul_gpu.h     src/gpu/matmul_gpu.cu     (U2-T02, U3-T01, U3-T02)
src/gpu/main_gpu.cu                                (U2-T03, U3-T03)
scripts/guard.sh  scripts/check_env.sh  scripts/setup_sandbox.sh
scripts/gpu_env_dump.sh  scripts/run_u2.sh  scripts/run_u3.sh
scripts/summarize.py  scripts/make_tables.py  scripts/make_figures.py
scripts/ncu_profile.sh  run_all.sh
docs/unidad2/README.md  docs/unidad3/README.md
results/unidad2/  results/unidad3/  logs/   (creados por scripts en el servidor)
```

## 4. Flujo de datos

```
make cpu            → build/cpu_baseline (g++, -march=native)        [reusa src/cpu/*, sin cambios]
make gpu            → build/bench_gpu    (nvcc, sm_60, enlaza libmatmul_cpu.a + -lcublas)
gpu_env_dump.sh     → results/unidadX/env/gpu_<run>_{before,after}.csv   (nvidia-smi sanitizado)
./build/cpu_baseline → results/unidadX/baseline/cpu_raw_<run>.csv  (CSV; tabla humana va al log)
./build/bench_gpu   → stdout=results/unidadX/raw_<run>.csv; validation_<run>.csv; stderr→log; exit 3 si falla validación
summarize.py        → cpu_summary_<run>.csv; summary_<run>.csv (mediana, GFLOPS, speedups vs T_CPU)
make_tables.py      → results/unidadX/tablas/*.tex + *.md
make_figures.py     → results/unidadX/figuras/*.png (matplotlib; salta con aviso si no hay)
ncu_profile.sh      → results/unidad3/ncu/ncu_<impl>_<n>.txt (opcional, sin privilegios)
run_u2.sh / run_u3.sh → orquestan etapas 3..10 de su unidad; run_all.sh compone ambas + check + sandbox + marcadores.
```

## 5. Compilación (flags exactos)

- CPU (sin cambios): `CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra -fPIC`. En el servidor `-march=native` → `skylake-avx512`; el valor efectivo se registra en `results/unidadX/env/build_<run>.txt`.
- GPU: `NVCCFLAGS := -O3 -std=c++17 -arch=sm_60 -Xcompiler "-O3 -Wall -Wextra"`. Enlace de `bench_gpu`: `nvcc $(NVCCFLAGS) $(GPU_OBJS) $(LIB) -lcublas -o build/bench_gpu`.
- **Sin** `--use_fast_math` (numeración idéntica a CPU; U1 registra `fast_math=no`).
- Se registra `nvcc --version` + NVCCFLAGS en el mismo `build_<run>.txt`.

## 6. Sandbox

```
.sandbox/{bin,state,cache,tmp,config,venv}    # venv solo si se necesitan figuras y no hay matplotlib del sistema
.sandbox/state/inicio                        # timestamp para la puerta de calidad (find)
.sandbox/state/<unit>_<stage>.ok             # marcador por etapa; "OK (ya presente)" y skip si existe
```
- Todo script exporta: `TMPDIR=$REPO_ROOT/.sandbox/tmp`, `XDG_CACHE_HOME=$REPO_ROOT/.sandbox/cache`, `XDG_CONFIG_HOME=$REPO_ROOT/.sandbox/config`.
- Única descarga opcional: `python3 -m venv .sandbox/venv` + `pip install matplotlib==3.8.4 numpy==1.26.4` (solo si no hay matplotlib del sistema y hay red). Se registra `pip freeze` en `results/unidadX/env/`. Sin checksum previo de wheels (limitación documentada). Rust **no** se instala.
- `--clean-sandbox`: borra `.sandbox/` completa (nunca `results/` ni `logs/`) y la recrea.

## 7. Recursos del sistema (solo lectura)

| Recurso | Uso |
|---|---|
| `nvcc` (PATH; si falta, `guard.sh`/`run_all.sh` prueban `/usr/local/cuda-12.2`, `/usr/local/cuda`, `/usr/local/cuda-12`, `/opt/cuda` y agregan `bin/` solo al PATH del proceso) | compilar; no se hardcodea ruta en Makefile |
| `/usr/local/cuda-12.2/lib64/libcudart.so.12`, `libcublas.so.12` | enlace (los resuelve nvcc) |
| `/usr/bin/nvidia-smi` | consultas sanitizadas |
| `g++`, `make`, `python3`, `awk`, `sort`, `date`, `curl` | toolchain |
| `/proc/loadavg`, `/proc/meminfo`, `df`, `nproc`, `lscpu` | lectura de entorno |

## 8. Guardia de raíz (patrón exacto)

Cada script de `scripts/` hace primero:
```bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)/guard.sh"
run_guard
```
`scripts/guard.sh` define:
```bash
run_guard() {
  REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[1]}")/.." && pwd -P)"
  case "$REPO_ROOT" in
    /data/FernandoPrieto/*) : ;;
    *) echo "error: raiz invalida ($REPO_ROOT): debe vivir bajo /data/FernandoPrieto" >&2; exit 2 ;;
  esac
  mkdir -p "$REPO_ROOT/.sandbox/tmp" "$REPO_ROOT/.sandbox/cache" "$REPO_ROOT/.sandbox/config"
  export TMPDIR="$REPO_ROOT/.sandbox/tmp" XDG_CACHE_HOME="$REPO_ROOT/.sandbox/cache" XDG_CONFIG_HOME="$REPO_ROOT/.sandbox/config"
}
```
`run_all.sh` usa el mismo patrón con `$0`. **Única excepción**: `run_all.sh --dry-run` no aborta con 2 (no ejecuta nada y debe poder probarse en local); imprime el aviso `[dry-run] guardia omitida`.

## 9. Contrato de run_all.sh

```
./run_all.sh [--check|--setup-only] [--unit 2|3|all] [--skip-cpu] [--use-sudo] [--clean-sandbox] [--dry-run]
```
- Exit codes: **0** ok · **1** error de etapa (ver `logs/FALLO_<ts>.md`) · **2** raíz inválida (fuera de `/data/FernandoPrieto` o bajo `/home`) · **3** validación numérica fallida (propagado de `bench_gpu`) · **4** entorno insuficiente (`check_env.sh`) · **130** interrumpido.
- Etapas de una unidad, en orden: `check` → `sandbox` → `build` → `env_before` → `cpu` (saltable con `--skip-cpu`; marcador `unitX_cpu.ok`) → `bench` → `summary` → `tables` → `figures` → `env_after` → `runs` (U3 añade `ncu` opcional tras `figures`). `--unit all` = U2 y luego U3. `--setup-only` = check+sandbox. `--check` = guardia + `check_env.sh` + `bash -n` de todos los scripts.
- Marcadores en `.sandbox/state/<unit>_<stage>.ok`; etapa completada + segunda corrida → `OK (ya presente)`, no se repite. `--clean-sandbox` los reinicia.
- `--use-sudo`: `sudo -v` una vez (nunca en hijos); fija `sudo nvidia-smi -lgc <sm> -lmc <mem>` con los relojes actuales leídos antes; `trap EXIT` revierte con `sudo nvidia-smi -rgc; sudo nvidia-smi -rmc`. Si `sudo -v` falla → abortar con mensaje.
- Logs: `logs/run_all_<ts>.log` (todo) y, en fallo, `logs/FALLO_<ts>.md` = {etapa, comando, exit code, hipótesis, últimas 40 líneas de stderr}. Nunca git add automático.
- Mensaje final exacto (nunca ejecuta git):
```
RESUMEN: N corridas OK. Validacion: OK. Figuras: <OK|pendiente local>.
git add results/unidad2 results/unidad3
git commit -m "feat: resultados GPU U2 (naive) y U3 (tiled) en Quadro GP100"
git push
# logs/ no se versiona. Para publicar un fallo: revisar logs/FALLO_<ts>.md y luego git add -f logs/FALLO_<ts>.md
# Figuras pendientes (si aplica): python3 scripts/make_figures.py --unit 2|3 --summary results/unidadX/all_summary.csv --out results/unidadX/figuras/
```

## 10. Esquemas CSV exactos (header + fila de ejemplo)

**a) `cpu_raw_<run>.csv`** (stdout de `cpu_baseline --csv`, ya existe — sin cambios):
`impl,n,phase,rep,t_s,gflops,checksum,seed`
`cpu_ikj,1024,measured,3,0.061441,34.952,-2050.77842318693,42`  (t en s, %.9f; checksum suma double)

**b) `cpu_summary_<run>.csv`** (summarize.py --mode cpu):
`impl,n,reps,t_min_s,t_median_s,t_mean_s,t_std_s,gflops,checksum,checksum_stable,seed`
`cpu_ikj,1024,5,0.060917614,0.061440794,0.063879967,0.004075006,34.952081641,-2050.77842318693,true,42`

**c) `raw_<run>.csv`** (stdout de `bench_gpu`):
`impl,kind,n,phase,rep,seed,t_kernel_s,t_total_s,gflops,checksum`
`gpu_naive,propia,1024,measured,3,42,0.012345678,0.015678901,173.523456,-2050.77842318693`
(phase ∈ {warmup, measured}; gflops = 2n³/(t_kernel·1e9); flush por fila)

**d) `validation_<run>.csv`** (bench_gpu --validation-file):
`run_id,impl,kind,n,reference,rel_error,max_abs_error,norm_fro,norm_fro_ref,tolerance,valid,checksum,checksum_ref`
`20261201-101530,gpu_naive,propia,1024,cpu_ikj,5.9e-07,6.8e-05,10922.544,10922.544,0.0001,true,-2050.77842318693,-2050.77842318693`

**e) `summary_<run>.csv`** (summarize.py --mode gpu --baseline):
`run_id,label,impl,kind,n,reps,t_kernel_min_s,t_kernel_median_s,t_kernel_mean_s,t_kernel_std_s,t_kernel_cv_pct,t_total_min_s,t_total_median_s,t_total_mean_s,t_total_std_s,gflops,gflops_best,gflops_mean,checksum,checksum_stable,seed,speedup_kernel,speedup_total`
`20261201-101530,,gpu_naive,propia,1024,5,0.012100000,0.012345678,0.012380000,0.000100000,0.8078,0.015500000,0.015678901,0.015710000,0.000090000,173.523456,176.992508,173.046295,-2050.77842318693,true,42,4.976203,3.918488`
(speedups vacíos si no hay `--baseline`; T_CPU = fila `cpu_ikj` del baseline con mismo `n`)

**f) `gpu_<run>_{before,after}.csv`** (gpu_env_dump.sh):
`run_id,moment,index,name,driver_version,memory_total_mib,memory_used_mib,memory_free_mib,compute_cap,clocks_sm_mhz,clocks_mem_mhz,temp_c,power_w,pstate,throttle_reasons,persistence_mode,pcie_gen_max,pcie_gen_current,pcie_width_current,compute_apps_count,compute_apps_used_mib,load1,extra`
`20261201-101530,before,0,Quadro GP100,580.178.04,16384,512,15872,6.0,1328,715,45.0,28.5,P2,"NONE",Enabled,3,3,16,1,320,0.05 0.10 0.08,`
(campos con comas van entrecomillados; `extra` = `locked_clocks=1328,715` con `--use-sudo`; nunca UUID/IP/hostname/usuario)

**g) `runs.csv`** (append de run_u2.sh / run_u3.sh):
`run_id,label,status,started_at,finished_at,unit,impls,sizes,reps,warmup,seed,tolerance,all_valid,gpu_name,gpu_cc,gpu_driver,nvcc_version,clocks_locked,notes`
`20261201-101530,,completed,2026-12-01T10:15:30Z,2026-12-01T10:20:05Z,2,gpu_naive,1024;2048;4096,5,1,42,0.0001,true,Quadro GP100,6.0,580.178.04,12.2.140,no,`

**h) `all_summary.csv`** = concatenación de `summary_<run>.csv` (header una sola vez, filas ordenadas por run_id).

## 11. Fórmulas

- `FLOPs = 2·N³` · `GFLOPS = 2·N³ / (t_kernel_mediana · 1e9)`
- `speedup_kernel = T_CPU / t_kernel_mediana` · `speedup_total = T_CPU / t_total_mediana` · `T_CPU = t_median_s de cpu_ikj (mismo N, servidor)`
- `rel_error = ‖C−C_ref‖_F / ‖C_ref‖_F` (en f64) · `max_abs = max|C−C_ref|` · válido ⇔ `rel_error ≤ tolerance` (1e-4)
- mediana: impar → elemento central; par → promedio de los dos centrales (regla U1). σ muestral (n−1). CV = 100·σ/media.

## 12. Contenido exacto de results/ (lo que consumirá el informe)

```
results/unidad2/
  runs.csv  all_summary.csv
  baseline/cpu_raw_<run>.csv  cpu_summary_<run>.csv
  raw_<run>.csv  summary_<run>.csv  validation_<run>.csv
  env/gpu_<run>_before.csv  gpu_<run>_after.csv  build_<run>.txt
  tablas/tabla_tiempos.{tex,md}  tabla_speedup.{tex,md}  tabla_validacion.{tex,md}
  figuras/tiempo_vs_n.png  gflops_vs_n.png  speedup_vs_n.png
results/unidad3/   (misma estructura, además:)
  tablas/tabla_padding.{tex,md}  tabla_comparativa.{tex,md}
  figuras/gflops_vs_n.png  speedup_vs_n.png  padding_vs_n.png  tiempo_vs_n.png
  ncu/ncu_<impl>_<n>.txt   (si ncu corre sin privilegios)
```

## 13. Convenciones de U1 copiadas literalmente

- Firmas (`include/matmul_cpu.h`): `void matmul_cpu_ikj(const float* __restrict A, const float* __restrict B, float* __restrict C, uint64_t N);` y `double matmul_cpu_ikj_timed(const float* __restrict A, const float* __restrict B, float* __restrict C, uint64_t N);`
- Registro (`harness/src/registry.rs`): nombre estable por impl; metadatos `kind ∈ {propia, referencia}`, `slow`, `threading`; función cronometrada devuelve tiempo en s.
- Tolerancia `1e-4`; Freivalds opcional; nunca comparar con `==`.
- Generador SplitMix64, semilla 42, A=stream 0, B=stream 1 (`matmul_util_fill_random`); checksum = suma en double (`matmul_util_checksum`). C++ y Rust bit a bit (aquí: bench_gpu reutiliza las funciones C++ existentes enlazando `libmatmul_cpu.a`).
- `run_id = YYYYMMDD-HHMMSS`; tiempos `%.9f`, gflops `%.6f`, checksum `%.17g`.
- Exit codes del arnés: 0 ok · 1 error · 2 opciones · 3 validación fallida. (run_all.sh añade 2=raíz y 4=entorno en su propio contrato; bench_gpu mantiene 0/1/2/3.)
- Commits: Conventional Commits en español, impersonal, ≤72 chars.

## 14. Sanitización (gate)

En `results/unidad2` y `results/unidad3` queda prohibido: hostname, usuario, IP, UUID/serial de GPU, rutas bajo `/home`, PID ajenos. Verificación: `grep -RniE "$(hostname)|$(id -un)|GPU-[0-9a-f]{8}-|/home/" results/unidad2 results/unidad3` → vacío.

## 15. Puertas de calidad

Ver `plan/INDICE.md` (§puertas): gate U0, gate U2 (antes de U3) y checklist final de clon limpio.
