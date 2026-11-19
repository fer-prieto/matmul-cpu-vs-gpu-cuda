# U2-T03 — src/gpu/main_gpu.cu (binario de benchmark, CSV y validación)

- **ID**: U2-T03
- **Depende de**: U2-T02
- **Objetivo**: `build/bench_gpu`: mide, valida contra cpu_ikj y emite CSV por stdout con exit codes 0/1/2/3.

## Archivos
- Crear: `src/gpu/main_gpu.cu`.

## Especificación
- Includes: `matmul_gpu.h`, `cuda_util.h`, `matmul_cpu.h`, `matmul_util.h` (enlaza `build/libmatmul_cpu.a`).
- CLI (parser estilo `main_cpu.cpp`): `--unit 2|3` (def 2) · `--impls a,b,c` (def según unit: 2→`gpu_naive`; 3→`gpu_naive,gpu_tiled,gpu_tiled_pad,gpu_cublas`) · `--sizes 1024 2048 4096` o lista posicional (def `1024 2048 4096`) · `--reps R` (def 5, ≥1) · `--warmup W` (def 1) · `--seed S` (def 42) · `--tolerance T` (def 1e-4) · `--no-validate` · `--validation-file PATH` · `--run-id ID` (def `$(date +%Y%m%d-%H%M%S)`) · `--device-info` (imprime `matmul_gpu_device_info()` a stdout y sale 0; lo usa run_u2.sh para runs.csv). Errores de uso → usage a stderr, exit 2. `--help` → usage a stdout, exit 0.
- Registro local (tabla estática, patrón de `kImplementations` en `main_cpu.cpp`): `{name, kind("propia"|"referencia"), run_timed, run_total_timed}` con punteros a función de `matmul_gpu.h`. U2 solo registra `gpu_naive` (propia). `--impls` filtra; nombre desconocido → exit 2.
- Flujo: `matmul_gpu_init()` → por cada N (orden dado): reservar host A,B,C (clase local `AlignedMatrix` idéntica en comportamiento a la de `main_cpu.cpp`: aligned_alloc 64 B, fill 0, RAII; NO incluir `main_cpu.cpp`) y device d_A,d_B,d_C (`cudaMalloc` con CUDA_CHECK, `cudaFree` al salir del N vía RAII); `matmul_util_fill_random(A, N², seed, 0)` y `(B, N², seed, 1)`; por cada impl: `W` warmup + `R` medidas; en cada ejecución: `t_kernel = run_timed(d_A,d_B,d_C,N)`, `t_total = run_total_timed(h_A,h_B,h_C,d_A,d_B,d_C,N)`, `checksum = matmul_util_checksum(h_C, N²)`; emitir a **stdout** (header una vez al inicio):
  `impl,kind,n,phase,rep,seed,t_kernel_s,t_total_s,gflops,checksum` — `%.9f,%.9f,%.6f,%.17g`; `gflops = 2n³/(t_kernel·1e9)`; `phase ∈ {warmup, measured}`; `fflush(stdout)` por fila.
- Validación (tras las reps medidas de cada impl, si no `--no-validate`): `matmul_cpu_ikj(h_A, h_B, h_Cref, N)` (Cref local); `rel = ‖C−Cref‖_F/‖Cref‖_F` y `max_abs = max|C−Cref|` en double; append a `--validation-file` (header si no existe): `run_id,impl,kind,n,reference,rel_error,max_abs_error,norm_fro,norm_fro_ref,tolerance,valid,checksum,checksum_ref` con `reference=cpu_ikj`, `valid=true|false`, `%.9e,%.9e,%.6f,%.6f,%.9e,%.17g,%.17g`. Si `rel > tolerance` → stderr `VALIDACION FALLO: <impl> N=<n> rel=... > tol=...` y al final exit **3** (siempre que no haya otro error previo). Si no hay `--validation-file`, imprimir la misma fila a stderr con prefijo `VALIDACION:`.
- Progreso/resumen humano → SOLO stderr. stdout = únicamente CSV.
- Exit codes: 0 ok · 1 error de runtime (cudaMalloc, etc.) · 2 uso · 3 validación.

## Casos límite
- N<1 o N>INT32_MAX → exit 2. `--reps 0` → exit 2. Solo warmup (R=0 no permitido).
- Semilla distinta entre A y B: siempre streams 0 y 1.

## Verificación LOCAL (sin nvcc: solo estructural)
```bash
grep -n 'matmul_util_fill_random' src/gpu/main_gpu.cu    # 2 llamadas (stream 0 y 1)
grep -n 'fflush(stdout)' src/gpu/main_gpu.cu             # presente en el bucle de filas
grep -n 'printf' src/gpu/main_gpu.cu | grep -v 'stderr\|fprintf'   # printf a stdout SOLO del header/fila CSV
```

## Verificación EN SERVIDOR (la corre el usuario; requiere U2-T04 para compilar)
```bash
make gpu
./build/bench_gpu --unit 2 --sizes 512,1024 --reps 3 --run-id TEST --validation-file /tmp/v.csv > /tmp/r.csv; echo "rc=$?"
head -1 /tmp/r.csv; wc -l /tmp/r.csv        # header + 2 impl*... filas: warmup 1 + 3 medidas por (impl,N)
head -1 /tmp/v.csv; wc -l /tmp/v.csv        # header + 2 filas, valid=true
./build/bench_gpu --unit 2 --sizes 512 --tolerance 0; echo "rc=$?"   # rc=3 esperado
```

## Criterios de aceptación
- [ ] rc=0 con validación OK en el smoke; rc=3 con `--tolerance 0`.
- [ ] stdout es CSV parseable (nada más); warmup excluido de estadísticas (lo filtra summarize.py con phase).
- [ ] checksum de `gpu_naive` ≈ checksum de `cpu_ikj` mismo N (|Δ| < 1e-2, misma semilla).

## Errores típicos (riesgo alto)
- Incluir warm-up en las estadísticas (phase los distingue; nunca resumir warmup).
- Leer `h_C` sin que el kernel haya terminado (total_timed sincroniza; no añadir lecturas fuera).
- Validar contra cpu_ikj con distinta semilla o distinto N.
- Imprimir tablas humanas en stdout (rompe el CSV).
- Olvidar propagar exit 3 (run_all.sh lo usa para abortar la etapa).

## Commit
`feat(gpu): binario de benchmark con CSV y validacion`
