# AGENTS.md — convenciones para agentes (matmul CPU/GPU)

## Stack y versiones (servidor de referencia)
- nvcc 12.2.140 (CUDA 12.2), target fijo `-arch=sm_60` (Quadro GP100, CC 6.0); g++ 11.4.0; make 4.3; bash; python3 >= 3.8 (stdlib; matplotlib solo para figuras).
- Sin Rust, sin cmake, sin sudo por defecto. Build = Makefile + nvcc; si `nvcc` no esta en PATH, `guard.sh`/`run_all.sh` prueban `/usr/local/cuda-12.2`, `/usr/local/cuda`, `/usr/local/cuda-12` y `/opt/cuda`. Ver `plan/00-contrato.md`.

## Lectura obligatoria antes de trabajar
1. `plan/00-contrato.md` (decisiones, esquemas CSV, fórmulas, contrato de run_all.sh).
2. `plan/INDICE.md` (orden, dependencias, puertas de calidad).
3. Solo la tarea asignada (`plan/U*/T*.md`). No improvisar cambios de diseño.

## Convenciones
- Código evaluable (kernels naive/tiled) escrito desde cero. cuBLAS solo se llama por su API (referencia). Prohibido copiar código ajeno (cuda-samples, SGEMM_CUDA, etc.).
- API C en `include/` (`extern "C"`), FP32 row-major (`m[i*N+j]`), N en `uint64_t`; kernels en `src/gpu/matmul_gpu.cu`; medición con cudaEvent dentro de `src/gpu/`; toda llamada CUDA/cuBLAS envuelta en `CUDA_CHECK`/`CUBLAS_CHECK`.
- CSV UTF-8 con header, una fila por ejecución, flush por fila. Datos por stdout, diagnóstico por stderr. Exit codes: 0 ok, 1 error, 2 uso inválido, 3 validación fallida, 4 entorno insuficiente.
- Scripts bash: primera acción = guardia de raíz (contrato §8), luego `export TMPDIR/XDG_CACHE_HOME/XDG_CONFIG_HOME` hacia `.sandbox/`.
- Cero escrituras fuera del repo. Nada sensible (usuario, hostname, IP, UUID, serial) en archivos versionados.

## Comandos
- Local (sin GPU): `bash -n scripts/*.sh run_all.sh` · `python3 -m py_compile scripts/*.py` · `./run_all.sh --dry-run`.
- Servidor: `chmod +x run_all.sh && ./run_all.sh --check`, luego `./run_all.sh --unit 2|3|all [--skip-cpu] [--use-sudo] [--clean-sandbox]`.
- Build manual: `make cpu gpu`; benchmark manual: `./build/bench_gpu --help`.

## NO tocar
`contexto/` · `results/unidad1/` · `src/cpu/` · `include/matmul_cpu.h` · `include/matmul_util.h` · `include/matmul_ref.h` · `harness/` · `docs/unidad1/` · `plan/` (solo lectura) · `README.md`.
Modificables SOLO si una tarea lo dice: `Makefile` (sección GPU), `.gitignore` (líneas nuevas), `include/matmul_gpu.h`, `include/cuda_util.h`, `src/gpu/*`.

## Commits
Conventional Commits en español, impersonal, <= 72 chars, sin firma de IA. Mensaje exacto en cada tarea. Un commit por tarea. Nunca commitear/empujar sin instrucción explícita.

## Protocolo de escalamiento
Si una tarea falla 2 veces: PARAR. Crear `plan/ESCALAR.md` con: ID de tarea, comando ejecutado, salida completa (stderr), hipótesis de causa. No reintentar con cambios de diseño: el arquitecto decide.

## Flujo clone -> resultados
`git clone <repo> && cd <repo> && chmod +x run_all.sh && ./run_all.sh`.
Al terminar, run_all.sh imprime los comandos git exactos; los ejecuta el humano, nunca el script.
