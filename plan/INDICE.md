# ÍNDICE de tareas — Unidades 2 y 3 (GPU)

Referencias: `plan/00-contrato.md` (contrato) · `AGENTS.md` (reglas de ejecución).
El ejecutor lee SOLO: AGENTS.md + contrato + la tarea asignada. Una tarea = un commit.

## Tabla de tareas

| ID | Archivos | Depende de | Riesgo | ¿Escalar si falla 2×? |
|---|---|---|---|---|
| U0-T01 | `scripts/guard.sh` | — | Bajo | No |
| U0-T02 | `scripts/check_env.sh` | U0-T01 | Bajo | No |
| U0-T03 | `scripts/setup_sandbox.sh` | U0-T01 | Bajo | No |
| U0-T04 | `scripts/gpu_env_dump.sh` | U0-T01 | Medio | No |
| U0-T05 | `run_all.sh` | U0-T01..T04 | Alto | Sí |
| U0-T06 | `.gitignore` (modificar) | — | Bajo | No |
| U2-T01 | `include/cuda_util.h`, `src/gpu/cuda_util.cu` | U0-T05 | Bajo | No |
| U2-T02 | `include/matmul_gpu.h`, `src/gpu/matmul_gpu.cu` (kernel naive) | U2-T01 | Alto | Sí |
| U2-T03 | `src/gpu/main_gpu.cu` (benchmark + validación) | U2-T02 | Alto | Sí |
| U2-T04 | `Makefile` (modificar: sección GPU) | U2-T03 | Medio | No |
| U2-T05 | `scripts/summarize.py` | — | Medio | No |
| U2-T06 | `scripts/make_tables.py` | U2-T05 | Bajo | No |
| U2-T07 | `scripts/make_figures.py` | — | Medio | No |
| U2-T08 | `scripts/run_u2.sh` | U2-T04..T07 | Alto | Sí |
| U2-T09 | `docs/unidad2/README.md` | U2-T08 | Bajo | No |
| U3-T01 | `include/matmul_gpu.h`, `src/gpu/matmul_gpu.cu` (tiled + padding) | U2-T02 | Alto | Sí |
| U3-T02 | `include/cuda_util.h`, `include/matmul_gpu.h`, `src/gpu/matmul_gpu.cu` (cuBLAS referencia) | U3-T01 | Alto | Sí |
| U3-T03 | `src/gpu/main_gpu.cu` (registro de impls U3) | U3-T02 | Alto | Sí |
| U3-T04 | `scripts/run_u3.sh` | U3-T03 | Alto | Sí |
| U3-T05 | `scripts/ncu_profile.sh` (opcional) | U3-T03 | Medio | No |
| U3-T06 | `docs/unidad3/README.md` | U3-T04 | Bajo | No |

Orden de ejecución: U0-T01 → … → U0-T06 → U2-T01 → … → U2-T09 → U3-T01 → … → U3-T06.
Las tareas "Alto" exigen sección *Errores típicos a evitar* en el archivo de tarea.

## Verificación local permitida al ejecutor (sin GPU, sin CUDA)

`bash -n scripts/*.sh run_all.sh` · `python3 -m py_compile scripts/*.py` · `./run_all.sh --dry-run` · greps estructurales que indique cada tarea. Todo lo demás es verificación EN SERVIDOR (la corre el usuario).

## PUERTA DE CALIDAD — fin de U0 (en el servidor, antes de U2)

1. `./run_all.sh --check` → exit 0; repetir → mismo resultado.
2. `touch .sandbox/state/inicio && ./run_all.sh --setup-only` y luego `find /home /tmp -newer .sandbox/state/inicio -type f 2>/dev/null` → sin archivos creados por el script (revisar manualmente lo ajeno).
3. `test ! -e ~/.cargo -a ! -e ~/.rustup` → cierto (nunca se instala Rust).
4. Hashes de `~/.bashrc` y `~/.profile` (md5sum antes/después) intactos.
5. Guardia: desde `/home/...`, `bash run_all.sh --check` → exit 2 con el mensaje de raíz inválida.
6. `bash -n` limpio de todos los scripts.

## PUERTA DE CALIDAD — fin de U2 (antes de empezar U3)

1. `./run_all.sh --unit 2` completa con exit 0; segunda corrida imprime `OK (ya presente)` por etapa.
2. `results/unidad2/all_summary.csv` tiene filas `gpu_naive` para N=1024/2048/4096 con `gflops` en [10, 2000] y `speedup_kernel ≥ 0.5` (cotas de cordura; si no, el timing está mal).
3. `validation_<run>.csv`: todas las filas `valid=true` con `rel_error ≤ 1e-4`.
4. `checksum_stable=true` en todas las filas de summary; checksum de `gpu_naive` ≈ checksum de `cpu_ikj` mismo N (±1e-3 absoluto).
5. Sanitización (§14 del contrato): grep vacío.
6. `results/unidad2/tablas/*.tex` y `figuras/*.png` existen (figuras: o existen o el mensaje final indica el comando local).

## CHECKLIST FINAL — clon limpio

1. `cd /data/FernandoPrieto && git clone <repo> && cd <repo> && chmod +x run_all.sh`.
2. `./run_all.sh --unit all` → exit 0; resultados en `results/unidad2` y `results/unidad3` completos (árbol §12 del contrato).
3. Sanitización: grep vacío en ambos directorios.
4. `./run_all.sh --unit all` segunda vez → todo `OK (ya presente)`.
5. El mensaje final imprime los comandos git exactos; los ejecuta el humano.
6. En local: `git pull` → redactar informes con `results/` (CSV, JSON, `.tex`, PNG).

## Escalamiento

Una tarea marcada "Sí" que falle 2 veces con el mismo error → el ejecutor para y escribe `plan/ESCALAR.md` (ID, comando, stderr completo, hipótesis). No se rediseña: decide el arquitecto.
