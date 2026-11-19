# U2-T08 — scripts/run_u2.sh (corrida completa de la Unidad 2)

- **ID**: U2-T08
- **Depende de**: U2-T04..T07
- **Objetivo**: ejecutar las etapas de medición de U2 (build, baseline CPU, bench, summary, tablas, figuras, runs.csv) con propagación de exit codes.

## Archivos
- Crear: `scripts/run_u2.sh` (ejecutable, `chmod +x`).

## Especificación
- Primera instrucción: `run_guard` (U0-T01). `set -u -o pipefail`.
- Args: `--stage all|build|env_before|cpu|bench|summary|tables|figures|env_after|runs` (def `all`) · `--run-id ID` (def `$(date +%Y%m%d-%H%M%S)`) · `--skip-cpu` · `--use-sudo`. Uso inválido → exit 2. run_all.sh invoca etapa por etapa (SIEMPRE con `--run-id`); el script también funciona standalone con `--stage all`.
- Directorios: `mkdir -p results/unidad2/{baseline,env,tablas,figuras}`.
- Etapas (cada una con `echo "== etapa <name> =="` y, en fallo, exit code correspondiente):
  1. `build`: `make cpu gpu` → fallo: exit 1.
  2. `env_before`: `scripts/gpu_env_dump.sh --moment before --run-id <id> --out results/unidad2/env/gpu_<id>_before.csv [--extra-key locked_clocks --extra-val "<v>"]`.
  3. `cpu` (omitida si `--skip-cpu`; si `results/unidad2/baseline/cpu_summary_*.csv` ya existe → `OK (ya presente)` y skip): `./build/cpu_baseline --warmup 1 --reps 5 --csv results/unidad2/baseline/cpu_raw_<id>.csv 1024 2048 4096 > results/unidad2/baseline/cpu_table_<id>.txt` (stdout = tabla humana) y luego `python3 scripts/summarize.py --input .../cpu_raw_<id>.csv --mode cpu --out .../cpu_summary_<id>.csv`.
  4. `bench`: `./build/bench_gpu --unit 2 --warmup 1 --reps 5 --seed 42 --run-id <id> --validation-file results/unidad2/validation_<id>.csv 1024 2048 4096 > results/unidad2/raw_<id>.csv` (stderr va al log de run_all); **rc 3 → salir 3** (validación fallida).
  5. `summary`: `python3 scripts/summarize.py --input raw_<id>.csv --mode gpu --baseline .../cpu_summary_<id>.csv --run-id <id> --out results/unidad2/summary_<id>.csv`; consolidar `all_summary.csv` así: `f=$(ls results/unidad2/summary_*.csv | sort | head -1); head -1 "$f" > results/unidad2/all_summary.csv; tail -n +2 results/unidad2/summary_*.csv >> results/unidad2/all_summary.csv` (header una vez, filas ordenadas por nombre de archivo = orden de run).
  6. `tables`: `python3 scripts/make_tables.py --unit 2 --summary results/unidad2/all_summary.csv --validation results/unidad2/validation_<id>.csv --out results/unidad2/tablas/`.
  7. `figures`: `python3 scripts/make_figures.py --unit 2 --summary results/unidad2/all_summary.csv --out results/unidad2/figuras/`; si rc=4 → intentar: `python3 -m venv .sandbox/venv && .sandbox/venv/bin/pip install -q matplotlib==3.8.4 numpy==1.26.4 && .sandbox/venv/bin/python scripts/make_figures.py --unit 2 ...`; si el venv/pip falla (sin red): stderr `figuras: pendiente local (comando al final)` y continuar (exit 0 para la etapa; el mensaje final de run_all lo indica). Guardar `pip freeze` a `results/unidad2/env/venv_freeze_<id>.txt` si el venv se creó.
  8. `env_after`: ídem env_before con `--moment after`.
  9. `runs`: append a `results/unidad2/runs.csv` (header si no existe; esquema contrato §10g). Valores: `all_valid` = todos los `valid=true` del validation file de esta corrida; `gpu_name,gpu_cc,gpu_driver` = parsear la salida de `./build/bench_gpu --device-info` (formato `name=...; cc=...; driver=...`, separar con sed); `nvcc_version` = `nvcc --version | tail -1`; `clocks_locked` = `yes` si `--use-sudo`, si no `no`; `status=completed`, timestamps ISO-8601 (`date -u +%Y-%m-%dT%H:%M:%SZ`).
- `--use-sudo` (solo afecta cpu/bench): ver contrato §9 (sudo -v, fijar relojes actuales, trap EXIT revierte). El trap se instala aquí, no en run_all.
- Nada se escribe fuera de `results/unidad2/`, `.sandbox/` y `logs/` (los logs los lleva run_all).

## Casos límite
- Corrida repetida con `--stage all`: genera un run nuevo (id nuevo); los consolidados acumulan filas (all_summary se regenera completo).
- `--skip-cpu` sin baseline previo → error claro: "no hay cpu_summary_*.csv; corre sin --skip-cpu".

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/run_u2.sh
bash scripts/run_u2.sh --stage all; echo "rc=$?"   # local: rc=2 (guardia), sin efectos
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./scripts/run_u2.sh --stage all; echo "rc=$?"      # rc=0, ~10 min
ls results/unidad2/ && head -2 results/unidad2/all_summary.csv
```

## Criterios de aceptación
- [ ] rc=0 con validación OK; rc=3 si bench_gpu valida mal (probar `--tolerance` no aplica aquí; verificar manualmente con un raw corrupto).
- [ ] Árbol results/unidad2 = contrato §12 (figuras o nota de pendiente).
- [ ] `all_summary.csv` con speedups no vacíos (hay baseline).
- [ ] Warm-up excluido (reps columna = solo medidas).

## Errores típicos (riesgo alto)
- No propagar exit 3 de bench_gpu.
- `tee`/pipe sin `pipefail` (exit code del último comando).
- Escribir tablas/figuras fuera de results/unidad2/.
- Incluir el warm-up en reps (bench_gpu lo marca con phase; summarize filtra).
- sudo sin `--use-sudo` o contraseña en variables/archivos.

## Commit
`feat(scripts): corrida completa de la Unidad 2`
