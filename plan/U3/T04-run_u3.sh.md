# U3-T04 — scripts/run_u3.sh (corrida completa de la Unidad 3)

- **ID**: U3-T04
- **Depende de**: U3-T03
- **Objetivo**: corrida de U3 con las 4 impls, reuso del baseline CPU y salidas del contrato §12.

## Archivos
- Crear: `scripts/run_u3.sh` (ejecutable, `chmod +x`).

## Especificación
- Misma estructura, args, guardias y lista de etapas (`all|build|env_before|cpu|bench|summary|tables|figures|env_after|runs`) que `run_u2.sh` (U2-T08), más la etapa `ncu` (opcional, tras `figures`), con estas diferencias:
  1. Directorios: `results/unidad3/{baseline,env,tablas,figuras,ncu}`.
  2. Etapa `cpu`: si `results/unidad3/baseline/cpu_summary_*.csv` NO existe y `results/unidad2/baseline/cpu_summary_*.csv` SÍ existe → `cp` del más reciente a `results/unidad3/baseline/` con el mismo nombre (el baseline es el mismo servidor; no re-medir). Si no existe en ningún sitio → correr `cpu_baseline` como en U2 (salvo `--skip-cpu`, que entonces es error con mensaje claro).
  3. Etapa `bench`: `./build/bench_gpu --unit 3 --warmup 1 --reps 5 --seed 42 --run-id <id> --validation-file results/unidad3/validation_<id>.csv 1024 2048 4096 > results/unidad3/raw_<id>.csv`; rc 3 → exit 3.
  4. `summary`/`tables`/`figures` con `--unit 3` y rutas `results/unidad3/...` (make_tables genera además `tabla_padding` y `tabla_comparativa`; make_figures genera 4 PNG).
  5. Etapa `ncu` (opcional): invocar `scripts/ncu_profile.sh --impls gpu_tiled,gpu_tiled_pad --sizes 1024 2048 4096 --out results/unidad3/ncu/`; el script ya gestiona su propia omisión (exit 0 si ncu no disponible/sin permisos) — nunca aborta la corrida.
  6. `runs.csv` propio en `results/unidad3/` con `unit=3` e `impls=gpu_naive;gpu_tiled;gpu_tiled_pad;gpu_cublas`.
- Resto (env_before/after, uso de sudo, propagación de exit codes, rutas venv para figuras): idéntico a U2-T08.

## Casos límite
- `--skip-cpu` sin baseline en ninguna unidad → error claro, exit 1.
- Corrida repetida: nuevo run id; `all_summary.csv` regenerado completo.

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/run_u3.sh
bash scripts/run_u3.sh --stage all; echo "rc=$?"   # local: rc=2 (guardia)
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./scripts/run_u3.sh --stage all; echo "rc=$?"      # rc=0, ~15 min
ls results/unidad3/ && head -2 results/unidad3/all_summary.csv
```

## Criterios de aceptación
- [ ] rc=0; árbol results/unidad3 = contrato §12.
- [ ] 4 filas por N en all_summary (naive, tiled, tiled_pad, cublas) con speedups no vacíos.
- [ ] `tabla_padding.tex` con Δ% positivo; ncu/ con archivos o vacío con nota.

## Errores típicos (riesgo alto)
- Copiar baseline de U2 a U3 con `cp` fuera de quotes (rutas con espacios).
- Abortar la corrida si ncu falla (es opcional).
- Escribir en results/unidad2 desde run_u3 (solo lectura de su baseline).

## Commit
`feat(scripts): corrida completa de la Unidad 3`
