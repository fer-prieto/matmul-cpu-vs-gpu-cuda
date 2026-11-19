# U3-T05 — scripts/ncu_profile.sh (bank conflicts con Nsight Compute, opcional)

- **ID**: U3-T05
- **Depende de**: U3-T03
- **Objetivo**: perfil de bank conflicts de `gpu_tiled` vs `gpu_tiled_pad`; se omite limpiamente si ncu no existe o no tiene permisos.

## Archivos
- Crear: `scripts/ncu_profile.sh` (ejecutable, `chmod +x`).

## Especificación
- Primera instrucción: `run_guard` (U0-T01). `set -u -o pipefail`.
- Args: `--impls a,b` (def `gpu_tiled,gpu_tiled_pad`) · `--sizes 1024 2048 4096` · `--out DIR` (def `results/unidad3/ncu`). Uso inválido → exit 2.
- Si `command -v ncu` vacío → stderr `ncu no disponible: etapa omitida` y **exit 0** (nunca aborta).
- Por cada (impl, n): `mkdir -p "$out"`; ejecutar desde `$REPO_ROOT/.sandbox/tmp`:
  `ncu --section MemoryWorkloadAnalysis -f --log-file "$out/ncu_<impl>_<n>.txt" -o "$REPO_ROOT/.sandbox/tmp/ncu_<impl>_<n>" "$REPO_ROOT/build/bench_gpu" --unit 3 --impls <impl> --sizes <n> --warmup 1 --reps 1 --no-validate`
  (el informe `.ncu-rep` queda en `.sandbox/tmp/`, ignorado por git; solo el log de texto va a results/).
- Resultado: si rc=0 → imprimir `ncu <impl> N=<n>: OK`; si stderr contiene `ERR_NVGPUCTRPERM` (o rc != 0 con permisos) → `ncu sin permisos de contadores: etapa omitida`, borrar el log parcial y **exit 0**. Otro error de ncu → stderr con las últimas 10 líneas y exit 0 (nunca rompe la corrida; queda registrado en el log).
- Extraer del log una línea resumen de shared bank conflicts (grep `shared` en `$out/*.txt` no es parte del script: lo hace el usuario en el informe).
- Cero escrituras fuera de `$out` y `.sandbox/tmp`.

## Casos límite
- ncu presente pero kernel tarda mucho con `--reps 1`: aceptable (2 kernels por rep).
- El `.ncu-rep` nunca toca el repo: `-o` apunta a `.sandbox/tmp/`.

## Verificación LOCAL (sí ejecutable)
```bash
bash -n scripts/ncu_profile.sh
bash scripts/ncu_profile.sh --out /tmp/ncu; echo "rc=$?"   # local: rc=2 (guardia)
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
./scripts/ncu_profile.sh --sizes 1024; echo "rc=$?"        # rc=0 siempre; con o sin archivos según permisos
ls results/unidad3/ncu/ 2>/dev/null
```

## Criterios de aceptación
- [ ] rc=0 en todos los escenarios (omisión documentada, no fallo).
- [ ] Con permisos: logs por (impl, n) con la sección MemoryWorkloadAnalysis.
- [ ] Sin archivos `.ncu-rep` sueltos en el repo.

## Errores típicos
- Tratar la falta de permisos como error de corrida (debe omitirse).
- Escribir informes de ncu fuera de results/unidad3/ncu/.
- Pasar rutas relativas a bench_gpu (usar `$REPO_ROOT/build/bench_gpu`).

## Commit
`feat(scripts): perfil ncu de bank conflicts (opcional)`
