# U2-T05 — scripts/summarize.py (agregación CSV, GFLOPS y speedups)

- **ID**: U2-T05
- **Depende de**: —
- **Objetivo**: convertir raw CSV en summary CSV (medianas, σ, GFLOPS, speedups vs T_CPU).

## Archivos
- Crear: `scripts/summarize.py` (python3, SOLO stdlib; ejecutable `chmod +x`; shebang `#!/usr/bin/env python3`).

## Especificación
- Args: `--input raw.csv --mode cpu|gpu --out summary.csv [--baseline cpu_summary.csv] [--run-id ID] [--label L]`. Uso inválido → stderr + exit 2. Input inexistente o sin filas `measured` → stderr + exit 1.
- `--mode cpu` (entrada = salida de `cpu_baseline --csv`, header `impl,n,phase,rep,t_s,gflops,checksum,seed`): filtra `phase==measured`; agrupa por (impl, n) en orden de primera aparición; calcula `reps, t_min_s, t_median_s, t_mean_s, t_std_s` (σ muestral n−1, vacío si reps=1), `gflops = 2·n³/(t_median·1e9)`, `checksum` de la última rep, `checksum_stable` (todas las reps medidas con el mismo checksum), `seed`. Header salida: `impl,n,reps,t_min_s,t_median_s,t_mean_s,t_std_s,gflops,checksum,checksum_stable,seed`.
- `--mode gpu` (entrada header del contrato §10c): filtra `measured`; por (impl, n) calcula min/median/mean/std de `t_kernel_s` y de `t_total_s`, `t_kernel_cv_pct = 100·σ/media`, `gflops` (mediana kernel), `gflops_best` (mín), `gflops_mean` (media de los gflops por rep), `checksum`, `checksum_stable`, `seed`; si `--baseline`: buscar la fila del baseline con `impl==cpu_ikj` y mismo `n` → `T_cpu = t_median_s` → `speedup_kernel = T_cpu/t_kernel_median`, `speedup_total = T_cpu/t_total_median` (redondear a 6 decimales); si no hay baseline o no existe esa fila → campos vacíos. Header salida = contrato §10e (con `run_id,label` al inicio).
- Formato numérico: tiempos `%.9f`, gflops `%.6f`, speedups `%.6f`, cv `%.4f`, checksum `%.17g`, `true/false` en minúsculas.
- Salida: escribe `--out` y además imprime el mismo CSV por stdout (composición).
- Mediana par = promedio de los dos centrales (contrato §11). Grupos con solo warmup → warning a stderr y se omiten.

## Casos límite
- reps=1: `t_std_s`/`t_kernel_cv_pct` vacíos; `checksum_stable` = true.
- n repetido con distinta semilla: agrupar SOLO por (impl, n); `seed` de la última fila (advertir a stderr si difieren).

## Verificación LOCAL (sí ejecutable)
```bash
python3 -m py_compile scripts/summarize.py
printf 'impl,kind,n,phase,rep,seed,t_kernel_s,t_total_s,gflops,checksum\ngpu_naive,propia,512,warmup,1,42,0.01,0.02,1.0,10.0\ngpu_naive,propia,512,measured,1,42,0.010,0.020,1.0,10.0\ngpu_naive,propia,512,measured,2,42,0.012,0.022,1.0,10.0\ngpu_naive,propia,512,measured,3,42,0.011,0.021,1.0,10.0\n' | python3 scripts/summarize.py --input /dev/stdin --mode gpu --out /dev/stdout
# esperado: 1 fila, reps=3, t_kernel_median_s=0.011000000, gflops=2*512^3/(0.011*1e9)=24.398..., checksum_stable=true
```

## Verificación EN SERVIDOR (la corre el usuario; con datos reales de U2-T08)
```bash
python3 scripts/summarize.py --input results/unidad2/baseline/cpu_raw_<run>.csv --mode cpu --out results/unidad2/baseline/cpu_summary_<run>.csv
head -2 results/unidad2/baseline/cpu_summary_<run>.csv   # fila cpu_ikj por cada N
```

## Criterios de aceptación
- [ ] Fixture local produce la fila esperada (mediana 0.011).
- [ ] Columnas y orden EXACTOS del contrato §10b/§10e.
- [ ] Speedups vacíos sin `--baseline`; correctos con él (verificable a mano en un caso).

## Errores típicos
- Incluir filas warmup en las estadísticas.
- σ con divisor n (debe ser n−1).
- GFLOPS con t_total (debe ser t_kernel).
- Speedup invertido (T_cpu va en el numerador).

## Commit
`feat(scripts): agregacion de CSV y speedups`
