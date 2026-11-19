# U2-T06 — scripts/make_tables.py (tablas .tex y .md para el informe)

- **ID**: U2-T06
- **Depende de**: U2-T05
- **Objetivo**: generar en `results/unidadX/tablas/` las tablas listas para `\input{}` (y su versión Markdown).

## Archivos
- Crear: `scripts/make_tables.py` (python3 stdlib; `chmod +x`).

## Especificación
- Args: `--unit 2|3 --summary all_summary.csv --validation validation_<run>.csv --out DIR`. Crea `DIR` con `mkdir -p`. Uso inválido → exit 2; input inexistente → exit 1.
- Entradas: summary (contrato §10e) y validation (contrato §10d). Orden de filas: impls en orden fijo `gpu_naive, gpu_tiled, gpu_tiled_pad, gpu_cublas` (solo las presentes), luego n ascendente.
- Salidas (`.tex` = fragmento `tabular` plano con `\hline`, SIN `\begin{table}` ni caption; `.md` = tabla pipe; números: t 4 dec, gflops 2 dec, speedup 2 dec, errores en `$1.2\times10^{-7}$` / `1.2e-07`):
  - `tabla_tiempos`: columnas `Implementación | Tipo | N | t_kernel mediana (s) | t_total mediana (s) | GFLOPS`.
  - `tabla_speedup`: `N | speedup_kernel | speedup_total` (una fila por impl y n; si vacío → `---`).
  - `tabla_validacion`: `Implementación | N | Error relativo | Máx. abs. | ¿Válido?`.
  - Solo `--unit 3`: `tabla_padding`: `N | GFLOPS gpu_tiled | GFLOPS gpu_tiled_pad | Δ (%)` (Δ = 100·(pad−tiled)/tiled); `tabla_comparativa`: `N | GFLOPS naive | tiled | tiled_pad | cuBLAS | % de cuBLAS` (% = gflops/gflops_cublas·100).
- Celda faltante (impl no medida) → `---`. Imprime por stdout la lista de archivos generados.

## Casos límite
- Unit 2 con summary que ya contiene impls de U3: mostrar solo `gpu_naive`.
- Filas con speedup vacío (sin baseline): `---` en .tex/.md.
- Validation inexistente o sin filas: `tabla_validacion` se genera igual con solo la cabecera (sin filas).

## Verificación LOCAL (sí ejecutable)
```bash
python3 -m py_compile scripts/make_tables.py
mkdir -p /tmp/tabtest && printf 'run_id,label,impl,kind,n,reps,t_kernel_min_s,t_kernel_median_s,t_kernel_mean_s,t_kernel_std_s,t_kernel_cv_pct,t_total_min_s,t_total_median_s,t_total_mean_s,t_total_std_s,gflops,gflops_best,gflops_mean,checksum,checksum_stable,seed,speedup_kernel,speedup_total\nx,,gpu_naive,propia,512,3,0.010,0.011,0.011,0.001,9.0,0.020,0.021,0.021,0.001,24.398,26.838,24.398,10.0,true,42,5.5,2.9\n' | python3 scripts/make_tables.py --unit 2 --summary /dev/stdin --validation /dev/null --out /tmp/tabtest 2>/dev/null; head -5 /tmp/tabtest/tabla_tiempos.tex; rm -rf /tmp/tabtest
```

## Verificación EN SERVIDOR (la corre el usuario; tras U2-T08)
```bash
python3 scripts/make_tables.py --unit 2 --summary results/unidad2/all_summary.csv --validation "$(ls results/unidad2/validation_*.csv | tail -1)" --out results/unidad2/tablas/
ls results/unidad2/tablas/
```

## Criterios de aceptación
- [ ] Fixture local genera `tabla_tiempos.tex` con la fila esperada (`24.40` GFLOPS).
- [ ] Los `.tex` compilan como `\input{}` dentro de un `tabular` de la plantilla (sin macros propias).
- [ ] Sin dependencias fuera de stdlib.

## Errores típicos
- Incluir `\begin{table}`/`\caption` (la plantilla del informe los pone).
- Caracteres `_` sin escapar en LaTeX (usar `\_` en nombres de impl).
- Orden de columnas distinto al especificado.

## Commit
`feat(scripts): tablas LaTeX y Markdown de resultados`
