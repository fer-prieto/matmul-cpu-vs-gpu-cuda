# U2-T07 — scripts/make_figures.py (figuras PNG para el informe)

- **ID**: U2-T07
- **Depende de**: —
- **Objetivo**: generar las figuras de results/unidadX/figuras/ con matplotlib (Agg), o fallar con exit 4 si no está disponible.

## Archivos
- Crear: `scripts/make_figures.py` (python3 + matplotlib; `chmod +x`).

## Especificación
- Args: `--unit 2|3 --summary all_summary.csv --out DIR`. `mkdir -p DIR`. Uso inválido → exit 2; input inexistente → exit 1.
- Si `import matplotlib` falla → stderr `FALLO: matplotlib no disponible (instalar en .sandbox/venv o generar en local)` y **exit 4**. Usar `matplotlib.use("Agg")`; sin pyplot interactivo; listas Python (no requiere numpy explícito).
- Estilo común: figsize (6.4, 4.0), dpi 300, grid, leyenda, títulos y ejes en español, marcadores por impl. Orden/colores fijos: `gpu_naive` azul, `gpu_tiled` naranja, `gpu_tiled_pad` verde, `gpu_cublas` rojo. Eje X = N (1024, 2048, 4096) como puntos equiespaciados con etiquetas de N.
- Unit 2 → 3 PNG: `tiempo_vs_n.png` (series `t_kernel_median_s` y `t_total_median_s`, eje y log), `gflops_vs_n.png` (y log), `speedup_vs_n.png` (series `speedup_kernel`/`speedup_total`, y log).
- Unit 3 → 4 PNG: `gflops_vs_n.png` (4 impls, y log), `speedup_vs_n.png` (series kernel/total por impl, y log), `padding_vs_n.png` (barras agrupadas por N: gflops `gpu_tiled` vs `gpu_tiled_pad`, y lineal), `tiempo_vs_n.png` (series kernel/total por impl, y log).
- Imprime por stdout los archivos generados. Sin escrituras fuera de `--out`.

## Casos límite
- Impl ausente en el summary: se omite su serie (sin error).
- Speedups vacíos (sin baseline): serie omitida con warning a stderr.

## Verificación LOCAL (sí ejecutable)
```bash
python3 -m py_compile scripts/make_figures.py
python3 scripts/make_figures.py --unit 2 --summary /nonexistent --out /tmp/f; echo "rc=$?"   # rc=1 (input) o 4 (sin matplotlib); ambos aceptables localmente
```

## Verificación EN SERVIDOR (la corre el usuario; tras U2-T08)
```bash
python3 scripts/make_figures.py --unit 2 --summary results/unidad2/all_summary.csv --out results/unidad2/figuras/; echo "rc=$?"
ls results/unidad2/figuras/    # 3 PNG (o rc=4 -> run_u2.sh activa la ruta venv)
```

## Criterios de aceptación
- [ ] Con matplotlib disponible: 3 PNG (U2) / 4 PNG (U3) con los nombres exactos del contrato §12.
- [ ] Sin matplotlib: exit 4 con mensaje accionable (run_u2.sh decide venv o salto local).
- [ ] PNG legibles (600×400 aprox. a 300 dpi), sin datos sensibles en ejes/títulos.

## Errores típicos
- `pyplot.show()` en servidor sin display (usar Agg).
- Depender de pandas/seaborn (solo matplotlib).
- Escribir fuera de `--out`.

## Commit
`feat(scripts): figuras de resultados`
