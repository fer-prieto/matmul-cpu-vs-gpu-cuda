# U2-T09 — docs/unidad2/README.md (guía de resultados para el informe)

- **ID**: U2-T09
- **Depende de**: U2-T08
- **Objetivo**: documentar qué contiene results/unidad2 y cómo consumirlo en el informe.

## Archivos
- Crear: `docs/unidad2/README.md` (máx. ~50 líneas).

## Especificación (contenido, secciones fijas)
1. `# Unidad 2 — GPU naive (memoria global)` — una frase de alcance: kernel naive 1 hilo/elemento, bloques 16×16, medición cudaEvent (kernel) y total (H2D+kernel+D2H), validación vs cpu_ikj (tolerancia 1e-4), baseline CPU re-medido en el servidor.
2. `## Archivos en results/unidad2/` — tabla (inventario del contrato §12 → sección del informe que lo consume).
3. `## Cómo reproducir` — comandos exactos: `make cpu gpu` · `./build/bench_gpu --unit 2 --warmup 1 --reps 5 --seed 42 ...` · `python3 scripts/summarize.py ...` · `python3 scripts/make_tables.py --unit 2 ...` · `python3 scripts/make_figures.py --unit 2 ...`.
4. `## Definiciones` — GFLOPS = 2N³/(t_kernel·1e9) con mediana; speedup_kernel/speedup_total con T_CPU = mediana de cpu_ikj del servidor; qué incluye t_total.
5. `## Limitaciones` — servidor compartido, relojes no fijados salvo `--use-sudo`, dispersión reportada (σ/CV en summary), transferencias PCIe medidas por producto.
6. `## Código propio vs referencia` — naive es propio; cuBLAS aparecerá en U3 como referencia.

## Casos límite
- No inventar cifras: si no hay datos aún, decir "pendiente de la corrida en el servidor".

## Verificación EN SERVIDOR
No aplica (solo documentación; no se ejecuta nada en el servidor).

## Verificación LOCAL (sí ejecutable)
```bash
wc -l docs/unidad2/README.md   # <= 60
grep -n 'results/unidad2' docs/unidad2/README.md   # rutas exactas, sin /home/ ni hostnames
```

## Criterios de aceptación
- [ ] Todas las rutas citadas existen en el contrato §12.
- [ ] Sin datos sensibles; sin cifras inventadas.

## Errores típicos
- Copiar cifras de results/unidad1 (otra máquina) — prohibido: solo datos del servidor.
- Prometer figuras que no se generaron (usar el estado real).

## Commit
`docs(unidad2): guia de resultados para el informe`
