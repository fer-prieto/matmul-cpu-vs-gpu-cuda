# U3-T06 — docs/unidad3/README.md (guía de resultados para el informe)

- **ID**: U3-T06
- **Depende de**: U3-T04
- **Objetivo**: documentar results/unidad3 y guiar el análisis (padding, techo cuBLAS).

## Archivos
- Crear: `docs/unidad3/README.md` (máx. ~50 líneas).

## Especificación (secciones fijas)
1. `# Unidad 3 — GPU tiled con memoria compartida` — alcance: TILE=32, As/Bs en shared, ablación padding (stride 33) contra bank conflicts, cuBLAS como techo (referencia), ncu opcional.
2. `## Archivos en results/unidad3/` — tabla inventario → sección del informe.
3. `## Cómo reproducir` — `make gpu` · `./build/bench_gpu --unit 3 --warmup 1 --reps 5 --seed 42 ...` · summarize/make_tables/make_figures con `--unit 3` · `scripts/ncu_profile.sh`.
4. `## Definiciones` — mismas fórmulas de U2 (referencia a docs/unidad2/README.md); % de cuBLAS = GFLOPS_impl / GFLOPS_cublas · 100; Δ padding = 100·(pad−tiled)/tiled.
5. `## Análisis sugerido` — por qué el padding elimina el conflicto 32-way en la lectura de As (fórmula de bancos del comentario de U3-T01); distancia naive→tiled→cuBLAS como evidencia de la jerarquía de memoria.
6. `## Limitaciones` — servidor compartido; relojes solo con `--use-sudo`; ncu condicionado a permisos (la ablación empírica sustituye si falta).

## Casos límite
- Sin cifras inventadas: referenciar los archivos, no números.

## Verificación EN SERVIDOR
No aplica (solo documentación; no se ejecuta nada en el servidor).

## Verificación LOCAL (sí ejecutable)
```bash
wc -l docs/unidad3/README.md   # <= 60
grep -n 'results/unidad3' docs/unidad3/README.md
```

## Criterios de aceptación
- [ ] Rutas exactas del contrato §12; sin datos sensibles.
- [ ] Explica el porqué del padding sin copiar código.

## Errores típicos
- Decir que el padding acelera "siempre" (solo elimina el conflicto de lectura de As con TILE=32; matizarlo).
- Citar resultados de U1 (otra máquina).

## Commit
`docs(unidad3): guia de resultados para el informe`
