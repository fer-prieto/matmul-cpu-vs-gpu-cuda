# U3-T03 — src/gpu/main_gpu.cu: registro de gpu_tiled, gpu_tiled_pad y gpu_cublas

- **ID**: U3-T03
- **Depende de**: U3-T02
- **Objetivo**: registrar las implementaciones de U3 en bench_gpu (`--unit 3` las mide todas y las valida).

## Archivos
- Modificar: `src/gpu/main_gpu.cu` (solo: tabla de registro, init/destroy de cuBLAS, defaults de `--unit 3`). Nada más.

## Especificación
- Tabla de registro (U2-T03) pasa a tener 4 entradas en este orden: `gpu_naive` (propia), `gpu_tiled` (propia), `gpu_tiled_pad` (propia), `gpu_cublas` (referencia). Punteros: los `*_timed`/`*_total_timed` de `matmul_gpu.h` correspondientes.
- `--unit 3` → impls por defecto = las 4. `--unit 2` → sigue siendo solo `gpu_naive` (comportamiento U2 intacto).
- Init: si las impls seleccionadas incluyen `gpu_cublas` → `matmul_gpu_cublas_init()` (exit 1 si falla); al terminar (también en caminos de error tras el init) → `matmul_gpu_cublas_destroy()`.
- Validación: las 4 impls se validan igual (vs `cpu_ikj`, tolerancia de `--tolerance`); `gpu_cublas` con `kind=referencia` también genera su fila en el validation CSV.
- `--device-info` ya existe (U2-T03); no cambia.

## Casos límite
- `--unit 3 --impls gpu_naive` (subconjunto): no inicializar cuBLAS (no se necesita).
- `--unit 2 --impls gpu_cublas`: permitido explícitamente (se inicializa cuBLAS igual).

## Verificación LOCAL (sin nvcc: solo estructural)
```bash
grep -n 'gpu_tiled\|gpu_tiled_pad\|gpu_cublas' src/gpu/main_gpu.cu | head -8   # entradas de registro + defaults
grep -n 'cublas_init\|cublas_destroy' src/gpu/main_gpu.cu                      # init/destroy presentes
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
make gpu
./build/bench_gpu --unit 3 --sizes 1024 --reps 3 --run-id TESTU3; echo "rc=$?"
# esperado: rc=0; 4 impls x 4 filas (1 warmup + 3 medidas); validation con 4 filas valid=true
./build/bench_gpu --unit 2 --sizes 512 --reps 1 | head -2   # U2 intacta: solo gpu_naive
```

## Criterios de aceptación
- [ ] `--unit 3` mide y valida las 4 impls; `--unit 2` sigue igual que en U2.
- [ ] Sin leaks: destroy de cuBLAS en todos los caminos (revisar flujo de error).
- [ ] GFLOPS esperados: naive < tiled ≤ tiled_pad < cublas (chequeo de cordura en N=1024).

## Errores típicos (riesgo alto)
- Medir cuBLAS sin init (falla con mensaje claro, no segfault).
- Registrar `gpu_cublas` como "propia" (es referencia).
- Cambiar el formato CSV o los exit codes (rompería run_u2/run_u3 y summarize.py).

## Commit
`feat(gpu): registro de tiled, tiled_pad y cublas`
