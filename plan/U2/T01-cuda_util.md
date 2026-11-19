# U2-T01 — include/cuda_util.h + src/gpu/cuda_util.cu (errores, timer, info de dispositivo)

- **ID**: U2-T01
- **Depende de**: U0-T05
- **Objetivo**: macro de error CUDA, timer con cudaEvent y cadena sanitizada de dispositivo, reutilizables por todo `src/gpu/`.

## Archivos
- Crear: `include/cuda_util.h`, `src/gpu/cuda_util.cu`.

## Especificación
`include/cuda_util.h` (guard `MATMUL_CUDA_UTIL_H`, comentarios en español, estilo de `include/matmul_cpu.h`):
```c
#include <cuda_runtime.h>
#define CUDA_CHECK(expr) do { cudaError_t e_ = (expr); if (e_ != cudaSuccess) { \
    fprintf(stderr, "CUDA error %s:%d en %s: %s\n", __FILE__, __LINE__, #expr, cudaGetErrorString(e_)); exit(1); } } while (0)
typedef struct { cudaEvent_t start, stop; } cuda_timer_t;
void cuda_timer_begin(cuda_timer_t* t);        // cudaEventCreate x2 (con CUDA_CHECK) + cudaEventRecord(start)
double cuda_timer_end_s(cuda_timer_t* t);      // record(stop) + cudaEventSynchronize(stop) + ms->s + destroy x2; devuelve segundos
const char* matmul_gpu_device_info(void);      // cadena estatica: "name=<n>; cc=<maj>.<min>; mem_total_mib=<m>; driver=<dv>; runtime=<rv>"
int matmul_gpu_init(void);                     // cudaSetDevice(0) con CUDA_CHECK; devuelve 0 siempre (exit 1 si falla)
```
`src/gpu/cuda_util.cu`:
- `cuda_timer_end_s`: `cudaEventElapsedTime(&ms, t->start, t->stop)` tras sincronizar; `ms * 1e-3`; destruir ambos eventos SIEMPRE (también si falla la sync → CUDA_CHECK aborta antes; documentar).
- `matmul_gpu_device_info`: construir con `cudaGetDeviceProperties` (`name`, `major`, `minor`, `totalGlobalMem >> 20`), `cudaDriverGetVersion`, `cudaRuntimeGetVersion`. **Sin** UUID (no se consulta), **sin** PCI bus ID. Guardar en `static std::string` (construcción una vez, patrón de `matmul_util.cpp:build_info`).

## Casos límite
- `cuda_timer_begin` dos veces sin `end` = leak (documentar: prohibido; el timer siempre se empareja en la misma función).
- Timer sobre el stream por defecto (0): todo el benchmark usa stream 0.

## Verificación LOCAL (sin nvcc: solo estructural)
```bash
grep -c 'CUDA_CHECK' include/cuda_util.h src/gpu/cuda_util.cu            # >= 1 / >= 4
grep -n 'uuid\|UUID\|pciBusID' include/cuda_util.h src/gpu/cuda_util.cu  # sin resultados
```

## Verificación EN SERVIDOR (la corre el usuario; compila con U2-T04)
```bash
make gpu 2>&1 | tail -3        # compila sin warnings
./build/bench_gpu --help 2>&1 | head -3   # (tras U2-T03) no imprime errores CUDA
```

## Criterios de aceptación
- [ ] Compila con nvcc sin warnings (`-Wall -Wextra` activos).
- [ ] `matmul_gpu_device_info()` devuelve `cc=6.0` y `name=Quadro GP100` en el servidor.
- [ ] Ninguna cadena sensible en el código (grep vacío).

## Errores típicos
- Olvidar `cudaEventSynchronize` (tiempos 0 o basura).
- No destruir eventos (leak por repetición).
- Incluir `pciBusID`/UUID en la info (viola §14 del contrato).
- `exit(1)` en código que no sea infraestructura de arranque (aquí SÍ es infraestructura: macro de error, permitido).

## Commit
`feat(gpu): utilidades CUDA (errores, timer, info de dispositivo)`
