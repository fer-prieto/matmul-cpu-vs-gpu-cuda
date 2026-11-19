# U3-T01 — kernel tiled con memoria compartida (TILE=32) + variante con padding

- **ID**: U3-T01
- **Depende de**: U2-T02
- **Objetivo**: añadir `gpu_tiled` (sin padding) y `gpu_tiled_pad` (As/Bs con stride 33) a la librería GPU, para la ablación de bank conflicts.

## Archivos
- Modificar: `include/matmul_gpu.h` (añadir declaraciones), `src/gpu/matmul_gpu.cu` (añadir kernels + wrappers). Nada más.

## Especificación
Header (extern "C", misma documentación que las funciones naive):
```c
void   matmul_gpu_tiled(const float* d_A, const float* d_B, float* d_C, uint64_t N);
void   matmul_gpu_tiled_pad(const float* d_A, const float* d_B, float* d_C, uint64_t N);
double matmul_gpu_tiled_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);        // solo kernel
double matmul_gpu_tiled_pad_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);
double matmul_gpu_tiled_total_timed(const float* h_A, const float* h_B, float* h_C,
                                    float* d_A, float* d_B, float* d_C, uint64_t N);              // H2D+kernel+D2H
double matmul_gpu_tiled_pad_total_timed(const float* h_A, const float* h_B, float* h_C,
                                        float* d_A, float* d_B, float* d_C, uint64_t N);
```
`.cu` — kernels `__global__` static, escritos desde cero a partir de ESTA especificación (sin pseudocódigo adicional):
- Constantes del archivo: `TILE = 32`, `PAD = TILE + 1` (= 33). grid `(ceil(N/32), ceil(N/32))`; block `(32, 32)`.
- `kernel_tiled`: `__shared__ float As[32][32], Bs[32][32]`.
- `kernel_tiled_pad`: `__shared__ float As[32][33], Bs[32][33]` (misma lógica, strides 33).
- Índices: `row = blockIdx.y*32 + threadIdx.y`; `col = blockIdx.x*32 + threadIdx.x`; `ty = threadIdx.y`, `tx = threadIdx.x`.
- Estructura del kernel (SIN escribir el cuerpo: el ejecutor lo implementa con estas reglas exactas):
  1. `acc` local en registro = 0.
  2. Bucle sobre tiles de K: `kt = 0, 32, 64, …` mientras `kt < N`:
     a. Carga de A: si `row < N` y `kt + tx < N` → `As[ty][tx] = A[row*N + (kt+tx)]`.
     b. Carga de B: si `kt + ty < N` y `col < N` → `Bs[ty][tx] = B[(kt+ty)*N + col]`.
     c. `__syncthreads()` (tras las DOS cargas, antes de leer).
     d. Bucle `k = 0..31`: si `kt + k < N` → `acc += As[ty][k] * Bs[k][tx]`.
     e. `__syncthreads()` (tras consumir el tile, antes de la siguiente carga).
  3. Si `row < N && col < N` → `C[row*N + col] = acc` (un solo write).
- Análisis de bancos para el comentario del .cu (el informe lo citará):
  - Sin padding: lectura `As[ty][k]` con k fijo → dirección `ty*32 + k` → banco `(ty*32+k) % 32 = k` para todo ty → **conflicto 32-way**. `Bs[k][tx]` → banco `tx` → sin conflicto.
  - Con padding (stride 33): banco de `As[ty][k]` = `(ty*33+k) % 32 = (ty+k) % 32` → sin conflicto; `Bs[k][tx]` → `(k*33+tx) % 32 = (k+tx) % 32` → sin conflicto.
- Wrappers host: idénticos en estructura a los de naive (U2-T02): lanzamiento, `*_timed` con cuda_timer alrededor del kernel, `*_total_timed` = H2D(A)+H2D(B)+kernel+D2H(C) entre timer. Mismas guardas de N (0 → no lanzar; > INT32_MAX → exit 1).

## Casos límite
- N=1024/2048/4096 múltiplos de 32 → guardas siempre verdaderas; quedan para N arbitrario.
- N no múltiplo de 32: las guardas evitan lecturas/escrituras fuera de rango; `As`/`Bs` fuera del tile válido pueden quedar sin inicializar pero NUNCA se leen (k con `kt+k >= N` no se acumula).

## Verificación LOCAL (sin nvcc: solo estructural)
```bash
grep -c '__syncthreads' src/gpu/matmul_gpu.cu   # >= 4 (2 por kernel x 2 kernels)
grep -n '32\]\[33\]\|As\[ty\]\[k\]\|Bs\[k\]\[tx\]' src/gpu/matmul_gpu.cu   # índices exactos presentes
grep -n 'matmul_gpu_tiled' include/matmul_gpu.h  # 6 declaraciones
```

## Verificación EN SERVIDOR (la corre el usuario; requiere U3-T03 para medirlas)
```bash
make gpu
./build/bench_gpu --unit 3 --impls gpu_tiled,gpu_tiled_pad --sizes 1024 --reps 3 --run-id TEST3
# esperado: rc=0; ambas validadas (rel_error <= 1e-4); gpu_tiled_pad >= gpu_tiled en GFLOPS (sin padding sufre el conflicto 32-way)
```

## Criterios de aceptación
- [ ] Compila sin warnings; ambas variantes validadas en N=1024.
- [ ] `gflops(gpu_tiled_pad) >= gflops(gpu_tiled)` en los tres N (evidencia del conflicto).
- [ ] `t_kernel_s > 0` y `t_total_s >= t_kernel_s` en todas las filas.

## Errores típicos (riesgo alto)
- `__syncthreads()` fuera del bucle de tiles o dentro de un `if` que no toman todos los hilos (deadlock).
- Leer `As`/`Bs` antes del primer sync o tras la siguiente carga sin sync.
- Guardas asimétricas entre la carga y el uso (elemento sin cargar que sí se lee → basura).
- `As[32][32]` en la variante pad (debe ser `[32][33]`) o stride 32 en lugar de 33.
- Escribir `C` dentro del bucle de tiles (solo una vez, al final).

## Commit
`feat(gpu): kernel tiled con memoria compartida (TILE=32)`
