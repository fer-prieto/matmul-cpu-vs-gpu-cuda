# U2-T02 — include/matmul_gpu.h + src/gpu/matmul_gpu.cu (kernel naive, memoria global)

- **ID**: U2-T02
- **Depende de**: U2-T01
- **Objetivo**: kernel naive (1 hilo por elemento, solo memoria global) con medición de kernel y de total, según contrato §5/Q2/Q3.

## Archivos
- Crear: `include/matmul_gpu.h`, `src/gpu/matmul_gpu.cu`.

## Especificación
`include/matmul_gpu.h` (guard `MATMUL_GPU_H`; `extern "C"`; FP32 row-major `m[i*N+j]`; documentar como `matmul_cpu.h`):
```c
void   matmul_gpu_naive(const float* d_A, const float* d_B, float* d_C, uint64_t N);      // sin medir
double matmul_gpu_naive_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);         // solo kernel, cudaEvent, s
double matmul_gpu_naive_total_timed(const float* h_A, const float* h_B, float* h_C,
                                    float* d_A, float* d_B, float* d_C, uint64_t N);               // H2D(A)+H2D(B)+kernel+D2H(C)
```
Contratos (comentarlos en el header): `d_*` = punteros de dispositivo ya reservados por el llamador (N² floats); `h_*` = host; C se sobrescribe por completo; N=0 no lanza nada y devuelve 0.0; N > INT32_MAX rechazado (error a stderr, exit 1).

`src/gpu/matmul_gpu.cu` (SIN pseudocódigo de implementación en la tarea; el kernel se escribe desde cero a partir de ESTA especificación):
- `__global__ void kernel_naive(const float* __restrict__ A, const float* __restrict__ B, float* __restrict__ C, int N)` — static, solo visible en este .cu.
  - grid: `(ceil(N/16), ceil(N/16), 1)`; block: `(16, 16, 1)`.
  - Índices: `i = blockIdx.y*16 + threadIdx.y`, `j = blockIdx.x*16 + threadIdx.x`.
  - Si `i < N && j < N`: `C[i*N+j] = Σ_{k=0..N-1} A[i*N+k] * B[k*N+j]` (acumulador local en registro, un solo write a C).
  - Nota pedagógica para el comentario del .cu: lectura de A coalescida por filas; B se lee por columnas con stride N (no coalescida) — es lo que el informe debe documentar.
- Host: `matmul_gpu_naive` lanza el kernel (CUDA_CHECK del lanzamiento). `matmul_gpu_naive_timed` = `cuda_timer_begin` → lanzar → `cuda_timer_end_s`. `matmul_gpu_naive_total_timed` = timer → `cudaMemcpy(d_A, h_A, N²·4, cudaMemcpyHostToDevice)` → ídem B → kernel → `cudaMemcpy(h_C, d_C, N²·4, cudaMemcpyDeviceToHost)` → end. N se convierte a `int` con guarda.
- Medición solo con `cuda_timer_*` (U2-T01). Sin `__syncthreads` aquí (no hay shared).

## Casos límite
- N=1024/2048/4096 son múltiplos de 16 → sin guardas necesarias; las guardas quedan para N arbitrario.
- N=0: no lanzar (grid 0 es error CUDA). N > INT32_MAX: stderr + exit 1.

## Verificación LOCAL (sin nvcc: solo estructural)
```bash
grep -c '__global__\|kernel_naive' src/gpu/matmul_gpu.cu                       # >= 2
grep -n 'cuda_timer_begin\|cuda_timer_end_s\|cudaMemcpy' src/gpu/matmul_gpu.cu # presentes en timed/total
grep -n '__syncthreads' src/gpu/matmul_gpu.cu                                  # vacío (no aplica aquí)
```

## Verificación EN SERVIDOR (la corre el usuario; compila con U2-T04 y corre con U2-T03)
```bash
make gpu
./build/bench_gpu --unit 2 --sizes 512 --reps 3   # exit 0; validacion OK (rel_error <= 1e-4)
```

## Criterios de aceptación
- [ ] Compila sin warnings; smoke test N=512 validado.
- [ ] `t_kernel_s > 0` y `t_total_s >= t_kernel_s` en todas las filas.
- [ ] Kernel escrito desde cero (sin código copiado de cuda-samples/SGEMM_CUDA).

## Errores típicos (riesgo alto)
- Usar `uint64_t` en el kernel (índices de 64 bits penalizan; usar `int` con guarda).
- Medir la primera ejecución (páginas no residentes): los buffers son persistentes y hay warm-up en bench_gpu; el timer NUNCA incluye cudaMalloc.
- No sincronizar el evento final (tiempo 0).
- Incluir transferencias en `*_timed` (eso es `*_total_timed`).
- `cudaMemcpy` sin `cudaMemcpyHostToDevice`/`DeviceToHost` explícito.

## Commit
`feat(gpu): kernel naive en memoria global`
