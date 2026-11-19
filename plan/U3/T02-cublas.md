# U3-T02 — cuBLAS como referencia (techo) con cublasSgemm

- **ID**: U3-T02
- **Depende de**: U3-T01
- **Objetivo**: envoltorio de `cublasSgemm` como `kind=referencia` (techo de la GPU), con medición igual a las impls propias.

## Archivos
- Modificar: `include/cuda_util.h` (añadir `CUBLAS_CHECK`), `include/matmul_gpu.h` (declaraciones), `src/gpu/matmul_gpu.cu` (implementación + handle estático).

## Especificación
`include/cuda_util.h` — junto a CUDA_CHECK:
```c
#include <cublas_v2.h>
#define CUBLAS_CHECK(expr) do { cublasStatus_t s_ = (expr); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "cuBLAS error %s:%d en %s: status %d\n", __FILE__, __LINE__, #expr, (int)s_); exit(1); } } while (0)
```
`include/matmul_gpu.h`:
```c
/* REFERENCIA (no evaluable): techo de la GPU con cuBLAS. Ver aclaracion de matmul_ref.h. */
int    matmul_gpu_cublas_init(void);    // cublasCreate del handle interno; 0 ok
void   matmul_gpu_cublas_destroy(void); // cublasDestroy (no-op si no creado)
double matmul_gpu_cublas_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);        // solo la llamada
double matmul_gpu_cublas_total_timed(const float* h_A, const float* h_B, float* h_C,
                                     float* d_A, float* d_B, float* d_C, uint64_t N);
```
`src/gpu/matmul_gpu.cu`:
- Handle `cublasHandle_t` como variable estática de archivo; `init` lo crea (CUBLAS_CHECK), `destroy` lo libera; `*_timed` aborta (exit 1, stderr) si se llama sin init.
- Llamada EXACTA (conversión row-major → column-major; comentar el porqué: en column-major `C = A·B` equivale en row-major a `Cᵀ = Bᵀ·Aᵀ`, por eso los operandos van intercambiados):
  `cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, (int)N, (int)N, (int)N, &alpha, d_B, (int)N, d_A, (int)N, &beta, d_C, (int)N)` con `alpha = 1.0f`, `beta = 0.0f` (C se pisa por completo).
- `*_timed` = timer alrededor de la llamada (con CUBLAS_CHECK). `*_total_timed` = timer alrededor de H2D(A)+H2D(B)+sgemm+D2H(C). N=0 → 0.0 sin llamar; N > INT32_MAX → exit 1.
- Comentario de cabecera: "Implementación de REFERENCIA, no evaluable" (mismo espíritu que `src/cpu/matmul_openblas.cpp`).

## Casos límite
- `cublas_init` dos veces: no-op la segunda (guardar flag). `destroy` sin init: no-op.
- Validación de cublas contra cpu_ikj espera `rel_error ≈ 1e-6` (misma tolerancia 1e-4).

## Verificación LOCAL (sin nvcc: solo estructural)
```bash
grep -n 'cublasSgemm' src/gpu/matmul_gpu.cu          # 1 llamada, con d_B antes que d_A
grep -n 'beta' src/gpu/matmul_gpu.cu | head -2       # beta = 0.0f documentado
grep -n 'CUBLAS_CHECK' include/cuda_util.h           # macro presente
```

## Verificación EN SERVIDOR (la corre el usuario; requiere U3-T03)
```bash
make gpu
./build/bench_gpu --unit 3 --impls gpu_cublas --sizes 1024 --reps 3 --run-id TESTC
# esperado: rc=0; validado vs cpu_ikj; gflops en [1000, 11000] (techo GP100 ~10.6 TFLOPS)
```

## Criterios de aceptación
- [ ] Compila y enlaza con `-lcublas` (ya en Makefile, U2-T04).
- [ ] `gpu_cublas` validado vs cpu_ikj; GFLOPS dentro de la cota de cordura.
- [ ] Comentario de referencia presente; sin código copiado (solo llamada a API).

## Errores típicos (riesgo alto)
- Orden de operandos: `(d_A, d_B)` daría Cᵀ — usar exactamente `(d_B, d_A)` con ambos `CUBLAS_OP_N`.
- `beta != 0` (contaminaría C). `alpha != 1`.
- Llamar sin `cublasCreate` (handle nulo → segfault/status error).
- Olvidar `-lcublas` en el enlace (falla "undefined reference to cublasSgemm_v2").

## Commit
`feat(gpu): cuBLAS como referencia (techo)`
