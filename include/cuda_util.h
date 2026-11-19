/**
 * @file cuda_util.h
 * @brief Utilidades de infraestructura CUDA compartidas por todo src/gpu/:
 *        macro de comprobacion de errores, cronometro con cudaEvent y datos
 *        del dispositivo.
 *
 * Proyecto: Multiplicacion de Matrices de Gran Escala, CPU vs GPU (IF557ZIN).
 * Unidades 2 y 3.
 *
 * Este archivo NO contiene codigo evaluable: solo infraestructura de medida y
 * de manejo de errores. Los kernels y sus envoltorios estan en matmul_gpu.h.
 *
 * - CUDA_CHECK / CUBLAS_CHECK: abortan con mensaje a stderr y exit(1) si la
 *   llamada falla. Se usan en TODAS las llamadas a CUDA y a cuBLAS.
 * - cuda_timer_t: par de eventos CUDA. cuda_timer_begin lo crea y arranca;
 *   cuda_timer_end_s lo detiene, sincroniza, devuelve segundos y destruye los
 *   eventos. Siempre se emparejan en la misma funcion (no reutilizar).
 * - matmul_gpu_device_info(): cadena estatica con nombre, compute capability,
 *   memoria total y versiones de driver/runtime. NO incluye UUID ni bus PCI
 *   (regla de sanitizacion del proyecto).
 */
#ifndef MATMUL_CUDA_UTIL_H
#define MATMUL_CUDA_UTIL_H

#include <cstdio>
#include <cstdlib>

#include <cublas_v2.h>
#include <cuda_runtime.h>

/** Aborta si la llamada CUDA no devuelve cudaSuccess. */
#define CUDA_CHECK(expr)                                                                     \
    do {                                                                                     \
        cudaError_t e_ = (expr);                                                             \
        if (e_ != cudaSuccess) {                                                             \
            std::fprintf(stderr, "CUDA error %s:%d en %s: %s\n", __FILE__, __LINE__, #expr,  \
                         cudaGetErrorString(e_));                                            \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

/** Aborta si la llamada cuBLAS no devuelve CUBLAS_STATUS_SUCCESS. */
#define CUBLAS_CHECK(expr)                                                                   \
    do {                                                                                     \
        cublasStatus_t s_ = (expr);                                                          \
        if (s_ != CUBLAS_STATUS_SUCCESS) {                                                   \
            std::fprintf(stderr, "cuBLAS error %s:%d en %s: status %d\n", __FILE__, __LINE__, \
                         #expr, static_cast<int>(s_));                                       \
            std::exit(1);                                                                    \
        }                                                                                    \
    } while (0)

#ifdef __cplusplus
extern "C" {
#endif

/** Par de eventos CUDA para medir el tiempo de una region del dispositivo. */
typedef struct {
    cudaEvent_t start;
    cudaEvent_t stop;
} cuda_timer_t;

/**
 * @brief Crea los dos eventos y registra el de arranque en el stream 0.
 * @param t Cronometro a inicializar (no debe estar ya iniciado).
 */
void cuda_timer_begin(cuda_timer_t* t);

/**
 * @brief Registra el evento final, espera a que el dispositivo termine y
 *        devuelve los segundos transcurridos. Destruye ambos eventos.
 * @param t Cronometro iniciado con cuda_timer_begin().
 * @return Tiempo en segundos.
 */
double cuda_timer_end_s(cuda_timer_t* t);

/**
 * @brief Describe el dispositivo 0 sin datos sensibles.
 *
 * Formato: `name=<n>; cc=<maj>.<min>; mem_total_mib=<m>; driver=<dv>; runtime=<rv>`.
 * La cadena es estatica y vive todo el programa.
 */
const char* matmul_gpu_device_info(void);

/**
 * @brief Selecciona el dispositivo 0 (CUDA_CHECK interno).
 * @return 0 (aborta el proceso si falla cudaSetDevice).
 */
int matmul_gpu_init(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MATMUL_CUDA_UTIL_H */
