/**
 * @file matmul_gpu.h
 * @brief API de las implementaciones GPU (CUDA) de C = A x B en precision simple.
 *
 * Proyecto: Multiplicacion de Matrices de Gran Escala, CPU vs GPU (IF557ZIN).
 * Unidad 2: kernel naive en memoria global (codigo EVALUABLE).
 * Unidad 3: kernel con memoria compartida (tiling) y su variante con padding
 *           (codigo EVALUABLE) + cuBLAS como REFERENCIA de rendimiento.
 *
 * Contrato comun (identico a include/matmul_cpu.h):
 *  - Matrices cuadradas N x N de `float` en orden por filas (row-major): el
 *    elemento (i, j) esta en m[i*N + j].
 *  - Los punteros `d_*` son de DISPOSITIVO y los reserva quien llama
 *    (N*N floats cada uno); `h_*` son de host. Las funciones no reservan ni
 *    liberan memoria de dispositivo.
 *  - A y B son de solo lectura; C se sobrescribe por completo (no es necesario
 *    inicializarla).
 *  - N = 0 no lanza ningun kernel y devuelve 0.0 en las variantes cronometradas.
 *  - N > 2^31-1 se rechaza con mensaje a stderr y exit(1) (los indices de los
 *    kernels son `int` de 32 bits).
 *  - Coste: 2*N^3 operaciones de punto flotante.
 *
 * Las funciones *_timed miden SOLO el kernel con cudaEvent (tiempo de computo
 * puro); las *_total_timed miden host->device + kernel + device->host, es decir
 * el tiempo total percibido por quien llama.
 *
 * La interfaz tiene enlace C para poder llamarla desde el binario de benchmark
 * (src/gpu/main_gpu.cu) y desde cualquier otro lenguaje.
 *
 * Implementacion: src/gpu/matmul_gpu.cu (escrita desde cero).
 */
#ifndef MATMUL_GPU_H
#define MATMUL_GPU_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ------------------------------------------------------------------------- */
/* Unidad 2: kernel naive (un hilo por elemento, solo memoria global)         */
/* ------------------------------------------------------------------------- */

/**
 * @brief C = A x B con el kernel naive (un hilo calcula C[i][j]).
 *
 * Cada hilo recorre toda la fila i de A y toda la columna j de B desde memoria
 * global: A se lee de forma coalescida por filas y B con salto de N elementos.
 *
 * @param d_A Matriz A en el dispositivo (N*N floats, solo lectura).
 * @param d_B Matriz B en el dispositivo (N*N floats, solo lectura).
 * @param d_C Matriz C en el dispositivo (N*N floats, salida).
 * @param N   Dimension.
 */
void matmul_gpu_naive(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/** @brief Igual que matmul_gpu_naive(), devolviendo el tiempo del kernel en segundos. */
double matmul_gpu_naive_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/**
 * @brief Igual que matmul_gpu_naive(), midiendo host->device + kernel + device->host.
 *
 * @param h_A Matriz A en host; h_B idem; h_C recibe el resultado.
 * @param d_A,d_B,d_C Buffers de dispositivo reutilizables (evitan reservar memoria
 *                    dentro de la medicion).
 */
double matmul_gpu_naive_total_timed(const float* h_A, const float* h_B, float* h_C,
                                    float* d_A, float* d_B, float* d_C, uint64_t N);

/* ------------------------------------------------------------------------- */
/* Unidad 3: kernel tiled (memoria compartida) y variante con padding         */
/* ------------------------------------------------------------------------- */

/** @brief C = A x B con tiles de memoria compartida TILE=32 (sin padding). */
void matmul_gpu_tiled(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/** @brief C = A x B con tiles de memoria compartida TILE=32 y padding (stride TILE+1). */
void matmul_gpu_tiled_pad(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/** @brief Tiempo del kernel tiled (solo kernel, cudaEvent). */
double matmul_gpu_tiled_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/** @brief Tiempo del kernel tiled con padding (solo kernel, cudaEvent). */
double matmul_gpu_tiled_pad_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/** @brief Tiempo total del kernel tiled (H2D + kernel + D2H). */
double matmul_gpu_tiled_total_timed(const float* h_A, const float* h_B, float* h_C,
                                    float* d_A, float* d_B, float* d_C, uint64_t N);

/** @brief Tiempo total del kernel tiled con padding (H2D + kernel + D2H). */
double matmul_gpu_tiled_pad_total_timed(const float* h_A, const float* h_B, float* h_C,
                                        float* d_A, float* d_B, float* d_C, uint64_t N);

/* ------------------------------------------------------------------------- */
/* REFERENCIA (no evaluable): cuBLAS como techo de rendimiento de la GPU      */
/* ------------------------------------------------------------------------- */

/**
 * @brief Inicializa el handle de cuBLAS (idempotente).
 * @return 0 si el handle esta listo (aborta con exit(1) si falla).
 */
int matmul_gpu_cublas_init(void);

/** @brief Destruye el handle de cuBLAS (no hace nada si no se creo). */
void matmul_gpu_cublas_destroy(void);

/**
 * @brief C = A x B con cublasSgemm. REFERENCIA: solo llama a la API de cuBLAS.
 * @return Tiempo de la llamada a cublasSgemm en segundos.
 */
double matmul_gpu_cublas_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N);

/** @brief Igual que matmul_gpu_cublas_timed(), midiendo H2D + gemm + D2H. */
double matmul_gpu_cublas_total_timed(const float* h_A, const float* h_B, float* h_C,
                                     float* d_A, float* d_B, float* d_C, uint64_t N);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MATMUL_GPU_H */
