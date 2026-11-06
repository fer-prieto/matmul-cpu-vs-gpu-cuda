/**
 * @file matmul_ref.h
 * @brief API de las implementaciones de REFERENCIA (NO evaluables) que envuelven
 *        bibliotecas externas, para comparar el código propio con el techo
 *        práctico de la CPU.
 *
 * Proyecto: Multiplicación de Matrices de Gran Escala, CPU vs GPU (IF557ZIN).
 * Unidad 1: línea base secuencial en CPU.
 *
 * ⚠️ Aclaración importante (consigna y plan de avance, sección 8): estas
 * funciones NO forman parte del código evaluable. Solo LLAMAN a OpenBLAS a
 * través de su API pública (CBLAS) para ofrecer un punto de comparación (techo
 * de rendimiento) y una referencia de validación en tamaños donde la versión
 * ingenua propia es demasiado lenta. El algoritmo evaluable está en
 * matmul_cpu.h, escrito desde cero.
 *
 * Este archivo solo se compila cuando el Makefile se invoca con OPENBLAS=1
 * (ver src/cpu/matmul_openblas.cpp). Sin esa opción, el proyecto funciona
 * exactamente igual sin OpenBLAS.
 *
 * Disposición de datos: matrices cuadradas N×N en FP32, arreglos 1D en orden
 * por filas (row-major): el elemento (i, j) está en `m[i*N + j]`, igual que en
 * matmul_cpu.h. `a`, `b` y `c` no deben solaparse.
 */

#ifndef MATMUL_REF_H
#define MATMUL_REF_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief C = A × B con OpenBLAS (`cblas_sgemm`) usando exactamente 1 hilo.
 *
 * Fuerza `OPENBLAS_NUM_THREADS=1` (vía `openblas_set_num_threads`) antes de
 * llamar a `cblas_sgemm`, para que la comparación con las implementaciones
 * propias (secuenciales) sea justa. Devuelve el tiempo de cómputo en segundos
 * medido con `std::chrono::steady_clock` solo alrededor de la llamada a
 * `cblas_sgemm`.
 *
 * @param a  Matriz A (N×N, solo lectura, row-major).
 * @param b  Matriz B (N×N, solo lectura, row-major).
 * @param c  Matriz de salida C (N×N, row-major). Se sobrescribe por completo
 *           (beta = 0), así que su contenido previo es irrelevante.
 * @param n  Dimensión N (> 0; en x86-64 `int` admite hasta 2^31, de sobra).
 * @return   Tiempo de la llamada a `cblas_sgemm` en segundos.
 */
double matmul_ref_openblas_1t_timed(const float* a, const float* b, float* c,
                                    uint64_t n);

/**
 * @brief C = A × B con OpenBLAS (`cblas_sgemm`) multihilo (solo informativa).
 *
 * Fija `threads` hilos (si `threads` es 0 se usan todos los disponibles, el
 * comportamiento por defecto de OpenBLAS) y mide igual que la variante de
 * 1 hilo. NO sirve para una comparación justa con el código secuencial: es el
 * techo absoluto de la CPU y la referencia que se usará contra la GPU en la
 * Unidad 2.
 *
 * @param threads  Número de hilos (0 = todos los que OpenBLAS considere).
 * @param ...      Resto de parámetros igual que matmul_ref_openblas_1t_timed.
 * @return         Tiempo de la llamada a `cblas_sgemm` en segundos.
 */
double matmul_ref_openblas_mt_timed(const float* a, const float* b, float* c,
                                    uint64_t n, uint64_t threads);

/**
 * @brief Información de la biblioteca OpenBLAS enlazada, para el registro del
 *        entorno de prueba (env_<run>.json).
 *
 * @param config        Cadena con la configuración completa
 *                      (`openblas_get_config`): versión, DYNAMIC_ARCH,
 *                      MAX_THREADS, etc.
 * @param corename      Nombre del micro-kernel seleccionado para esta CPU
 *                      (`openblas_get_corename`).
 * @param parallel_mode 0 = secuencial, 1 = pthreads, 2 = OpenMP
 *                      (`openblas_get_parallel`).
 * @param num_procs     Número de unidades de ejecución que OpenBLAS puede usar
 *                      (`openblas_get_num_procs`).
 *
 * Los cuatro punteros de salida reciben valores válidos de OpenBLAS; si la
 * biblioteca no informara alguno, se recibe una cadena vacía o 0.
 */
void matmul_ref_openblas_info(const char** config, const char** corename,
                              int* parallel_mode, int* num_procs);

#ifdef __cplusplus
}
#endif

#endif /* MATMUL_REF_H */
