/**
 * @file matmul_cpu.h
 * @brief API propia (código EVALUABLE) de la multiplicación de matrices densas
 *        en CPU, secuencial, precisión simple (FP32).
 *
 * Proyecto: Multiplicación de Matrices de Gran Escala, CPU vs GPU (IF557ZIN).
 * Unidad 1: línea base secuencial en CPU.
 *
 * Todas las funciones calculan C = A × B con matrices cuadradas de N × N
 * elementos `float`, guardadas como arreglos 1D en orden por filas
 * (row-major): el elemento (i, j) de una matriz M está en M[i*N + j].
 * Es la misma disposición que se usará en los kernels CUDA (Unidades 2 y 3).
 *
 * Contrato común a todas las funciones:
 *  - A y B son de solo lectura; C es la salida.
 *  - C NO debe solaparse con A ni con B (los punteros se declaran `restrict`).
 *    A y B sí pueden ser la misma matriz (por ejemplo, para calcular A × A),
 *    porque ninguna de las dos se modifica.
 *  - El contenido previo de C se ignora: i-j-k escribe cada elemento de C una
 *    sola vez y i-k-j **pone C a cero** antes de acumular sobre ella. Por eso
 *    da igual que C llegue con basura.
 *  - Las funciones no reservan memoria: la gestiona quien llama. No se exige
 *    alineación especial (se recomienda alinear a 64 bytes, una línea de caché).
 *  - N es un entero de 64 bits sin signo, para que `i*N + j` no desborde.
 *  - N = 0 es válido y no hace nada.
 *  - Coste: 2·N³ operaciones de punto flotante (N³ multiplicaciones y N³ sumas).
 *
 * La interfaz tiene enlace C (`extern "C"`) para poder llamarla desde el arnés
 * en Rust (`harness/`, crate `matbench`) y desde cualquier otro lenguaje.
 *
 * Implementación: src/cpu/matmul_cpu.cpp (escrita desde cero).
 */
#ifndef MATMUL_CPU_H
#define MATMUL_CPU_H

#include <stdint.h>

/* `restrict` es estándar en C99; en C++ se usa la extensión `__restrict`,
 * que aceptan GCC, Clang y MSVC. */
#ifndef MATMUL_RESTRICT
#ifdef __cplusplus
#define MATMUL_RESTRICT __restrict
#else
#define MATMUL_RESTRICT restrict
#endif
#endif

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Producto C = A × B con el orden de bucles i-j-k (versión ingenua).
 *
 * Es la forma "de libro": para cada elemento C[i][j] recorre la fila i de A y
 * la columna j de B. El acceso a B por columnas salta N floats en cada paso,
 * por lo que aprovecha mal la caché. Sirve como **referencia de correctitud**
 * y para mostrar el efecto del orden de los bucles en el rendimiento.
 *
 * @param A  Matriz N×N de entrada (row-major), solo lectura.
 * @param B  Matriz N×N de entrada (row-major), solo lectura.
 * @param C  Matriz N×N de salida (row-major). Se sobrescribe por completo.
 * @param N  Dimensión de las matrices.
 */
void matmul_cpu_ijk(const float* MATMUL_RESTRICT A,
                    const float* MATMUL_RESTRICT B,
                    float* MATMUL_RESTRICT C,
                    uint64_t N);

/**
 * @brief Producto C = A × B con el orden de bucles i-k-j (versión optimizada).
 *
 * Intercambia los dos bucles internos: el bucle más interno recorre una fila
 * de B y una fila de C de forma contigua, lo que aprovecha las líneas de caché
 * y permite la vectorización SIMD automática. **Es la línea base oficial** del
 * proyecto (T_CPU en los speedups de las Unidades 2 y 3).
 *
 * Pone C a cero y luego acumula sobre ella.
 *
 * @param A  Matriz N×N de entrada (row-major), solo lectura.
 * @param B  Matriz N×N de entrada (row-major), solo lectura.
 * @param C  Matriz N×N de salida (row-major). Se sobrescribe por completo.
 * @param N  Dimensión de las matrices.
 */
void matmul_cpu_ikj(const float* MATMUL_RESTRICT A,
                    const float* MATMUL_RESTRICT B,
                    float* MATMUL_RESTRICT C,
                    uint64_t N);

/**
 * @brief Igual que matmul_cpu_ijk(), pero devuelve el tiempo de cómputo.
 *
 * El tiempo se mide con `std::chrono::steady_clock` (reloj monótono) solo
 * alrededor del triple bucle. No incluye reservas de memoria ni la
 * inicialización de las matrices.
 *
 * @return Tiempo de cómputo en segundos.
 */
double matmul_cpu_ijk_timed(const float* MATMUL_RESTRICT A,
                            const float* MATMUL_RESTRICT B,
                            float* MATMUL_RESTRICT C,
                            uint64_t N);

/**
 * @brief Igual que matmul_cpu_ikj(), pero devuelve el tiempo de cómputo.
 *
 * El tiempo se mide con `std::chrono::steady_clock` solo alrededor del triple
 * bucle. La puesta a cero de C se hace ANTES de empezar a medir: es O(N²) y,
 * frente al O(N³) del producto, no cambia el resultado (el arnés en Rust
 * registra aparte el tiempo de la llamada completa, `t_call_s`, que sí la
 * incluye, por si se quiere comparar).
 *
 * @return Tiempo de cómputo en segundos.
 */
double matmul_cpu_ikj_timed(const float* MATMUL_RESTRICT A,
                            const float* MATMUL_RESTRICT B,
                            float* MATMUL_RESTRICT C,
                            uint64_t N);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MATMUL_CPU_H */
