/**
 * @file matmul_util.h
 * @brief Utilidades AUXILIARES (no evaluables) compartidas por el binario C++
 *        autónomo y el arnés en Rust.
 *
 * No forman parte del algoritmo que se evalúa (ese está en matmul_cpu.h); solo
 * existen para que las dos herramientas de medición trabajen con datos
 * idénticos y dejen constancia de cómo se compiló la biblioteca:
 *
 *  - Inicialización pseudoaleatoria reproducible con semilla (SplitMix64).
 *    El arnés en Rust implementa el mismo generador (harness/src/rng.rs) y un
 *    test comprueba que ambas versiones producen exactamente los mismos
 *    valores. Así, con la misma semilla, `cpu_baseline` y `matbench` multiplican
 *    las mismas matrices y sus checksums son comparables.
 *  - Checksum de una matriz (suma de sus elementos en doble precisión).
 *  - Información de compilación (compilador, flags, extensiones SIMD activas).
 *
 * Implementación: src/cpu/matmul_util.cpp.
 */
#ifndef MATMUL_UTIL_H
#define MATMUL_UTIL_H

#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/**
 * @brief Rellena `count` floats con valores pseudoaleatorios en [-1, 1).
 *
 * Generador SplitMix64 (Steele, Lea y Flood, 2014). Estado inicial:
 * `seed XOR (stream · 0xD1B54A32D192ED03)`. Cada valor usa los 24 bits altos
 * de la salida de 64 bits: v = (k - 2^23) / 2^23 con k en [0, 2^24), que es
 * exactamente representable en FP32 (no hay redondeo, así que C++ y Rust
 * obtienen los mismos bits).
 *
 * Convención del proyecto: A usa stream 0 y B usa stream 1.
 *
 * @param M       Destino (al menos `count` floats).
 * @param count   Número de elementos (N·N para una matriz).
 * @param seed    Semilla.
 * @param stream  Identificador de la secuencia (0 = A, 1 = B, ...).
 */
void matmul_util_fill_random(float* M, uint64_t count, uint64_t seed, uint64_t stream);

/**
 * @brief Suma de los `count` elementos de M acumulada en double, en orden.
 *
 * Sirve para comprobar que el cálculo no se ha eliminado y para comparar
 * ejecuciones (mismos datos ⇒ mismo checksum, bit a bit).
 */
double matmul_util_checksum(const float* M, uint64_t count);

/**
 * @brief Describe cómo se compiló la biblioteca.
 *
 * Cadena estática con pares `clave=valor` separados por "; ", por ejemplo:
 * `compiler=gcc 16.2.1; cxx_std=201703; flags=-O3 -march=native ...;
 *  isa=sse sse2 ... avx; fma=no; avx2=no; avx512f=no; optimize=yes; fast_math=no`.
 * Los flags llegan desde el Makefile (macro MATMUL_BUILD_FLAGS); las extensiones
 * SIMD salen de las macros que define el propio compilador (__AVX__, etc.).
 */
const char* matmul_util_build_info(void);

#ifdef __cplusplus
} /* extern "C" */
#endif

#endif /* MATMUL_UTIL_H */
