/**
 * @file matmul_openblas.cpp
 * @brief Implementación de REFERENCIA (NO evaluable): envoltorio cronometrado
 *        de `cblas_sgemm` de OpenBLAS.
 *
 * ⚠️ Este archivo NO contiene algoritmo propio: solo llama a la API pública
 * CBLAS de OpenBLAS. Existe para dar un techo de rendimiento con el que
 * comparar las implementaciones propias (matmul_cpu.cpp) y una referencia de
 * validación en N grande, donde la versión ingenua i-j-k es demasiado lenta.
 * La consigna y el plan de avance (sección 8) lo dejan claro: las bibliotecas
 * optimizadas se usan solo como referencia comparativa, no como objeto de
 * reimplementación.
 *
 * Control de hilos (clave para una comparación justa):
 *  - matmul_ref_openblas_1t_timed fija 1 hilo antes de multiplicar, igual que
 *    las implementaciones propias (secuenciales). Es la referencia contra la
 *    que se calcula `pct_of_openblas`.
 *  - matmul_ref_openblas_mt_timed fija el número de hilos que se le pida. Es
 *    solo informativa (techo absoluto de la CPU) y no se compara directamente
 *    con el código secuencial.
 *
 * Compilación: solo el Makefile la compila, y solo con `make OPENBLAS=1`
 * (ver README). Si OpenBLAS no está instalada, este archivo no se usa y el
 * proyecto funciona igual sin él.
 */

#include "matmul_ref.h"

#include <chrono>
#include <cstdint>
#include <cstddef>

// CBLAS: la interfaz C de OpenBLAS (compatible con la de netlib). El header
// puede venir del paquete del sistema o de una ruta indicada al Makefile con
// OPENBLAS_CFLAGS="-I<ruta>".
#include <cblas.h>

// Extensiones de OpenBLAS que no están en la interfaz CBLAS estándar. Se
// declaran aquí (y no en un header ajeno) con la firma documentada en
// https://github.com/OpenMathLib/OpenBLAS/blob/develop/cblas.h
extern "C" {
/// Fija el número de hilos que usan las llamadas posteriores.
void openblas_set_num_threads(int num_threads);
/// Configuración completa: "OpenBLAS <versión> <modo> <target> MAX_THREADS=...".
const char* openblas_get_config(void);
/// Nombre del micro-kernel seleccionado para esta CPU (p. ej. "SandyBridge").
const char* openblas_get_corename(void);
/// 0 = secuencial, 1 = pthreads, 2 = OpenMP.
int openblas_get_parallel(void);
/// Unidades de ejecución que OpenBLAS puede usar en esta máquina.
int openblas_get_num_procs(void);
}

namespace {

/// Llama a cblas_sgemm: C = 1·A×B + 0·C, row-major, sin transponer.
///
/// Con beta = 0 el contenido previo de `c` no interviene (el contrato del
/// header: C se sobrescribe por completo). OpenBLAS empaqueta los bloques y
/// usa micro-kernels en ensamblador; aquí no hay nada que decidir, por eso
/// esta función es de referencia y no evaluable.
void sgemm_nn(const float* a, const float* b, float* c, int n) {
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n,
                1.0f, a, n,
                b, n,
                0.0f, c, n);
}

}  // namespace

extern "C" {

double matmul_ref_openblas_1t_timed(const float* a, const float* b, float* c,
                                    uint64_t n) {
    // La comparación con el código propio (secuencial) debe ser justa: 1 hilo.
    openblas_set_num_threads(1);
    const auto t0 = std::chrono::steady_clock::now();
    sgemm_nn(a, b, c, static_cast<int>(n));
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

double matmul_ref_openblas_mt_timed(const float* a, const float* b, float* c,
                                    uint64_t n, uint64_t threads) {
    // 0 = todos los hilos que OpenBLAS considere (comportamiento por defecto,
    // controlable además con la variable de entorno OPENBLAS_NUM_THREADS).
    if (threads > 0 && threads <= static_cast<uint64_t>(INT32_MAX)) {
        openblas_set_num_threads(static_cast<int>(threads));
    }
    const auto t0 = std::chrono::steady_clock::now();
    sgemm_nn(a, b, c, static_cast<int>(n));
    const auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double>(t1 - t0).count();
}

void matmul_ref_openblas_info(const char** config, const char** corename,
                              int* parallel_mode, int* num_procs) {
    if (config != nullptr) {
        const char* s = openblas_get_config();
        *config = (s != nullptr) ? s : "";
    }
    if (corename != nullptr) {
        const char* s = openblas_get_corename();
        *corename = (s != nullptr) ? s : "";
    }
    if (parallel_mode != nullptr) {
        *parallel_mode = openblas_get_parallel();
    }
    if (num_procs != nullptr) {
        *num_procs = openblas_get_num_procs();
    }
}

}  // extern "C"
