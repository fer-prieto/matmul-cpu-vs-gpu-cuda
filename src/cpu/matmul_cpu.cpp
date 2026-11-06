/**
 * @file matmul_cpu.cpp
 * @brief Implementaciones propias (código EVALUABLE) de C = A × B en CPU:
 *        órdenes de bucles i-j-k (referencia) e i-k-j (línea base optimizada).
 *
 * Escrito desde cero para el proyecto. La documentación de la API está en
 * include/matmul_cpu.h.
 *
 * ---------------------------------------------------------------------------
 * ¿Por qué i-k-j es mucho más rápido que i-j-k?
 * ---------------------------------------------------------------------------
 * Las dos versiones hacen exactamente las mismas 2·N³ operaciones; solo cambia
 * el ORDEN en que recorren la memoria. Con matrices row-major:
 *
 *  i-j-k:  C[i][j] = Σ_k A[i][k] · B[k][j]
 *          El bucle interno (k) recorre una fila de A (contigua) pero una
 *          COLUMNA de B: entre dos iteraciones seguidas salta N floats
 *          (4 KiB con N = 1024). Cada acceso a B cae en una línea de caché
 *          distinta (de los 64 bytes que se traen solo se usan 4), en una
 *          página distinta para N grande (fallos de TLB) y, como el salto es
 *          potencia de 2, todas esas líneas compiten por los mismos conjuntos
 *          de la caché (fallos por conflicto). Además el bucle es una
 *          reducción (`suma += ...`): sin -ffast-math el compilador no puede
 *          reordenar las sumas en punto flotante. GCC 16 con -O3 solo logra
 *          una vectorización parcial: carga 4 elementos de B uno por uno, los
 *          multiplica con una instrucción SIMD de 4 floats y luego suma los
 *          4 productos de uno en uno y en orden (reducción "in-order"). El
 *          resultado es idéntico al de la versión escalar y apenas gana
 *          velocidad. `make asm` guarda el informe del vectorizador y el
 *          ensamblador en build/asm/ para comprobarlo.
 *
 *  i-k-j:  C[i][:] += A[i][k] · B[k][:]
 *          El bucle interno (j) recorre una fila de B y una fila de C de forma
 *          CONTIGUA (salto de 1): cada línea de caché de 64 bytes aporta
 *          16 floats que se usan seguidos (localidad espacial) y el prefetcher
 *          por hardware detecta el patrón secuencial. A[i][k] se lee una vez y
 *          queda en un registro durante todo el bucle j (localidad temporal), y
 *          la fila i de C (N floats) se reutiliza en cada iteración de k, así
 *          que permanece en L1/L2. Por último, las iteraciones de j son
 *          independientes entre sí, de modo que el compilador vectoriza el
 *          bucle por completo con SIMD (con -O3 -march=native en una CPU con
 *          AVX usa vectores de 32 bytes: 8 floats por instrucción).
 *
 * Nota numérica: en ambos órdenes cada C[i][j] se acumula en el mismo orden
 * (k = 0, 1, ..., N-1) y en FP32, así que, si el compilador no fusiona la
 * multiplicación y la suma (FMA) de forma distinta en cada versión, los dos
 * resultados coinciden bit a bit. Bibliotecas como OpenBLAS suman en otro
 * orden y difieren en el orden de 1e-6 (relativo); por eso la validación se
 * hace siempre con tolerancia.
 *
 * ---------------------------------------------------------------------------
 * Medición del tiempo
 * ---------------------------------------------------------------------------
 * Las variantes *_timed usan std::chrono::steady_clock (monótono, no le afectan
 * los cambios de hora del sistema) alrededor del triple bucle únicamente. Los
 * núcleos de cálculo se marcan `noinline` para que la versión cronometrada y la
 * no cronometrada ejecuten exactamente el mismo código máquina.
 */
#include "matmul_cpu.h"

#include <algorithm>
#include <atomic>
#include <chrono>

#if defined(__GNUC__) || defined(__clang__)
#define MATMUL_NOINLINE __attribute__((noinline))
#elif defined(_MSC_VER)
#define MATMUL_NOINLINE __declspec(noinline)
#else
#define MATMUL_NOINLINE
#endif

namespace {

using Clock = std::chrono::steady_clock;

/// Barrera de compilador: impide que el optimizador mueva accesos a memoria
/// a través de las lecturas del reloj. No emite ninguna instrucción.
inline void compiler_barrier() {
    std::atomic_signal_fence(std::memory_order_seq_cst);
}

/// Núcleo i-j-k: un producto escalar fila(A) · columna(B) por cada C[i][j].
MATMUL_NOINLINE void kernel_ijk(const float* __restrict A,
                                const float* __restrict B,
                                float* __restrict C,
                                uint64_t N) {
    for (uint64_t i = 0; i < N; ++i) {
        for (uint64_t j = 0; j < N; ++j) {
            float sum = 0.0f;
            for (uint64_t k = 0; k < N; ++k) {
                // A se lee por filas (contiguo); B por columnas (salto de N).
                sum += A[i * N + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}

/// Núcleo i-k-j: acumula A[i][k] · fila k de B sobre la fila i de C.
/// Requiere que C esté a cero al empezar (lo garantizan las funciones públicas).
MATMUL_NOINLINE void kernel_ikj(const float* __restrict A,
                                const float* __restrict B,
                                float* __restrict C,
                                uint64_t N) {
    for (uint64_t i = 0; i < N; ++i) {
        const float* __restrict a_row = A + i * N;  // fila i de A
        float* __restrict c_row = C + i * N;        // fila i de C
        for (uint64_t k = 0; k < N; ++k) {
            const float a_ik = a_row[k];            // escalar reutilizado en todo el bucle j
            const float* __restrict b_row = B + k * N;  // fila k de B
            for (uint64_t j = 0; j < N; ++j) {
                // B y C se recorren de forma contigua: bucle vectorizable.
                c_row[j] += a_ik * b_row[j];
            }
        }
    }
}

/// Pone a cero las N·N posiciones de C (O(N²)). Con N = 0 no accede a memoria.
inline void zero_matrix(float* C, uint64_t N) {
    std::fill(C, C + N * N, 0.0f);
}

/// Ejecuta `kernel` y devuelve los segundos transcurridos (steady_clock).
template <typename Kernel>
double time_kernel(Kernel kernel,
                   const float* A,
                   const float* B,
                   float* C,
                   uint64_t N) {
    compiler_barrier();
    const Clock::time_point t0 = Clock::now();
    compiler_barrier();
    kernel(A, B, C, N);
    compiler_barrier();
    const Clock::time_point t1 = Clock::now();
    compiler_barrier();
    return std::chrono::duration<double>(t1 - t0).count();
}

}  // namespace

extern "C" {

void matmul_cpu_ijk(const float* __restrict A,
                    const float* __restrict B,
                    float* __restrict C,
                    uint64_t N) {
    kernel_ijk(A, B, C, N);
}

void matmul_cpu_ikj(const float* __restrict A,
                    const float* __restrict B,
                    float* __restrict C,
                    uint64_t N) {
    zero_matrix(C, N);
    kernel_ikj(A, B, C, N);
}

double matmul_cpu_ijk_timed(const float* __restrict A,
                            const float* __restrict B,
                            float* __restrict C,
                            uint64_t N) {
    return time_kernel(kernel_ijk, A, B, C, N);
}

double matmul_cpu_ikj_timed(const float* __restrict A,
                            const float* __restrict B,
                            float* __restrict C,
                            uint64_t N) {
    zero_matrix(C, N);  // fuera de la medición (ver matmul_cpu.h)
    return time_kernel(kernel_ikj, A, B, C, N);
}

}  // extern "C"
