/**
 * @file matmul_gpu.cu
 * @brief Implementaciones GPU (CUDA) de C = A x B en FP32.
 *
 * Escrito desde cero para el proyecto (Unidades 2 y 3). La documentacion de la
 * API esta en include/matmul_gpu.h.
 *
 * ---------------------------------------------------------------------------
 * Que hay aqui
 * ---------------------------------------------------------------------------
 *  1. kernel_naive   (Unidad 2): un hilo calcula C[i][j] recorriendo la fila i
 *                    de A y la columna j de B desde memoria global.
 *  2. kernel_tiled   (Unidad 3): bloques de 32x32 hilos que cooperan cargando
 *                    tiles de A y B en memoria compartida.
 *  3. kernel_tiled_pad: identico, con padding de una columna en los tiles para
 *                    eliminar los conflictos de banco.
 *  4. Envoltorio de cuBLAS (REFERENCIA, no evaluable): solo llama a la API.
 *
 * ---------------------------------------------------------------------------
 * Conflictos de banco en los tiles (justificacion del padding)
 * ---------------------------------------------------------------------------
 * sm_60 tiene 32 bancos de 4 bytes. Con tiles de lado 32 y stride 32:
 *   - As[ty][k]  -> direccion ty*32 + k -> banco (ty*32 + k) % 32 = k, igual
 *     para los 32 hilos de una columna del tile: conflicto de 32 vias.
 *   - Bs[k][tx]  -> banco (k*32 + tx) % 32 = tx, distinto por hilo: sin
 *     conflicto.
 * Con stride 33 (padding):
 *   - As[ty][k]  -> banco (ty*33 + k) % 32 = (ty + k) % 32: distinto por hilo.
 *   - Bs[k][tx]  -> banco (k*33 + tx) % 32 = (k + tx) % 32: distinto por hilo.
 * De ahi que la variante con padding sea mas rapida aunque reserve mas memoria
 * compartida: la comparacion empirica entre ambas es la ablacion de la Unidad 3.
 *
 * ---------------------------------------------------------------------------
 * Medicion
 * ---------------------------------------------------------------------------
 * Las variantes *_timed usan cudaEvent (include/cuda_util.h) alrededor del
 * lanzamiento del kernel; las *_total_timed incluyen ademas las copias
 * host->device y device->host. En ningun caso se reserva memoria dentro de la
 * medicion: los buffers los gestiona el llamador (main_gpu.cu).
 */
#include "matmul_gpu.h"

#include "cuda_util.h"

#include <climits>
#include <cstdio>
#include <cstdlib>

namespace {

constexpr int kNaiveTile = 16;  ///< Lado del bloque del kernel naive.
constexpr int kTile = 32;       ///< Lado del tile del kernel con memoria compartida.
constexpr int kPad = kTile + 1; ///< Stride del tile con padding (evita conflictos).

/// Convierte N a int para los indices del kernel; aborta si no cabe.
int device_size(uint64_t N) {
    if (N > static_cast<uint64_t>(INT_MAX)) {
        std::fprintf(stderr, "error: N = %llu supera el maximo admitido (%d)\n",
                     static_cast<unsigned long long>(N), INT_MAX);
        std::exit(1);
    }
    return static_cast<int>(N);
}

/// Copia N*N floats de host a dispositivo.
void copy_in(float* dst, const float* src, uint64_t N) {
    CUDA_CHECK(cudaMemcpy(dst, src, static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float),
                          cudaMemcpyHostToDevice));
}

/// Copia N*N floats de dispositivo a host.
void copy_out(float* dst, const float* src, uint64_t N) {
    CUDA_CHECK(cudaMemcpy(dst, src, static_cast<size_t>(N) * static_cast<size_t>(N) * sizeof(float),
                          cudaMemcpyDeviceToHost));
}

/* ------------------------------------------------------------------------- */
/* Kernels                                                                   */
/* ------------------------------------------------------------------------- */

/**
 * @brief Un hilo por elemento: C[i][j] = suma_k A[i][k] * B[k][j].
 *
 * El hilo recorre la fila i de A (contigua) y la columna j de B (stride N).
 * Cada elemento de C se escribe una sola vez, al final.
 */
__global__ void kernel_naive(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int N) {
    const int i = blockIdx.y * kNaiveTile + threadIdx.y;
    const int j = blockIdx.x * kNaiveTile + threadIdx.x;
    if (i >= N || j >= N) {
        return;
    }
    float acc = 0.0f;
    for (int k = 0; k < N; ++k) {
        acc += A[i * N + k] * B[k * N + j];
    }
    C[i * N + j] = acc;
}

/**
 * @brief Kernel con tiles de memoria compartida, sin padding.
 *
 * Cada bloque calcula un tile de C de 32x32. En cada iteracion del bucle sobre
 * K, el bloque carga el tile de A (filas) y el de B (columnas) en memoria
 * compartida, se sincroniza, acumula los 32 productos y se sincroniza de nuevo
 * antes de reutilizar el almacenamiento para el siguiente tile.
 */
__global__ void kernel_tiled(const float* __restrict__ A, const float* __restrict__ B,
                             float* __restrict__ C, int N) {
    __shared__ float As[kTile][kTile];
    __shared__ float Bs[kTile][kTile];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    float acc = 0.0f;
    for (int kt = 0; kt < N; kt += kTile) {
        if (row < N && kt + tx < N) {
            As[ty][tx] = A[row * N + kt + tx];
        }
        if (kt + ty < N && col < N) {
            Bs[ty][tx] = B[(kt + ty) * N + col];
        }
        __syncthreads();
        for (int k = 0; k < kTile; ++k) {
            if (kt + k < N) {
                acc += As[ty][k] * Bs[k][tx];
            }
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

/**
 * @brief Igual que kernel_tiled pero con stride kPad en los tiles compartidos.
 *
 * El padding desplaza el inicio de cada fila del tile un elemento, de modo que
 * los 32 hilos de una columna caen en bancos distintos (ver cabecera del
 * archivo). El resto de la logica es identica.
 */
__global__ void kernel_tiled_pad(const float* __restrict__ A, const float* __restrict__ B,
                                 float* __restrict__ C, int N) {
    __shared__ float As[kTile][kPad];
    __shared__ float Bs[kTile][kPad];

    const int row = blockIdx.y * kTile + threadIdx.y;
    const int col = blockIdx.x * kTile + threadIdx.x;
    const int ty = threadIdx.y;
    const int tx = threadIdx.x;

    float acc = 0.0f;
    for (int kt = 0; kt < N; kt += kTile) {
        if (row < N && kt + tx < N) {
            As[ty][tx] = A[row * N + kt + tx];
        }
        if (kt + ty < N && col < N) {
            Bs[ty][tx] = B[(kt + ty) * N + col];
        }
        __syncthreads();
        for (int k = 0; k < kTile; ++k) {
            if (kt + k < N) {
                acc += As[ty][k] * Bs[k][tx];
            }
        }
        __syncthreads();
    }
    if (row < N && col < N) {
        C[row * N + col] = acc;
    }
}

/* ------------------------------------------------------------------------- */
/* Lanzamientos y medicion                                                   */
/* ------------------------------------------------------------------------- */

typedef void (*launch_fn)(const float*, const float*, float*, int);

void launch_naive(const float* A, const float* B, float* C, int n) {
    const dim3 block(kNaiveTile, kNaiveTile);
    const dim3 grid((n + kNaiveTile - 1) / kNaiveTile, (n + kNaiveTile - 1) / kNaiveTile);
    kernel_naive<<<grid, block>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tiled(const float* A, const float* B, float* C, int n) {
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    kernel_tiled<<<grid, block>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
}

void launch_tiled_pad(const float* A, const float* B, float* C, int n) {
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (n + kTile - 1) / kTile);
    kernel_tiled_pad<<<grid, block>>>(A, B, C, n);
    CUDA_CHECK(cudaGetLastError());
}

/// Tiempo de computo puro: solo el kernel.
double timed_kernel(launch_fn launch, const float* A, const float* B, float* C, uint64_t N) {
    if (N == 0) {
        return 0.0;
    }
    const int n = device_size(N);
    cuda_timer_t timer;
    cuda_timer_begin(&timer);
    launch(A, B, C, n);
    return cuda_timer_end_s(&timer);
}

/// Tiempo total: H2D(A) + H2D(B) + kernel + D2H(C).
double total_timed(launch_fn launch, const float* h_A, const float* h_B, float* h_C,
                   float* d_A, float* d_B, float* d_C, uint64_t N) {
    if (N == 0) {
        return 0.0;
    }
    const int n = device_size(N);
    cuda_timer_t timer;
    cuda_timer_begin(&timer);
    copy_in(d_A, h_A, N);
    copy_in(d_B, h_B, N);
    launch(d_A, d_B, d_C, n);
    copy_out(h_C, d_C, N);
    return cuda_timer_end_s(&timer);
}

/* ------------------------------------------------------------------------- */
/* Handle de cuBLAS (referencia)                                             */
/* ------------------------------------------------------------------------- */

cublasHandle_t g_handle = nullptr;

void cublas_gemm(const float* d_A, const float* d_B, float* d_C, int n) {
    if (g_handle == nullptr) {
        std::fprintf(stderr, "cuBLAS: falta matmul_gpu_cublas_init()\n");
        std::exit(1);
    }
    // Row-major C = A*B equivale en column-major a C^T = B^T * A^T: por eso se
    // intercambian los operandos y se usan ambas operaciones sin transponer.
    const float alpha = 1.0f;
    const float beta = 0.0f;
    CUBLAS_CHECK(cublasSgemm(g_handle, CUBLAS_OP_N, CUBLAS_OP_N, n, n, n, &alpha, d_B, n, d_A, n,
                             &beta, d_C, n));
}

double cublas_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    if (N == 0) {
        return 0.0;
    }
    const int n = device_size(N);
    cuda_timer_t timer;
    cuda_timer_begin(&timer);
    cublas_gemm(d_A, d_B, d_C, n);
    return cuda_timer_end_s(&timer);
}

double cublas_total_timed(const float* h_A, const float* h_B, float* h_C, float* d_A, float* d_B,
                          float* d_C, uint64_t N) {
    if (N == 0) {
        return 0.0;
    }
    const int n = device_size(N);
    cuda_timer_t timer;
    cuda_timer_begin(&timer);
    copy_in(d_A, h_A, N);
    copy_in(d_B, h_B, N);
    cublas_gemm(d_A, d_B, d_C, n);
    copy_out(h_C, d_C, N);
    return cuda_timer_end_s(&timer);
}

}  // namespace

extern "C" {

/* --- Unidad 2: naive ------------------------------------------------------ */

void matmul_gpu_naive(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    if (N == 0) {
        return;
    }
    launch_naive(d_A, d_B, d_C, device_size(N));
}

double matmul_gpu_naive_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    return timed_kernel(launch_naive, d_A, d_B, d_C, N);
}

double matmul_gpu_naive_total_timed(const float* h_A, const float* h_B, float* h_C, float* d_A,
                                    float* d_B, float* d_C, uint64_t N) {
    return total_timed(launch_naive, h_A, h_B, h_C, d_A, d_B, d_C, N);
}

/* --- Unidad 3: tiled y tiled con padding ---------------------------------- */

void matmul_gpu_tiled(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    if (N == 0) {
        return;
    }
    launch_tiled(d_A, d_B, d_C, device_size(N));
}

void matmul_gpu_tiled_pad(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    if (N == 0) {
        return;
    }
    launch_tiled_pad(d_A, d_B, d_C, device_size(N));
}

double matmul_gpu_tiled_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    return timed_kernel(launch_tiled, d_A, d_B, d_C, N);
}

double matmul_gpu_tiled_pad_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    return timed_kernel(launch_tiled_pad, d_A, d_B, d_C, N);
}

double matmul_gpu_tiled_total_timed(const float* h_A, const float* h_B, float* h_C, float* d_A,
                                    float* d_B, float* d_C, uint64_t N) {
    return total_timed(launch_tiled, h_A, h_B, h_C, d_A, d_B, d_C, N);
}

double matmul_gpu_tiled_pad_total_timed(const float* h_A, const float* h_B, float* h_C,
                                        float* d_A, float* d_B, float* d_C, uint64_t N) {
    return total_timed(launch_tiled_pad, h_A, h_B, h_C, d_A, d_B, d_C, N);
}

/* --- REFERENCIA: cuBLAS --------------------------------------------------- */

int matmul_gpu_cublas_init(void) {
    if (g_handle != nullptr) {
        return 0;
    }
    CUBLAS_CHECK(cublasCreate(&g_handle));
    return 0;
}

void matmul_gpu_cublas_destroy(void) {
    if (g_handle != nullptr) {
        cublasDestroy(g_handle);
        g_handle = nullptr;
    }
}

double matmul_gpu_cublas_timed(const float* d_A, const float* d_B, float* d_C, uint64_t N) {
    return cublas_timed(d_A, d_B, d_C, N);
}

double matmul_gpu_cublas_total_timed(const float* h_A, const float* h_B, float* h_C, float* d_A,
                                     float* d_B, float* d_C, uint64_t N) {
    return cublas_total_timed(h_A, h_B, h_C, d_A, d_B, d_C, N);
}

} /* extern "C" */
