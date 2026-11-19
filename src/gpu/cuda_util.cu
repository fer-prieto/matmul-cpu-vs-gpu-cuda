/**
 * @file cuda_util.cu
 * @brief Implementacion de las utilidades CUDA de include/cuda_util.h.
 *
 * Infraestructura de medida y diagnostico: no contiene codigo evaluable.
 */
#include "cuda_util.h"

#include <cstdio>
#include <cstring>

extern "C" {

void cuda_timer_begin(cuda_timer_t* t) {
    CUDA_CHECK(cudaEventCreate(&t->start));
    CUDA_CHECK(cudaEventCreate(&t->stop));
    CUDA_CHECK(cudaEventRecord(t->start, 0));
}

double cuda_timer_end_s(cuda_timer_t* t) {
    // El evento de parada se registra despues de todo el trabajo encolado en el
    // stream 0; cudaEventSynchronize espera a que el dispositivo lo alcance.
    CUDA_CHECK(cudaEventRecord(t->stop, 0));
    CUDA_CHECK(cudaEventSynchronize(t->stop));
    float ms = 0.0f;
    CUDA_CHECK(cudaEventElapsedTime(&ms, t->start, t->stop));
    CUDA_CHECK(cudaEventDestroy(t->start));
    CUDA_CHECK(cudaEventDestroy(t->stop));
    return static_cast<double>(ms) * 1e-3;
}

const char* matmul_gpu_device_info(void) {
    static char info[512] = {0};
    if (info[0] != '\0') {
        return info;
    }
    cudaDeviceProp prop;
    std::memset(&prop, 0, sizeof(prop));
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    int driver = 0;
    int runtime = 0;
    CUDA_CHECK(cudaDriverGetVersion(&driver));
    CUDA_CHECK(cudaRuntimeGetVersion(&runtime));
    std::snprintf(info, sizeof(info), "name=%s; cc=%d.%d; mem_total_mib=%zu; driver=%d; runtime=%d",
                  prop.name, prop.major, prop.minor,
                  static_cast<size_t>(prop.totalGlobalMem >> 20), driver, runtime);
    return info;
}

int matmul_gpu_init(void) {
    CUDA_CHECK(cudaSetDevice(0));
    return 0;
}

} /* extern "C" */
