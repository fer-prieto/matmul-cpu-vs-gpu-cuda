/**
 * @file matmul_util.cpp
 * @brief Utilidades AUXILIARES (no evaluables): generador SplitMix64, checksum
 *        e información de compilación. Ver include/matmul_util.h.
 */
#include "matmul_util.h"

namespace {

// Constantes de SplitMix64 (Steele, Lea y Flood, 2014).
constexpr uint64_t kGoldenGamma = 0x9E3779B97F4A7C15ULL;
constexpr uint64_t kMix1 = 0xBF58476D1CE4E5B9ULL;
constexpr uint64_t kMix2 = 0x94D049BB133111EBULL;
// Constante para separar secuencias (streams) a partir de una misma semilla.
constexpr uint64_t kStreamMix = 0xD1B54A32D192ED03ULL;

inline uint64_t splitmix64_next(uint64_t& state) {
    state += kGoldenGamma;
    uint64_t z = state;
    z = (z ^ (z >> 30)) * kMix1;
    z = (z ^ (z >> 27)) * kMix2;
    return z ^ (z >> 31);
}

// --- Información de compilación ---------------------------------------------

#define MATMUL_STR_IMPL(x) #x
#define MATMUL_STR(x) MATMUL_STR_IMPL(x)

#ifndef MATMUL_BUILD_FLAGS
#define MATMUL_BUILD_FLAGS "desconocidos (compilado sin el Makefile del proyecto)"
#endif

#if defined(__clang__)
#define MATMUL_COMPILER "clang " __clang_version__
#elif defined(__GNUC__)
#define MATMUL_COMPILER "gcc " __VERSION__
#elif defined(_MSC_VER)
#define MATMUL_COMPILER "msvc " MATMUL_STR(_MSC_VER)
#else
#define MATMUL_COMPILER "desconocido"
#endif

#ifdef __SSE__
#define MATMUL_ISA_SSE " sse"
#else
#define MATMUL_ISA_SSE ""
#endif
#ifdef __SSE2__
#define MATMUL_ISA_SSE2 " sse2"
#else
#define MATMUL_ISA_SSE2 ""
#endif
#ifdef __SSE3__
#define MATMUL_ISA_SSE3 " sse3"
#else
#define MATMUL_ISA_SSE3 ""
#endif
#ifdef __SSSE3__
#define MATMUL_ISA_SSSE3 " ssse3"
#else
#define MATMUL_ISA_SSSE3 ""
#endif
#ifdef __SSE4_1__
#define MATMUL_ISA_SSE41 " sse4.1"
#else
#define MATMUL_ISA_SSE41 ""
#endif
#ifdef __SSE4_2__
#define MATMUL_ISA_SSE42 " sse4.2"
#else
#define MATMUL_ISA_SSE42 ""
#endif
#ifdef __AVX__
#define MATMUL_ISA_AVX " avx"
#else
#define MATMUL_ISA_AVX ""
#endif
#ifdef __AVX2__
#define MATMUL_ISA_AVX2 " avx2"
#define MATMUL_HAS_AVX2 "yes"
#else
#define MATMUL_ISA_AVX2 ""
#define MATMUL_HAS_AVX2 "no"
#endif
#ifdef __FMA__
#define MATMUL_ISA_FMA " fma"
#define MATMUL_HAS_FMA "yes"
#else
#define MATMUL_ISA_FMA ""
#define MATMUL_HAS_FMA "no"
#endif
#ifdef __AVX512F__
#define MATMUL_ISA_AVX512F " avx512f"
#define MATMUL_HAS_AVX512F "yes"
#else
#define MATMUL_ISA_AVX512F ""
#define MATMUL_HAS_AVX512F "no"
#endif
#ifdef __ARM_NEON
#define MATMUL_ISA_NEON " neon"
#else
#define MATMUL_ISA_NEON ""
#endif

#define MATMUL_ISA_LIST                                                    \
    MATMUL_ISA_SSE MATMUL_ISA_SSE2 MATMUL_ISA_SSE3 MATMUL_ISA_SSSE3        \
        MATMUL_ISA_SSE41 MATMUL_ISA_SSE42 MATMUL_ISA_AVX MATMUL_ISA_AVX2   \
            MATMUL_ISA_FMA MATMUL_ISA_AVX512F MATMUL_ISA_NEON

#ifdef __OPTIMIZE__
#define MATMUL_OPTIMIZE "yes"
#else
#define MATMUL_OPTIMIZE "no"
#endif

#ifdef __FAST_MATH__
#define MATMUL_FAST_MATH "yes"
#else
#define MATMUL_FAST_MATH "no"
#endif

#ifdef __cplusplus
#define MATMUL_CXX_STD MATMUL_STR(__cplusplus)
#else
#define MATMUL_CXX_STD "c"
#endif

// La lista de ISA empieza con un espacio (" sse ..."); "isa=" lo absorbe.
constexpr const char kBuildInfo[] =
    "compiler=" MATMUL_COMPILER
    "; cxx_std=" MATMUL_CXX_STD
    "; flags=" MATMUL_BUILD_FLAGS
    "; isa=" MATMUL_ISA_LIST
    "; fma=" MATMUL_HAS_FMA
    "; avx2=" MATMUL_HAS_AVX2
    "; avx512f=" MATMUL_HAS_AVX512F
    "; optimize=" MATMUL_OPTIMIZE
    "; fast_math=" MATMUL_FAST_MATH;

}  // namespace

extern "C" {

void matmul_util_fill_random(float* M, uint64_t count, uint64_t seed, uint64_t stream) {
    uint64_t state = seed ^ (stream * kStreamMix);
    for (uint64_t i = 0; i < count; ++i) {
        const uint64_t z = splitmix64_next(state);
        // 24 bits altos -> entero en [-2^23, 2^23) -> escala exacta a [-1, 1).
        const int32_t k = static_cast<int32_t>(z >> 40) - (1 << 23);
        M[i] = static_cast<float>(k) * (1.0f / 8388608.0f);
    }
}

double matmul_util_checksum(const float* M, uint64_t count) {
    double sum = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        sum += static_cast<double>(M[i]);
    }
    return sum;
}

const char* matmul_util_build_info(void) {
    return kBuildInfo;
}

}  // extern "C"
