/**
 * @file main_cpu.cpp
 * @brief Binario autónomo de la línea base en CPU (no necesita Rust).
 *
 * Mide las implementaciones propias i-j-k e i-k-j (include/matmul_cpu.h) y
 * muestra una tabla `impl | N | tiempo | GFLOPS | checksum`. Con `--csv` guarda
 * además cada ejecución (calentamientos incluidos) en un archivo CSV.
 *
 * Uso:
 *   ./build/cpu_baseline [opciones] [N ...]
 *
 * Por defecto mide N = 1024, 2048 y 4096 con i-k-j, y con i-j-k solo hasta
 * N = 2048, porque en 4096 tarda varios minutos (para incluirlo: --include-slow).
 *
 * Metodología (la misma que el arnés en Rust):
 *  - A y B se inicializan con valores pseudoaleatorios en [-1, 1) con semilla
 *    fija (SplitMix64, include/matmul_util.h): A usa stream 0 y B stream 1.
 *  - Las matrices se reservan alineadas a 64 bytes y se escriben antes de
 *    medir, así que no hay fallos de página dentro de la medición.
 *  - El tiempo lo mide la propia biblioteca con std::chrono::steady_clock,
 *    solo alrededor del triple bucle.
 *  - GFLOPS = 2·N³ / (t · 10⁹), con t = mediana de las repeticiones.
 *  - El checksum (suma de C en double) evita que el compilador elimine el
 *    cálculo y debe salir idéntico entre ejecuciones y entre herramientas.
 */
#include "matmul_cpu.h"
#include "matmul_util.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 64;  // una línea de caché

/// Matriz N×N de floats alineada a 64 bytes, con liberación automática (RAII).
class AlignedMatrix {
public:
    explicit AlignedMatrix(uint64_t n) : n_(n) {
        const std::size_t count = static_cast<std::size_t>(n * n);
        std::size_t bytes = count * sizeof(float);
        // std::aligned_alloc exige que el tamaño sea múltiplo de la alineación.
        bytes = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        if (bytes == 0) {
            bytes = kAlignment;
        }
        data_ = static_cast<float*>(std::aligned_alloc(kAlignment, bytes));
        if (data_ == nullptr) {
            std::fprintf(stderr, "error: no hay memoria para una matriz de %" PRIu64 "x%" PRIu64 "\n", n, n);
            std::exit(1);
        }
        // Escribir toda la matriz fuerza a reservar sus páginas ahora, fuera de la medición.
        std::fill(data_, data_ + count, 0.0f);
    }
    ~AlignedMatrix() { std::free(data_); }
    AlignedMatrix(const AlignedMatrix&) = delete;
    AlignedMatrix& operator=(const AlignedMatrix&) = delete;

    float* data() { return data_; }
    const float* data() const { return data_; }
    uint64_t count() const { return n_ * n_; }

private:
    uint64_t n_;
    float* data_ = nullptr;
};

using TimedFn = double (*)(const float*, const float*, float*, uint64_t);

struct Implementation {
    const char* name;
    TimedFn run;
    bool slow;  // se omite para N > slow_max_n salvo con --include-slow
};

const Implementation kImplementations[] = {
    {"cpu_ijk", matmul_cpu_ijk_timed, true},
    {"cpu_ikj", matmul_cpu_ikj_timed, false},
};

struct Options {
    std::vector<uint64_t> sizes;
    uint64_t reps = 1;
    uint64_t warmup = 0;
    uint64_t seed = 42;
    bool include_slow = false;
    uint64_t slow_max_n = 2048;
    std::string csv_path;
};

void print_usage(const char* prog) {
    std::printf(
        "Uso: %s [opciones] [N ...]\n"
        "\n"
        "Mide C = A x B (FP32) con las implementaciones propias cpu_ijk y cpu_ikj.\n"
        "\n"
        "  N ...              Tamaños de matriz (por defecto: 1024 2048 4096)\n"
        "  --reps R           Repeticiones medidas por (impl, N) (por defecto: 1)\n"
        "  --warmup W         Ejecuciones de calentamiento no medidas (por defecto: 0)\n"
        "  --seed S           Semilla de inicialización (por defecto: 42)\n"
        "  --include-slow     Ejecuta también cpu_ijk para N > slow-max-n (lento)\n"
        "  --slow-max-n N     Mayor N para el que se ejecuta cpu_ijk (por defecto: 2048)\n"
        "  --csv ARCHIVO      Guarda cada ejecución (incluidos calentamientos) en CSV\n"
        "  -h, --help         Muestra esta ayuda\n"
        "\n"
        "Ejemplos:\n"
        "  %s 512\n"
        "  %s --warmup 1 --reps 5 1024 2048\n"
        "\n"
        "Para el análisis completo (estadísticas, validación, entorno) usa el arnés:\n"
        "  make bench\n",
        prog, prog, prog);
}

/// Convierte un texto en entero sin signo; rechaza signos, vacíos y basura.
bool parse_u64(const char* text, uint64_t& out) {
    if (text == nullptr || *text == '\0' || *text == '-' || *text == '+') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const unsigned long long value = std::strtoull(text, &end, 10);
    if (errno != 0 || end == nullptr || *end != '\0') {
        return false;
    }
    out = static_cast<uint64_t>(value);
    return true;
}

/// Devuelve 0 si todo fue bien, 1 si se mostró la ayuda y 2 si hubo un error.
int parse_args(int argc, char** argv, Options& opt) {
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        auto take_u64 = [&](uint64_t& dst) -> bool {
            if (i + 1 >= argc || !parse_u64(argv[i + 1], dst)) {
                std::fprintf(stderr, "error: %s necesita un entero no negativo\n", arg.c_str());
                return false;
            }
            ++i;
            return true;
        };

        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 1;
        } else if (arg == "--reps") {
            if (!take_u64(opt.reps)) return 2;
        } else if (arg == "--warmup") {
            if (!take_u64(opt.warmup)) return 2;
        } else if (arg == "--seed") {
            if (!take_u64(opt.seed)) return 2;
        } else if (arg == "--slow-max-n") {
            if (!take_u64(opt.slow_max_n)) return 2;
        } else if (arg == "--include-slow") {
            opt.include_slow = true;
        } else if (arg == "--csv") {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "error: --csv necesita la ruta de un archivo\n");
                return 2;
            }
            opt.csv_path = argv[++i];
        } else if (arg.size() > 1 && arg[0] == '-') {
            std::fprintf(stderr, "error: opción desconocida '%s' (usa --help)\n", arg.c_str());
            return 2;
        } else {
            uint64_t n = 0;
            if (!parse_u64(arg.c_str(), n) || n == 0) {
                std::fprintf(stderr, "error: tamaño inválido '%s' (debe ser un entero > 0)\n", arg.c_str());
                return 2;
            }
            opt.sizes.push_back(n);
        }
    }
    if (opt.sizes.empty()) {
        opt.sizes = {1024, 2048, 4096};
    }
    if (opt.reps == 0) {
        std::fprintf(stderr, "error: --reps debe ser al menos 1\n");
        return 2;
    }
    return 0;
}

double gflops(uint64_t n, double seconds) {
    const double nd = static_cast<double>(n);
    return seconds > 0.0 ? 2.0 * nd * nd * nd / (seconds * 1e9) : 0.0;
}

double median(std::vector<double> values) {
    std::sort(values.begin(), values.end());
    const std::size_t mid = values.size() / 2;
    return (values.size() % 2 == 1) ? values[mid] : 0.5 * (values[mid - 1] + values[mid]);
}

void write_csv_row(std::FILE* csv, const char* impl, uint64_t n, const char* phase,
                   uint64_t rep, double t, double checksum, uint64_t seed) {
    if (csv == nullptr) {
        return;
    }
    std::fprintf(csv, "%s,%" PRIu64 ",%s,%" PRIu64 ",%.9f,%.6f,%.17g,%" PRIu64 "\n",
                 impl, n, phase, rep, t, gflops(n, t), checksum, seed);
    std::fflush(csv);  // si se interrumpe la ejecución, lo medido ya está guardado
}

}  // namespace

int main(int argc, char** argv) {
    Options opt;
    const int parsed = parse_args(argc, argv, opt);
    if (parsed == 1) {
        return 0;
    }
    if (parsed != 0) {
        return 2;
    }

    std::FILE* csv = nullptr;
    if (!opt.csv_path.empty()) {
        csv = std::fopen(opt.csv_path.c_str(), "w");
        if (csv == nullptr) {
            std::perror(opt.csv_path.c_str());
            return 1;
        }
        std::fprintf(csv, "impl,n,phase,rep,t_s,gflops,checksum,seed\n");
    }

    std::printf("=== Linea base CPU: C = A x B (FP32, row-major) ===\n");
    std::printf("Compilacion : %s\n", matmul_util_build_info());
    std::printf("Semilla %" PRIu64 " | repeticiones %" PRIu64 " | calentamiento %" PRIu64
                " | cpu_ijk hasta N=%" PRIu64 "%s\n",
                opt.seed, opt.reps, opt.warmup, opt.slow_max_n,
                opt.include_slow ? " (--include-slow: sin limite)" : "");
    std::printf("GFLOPS = 2*N^3 / (t * 1e9), con t = mediana de las repeticiones\n\n");
    std::printf("%-8s %6s %5s %14s %14s %10s %24s\n",
                "impl", "N", "reps", "t_mediana(s)", "t_min(s)", "GFLOPS", "checksum(C)");
    std::printf("%-8s %6s %5s %14s %14s %10s %24s\n",
                "--------", "------", "-----", "--------------", "--------------",
                "----------", "------------------------");
    std::fflush(stdout);

    for (const uint64_t n : opt.sizes) {
        AlignedMatrix A(n);
        AlignedMatrix B(n);
        AlignedMatrix C(n);
        matmul_util_fill_random(A.data(), A.count(), opt.seed, 0);
        matmul_util_fill_random(B.data(), B.count(), opt.seed, 1);

        double t_ijk = -1.0;
        double t_ikj = -1.0;
        for (const Implementation& impl : kImplementations) {
            if (impl.slow && n > opt.slow_max_n && !opt.include_slow) {
                std::printf("%-8s %6" PRIu64 "   omitida: lenta para N > %" PRIu64
                            " (usa --include-slow)\n",
                            impl.name, n, opt.slow_max_n);
                continue;
            }
            for (uint64_t w = 1; w <= opt.warmup; ++w) {
                const double t = impl.run(A.data(), B.data(), C.data(), n);
                write_csv_row(csv, impl.name, n, "warmup", w, t,
                              matmul_util_checksum(C.data(), C.count()), opt.seed);
            }
            std::vector<double> times;
            double checksum = 0.0;
            for (uint64_t r = 1; r <= opt.reps; ++r) {
                const double t = impl.run(A.data(), B.data(), C.data(), n);
                checksum = matmul_util_checksum(C.data(), C.count());
                times.push_back(t);
                write_csv_row(csv, impl.name, n, "measured", r, t, checksum, opt.seed);
            }
            const double t_med = median(times);
            const double t_min = *std::min_element(times.begin(), times.end());
            std::printf("%-8s %6" PRIu64 " %5" PRIu64 " %14.6f %14.6f %10.3f %24.12g\n",
                        impl.name, n, opt.reps, t_med, t_min, gflops(n, t_med), checksum);
            std::fflush(stdout);
            if (impl.slow) {
                t_ijk = t_med;
            } else {
                t_ikj = t_med;
            }
        }
        if (t_ijk > 0.0 && t_ikj > 0.0) {
            std::printf("%-8s %6" PRIu64 "   speedup cpu_ikj vs cpu_ijk = %.2fx\n",
                        "", n, t_ijk / t_ikj);
        }
    }

    if (csv != nullptr) {
        std::fclose(csv);
        std::printf("\nMediciones guardadas en %s\n", opt.csv_path.c_str());
    }
    return 0;
}
