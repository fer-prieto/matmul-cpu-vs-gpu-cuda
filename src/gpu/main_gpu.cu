/**
 * @file main_gpu.cu
 * @brief Binario de benchmark GPU (Unidades 2 y 3): mide, valida y emite CSV.
 *
 * Uso:
 *   ./build/bench_gpu [opciones] [N ...]
 *
 * Comportamiento:
 *  - Inicializa A y B con el mismo generador SplitMix64 de la Unidad 1
 *    (matmul_util.h): A usa el stream 0 y B el stream 1, valores en [-1, 1).
 *  - Mide cada implementacion con cudaEvent: t_kernel_s (solo kernel) y
 *    t_total_s (host->device + kernel + device->host).
 *  - Valida cada implementacion contra cpu_ikj (referencia propia ya validada
 *    en la Unidad 1) con error relativo de Frobenius y error absoluto maximo.
 *  - Escribe por **stdout** un CSV con una fila por ejecucion (calentamientos
 *    incluidos, con phase=warmup). El diagnostico y las tablas humanas van a
 *    **stderr**.
 *
 * Exit codes: 0 ok · 1 error de runtime · 2 uso invalido · 3 validacion fallida.
 */
#include "cuda_util.h"
#include "matmul_cpu.h"
#include "matmul_gpu.h"
#include "matmul_util.h"

#include <algorithm>
#include <cerrno>
#include <cinttypes>
#include <cmath>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

namespace {

constexpr std::size_t kAlignment = 64;  // una linea de cache

/// Matriz N x N de floats en host, alineada a 64 bytes, con liberacion automatica.
class HostMatrix {
public:
    explicit HostMatrix(uint64_t n) : n_(n) {
        const std::size_t count = static_cast<std::size_t>(n * n);
        std::size_t bytes = count * sizeof(float);
        bytes = (bytes + kAlignment - 1) / kAlignment * kAlignment;
        if (bytes == 0) {
            bytes = kAlignment;
        }
        data_ = static_cast<float*>(std::aligned_alloc(kAlignment, bytes));
        if (data_ == nullptr) {
            std::fprintf(stderr, "error: no hay memoria para una matriz de %" PRIu64 "x%" PRIu64 "\n", n,
                         n);
            std::exit(1);
        }
        // Se escribe toda la matriz para que las paginas queden residentes antes
        // de la medicion.
        std::fill(data_, data_ + count, 0.0f);
    }
    ~HostMatrix() { std::free(data_); }
    HostMatrix(const HostMatrix&) = delete;
    HostMatrix& operator=(const HostMatrix&) = delete;

    float* data() { return data_; }
    const float* data() const { return data_; }
    uint64_t count() const { return n_ * n_; }

private:
    uint64_t n_;
    float* data_ = nullptr;
};

/// Buffer de dispositivo N x N de floats con liberacion automatica.
class DeviceMatrix {
public:
    explicit DeviceMatrix(uint64_t n) {
        const std::size_t bytes = static_cast<std::size_t>(n * n) * sizeof(float);
        void* raw = nullptr;
        CUDA_CHECK(cudaMalloc(&raw, bytes));
        data_ = static_cast<float*>(raw);
    }
    ~DeviceMatrix() {
        if (data_ != nullptr) {
            cudaFree(data_);
        }
    }
    DeviceMatrix(const DeviceMatrix&) = delete;
    DeviceMatrix& operator=(const DeviceMatrix&) = delete;

    float* data() { return data_; }
    const float* data() const { return data_; }

private:
    float* data_ = nullptr;
};

/// Firma de las funciones cronometradas del kernel (solo computo).
using KernelFn = double (*)(const float*, const float*, float*, uint64_t);
/// Firma de las funciones cronometradas completas (incluye transferencias).
using TotalFn = double (*)(const float*, const float*, float*, float*, float*, float*, uint64_t);

struct Implementation {
    const char* name;
    const char* kind;  ///< "propia" (evaluable) o "referencia" (cuBLAS).
    KernelFn run_kernel;
    TotalFn run_total;
    int unit;  ///< Unidad en la que se introduce (2 = naive, 3 = resto).
};

const Implementation kRegistry[] = {
    {"gpu_naive", "propia", matmul_gpu_naive_timed, matmul_gpu_naive_total_timed, 2},
    {"gpu_tiled", "propia", matmul_gpu_tiled_timed, matmul_gpu_tiled_total_timed, 3},
    {"gpu_tiled_pad", "propia", matmul_gpu_tiled_pad_timed, matmul_gpu_tiled_pad_total_timed, 3},
    {"gpu_cublas", "referencia", matmul_gpu_cublas_timed, matmul_gpu_cublas_total_timed, 3},
};
constexpr std::size_t kRegistrySize = sizeof(kRegistry) / sizeof(kRegistry[0]);

bool is_cublas(const Implementation& impl) { return std::strcmp(impl.name, "gpu_cublas") == 0; }

struct Options {
    std::vector<uint64_t> sizes;
    std::vector<std::string> impl_names;
    uint64_t reps = 5;
    uint64_t warmup = 1;
    uint64_t seed = 42;
    double tolerance = 1e-4;
    bool validate = true;
    bool device_info = false;
    int unit = 2;
    std::string validation_file;
    std::string run_id;
    std::string label;
};

void print_usage(const char* prog) {
    std::fprintf(stderr,
                 "Uso: %s [opciones] [N ...]\n"
                 "\n"
                 "Mide C = A x B (FP32) en GPU y valida contra cpu_ikj.\n"
                 "\n"
                 "  N ...                 Tam. de matriz (por defecto: 1024 2048 4096)\n"
                 "  --sizes L             Igual, lista separada por comas (1024,2048,4096)\n"
                 "  --unit 2|3            Conjunto de implementaciones por defecto (por defecto: 2)\n"
                 "  --impls a,b,c         Implementaciones a medir (gpu_naive, gpu_tiled,\n"
                 "                        gpu_tiled_pad, gpu_cublas)\n"
                 "  --reps R              Repeticiones medidas (por defecto: 5)\n"
                 "  --warmup W            Ejecuciones de calentamiento no medidas (por defecto: 1)\n"
                 "  --seed S              Semilla de inicializacion (por defecto: 42)\n"
                 "  --tolerance T         Tolerancia del error relativo (por defecto: 1e-4)\n"
                 "  --no-validate         No valida contra cpu_ikj\n"
                 "  --validation-file F   Escribe las filas de validacion en el CSV F\n"
                 "  --run-id ID           Identificador de la corrida\n"
                 "  --label L             Etiqueta libre (incluida en las filas de validacion)\n"
                 "  --device-info         Imprime los datos del dispositivo y sale\n"
                 "  -h, --help            Muestra esta ayuda\n"
                 "\n"
                 "El CSV de mediciones sale por stdout:\n"
                 "  impl,kind,n,phase,rep,seed,t_kernel_s,t_total_s,gflops,checksum\n"
                 "\n"
                 "Exit codes: 0 ok, 1 error, 2 uso invalido, 3 validacion fallida.\n",
                 prog);
}

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

bool parse_double(const char* text, double& out) {
    if (text == nullptr || *text == '\0') {
        return false;
    }
    errno = 0;
    char* end = nullptr;
    const double value = std::strtod(text, &end);
    if (errno != 0 || end == nullptr || *end != '\0') {
        return false;
    }
    out = value;
    return true;
}

/// "1024,2048" -> {1024, 2048} (rechaza vacios y ceros).
bool parse_size_list(const std::string& text, std::vector<uint64_t>& sizes) {
    std::string current;
    for (std::size_t i = 0; i <= text.size(); ++i) {
        if (i == text.size() || text[i] == ',') {
            if (current.empty()) {
                return false;
            }
            uint64_t n = 0;
            if (!parse_u64(current.c_str(), n) || n == 0) {
                return false;
            }
            sizes.push_back(n);
            current.clear();
        } else {
            current.push_back(text[i]);
        }
    }
    return true;
}

const char* default_impls(int unit) {
    return unit == 2 ? "gpu_naive" : "gpu_naive,gpu_tiled,gpu_tiled_pad,gpu_cublas";
}

/// Devuelve 0 si todo fue bien, 1 si se mostro la ayuda, 2 si hubo un error de uso.
int parse_args(int argc, char** argv, Options& opt) {
    auto take = [&](int& i, const char* flag) -> const char* {
        if (i + 1 >= argc) {
            std::fprintf(stderr, "error: %s necesita un valor\n", flag);
            return nullptr;
        }
        return argv[++i];
    };

    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "-h" || arg == "--help") {
            print_usage(argv[0]);
            return 1;
        } else if (arg == "--unit") {
            const char* value = take(i, "--unit");
            uint64_t unit = 0;
            if (value == nullptr || !parse_u64(value, unit) || (unit != 2 && unit != 3)) {
                std::fprintf(stderr, "error: --unit debe ser 2 o 3\n");
                return 2;
            }
            opt.unit = static_cast<int>(unit);
        } else if (arg == "--impls") {
            const char* value = take(i, "--impls");
            if (value == nullptr) {
                return 2;
            }
            std::string list = value;
            std::string current;
            opt.impl_names.clear();
            for (std::size_t p = 0; p <= list.size(); ++p) {
                if (p == list.size() || list[p] == ',') {
                    if (current.empty()) {
                        std::fprintf(stderr, "error: --impls tiene un nombre vacio\n");
                        return 2;
                    }
                    opt.impl_names.push_back(current);
                    current.clear();
                } else {
                    current.push_back(list[p]);
                }
            }
        } else if (arg == "--reps") {
            const char* value = take(i, "--reps");
            if (value == nullptr || !parse_u64(value, opt.reps) || opt.reps == 0) {
                std::fprintf(stderr, "error: --reps necesita un entero >= 1\n");
                return 2;
            }
        } else if (arg == "--warmup") {
            const char* value = take(i, "--warmup");
            if (value == nullptr || !parse_u64(value, opt.warmup)) {
                std::fprintf(stderr, "error: --warmup necesita un entero >= 0\n");
                return 2;
            }
        } else if (arg == "--seed") {
            const char* value = take(i, "--seed");
            if (value == nullptr || !parse_u64(value, opt.seed)) {
                std::fprintf(stderr, "error: --seed necesita un entero no negativo\n");
                return 2;
            }
        } else if (arg == "--tolerance") {
            const char* value = take(i, "--tolerance");
            if (value == nullptr || !parse_double(value, opt.tolerance) || opt.tolerance < 0.0) {
                std::fprintf(stderr, "error: --tolerance necesita un numero >= 0\n");
                return 2;
            }
        } else if (arg == "--no-validate") {
            opt.validate = false;
        } else if (arg == "--device-info") {
            opt.device_info = true;
        } else if (arg == "--validation-file") {
            const char* value = take(i, "--validation-file");
            if (value == nullptr) {
                return 2;
            }
            opt.validation_file = value;
        } else if (arg == "--run-id") {
            const char* value = take(i, "--run-id");
            if (value == nullptr) {
                return 2;
            }
            opt.run_id = value;
        } else if (arg == "--label") {
            const char* value = take(i, "--label");
            if (value == nullptr) {
                return 2;
            }
            opt.label = value;
        } else if (arg == "--sizes") {
            const char* value = take(i, "--sizes");
            if (value == nullptr || !parse_size_list(value, opt.sizes)) {
                std::fprintf(stderr, "error: --sizes necesita una lista como 1024,2048,4096\n");
                return 2;
            }
        } else if (arg.size() > 1 && arg[0] == '-') {
            std::fprintf(stderr, "error: opcion desconocida '%s' (usa --help)\n", arg.c_str());
            return 2;
        } else {
            uint64_t n = 0;
            if (!parse_u64(arg.c_str(), n) || n == 0) {
                std::fprintf(stderr, "error: tamano invalido '%s' (debe ser un entero > 0)\n", arg.c_str());
                return 2;
            }
            opt.sizes.push_back(n);
        }
    }
    if (opt.sizes.empty()) {
        opt.sizes = {1024, 2048, 4096};
    }
    if (opt.impl_names.empty()) {
        std::string list = default_impls(opt.unit);
        std::string current;
        for (std::size_t p = 0; p <= list.size(); ++p) {
            if (p == list.size() || list[p] == ',') {
                opt.impl_names.push_back(current);
                current.clear();
            } else {
                current.push_back(list[p]);
            }
        }
    }
    return 0;
}

double gflops(uint64_t n, double seconds) {
    const double nd = static_cast<double>(n);
    return seconds > 0.0 ? 2.0 * nd * nd * nd / (seconds * 1e9) : 0.0;
}

void write_header_stdout() {
    std::printf("impl,kind,n,phase,rep,seed,t_kernel_s,t_total_s,gflops,checksum\n");
    std::fflush(stdout);
}

void write_row_stdout(const char* impl, const char* kind, uint64_t n, const char* phase, uint64_t rep,
                      uint64_t seed, double t_kernel, double t_total, double checksum) {
    std::printf("%s,%s,%" PRIu64 ",%s,%" PRIu64 ",%" PRIu64 ",%.9f,%.9f,%.6f,%.17g\n", impl, kind, n,
                phase, rep, seed, t_kernel, t_total, gflops(n, t_kernel), checksum);
    std::fflush(stdout);  // si la corrida se interrumpe, lo medido ya esta guardado
}

const char* kValidationHeader =
    "run_id,impl,kind,n,reference,rel_error,max_abs_error,norm_fro,norm_fro_ref,tolerance,valid,"
    "checksum,checksum_ref";

void write_validation(std::FILE* file, const Options& opt, const char* impl, const char* kind, uint64_t n,
                      const char* reference, double rel, double max_abs, double norm, double norm_ref,
                      bool valid, double checksum, double checksum_ref) {
    char line[1024];
    std::snprintf(line, sizeof(line), "%s,%s,%s,%" PRIu64 ",%s,%.9e,%.9e,%.6f,%.6f,%.9e,%s,%.17g,%.17g\n",
                  opt.run_id.c_str(), impl, kind, n, reference, rel, max_abs, norm, norm_ref,
                  opt.tolerance, valid ? "true" : "false", checksum, checksum_ref);
    if (file != nullptr) {
        std::fputs(line, file);
        std::fflush(file);
    } else {
        std::fprintf(stderr, "VALIDACION: %s", line);
    }
}

/// Error relativo de Frobenius y error absoluto maximo, calculados en double.
void compare(const float* c, const float* ref, uint64_t count, double& rel, double& max_abs, double& norm,
             double& norm_ref) {
    double diff2 = 0.0;
    double c2 = 0.0;
    double ref2 = 0.0;
    double worst = 0.0;
    for (uint64_t i = 0; i < count; ++i) {
        const double x = static_cast<double>(c[i]);
        const double y = static_cast<double>(ref[i]);
        const double d = x - y;
        diff2 += d * d;
        c2 += x * x;
        ref2 += y * y;
        const double abs_d = d < 0.0 ? -d : d;
        if (abs_d > worst) {
            worst = abs_d;
        }
    }
    norm = std::sqrt(c2);
    norm_ref = std::sqrt(ref2);
    const double num = std::sqrt(diff2);
    rel = norm_ref > 0.0 ? num / norm_ref : (num == 0.0 ? 0.0 : 1e300);
    max_abs = worst;
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

    matmul_gpu_init();

    if (opt.device_info) {
        std::printf("%s\n", matmul_gpu_device_info());
        return 0;
    }

    if (opt.run_id.empty()) {
        const char* id = std::getenv("BENCH_GPU_RUN_ID");
        opt.run_id = id != nullptr ? id : "local";
    }

    // Resolucion del registro: se conserva el orden pedido y se rechaza lo desconocido.
    std::vector<const Implementation*> impls;
    for (const std::string& name : opt.impl_names) {
        bool found = false;
        for (std::size_t i = 0; i < kRegistrySize; ++i) {
            if (name == kRegistry[i].name) {
                impls.push_back(&kRegistry[i]);
                found = true;
                break;
            }
        }
        if (!found) {
            std::fprintf(stderr,
                         "error: implementacion desconocida '%s' (disponibles: gpu_naive, gpu_tiled, "
                         "gpu_tiled_pad, gpu_cublas)\n",
                         name.c_str());
            return 2;
        }
    }

    bool needs_cublas = false;
    for (const Implementation* impl : impls) {
        if (is_cublas(*impl)) {
            needs_cublas = true;
        }
    }
    if (needs_cublas) {
        matmul_gpu_cublas_init();
    }

    std::FILE* validation = nullptr;
    if (opt.validate && !opt.validation_file.empty()) {
        const bool fresh = std::fopen(opt.validation_file.c_str(), "r") == nullptr;
        validation = std::fopen(opt.validation_file.c_str(), "a");
        if (validation == nullptr) {
            std::perror(opt.validation_file.c_str());
            if (needs_cublas) {
                matmul_gpu_cublas_destroy();
            }
            return 1;
        }
        if (fresh) {
            std::fprintf(validation, "%s\n", kValidationHeader);
            std::fflush(validation);
        }
    }

    std::fprintf(stderr, "bench_gpu · dispositivo: %s\n", matmul_gpu_device_info());
    std::fprintf(stderr, "run_id=%s · unidad=%d · semilla=%" PRIu64 " · reps=%" PRIu64
                         " · calentamiento=%" PRIu64 " · tolerancia=%g\n",
                 opt.run_id.c_str(), opt.unit, opt.seed, opt.reps, opt.warmup, opt.tolerance);

    write_header_stdout();

    int failures = 0;
    for (const uint64_t n : opt.sizes) {
        HostMatrix h_A(n);
        HostMatrix h_B(n);
        HostMatrix h_C(n);
        HostMatrix h_ref(n);
        DeviceMatrix d_A(n);
        DeviceMatrix d_B(n);
        DeviceMatrix d_C(n);

        matmul_util_fill_random(h_A.data(), h_A.count(), opt.seed, 0);
        matmul_util_fill_random(h_B.data(), h_B.count(), opt.seed, 1);

        std::fprintf(stderr, "== N = %" PRIu64 " (%.1f MiB por matriz) ==\n", n,
                     static_cast<double>(n) * static_cast<double>(n) * sizeof(float) / (1024.0 * 1024.0));

        for (const Implementation* impl : impls) {
            for (uint64_t w = 1; w <= opt.warmup; ++w) {
                const double t_kernel =
                    impl->run_kernel(d_A.data(), d_B.data(), d_C.data(), n);
                const double t_total =
                    impl->run_total(h_A.data(), h_B.data(), h_C.data(), d_A.data(), d_B.data(), d_C.data(), n);
                const double checksum = matmul_util_checksum(h_C.data(), h_C.count());
                write_row_stdout(impl->name, impl->kind, n, "warmup", w, opt.seed, t_kernel, t_total, checksum);
            }

            std::vector<double> kernel_times;
            std::vector<double> total_times;
            double checksum = 0.0;
            for (uint64_t r = 1; r <= opt.reps; ++r) {
                const double t_kernel = impl->run_kernel(d_A.data(), d_B.data(), d_C.data(), n);
                const double t_total =
                    impl->run_total(h_A.data(), h_B.data(), h_C.data(), d_A.data(), d_B.data(), d_C.data(), n);
                checksum = matmul_util_checksum(h_C.data(), h_C.count());
                kernel_times.push_back(t_kernel);
                total_times.push_back(t_total);
                write_row_stdout(impl->name, impl->kind, n, "measured", r, opt.seed, t_kernel, t_total, checksum);
            }

            std::sort(kernel_times.begin(), kernel_times.end());
            std::sort(total_times.begin(), total_times.end());
            const double t_med = kernel_times[kernel_times.size() / 2];
            std::fprintf(stderr, "  %-14s [%s] mediana kernel %.6f s · total %.6f s · %8.2f GFLOPS\n",
                         impl->name, impl->kind, t_med, total_times[total_times.size() / 2],
                         gflops(n, t_med));

            if (opt.validate) {
                // Referencia: cpu_ikj (codigo propio ya validado en la Unidad 1).
                matmul_cpu_ikj(h_A.data(), h_B.data(), h_ref.data(), n);
                const double checksum_ref = matmul_util_checksum(h_ref.data(), h_ref.count());
                double rel = 0.0;
                double max_abs = 0.0;
                double norm = 0.0;
                double norm_ref = 0.0;
                compare(h_C.data(), h_ref.data(), h_C.count(), rel, max_abs, norm, norm_ref);
                const bool valid = rel <= opt.tolerance;
                if (!valid) {
                    failures += 1;
                    std::fprintf(stderr,
                                 "VALIDACION FALLO: %s N=%" PRIu64 " rel=%.3e > tol=%.3e (max_abs=%.3e)\n",
                                 impl->name, n, rel, opt.tolerance, max_abs);
                }
                write_validation(validation, opt, impl->name, impl->kind, n, "cpu_ikj", rel, max_abs, norm,
                                 norm_ref, valid, checksum, checksum_ref);
            }
        }
    }

    if (validation != nullptr) {
        std::fclose(validation);
    }
    if (needs_cublas) {
        matmul_gpu_cublas_destroy();
    }

    if (failures > 0) {
        std::fprintf(stderr, "validacion: %d comprobacion(es) fallida(s)\n", failures);
        return 3;
    }
    std::fprintf(stderr, "validacion: OK\n");
    return 0;
}
