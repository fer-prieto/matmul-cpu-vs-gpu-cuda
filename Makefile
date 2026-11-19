# =============================================================================
#  Makefile: Multiplicación de matrices CPU vs GPU (IF557ZIN), Unidad 1
#
#  Es la ÚNICA fuente de verdad para compilar el código C++: Cargo
#  (harness/build.rs) solo ENLAZA build/libmatmul_cpu.a y nunca la recompila
#  con otros flags.
#
#  Uso rápido:
#    make                 biblioteca + binario autónomo C++ (no necesita Rust)
#    make test            tests de correctitud (Rust)
#    make bench           benchmark completo (Rust)  -> results/unidad1/
#    make help            lista de todos los objetivos
# =============================================================================

SHELL := /bin/bash
.DEFAULT_GOAL := all

# --- Configuración (todo se puede sobrescribir: make CXX=clang++ ARGS="...") --
CXX   ?= g++
AR    ?= ar
CARGO ?= cargo

# Flags de compilación del código C++ (ver README, sección Metodología):
#   -O3 -march=native  optimización máxima con las extensiones SIMD de ESTA CPU
#                      (en la laptop de desarrollo, -march=native = sandybridge: AVX).
#   -fPIC              permite enlazar la biblioteca estática en el ejecutable
#                      PIE que genera Rust; no afecta a los bucles de cálculo.
# Se asigna con := para que una variable CXXFLAGS del entorno no la cambie sin
# querer; para cambiarla a propósito: make CXXFLAGS="...". El valor efectivo
# queda en build/build_config.txt y en cada results/.../env_<run>.json.
CXXFLAGS := -O3 -march=native -std=c++17 -Wall -Wextra -fPIC
CPPFLAGS := -Iinclude
LDFLAGS  :=

# 1 = compilar también la referencia OpenBLAS (opcional, ver Fase 6 del plan).
# Con OpenBLAS instalado (en Artix/Arch: `doas pacman -S openblas`) basta
# `make bench-ref`. Las dos variables siguientes permiten encontrar el header
# cblas.h y la biblioteca si viven fuera de las rutas por defecto, p. ej.:
#   make bench-ref OPENBLAS_CFLAGS="-I/usr/include/openblas"
OPENBLAS ?= 0
OPENBLAS_CFLAGS ?=
OPENBLAS_LIBS ?= -lopenblas

# Argumentos extra para el arnés o el binario C++, p. ej.:
#   make bench ARGS="--sizes 256,512"      make baseline ARGS="512 1024"
ARGS ?=

# --- Rutas -------------------------------------------------------------------
BUILD_DIR      := build
OBJ_DIR        := $(BUILD_DIR)/obj
ASM_DIR        := $(BUILD_DIR)/asm
LIB            := $(BUILD_DIR)/libmatmul_cpu.a
CPU_BIN        := $(BUILD_DIR)/cpu_baseline
BUILD_CONFIG   := $(BUILD_DIR)/build_config.txt
RESULTS_DIR    := results/unidad1
QUICK_DIR      := results/pruebas
CARGO_MANIFEST := harness/Cargo.toml

# --- Fuentes -----------------------------------------------------------------
# Código propio (evaluable) + utilidades auxiliares (generador, checksum, info).
LIB_SRCS       := src/cpu/matmul_cpu.cpp src/cpu/matmul_util.cpp
CARGO_FEATURES :=

# Con OPENBLAS=1 la biblioteca incluye el envoltorio de REFERENCIA (no
# evaluable) de cblas_sgemm y el arnés se compila con la feature `openblas`,
# que registra cpu_openblas_1t y cpu_openblas_mt. Sin esa opción no se compila
# nada relacionado y el proyecto funciona igual.
ifeq ($(OPENBLAS),1)
LIB_SRCS       += src/cpu/matmul_openblas.cpp
CPPFLAGS       += $(OPENBLAS_CFLAGS)
LDFLAGS        += $(OPENBLAS_LIBS)
CARGO_FEATURES := --features openblas
endif

LIB_OBJS := $(patsubst src/cpu/%.cpp,$(OBJ_DIR)/%.o,$(LIB_SRCS))
MAIN_OBJ := $(OBJ_DIR)/main_cpu.o
DEPS     := $(LIB_OBJS:.o=.d) $(MAIN_OBJ:.o=.d)

# --- Datos de la compilación (se registran junto a los resultados) ----------
CXX_VERSION  := $(shell $(CXX) --version 2>/dev/null | head -n 1)
MARCH_NATIVE := $(shell $(CXX) -march=native -Q --help=target 2>/dev/null | awk '$$1 == "-march=" { print $$2; exit }')

# build/build_config.txt describe con qué se compiló la biblioteca. Solo se
# reescribe si cambia, y todos los objetos dependen de él: cambiar CXXFLAGS,
# CXX u OPENBLAS fuerza a recompilar. harness/build.rs lo lee para enlazar y
# para registrar estos datos en los resultados.
define BUILD_CONFIG_TEXT
# Generado por el Makefile; lo lee harness/build.rs. No editar a mano.
CXX=$(CXX)
CXX_VERSION=$(CXX_VERSION)
CXXFLAGS=$(CXXFLAGS)
CPPFLAGS=$(CPPFLAGS)
MARCH_NATIVE=$(MARCH_NATIVE)
NVCC=$(NVCC)
NVCC_VERSION=$(NVCC_VERSION)
OPENBLAS=$(OPENBLAS)
OPENBLAS_CFLAGS=$(OPENBLAS_CFLAGS)
OPENBLAS_LIBS=$(OPENBLAS_LIBS)
endef
export BUILD_CONFIG_TEXT

# =============================================================================
#  Objetivos principales
# =============================================================================
.PHONY: all lib cpu gpu bench bench-quick bench-ref test test-ref baseline env asm consolidate clean help FORCE

all: lib cpu        ## Compila la biblioteca y el binario autónomo C++

lib: $(LIB)         ## Solo la biblioteca estática build/libmatmul_cpu.a

cpu: $(CPU_BIN)     ## Solo el binario autónomo build/cpu_baseline

bench: lib          ## Benchmark completo con el arnés Rust (ARGS="..." para opciones)
	$(CARGO) run --release --manifest-path $(CARGO_MANIFEST) $(CARGO_FEATURES) -- $(ARGS)

bench-quick: lib    ## Prueba rápida de extremo a extremo (N=256,512) en results/pruebas/
	$(CARGO) run --release --manifest-path $(CARGO_MANIFEST) $(CARGO_FEATURES) -- \
		--sizes 256,512 --label prueba --out $(QUICK_DIR)

test: lib           ## Tests de correctitud (cargo test --release)
	$(CARGO) test --release --manifest-path $(CARGO_MANIFEST) $(CARGO_FEATURES)

bench-ref:         ## Benchmark con OpenBLAS de referencia (compila con OPENBLAS=1; ARGS="..." para opciones)
	$(MAKE) lib OPENBLAS=1
	$(CARGO) run --release --manifest-path $(CARGO_MANIFEST) --features openblas -- $(ARGS)

test-ref:          ## Tests de correctitud con OpenBLAS (compila con OPENBLAS=1)
	$(MAKE) lib OPENBLAS=1
	$(CARGO) test --release --manifest-path $(CARGO_MANIFEST) --features openblas

baseline: $(CPU_BIN) | $(RESULTS_DIR)  ## Ejecuta cpu_baseline y guarda tabla (.txt) y mediciones (.csv)
	@set -o pipefail; ts=$$(date +%Y%m%d-%H%M%S); out="$(RESULTS_DIR)/cpp_baseline_$$ts"; \
	  ./$(CPU_BIN) --csv "$$out.csv" $(ARGS) | tee "$$out.txt"; \
	  echo "Tabla guardada en $$out.txt"

env: | $(RESULTS_DIR)  ## Volcado manual del entorno (scripts/collect_env.sh) a results/unidad1/
	@set -o pipefail; out="$(RESULTS_DIR)/env_manual_$$(date +%Y%m%d-%H%M%S).txt"; \
	  bash scripts/collect_env.sh | tee "$$out"; echo; echo "Guardado en $$out"

asm: $(BUILD_CONFIG) | $(ASM_DIR)  ## Ensamblador e informe del vectorizador de GCC en build/asm/
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -S -fverbose-asm src/cpu/matmul_cpu.cpp -o $(ASM_DIR)/matmul_cpu.s
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c src/cpu/matmul_cpu.cpp -o /dev/null \
		-fopt-info-vec-optimized=$(ASM_DIR)/vec_optimized.txt
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c src/cpu/matmul_cpu.cpp -o /dev/null \
		-fopt-info-vec-missed=$(ASM_DIR)/vec_missed.txt
	@echo "Bucles vectorizados ($(ASM_DIR)/vec_optimized.txt; no vectorizados en vec_missed.txt):"
	@sed 's|^.*/src/|src/|' $(ASM_DIR)/vec_optimized.txt

consolidate:        ## Regenera runs.csv y all_*.csv de results/unidad1/ a partir de cada corrida
	$(CARGO) run --release --manifest-path $(CARGO_MANIFEST) $(CARGO_FEATURES) -- --consolidate $(ARGS)

clean:              ## Borra build/ y harness/target/ (los resultados NO se tocan)
	rm -rf $(BUILD_DIR) harness/target

help:               ## Muestra esta ayuda
	@echo "Objetivos disponibles:"
	@grep -hE '^[a-zA-Z_-]+:.*## ' $(MAKEFILE_LIST) | \
	  sed -E 's/^([a-zA-Z_-]+):.*## (.*)$$/\1|\2/' | \
	  awk -F'|' '{ printf "  make %-14s %s\n", $$1, $$2 }'
	@echo
	@echo "Variables: CXX=$(CXX)  OPENBLAS=$(OPENBLAS)  ARGS=\"$(ARGS)\""
	@echo "CXXFLAGS=$(CXXFLAGS)"

# =============================================================================
#  GPU (Unidades 2 y 3): kernels CUDA y binario de benchmark
#
#  Es independiente de los objetivos de la Unidad 1: `make` y `make cpu`
#  siguen compilando exactamente lo mismo que antes. El binario GPU enlaza la
#  biblioteca C++ (libmatmul_cpu.a) para reutilizar el generador, el checksum y
#  la referencia cpu_ikj, y ademas libcublas (solo como referencia comparativa).
# =============================================================================
NVCC       ?= nvcc
CUDA_ARCH  ?= 60
NVCCFLAGS  := -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -Xcompiler "-O3 -Wall -Wextra"
GPU_SRCS   := src/gpu/cuda_util.cu src/gpu/matmul_gpu.cu src/gpu/main_gpu.cu
GPU_OBJS   := $(patsubst src/gpu/%.cu,$(OBJ_DIR)/%.o,$(GPU_SRCS))
GPU_BIN    := $(BUILD_DIR)/bench_gpu

NVCC_VERSION := $(shell $(NVCC) --version 2>/dev/null | tail -n 1)

gpu: $(GPU_BIN)        ## Compila el binario de benchmark GPU (build/bench_gpu)

$(OBJ_DIR)/%.o: src/gpu/%.cu $(BUILD_CONFIG) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BIN): $(GPU_OBJS) $(LIB)
	$(NVCC) $(NVCCFLAGS) $(GPU_OBJS) $(LIB) -lcublas -o $@

# =============================================================================
#  Reglas de compilación
# =============================================================================
$(BUILD_CONFIG): FORCE | $(BUILD_DIR)
	@printf '%s\n' "$$BUILD_CONFIG_TEXT" > $@.tmp
	@if cmp -s $@.tmp $@; then rm -f $@.tmp; else mv -f $@.tmp $@; echo "  [config] $@ actualizado"; fi

# Los flags efectivos se incrustan en la biblioteca (matmul_util_build_info()).
$(OBJ_DIR)/matmul_util.o: EXTRA_FLAGS = -DMATMUL_BUILD_FLAGS='"$(CXXFLAGS)"'

$(OBJ_DIR)/%.o: src/cpu/%.cpp $(BUILD_CONFIG) | $(OBJ_DIR)
	$(CXX) $(CPPFLAGS) $(CXXFLAGS) $(EXTRA_FLAGS) -MMD -MP -c $< -o $@

$(LIB): $(LIB_OBJS)
	@rm -f $@
	$(AR) rcs $@ $^

$(CPU_BIN): $(MAIN_OBJ) $(LIB)
	$(CXX) $(CXXFLAGS) $(MAIN_OBJ) $(LIB) $(LDFLAGS) -o $@

$(BUILD_DIR) $(OBJ_DIR) $(ASM_DIR) $(RESULTS_DIR):
	@mkdir -p $@

FORCE:

-include $(DEPS)
