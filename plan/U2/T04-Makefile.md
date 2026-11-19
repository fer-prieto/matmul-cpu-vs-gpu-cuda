# U2-T04 — Makefile (modificar: sección GPU)

- **ID**: U2-T04
- **Depende de**: U2-T03
- **Objetivo**: compilar `src/gpu/*.cu` con nvcc/sm_60 y enlazar `build/bench_gpu`.

## Archivos
- Modificar: `Makefile`. Insertar UN bloque nuevo **al final** (antes de `-include $(DEPS)` si no molesta, o justo antes; si dudas, al final del archivo) y añadir `gpu` al bloque `help` existente con una línea `gpu: ## Compila el binario de benchmark GPU (build/bench_gpu)`.
- NO tocar: variables CPU, reglas existentes, `all`, `clean`, `bench*`, `test*`.

## Especificación (texto exacto del bloque)
```make
# --- GPU (Unidades 2 y 3) ---------------------------------------------------
NVCC       ?= nvcc
CUDA_ARCH  ?= 60
NVCCFLAGS  := -O3 -std=c++17 -arch=sm_$(CUDA_ARCH) -Xcompiler "-O3 -Wall -Wextra"
GPU_SRCS   := src/gpu/cuda_util.cu src/gpu/matmul_gpu.cu src/gpu/main_gpu.cu
GPU_OBJS   := $(patsubst src/gpu/%.cu,$(OBJ_DIR)/%.o,$(GPU_SRCS))
GPU_BIN    := $(BUILD_DIR)/bench_gpu

NVCC_VERSION := $(shell $(NVCC) --version 2>/dev/null | tail -1)

gpu: $(GPU_BIN)        ## Compila el binario de benchmark GPU (build/bench_gpu)

$(OBJ_DIR)/%.o: src/gpu/%.cu $(BUILD_CONFIG) | $(OBJ_DIR)
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(GPU_BIN): $(GPU_OBJS) $(LIB)
	$(NVCC) $(NVCCFLAGS) $(GPU_OBJS) $(LIB) -lcublas -o $@
```
- Añadir `NVCC=$(NVCC)` y `NVCC_VERSION=$(NVCC_VERSION)` al `define BUILD_CONFIG_TEXT` (así queda registrado el toolchain); si `nvcc` no existe, `NVCC_VERSION` queda vacío y `make gpu` falla con el error natural de nvcc.
- `build_config.txt` ya fuerza recompilación al cambiar flags (mecánica existente).
- `clean` ya borra `build/` completo: no hay que tocar nada.

## Casos límite
- `make GPU...` no existe; override: `make gpu CUDA_ARCH=70` para otra GPU (el servidor usa el default 60).
- `make -n gpu` debe mostrar comandos nvcc sin ejecutar.

## Verificación LOCAL (sí ejecutable)
```bash
make -n gpu | head -6            # imprime los comandos nvcc SIN ejecutarlos
make gpu 2>&1 | tail -2          # local: falla por "nvcc: no encontrado" (mensaje claro, esperado)
```

## Verificación EN SERVIDOR (la corre el usuario)
```bash
make gpu && ls -la build/bench_gpu && ./build/bench_gpu --help | head -3
```

## Criterios de aceptación
- [ ] `make gpu` produce `build/bench_gpu` en el servidor; `make cpu gpu` compila ambas sin warnings.
- [ ] `git diff Makefile` = solo el bloque GPU + línea de help + 2 líneas en BUILD_CONFIG_TEXT.
- [ ] `make clean && make cpu` sigue funcionando igual que en U1 (nada CPU roto).

## Errores típicos
- Hardcodear `-arch=sm_60` ignorando `CUDA_ARCH`.
- Compilar `.cu` con `$(CXX)` (debe ser `$(NVCC)`).
- Olvidar `-lcublas` (falla el enlace de U3-T02).
- Cambiar `all` (el flujo de U1 debe quedar intacto).

## Commit
`build(gpu): objetivos nvcc y binario bench_gpu`
