# Referencias y material consultado

> Base para la sección de antecedentes / estado del arte del informe IEEE final
> y del reporte breve de cada unidad. Todo el material que se cita aquí se
> **estudió para entender** los conceptos (localidad de caché, tiling, SIMD);
> **nunca se copió código**. Las implementaciones evaluables están escritas
> desde cero; las bibliotecas solo se llaman por su API pública.

## Estado del arte / bibliografía

- Goto, K., & van de Geijn, R. A. (2008). *Anatomy of High-Performance Matrix
  Multiplication*. ACM Transactions on Mathematical Software. Base del diseño
  de OpenBLAS y BLIS (empaquetado, micro-kernels, tiling multinivel).
- Volkov, V., & Demmel, J. W. (2008). *Benchmarking GPUs to Tune Dense Linear
  Algebra*. SC'08.
- Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful
  Visual Performance Model for Multicore Architectures*. Communications of the
  ACM.
- Documentación de NVIDIA CUTLASS: el enfoque actual de NVIDIA para GEMM en GPU
  (jerarquía de tiles, Tensor Cores).

## Material educativo consultado (para entender, no copiar)

| Recurso | Utilidad |
|---|---|
| Simon Boehm, blog *"How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance"* y repo `siboehm/SGEMM_CUDA` | Camino progresivo naive → coalescing → shared memory → tiling (Unidades 2 y 3) |
| NVIDIA `cuda-samples` → `matrixMul` | Kernel tiled clásico de la CUDA Programming Guide (`__shared__`) |
| `flame/how-to-optimize-gemm` | Optimización paso a paso de GEMM en CPU, más allá de i-k-j |
| Simon Boehm, *"Fast Multidimensional Matrix Multiplication on CPU from Scratch"* | Caché, tiling y SIMD en CPU (apoyo al reporte de la Unidad 1) |
| Algorithmica (S. Slotin), capítulo *Matrix Multiplication* | Efecto de la caché, justificación del orden i-k-j |
| Kirk & Hwu, *Programming Massively Parallel Processors* | Teoría del tiling (base del informe) |

## Bibliotecas de referencia (solo comparación)

| Plataforma | Bibliotecas | Uso en el proyecto |
|---|---|---|
| CPU | OpenBLAS, BLIS, Intel MKL, Eigen | OpenBLAS (Unidad 1, opcional): techo y validación |
| GPU | cuBLAS, CUTLASS | cuBLAS (Unidad 3): techo en GPU |
| Rust | `matrixmultiply`, `gemm`, `faer`, `ndarray` | Posible comparación en una extensión C++ vs Rust |

## Lecciones verificadas experimentalmente (Unidades 2 y 3)

| Observación | Evidencia | Lectura |
|---|---|---|
| El *padding* del tile no mejora el rendimiento con bloques de 32×32 | `results/unidad3/tablas/tabla_padding.tex` (+0.2 %, −1.4 %, −2.9 %) | Con un warp de 32×1 el acceso `As[ty][k]` es uniforme (*broadcast*) y `Bs[k][tx]` recorre bancos consecutivos: no hay conflicto que eliminar. El conflicto clásico aparece con warps que abarcan varias filas del tile. |
| El kernel naive rinde 4× peor en N=1024 que en N=2048 | `results/unidad2/summary_*.csv` (CV ≤ 5.5 %) | Patrón *strided* sobre la columna de B: el salto de 4 KiB concentra líneas en pocos conjuntos de caché. Hipótesis pendiente de perfilador. |
| La transferencia PCIe domina con kernels rápidos | `results/unidad3/tablas/*` (hasta 72.9 % del tiempo total con cuBLAS) | Memoria paginable sin solape: 4.5--5.2 GB/s efectivos frente a ~12 GB/s prácticos de Gen3 ×16. |
| Los kernels propios reproducen `cpu_ikj` bit a bit | `results/unidad*/validation_*.csv` (error 0.00) | Mismo orden de acumulación (k creciente) y FMA en ambas plataformas. |

## Herramientas de infraestructura

- Crates de Rust: `clap`, `csv`, `serde`, `chrono`, `libc`.
- `criterion`: descartado (microbenchmarks; no encaja con ejecuciones de
  decenas de segundos).
- `hyperfine`: comprobación rápida de binarios completos.
- Plantilla **IEEEtran** para los informes (LaTeX); en este proyecto se usa la
  plantilla *Template-Informe* de la UNSAAC.
- **Nsight Compute**: se intentó usar para contadores de *bank conflicts*; en el
  servidor del curso no produce perfiles por falta de permisos de contadores, de
  modo que la ablación empírica sustituye a la métrica del perfilador.

> Antes de usar un enlace o repositorio, verificar que sigue activo y revisar
> su licencia (mayoritariamente MIT/BSD). Citar todo lo consultado, aunque no
> se copie código.
