# Multiplicación de matrices de gran escala (CPU vs GPU con CUDA)
## Reporte breve — Unidad 1: línea base secuencial en CPU

**Curso:** Arquitecturas de Alto Rendimiento (IF557ZIN)
**Autor:** David Fernando Prieto Cardoso
**Docente:** Henry Samuel Dueñas de la Cruz
**Fecha:** 8 de noviembre de 2026

---

## Resumen

Este reporte documenta la primera entrega del proyecto de multiplicación de
matrices de gran escala `C = A × B` en precisión simple (FP32). Se implementa
desde cero la versión secuencial en CPU en C/C++ con dos órdenes de bucles —la
ingenua `i-j-k` y la optimizada `i-k-j`— y se miden el tiempo de cómputo, el
speedup y los GFLOPS para matrices de N = 1024, 2048 y 4096. La versión
`i-k-j` alcanza un speedup de ~36× frente a `i-j-k` y entre el 11 % y el 25 %
del techo práctico de la CPU (OpenBLAS a 1 hilo). Estos resultados sirven como
denominador de referencia (`T_CPU`) para los speedups frente a los kernels CUDA
de las Unidades 2 y 3.

---

## 1. Introducción y objetivos

La multiplicación de matrices densas (GEMM) es una de las operaciones más
estudiadas en computación de alto rendimiento y el bloque de construcción de
muchos algoritmos numéricos y de aprendizaje automático. El objetivo de la
Unidad 1 es establecer una **línea base secuencial en CPU** correcta, medible y
reproducible que sirva de:

1. **Referencia de correctitud** para las versiones en GPU.
2. **Denominador de speedup** (`S = T_CPU / T_GPU`) en las Unidades 2 y 3.
3. **Punto de partida** para justificar la optimización con memoria compartida
   (tiling) y para cuantificar la distancia al techo práctico de la CPU
   (OpenBLAS).

El alcance se limita a precisión simple (FP32), matrices cuadradas de
1024×1024, 2048×2048 y 4096×4096, y a un análisis cuantitativo de la jerarquía
de memoria.

---

## 2. Descripción del algoritmo

### 2.1 Producto de matrices y FLOPs

Dadas dos matrices `A` y `B` de N×N, el producto `C = A × B` se define como:

> `C[i][j] = Σₖ A[i][k] · B[k][j]`, para `k = 0 … N−1`

Cada elemento `C[i][j]` requiere N multiplicaciones y N−1 sumas. Como hay N²
elementos, el número total de operaciones en punto flotante es:

> **FLOPs = 2·N³**

El rendimiento se expresa en GFLOPS (`1 GFLOP = 10⁹ operaciones/segundo`):

> **GFLOPS = 2·N³ / (t · 10⁹)**

### 2.2 Disposición en memoria

Las matrices se almacenan como **arreglos unidimensionales en orden por filas
(row-major)**: el elemento `(i, j)` está en `m[i·N + j]`. Esta es la misma
disposición que usará CUDA, de modo que los kernels futuros y la versión CPU
comparten la convención. Los datos se alinean a 64 bytes (una línea de caché) y
se generan con un generador pseudoaleatorio reproducible (SplitMix64, semilla
fija) en el intervalo [−1, 1].

### 2.3 Órdenes de bucles: i-j-k frente a i-k-j

Se implementan dos variantes que recorren los mismos datos en distinto orden:

**`i-j-k` (ingenua, referencia de correctitud).** El bucle interno recorre `k`:

```c
for (i) for (j) { suma = 0; for (k) suma += A[i][k] * B[k][j]; C[i][j] = suma; }
```

Cada iteración lee `B[k][j]` saltando una fila completa en memoria (strided
access), lo que produce **fallos de caché por conflicto** cuando N es potencia
de 2 y, al ser una reducción (`suma += …`), el compilador no puede reordenar las
sumas en FP sin `-ffast-math`. Resultado: apenas se vectoriza y es la versión
más lenta.

**`i-k-j` (optimizada, línea base oficial).** El bucle interno recorre `j`:

```c
for (i) for (k) { aik = A[i][k]; for (j) C[i][j] += aik * B[k][j]; }
```

- **Localidad espacial:** el bucle interno recorre una fila de `B` y una fila de
  `C` de forma contigua (elementos consecutivos en memoria).
- **Localidad temporal:** `aik` queda en un registro durante todo el bucle `j`,
  y la fila `i` de `C` se reutiliza en cada iteración de `k`, permaneciendo en
  L1/L2.
- **Vectorización:** las iteraciones de `j` son independientes, por lo que el
  compilador vectoriza el bucle con SIMD (en esta CPU, AVX2 con FMA: 8 floats
  por instrucción y multiply-add fusionado).

La implementación está en `include/matmul_cpu.h` y `src/cpu/matmul_cpu.cpp`
(escrita desde cero). Las funciones con interfaz C (`extern "C"`) reciben los
punteros a `A`, `B` y `C` (con `__restrict`), el tamaño `N` en 64 bits, ponen
`C` a cero antes de acumular y **no reservan memoria** (la gestiona el llamador).

---

## 3. Antecedentes

GEMM es probablemente la operación más optimizada de la computación: las
bibliotecas BLAS (y en particular el trabajo de Goto y van de Geijn sobre
empaquetado de datos, tiling multinivel y micro-kernels en ensamblador) definen
el estado del arte en CPU. En este trabajo **la implementación es propia y
escrita desde cero** porque es precisamente lo que evalúa la consigna; las
bibliotecas optimizadas (OpenBLAS) se usan **únicamente como referencia
comparativa** para mostrar el techo práctico de la CPU y como referencia de
validación, nunca como objeto de reimplementación. El material educativo
consultado se estudia y se cita en `docs/referencias.md`.

---

## 4. Entorno de prueba

Los valores de esta tabla provienen de `results/unidad1/env_<run>.json`,
generado automáticamente por el arnés antes de la corrida.

| Componente                 | Valor                                               |
| -------------------------- | --------------------------------------------------- |
| CPU                        | Intel Core i7-12700KF (Alder Lake)                  |
| Núcleos / hilos            | 10 núcleos / 20 hilos (vista de WSL2)               |
| SIMD soportado             | SSE4.2, AVX, AVX2, FMA (sin AVX-512)                |
| Caché L1d                  | 48 KiB                                              |
| Caché L2                   | 1.25 MiB (por núcleo)                               |
| Caché L3                   | 25 MiB (compartida)                                 |
| RAM                        | 15.5 GiB                                            |
| Sistema operativo          | Ubuntu 24.04.5 LTS (WSL2, kernel 6.18.54-microsoft) |
| Compilador                 | g++ (GCC) 13.3.0                                    |
| Flags de compilación       | `-O3 -march=native -std=c++17 -Wall -Wextra -fPIC`  |
| `-march=native` resuelve a | `alderlake` (AVX2, FMA)                             |
| Rust / make                | rustc 1.99.1 · GNU Make 4.3                         |
| Gobernador de CPU          | no disponible (WSL2)                                |
| OpenBLAS                   | 0.3.26 DYNAMIC_ARCH (MAX_THREADS=64)                |

> **Nota sobre WSL2:** la medición se ejecutó dentro de WSL2, que **no expone**
> los sensores de temperatura ni el gobernador de frecuencias del host. Por eso
> el reporte no incluye temperatura ni throttling térmico (ver Sección 7).

---

## 5. Metodología

- **Generación de datos:** matrices en [−1, 1] con semilla fija `42` (streams
  independientes para `A` y `B`), idénticas en el binario C++ y en el arnés.
- **Medición de tiempo:** con `std::chrono::steady_clock` **dentro de C++**,
  rodeando exclusivamente el triple bucle (o la llamada a `cblas_sgemm`). No se
  incluye la generación de datos ni la reserva de memoria.
- **Tiempo de cómputo puro vs. total:** en CPU el tiempo reportado es el de
  cómputo puro. La separación "cómputo vs. transferencia Host↔Device"
  (`cudaEventRecord`/`cudaMemcpy`) corresponde a la Unidad 2, donde sí habrá
  copias por PCIe.
- **Calentamiento y repeticiones:** 1 warm-up no medido + 5 repeticiones
  medidas por (implementación, N); para N = 4096 se usan 3 repeticiones. Se
  reporta la **mediana** (robusta a valores atípicos), además del mínimo, la
  media y la desviación estándar.
- **GFLOPS y speedup:** `GFLOPS = 2·N³ / (t_mediana · 10⁹)`; el speedup de
  referencia de esta unidad es `S = T_ijk / T_ikj` (efecto de la localidad de
  caché).
- **Validación de correctitud:** comparación contra una referencia calculada en
  `f64` (error relativo de Frobenius y error absoluto máximo) y verificación
  probabilística de Freivalds; tolerancia `1e-4`. La referencia preferida es
  `cpu_openblas_1t`.
- **Control de hilos de OpenBLAS:** se mide `cpu_openblas_1t` (1 hilo, la
  comparación justa con el código secuencial) y `cpu_openblas_mt` (20 hilos,
  solo informativa como techo absoluto). El porcentaje frente a OpenBLAS se
  calcula contra la variante de 1 hilo.

---

## 6. Resultados base

Los valores provienen de `results/unidad1/summary_<run>.csv`. Los GFLOPS usan
la mediana.

### 6.1 Tiempo de cómputo (mediana) y GFLOPS

| Implementación  | Tipo       | N    | t_mediana (s) | GFLOPS |
| --------------- | ---------- | ---- | ------------- | ------ |
| cpu_ijk         | propia     | 1024 | 2.212         | 0.97   |
| cpu_ikj         | propia     | 1024 | 0.0614        | 34.95  |
| cpu_ijk         | propia     | 2048 | 24.207        | 0.71   |
| cpu_ikj         | propia     | 2048 | 0.654         | 26.26  |
| cpu_ijk         | propia     | 4096 | —             | —      |
| cpu_ikj         | propia     | 4096 | 8.882         | 15.47  |
| cpu_openblas_1t | referencia | 1024 | 0.0157        | 137.01 |
| cpu_openblas_1t | referencia | 2048 | 0.122         | 140.61 |
| cpu_openblas_1t | referencia | 4096 | 0.959         | 143.38 |
| cpu_openblas_mt | referencia | 1024 | 0.0035        | 611.35 |
| cpu_openblas_mt | referencia | 2048 | 0.0237        | 724.88 |
| cpu_openblas_mt | referencia | 4096 | 0.175         | 784.37 |

> `cpu_ijk` se omite en N = 4096 por ser demasiado lenta para N > 2048
> (tarda ~24 s solo en N = 2048).

### 6.2 Speedup (ikj frente a ijk) y % de OpenBLAS

| N    | S = T_ijk / T_ikj | % de OpenBLAS (GFLOPS_ikj / GFLOPS_openblas_1t × 100) |
| ---- | ----------------- | ----------------------------------------------------- |
| 1024 | 36.0              | 25.5 %                                                |
| 2048 | 37.0              | 18.7 %                                                |
| 4096 | —                 | 10.8 %                                                |

> El speedup de N = 4096 no se reporta porque `cpu_ijk` se omitió en ese tamaño.

### 6.3 Validación de correctitud

| Implementación | N    | Error relativo (Frobenius) | Freivalds |
| -------------- | ---- | -------------------------- | --------- |
| cpu_ikj        | 1024 | 5.85×10⁻⁷                  | 5.41×10⁻⁷ |
| cpu_ikj        | 2048 | 8.45×10⁻⁷                  | 8.48×10⁻⁷ |
| cpu_ikj        | 4096 | 1.19×10⁻⁶                  | 1.16×10⁻⁶ |

> Todos los errores quedan varios órdenes por debajo de la tolerancia `1e-4`
> (el error absoluto máximo es de ~3.7×10⁻⁴ en N = 4096). `cpu_ijk` presenta
> errores equivalentes.

---

## 7. Observaciones y limitaciones

- **Entorno WSL2 sin sensores:** la máquina de prueba es WSL2 sobre Windows,
  por lo que **no se pudo registrar temperatura, frecuencia ni throttling**
  térmico (el arnés los deja vacíos). No es una limitación del código sino del
  entorno virtualizado.
- **Estabilidad de la medición:** el coeficiente de variación (CV) se mantiene
  bajo (≤ 6.4 % en casi todos los casos), lo que indica una medición estable.
- **Speedup y localidad de caché:** `i-k-j` es ~36–37× más rápido que `i-j-k`
  gracias a la localidad espacial/temporal y a la vectorización AVX2+FMA.
- **Distancia al techo de OpenBLAS:** `i-k-j` alcanza 25.5 % (N=1024), 18.7 %
  (N=2048) y 10.8 % (N=4096) del techo secuencial. La brecha **crece con N**
  porque `i-k-j` no aplica tiling multinivel, empaquetado de datos ni
  micro-kernels en ensamblador; cuando N excede la caché L2 (1.25 MiB), la
  matriz ya no cabe en caché y el rendimiento cae de ~35 a ~15 GFLOPS. Esta
  diferencia es justamente la motivación para el tiling en las Unidades 2 y 3.
- **Precisión:** el análisis se limita a FP32; otras precisiones quedan como
  extensión.

---

## 8. Siguientes pasos

1. Implementar el **kernel CUDA naive** (un hilo por elemento, acceso directo a
   memoria global) en esta máquina con GPU NVIDIA (RTX 4060 Ti) (Unidad 2).
2. Instrumentar con `cudaEventRecord` (tiempo de kernel) y `cudaMemcpy`
   (tiempo de transferencia Host↔Device) para separar cómputo y PCIe.
3. Comparar `S = T_CPU / T_GPU` contra la línea base aquí establecida.
4. En la Unidad 3, optimizar con **memoria compartida (tiling)** y contrastar
   con `cuBLAS` como techo en GPU.

---

## 9. Referencias

1. Goto, K., & van de Geijn, R. A. (2008). *Anatomy of High-Performance Matrix
   Multiplication*. ACM TOMS.
2. Hennessy, J. L., & Patterson, D. A. *Computer Architecture: A Quantitative
   Approach*. Morgan Kaufmann.
3. Kirk, D. B., & Hwu, W. W. *Programming Massively Parallel Processors: A
   Hands-on Approach*. Morgan Kaufmann.
4. Williams, S., Waterman, A., & Patterson, D. (2009). *Roofline: An Insightful
   Visual Performance Model for Multicore Architectures*. Communications of the
   ACM, 52(4), 65–76.
5. NVIDIA Corporation. *CUDA C++ Programming Guide*.
6. Material educativo consultado (no copiado): ver `docs/referencias.md`.
