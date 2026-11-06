<!--
═══════════════════════════════════════════════════════════════════════════
GUÍA DE RELLENO (leer antes de migrar a LaTeX) — borra este comentario al final
═══════════════════════════════════════════════════════════════════════════
Este reporte está completo salvo los NÚMEROS de la corrida oficial. Cada valor
pendiente aparece como un token entre llaves dobles {{...}}. Busca el token,
cópialo y sustitúyelo por el número real leyéndolo de los archivos que genera
el arnés (todos en results/unidad1/):

  • env_<run>.json          → entorno (CPU, cachés, RAM, SO, flags, OpenBLAS)
  • summary_<run>.csv       → una fila por (impl × N) con los estadísticos
  • validation_<run>.csv    → errores relativos/máximos y Freivalds
  • table_<run>.md          → misma tabla de resultados, ya formateada
  • all_summary.csv         → consolidado de todas las corridas (por si hay varias)

Qué hay que rellenar (solo 3 cosas dependen de la corrida oficial):
  1) TABLAS DE RESULTADOS (Sección 6): tokens {{summary...}} y {{validation...}}.
  2) GOBERNADOR de CPU (Sección 4): token {{env.frequency.governor}}.
  3) OPENBLAS y THROTTLING (Secciones 4 y 7): {{env.openblas.config}} y
     {{env.thermal.throttle_pkg_ms_during_run}} (solo si usaste make bench-ref).

  Todo lo demás de la Sección 4 (CPU, cachés, RAM, SO, compilador) ya viene
  PRE-LLENADO: son constantes de ESTA máquina, confirmadas contra env_<run>.json.
  Si cambias de equipo, vuelve a copiarlos desde el env.json correspondiente.

Convención de tokens (ruta real dentro de env_<run>.json entre puntos):
  {{summary.<impl>.<N>.<columna>}}          ejemplo {{summary.cpu_ikj.1024.t_median_s}}
  {{env.<clave>.<subclave>}}                 ejemplo {{env.frequency.governor}}
  {{validation.<impl>.<N>.<columna>}}        ejemplo {{validation.cpu_ikj.4096.rel_error}}

Columnas de summary_<run>.csv (para los tokens {{summary...}}):
  impl, kind, n, reps, t_min_s, t_median_s, t_mean_s, t_std_s,
  gflops, speedup_vs_ijk, pct_of_openblas, validated_against, rel_error

Columnas de validation_<run>.csv (para los tokens {{validation...}}):
  rel_error, max_abs_error, freivalds_rel_error, valid, ...

CÓMO OBTENER LOS DATOS:
  1) Prepara la máquina (corriente conectada, gobernador performance, sin apps).
  2) make test            → debe pasar.
  3) make bench           → resultados oficiales en results/unidad1/.
     (opcional) make bench-ref  → añade cpu_openblas_1t / cpu_openblas_mt.
  4) Los tokens se rellenan con los valores de summary_<run>.csv y env_<run>.json.

IMAGEN (opcional, solo si aporta — NO hace falta ninguna foto):
  Si quieres una figura, lo valioso es UN gráfico de barras "GFLOPS por
  implementación y tamaño" (o de líneas "tiempo mediano vs N"). Se genera con
  matplotlib desde results/unidad1/all_summary.csv, NO es una foto de pantalla:
      x = N (1024, 2048, 4096), agrupado por impl (cpu_ijk, cpu_ikj, cpu_openblas_1t)
      y = gflops (escala log)
  Guárdalo como docs/unidad1/figuras/gflops_vs_n.png y en la Sección 6 sustituye
  la línea marcada [FIGURA-GFLOPS] por:
      ![GFLOPS por implementación y tamaño](figuras/gflops_vs_n.png)
  Si decides no incluir gráfico (es opcional en esta unidad), borra la línea [FIGURA-GFLOPS].
  No se necesita capturar ninguna fotografía del hardware ni de la pantalla.
═══════════════════════════════════════════════════════════════════════════
-->

# Multiplicación de matrices de gran escala (CPU vs GPU con CUDA)
## Reporte breve — Unidad 1: línea base secuencial en CPU

**Curso:** Arquitecturas de Alto Rendimiento (IF557ZIN)
**Autor:** David Fernando Prieto Cardoso
**Docente:** Henry Samuel Dueñas de la Cruz
**Fecha:** {{fecha}}

---

## Resumen

Este reporte documenta la primera entrega del proyecto de multiplicación de
matrices de gran escala `C = A × B` en precisión simple (FP32). Se implementa
desde cero la versión secuencial en CPU en C/C++ con dos órdenes de bucles —el
ingenua `i-j-k` y la optimizada `i-k-j`— y se miden el tiempo de cómputo, el
speedup y los GFLOPS para matrices de N = 1024, 2048 y 4096. Se describe el
algoritmo, el entorno de prueba y los resultados base, que servirán como
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
  compilador vectoriza el bucle con SIMD (en esta CPU, AVX de 8 floats por
  instrucción con `-O3 -march=native`).

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
generado automáticamente por el arnés antes de cada corrida.

| Componente | Valor |
|---|---|
| CPU | Intel Core i5-2540M @ 2.60 GHz (Sandy Bridge) |
| Núcleos / hilos | 2 núcleos / 4 hilos |
| SIMD soportado | SSE4.2, AVX (sin AVX2, FMA ni AVX-512) |
| Caché L1d | 32 KiB |
| Caché L2 | 256 KiB (por núcleo) |
| Caché L3 | 3 MiB (compartida) |
| RAM | 12 GiB (12 412 735 488 bytes) |
| Sistema operativo | Artix Linux (rolling), kernel 7.2.10-hardened1-1-hardened |
| Compilador | g++ (GCC) 16.2.1 20260810 |
| Flags de compilación | `-O3 -march=native -std=c++17 -Wall -Wextra -fPIC` |
| `-march=native` resuelve a | `sandybridge` (AVX) |
| Rust / make | rustc 1.99.1 · GNU Make 4.4.1 |
| Gobernador de CPU | {{env.frequency.governor}} |
| OpenBLAS (si aplica) | {{env.openblas.config}} |

> Nota: al ser una laptop, los resultados dependen del estado de alimentación y
> temperatura; ver la Sección 7.

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
  caché; pendiente de confirmar con el docente).
- **Validación de correctitud:** comparación contra una referencia calculada en
  `f64` (error relativo de Frobenius y error absoluto máximo) y verificación
  probabilística de Freivalds; tolerancia `1e-4`. La referencia preferida es
  `cpu_openblas_1t` si está compilado y, si no, `cpu_ijk`.
- **Control de hilos de OpenBLAS:** se mide `cpu_openblas_1t` (1 hilo, la
  comparación justa con el código secuencial) y `cpu_openblas_mt` (todos los
  hilos, solo informativa como techo absoluto). El porcentaje frente a OpenBLAS
  se calcula contra la variante de 1 hilo.

---

## 6. Resultados base

Los valores provienen de `results/unidad1/summary_<run>.csv` (o
`all_summary.csv`). Los GFLOPS usan la mediana.

### 6.1 Tiempo de cómputo (mediana) y GFLOPS

| Implementación | Tipo | N | t_mediana (s) | GFLOPS |
|---|---|---|---|---|
| cpu_ijk | propia | 1024 | {{summary.cpu_ijk.1024.t_median_s}} | {{summary.cpu_ijk.1024.gflops}} |
| cpu_ikj | propia | 1024 | {{summary.cpu_ikj.1024.t_median_s}} | {{summary.cpu_ikj.1024.gflops}} |
| cpu_ijk | propia | 2048 | {{summary.cpu_ijk.2048.t_median_s}} | {{summary.cpu_ijk.2048.gflops}} |
| cpu_ikj | propia | 2048 | {{summary.cpu_ikj.2048.t_median_s}} | {{summary.cpu_ikj.2048.gflops}} |
| cpu_ijk | propia | 4096 | {{summary.cpu_ijk.4096.t_median_s}} | {{summary.cpu_ijk.4096.gflops}} |
| cpu_ikj | propia | 4096 | {{summary.cpu_ikj.4096.t_median_s}} | {{summary.cpu_ikj.4096.gflops}} |
| cpu_openblas_1t | referencia | 1024 | {{summary.cpu_openblas_1t.1024.t_median_s}} | {{summary.cpu_openblas_1t.1024.gflops}} |
| cpu_openblas_1t | referencia | 2048 | {{summary.cpu_openblas_1t.2048.t_median_s}} | {{summary.cpu_openblas_1t.2048.gflops}} |
| cpu_openblas_1t | referencia | 4096 | {{summary.cpu_openblas_1t.4096.t_median_s}} | {{summary.cpu_openblas_1t.4096.gflops}} |
| cpu_openblas_mt | referencia | 1024 | {{summary.cpu_openblas_mt.1024.t_median_s}} | {{summary.cpu_openblas_mt.1024.gflops}} |
| cpu_openblas_mt | referencia | 2048 | {{summary.cpu_openblas_mt.2048.t_median_s}} | {{summary.cpu_openblas_mt.2048.gflops}} |
| cpu_openblas_mt | referencia | 4096 | {{summary.cpu_openblas_mt.4096.t_median_s}} | {{summary.cpu_openblas_mt.4096.gflops}} |

> Si no se compiló OpenBLAS (`make bench-ref`), se omiten las filas
> `cpu_openblas_*` y la columna `% OpenBLAS` queda vacía.

### 6.2 Speedup (ikj frente a ijk) y % de OpenBLAS

| N | S = T_ijk / T_ikj | % de OpenBLAS (GFLOPS_ikj / GFLOPS_openblas_1t × 100) |
|---|---|---|
| 1024 | {{summary.cpu_ikj.1024.speedup_vs_ijk}} | {{summary.cpu_ikj.1024.pct_of_openblas}} |
| 2048 | {{summary.cpu_ikj.2048.speedup_vs_ijk}} | {{summary.cpu_ikj.2048.pct_of_openblas}} |
| 4096 | {{summary.cpu_ikj.4096.speedup_vs_ijk}} | {{summary.cpu_ikj.4096.pct_of_openblas}} |

[FIGURA-GFLOPS]

### 6.3 Validación de correctitud

| Implementación | N | Error relativo (Frobenius) | Freivalds |
|---|---|---|---|
| cpu_ikj | 1024 | {{validation.cpu_ikj.1024.rel_error}} | {{validation.cpu_ikj.1024.freivalds_rel_error}} |
| cpu_ikj | 2048 | {{validation.cpu_ikj.2048.rel_error}} | {{validation.cpu_ikj.2048.freivalds_rel_error}} |
| cpu_ikj | 4096 | {{validation.cpu_ikj.4096.rel_error}} | {{validation.cpu_ikj.4096.freivalds_rel_error}} |

> Todos los errores quedan por debajo de la tolerancia `1e-4` (en la práctica,
> al sumar en el mismo orden, `cpu_ikj` y `cpu_ijk` coinciden bit a bit; frente
> a OpenBLAS el error relativo es del orden de `1e-7`).

---

## 7. Observaciones y limitaciones

- **Variabilidad en laptop:** con el gobernador `schedutil`, la frecuencia
  oscila durante la rampa y el coeficiente de variación (CV) puede superar el
  20 % en tamaños pequeños. Por eso se reporta la mediana y se recomienda
  `performance` + corriente conectada para la corrida oficial.
- **Throttling térmico:** es una limitación propia de laptops; el arnés registra
  el throttling y la temperatura por ejecución. En la corrida oficial fue de
  {{env.thermal.throttle_pkg_ms_during_run}} ms.
- **Distancia al techo de OpenBLAS:** la implementación `i-k-j` alcanza
  {{summary.cpu_ikj.4096.pct_of_openblas}} % del techo secuencial. La brecha se
  explica por las técnicas que no aplica: tiling multinivel, empaquetado de
  datos y micro-kernels en ensamblador. Esta diferencia es justamente la
  motivación para el tiling en las Unidades 2 y 3.
- **Precisión:** el análisis se limita a FP32; otras precisiones quedan como
  extensión.

---

## 8. Siguientes pasos

1. Implementar el **kernel CUDA naive** (un hilo por elemento, acceso directo a
   memoria global) en la máquina con GPU de la universidad (Unidad 2).
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
