# Multiplicación de Matrices de Gran Escala: Comparación CPU vs GPU con CUDA

Kernel CUDA naive y comparación CPU vs GPU en FP32

Alumno: David Fernando Prieto Cardoso Profesor: Henry Samuel Dueñas de la Cruz Curso: Arquitecturas de Alto Rendimiento (IF557ZIN)

Fecha de realización: 18 de noviembre de 2026 Cusco, Perú

## Resumen

Este reporte documenta la segunda entrega del proyecto de multiplicación de matrices de gran escala _C_ = _A × B_ en precisión simple (FP32). Se implementa desde cero un **kernel CUDA naive** en el que cada hilo calcula un elemento de _C_ accediendo directamente a memoria global, y se mide con cudaEvent el **tiempo de cómputo puro** del kernel y con cudaMemcpy el **tiempo total** , que incluye las transferencias Host _↔_ Device. La línea base en CPU se re-mide en el mismo servidor para que el speedup sea válido. Para _N_ = 1024, 2048 y 4096 el kernel alcanza 89 _,_ 66, 371 _,_ 35 y 346 _,_ 18 GFLOPS, con speedups de 6 _,_ 74 _×_ , 29 _,_ 54 _×_ y 43 _,_ 25 _×_ frente a la implementación secuencial _i_ - _k_ - _j_ del mismo equipo. Las transferencias por PCIe cuestan entre el 10 % y el 19 % del tiempo total y se comportan como un enlace de _≈_ 4 _,_ 5 GB/s.

## Índice de Contenidos

|**1.**|**Intr**|**oducción y objetivos**|**1**|
|---|---|---|---|
||1.1.|Antecedentes . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>1|
|**2.**|**Des**|**cripción del algoritmo**|**1**|
||2.1.|Producto de matrices y FLOPs<br>. . . . . . . . . . . . . . . . . . . . . . .|.<br>2|
||2.2.|Diseño del kernel naive . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>2|
||2.3.|Separación de tiempos<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>3|
|**3.**|**Ent**|**orno de prueba**|**3**|
|**4.**|**Met**|**odología**|**4**|
|**5.**|**Res**|**ultados**|**5**|
||5.1.|Línea base en CPU del servidor<br>. . . . . . . . . . . . . . . . . . . . . . .|.<br>5|
||5.2.|Kernel naive en la GPU . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>5|
||5.3.|Speedup frente a la CPU . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>6|
||5.4.|Análisis<br>. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>6|
||5.5.|Figuras . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>7|
|**6.**|**Obs**|**ervaciones y limitaciones**|**8**|
||6.1.|Siguientes pasos . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|.<br>9|
|**Re**|**feren**|**cias**|**10**|
|**Ín**|**di**|**ce de Figuras**||
|1.|Ti|empo de cómputo puro del kernel naive frente a _N_ (escala logarítmica).<br>.|.<br>7|
|2.|R|endimiento del kernel naive en GFLOPS frente a _N_.<br>. . . . . . . . . . . .|.<br>8|
|3.|Sp|eedup del kernel naive frente a cpu_ikj del servidor: _S_kernel y _S_total.<br>. . .|.<br>8|
|**Ín**|**di**|**ce de Tablas**||

|1.|Entorno de prueba de la Unidad 2 (servidor del curso). . . . . . . . . . . . . .<br>3|
|---|---|
|2.|Línea base en CPU medida en el servidor (_T_CPU de los speedups). . . . . . . .<br>5|
|3.|Kernel naive: tiempo de cómputo puro, tiempo total y rendimiento. . . . . . .<br>5|
|4.|Speedup del kernel naive respecto de cpu_ikj del mismo servidor. . . . . . . .<br>6|
|5.|Checksums y validación del kernel naive contra cpu_ikj. . . . . . . . . . . . .<br>7|

## 1. Introducción y objetivos

La Unidad 1 estableció la línea base secuencial en CPU del producto de matrices densas _C_ = _A × B_ y dejó abierta la pregunta central del proyecto: cuánto se gana al trasladar el cómputo a una GPU. El objetivo de la Unidad 2 es responderla con la implementación más simple posible, el **kernel naive** , que sirve además de punto de partida para todas las optimizaciones posteriores:

1. **Implementar** en CUDA un kernel en el que cada hilo calcula un elemento _C_ [ _i_ ][ _j_ ] leyendo _A_ y _B_ directamente de memoria global.

2. **Separar** en la medición el tiempo de cómputo puro del kernel (eventos CUDA) del tiempo total, que incluye las copias Host _↔_ Device por el bus PCIe.

3. **Comparar** ambos tiempos con la versión _i_ - _k_ - _j_ en CPU **medida en el mismo servidor** , de modo que el speedup _S_ = _T_ CPU _/T_ GPU no mezcle máquinas distintas.

4. **Validar** numéricamente el resultado del dispositivo contra la referencia en CPU con tolerancia 10<sup>_−_4</sup> .

El alcance se mantiene en FP32, matrices cuadradas de 1024<sup>2</sup> , 2048<sup>2</sup> y 4096<sup>2</sup> , y transferencias secuenciales sin solapamiento (sin _streams_ ni memoria fijada).

### 1.1. Antecedentes

El modelo de ejecución SIMT agrupa los hilos en _warps_ de 32 hebras que ejecutan la misma instrucción sobre datos distintos; el paralelismo de datos se explota repartiendo el trabajo entre miles de hilos y el rendimiento queda limitado por la jerarquía de memoria (registros, memoria compartida, L1/L2 y memoria global) [1]. La guía de programación de CUDA documenta el modelo de grillas y bloques, la sincronización con __syncthreads() y la medición con eventos [2].

El kernel naive es el ejemplo canónico de la guía y, aunque su rendimiento es pobre, resulta indispensable como referencia: cualquier optimización posterior (memoria compartida, vectorización, _register tiling_ ) se juzga contra él. El estado del arte en GPU está representado por bibliotecas como cuBLAS, que se emplearán como techo comparativo en la Unidad 3 y que, igual que OpenBLAS en la Unidad 1, **solo se llaman por su API** y no forman parte del código evaluable [3].

El modelo de _roofline_ permite interpretar los resultados en términos de intensidad aritmética y ancho de banda [4]; se usa en la sección de resultados para explicar por qué el kernel naive es un problema limitado por memoria mientras que la versión con tiling se acerca al límite de cómputo.

## 2. Descripción del algoritmo

### 2.1. Producto de matrices y FLOPs

Dadas dos matrices _A_ y _B_ de _N × N_ , el producto _C_ = _A × B_ se define como:

<!-- Start of picture text -->
N − 1<br>C [ i ][ j ] = � A [ i ][ k ] · B [ k ][ j ] (1)<br>k =0<br><!-- End of picture text -->

Cada elemento de _C_ requiere _N_ multiplicaciones y _N −_ 1 sumas, de modo que el total de operaciones es FLOPs = 2 _N_<sup>3</sup> y el rendimiento se expresa como:

<!-- Start of picture text -->
GFLOPS = 2 N 3 (2)<br>t · 10 9<br><!-- End of picture text -->

Las matrices se almacenan en orden por filas ( _row-major_ ): el elemento ( _i, j_ ) está en _m_ [ _i · N_ + _j_ ], la misma convención de la Unidad 1 y la que impone el arreglo contiguo que se copia al dispositivo.

### 2.2. Diseño del kernel naive

El kernel asigna **un hilo por elemento** de _C_ . La grilla y los bloques se eligen con una forma bidimensional que replica la disposición de la matriz:

- **Bloque:** 16 _×_ 16 = 256 hilos (dim3(16,16)).

- **Grilla:** _⌈N/_ 16 _⌉× ⌈N/_ 16 _⌉_ bloques. Para _N_ = 4096 son 256 _×_ 256 = 65 536 bloques.

- **Índices:** _i_ = blockIdx.y _·_ 16 + threadIdx.y y _j_ = blockIdx.x _·_ 16 + threadIdx.x; el hilo se descarta si _i ≥ N_ o _j ≥ N_ .

El cuerpo del hilo recorre el eje _k_ completo acumulando en un registro y escribe _C_ [ _i_ ][ _j_ ] una sola vez:

float acc = 0.0f;

for (int k = 0; k < N; ++k)

acc += A[i * N + k] * B[k * N + j]; C[i * N + j] = acc;

El patrón de acceso es asimétrico y esa asimetría explica el resultado. Para un warp (32 hilos con threadIdx.y fijo y threadIdx.x variable) los 16 hilos de una misma fila leen _A_ [ _i_ ][ _k_ ] de forma **contigua** —una sola transacción de memoria por fila—, pero leen _B_ [ _k_ ][ _j_ ] en elementos separados _N_ posiciones: la columna de _B_ se recorre con un salto de _N ·_ 4 bytes (4 KiB para _N_ = 1024, 16 KiB para _N_ = 4096). Cada iteración de _k_ toca, por tanto, una línea distinta de memoria y la memoria caché no logra reutilizar nada: el kernel es **limitado por ancho de banda** y no por capacidad de cómputo.

### 2.3. Separación de tiempos

A diferencia de la Unidad 1, aquí interesa distinguir dos magnitudes que la consigna exige por separado:

- **Tiempo de cómputo puro** ( _t_ kernel): dos eventos CUDA registrados alrededor del lanzamiento del kernel. No incluye ninguna copia.

- **Tiempo total** ( _t_ total): los mismos eventos rodeando la secuencia completa _AH→D_ , _BH→D_ , kernel y _CD→H_ .

La diferencia _t_ total _− t_ kernel es el costo de comunicación del bus PCIe, que se analiza como ancho de banda efectivo a partir del volumen transferido, 3 _N_<sup>2</sup> _·_ 4 bytes. Los buffers de dispositivo se reservan **fuera** de la medición y se reutilizan entre repeticiones, de modo que cudaMalloc nunca contamina los tiempos.

## 3. Entorno de prueba

Los valores de esta tabla provienen de results/unidad2/env/ (hardware_<run>.txt, gpu_<run>_{before,after}.csv y build_<run>.txt), generados por los scripts del repositorio antes y después de la corrida.

Tabla 1: Entorno de prueba de la Unidad 2 (servidor del curso).

|**Componente**|**Valor**|
|---|---|
|CPU|2 _×_ Intel Xeon Gold 6134 @ 3.20 GHz|
|Núcleos / hilos|16 núcleos / 32 hilos (2 sockets, 8 núcleos por socket)|
|Cachés|L1d 512 KiB, L1i 512 KiB, L2 16 MiB, L3 49,5 MiB|
|Nodos NUMA|2|
|Memoria RAM|125 GiB|
|GPU|NVIDIA Quadro GP100, _compute capability_ 6.0 (sm_60)|
|Memoria de la GPU|16 384 MiB (HBM2)|
|Enlace GPU–CPU|PCIe Gen3 _×_16|
|Reloj máximo del SM|1556 MHz|
|Límite de potencia|235 W|
|Driver NVIDIA|580.178.04|
|Toolkit CUDA|nvcc 12.2.140 (CUDA 12.2)|
|Compilador CPU|g++ 11.4.0 (-O3 -march=native)|
|Flags del kernel|-O3 -std=c++17 -arch=sm_60 (sin –use_fast_math)|
|Sistema operativo|Linux 5.15 (Ubuntu 22.04)|
|Make / Python|GNU Make 4.3 _·_ Python 3.10.12|

Con 3584 núcleos CUDA y el reloj máximo registrado por el dispositivo, el techo teórico de FP32 es 3584 _·_ 2 _·_ 1 _,_ 556 GHz _≈_ 11 _,_ 1 TFLOP/s. Ese valor se usa solo como referencia de

escala; la comparación principal se hace contra cuBLAS medido en el mismo equipo (Unidad 3).

**Sobre los relojes:** el servidor es compartido y no se fijaron las frecuencias (la opción – use-sudo del repositorio permite hacerlo, pero modifica un recurso global). Antes y después de cada corrida se registraron relojes, temperatura, potencia y motivos de _throttling_ ; la dispersión de cada medición queda reportada como desviación estándar y coeficiente de variación en el resumen de resultados.

## 4. Metodología

- **Generación de datos:** matrices en [ _−_ 1 _,_ 1] con semilla fija 42 y streams independientes para _A_ (stream 0) y _B_ (stream 1), con el mismo generador SplitMix64 de la Unidad 1. Como el binario de GPU reutiliza la biblioteca C++ de la Unidad 1, las matrices y el checksum son **idénticos** en ambas herramientas.

- **Medición del kernel:** dos eventos CUDA (cudaEventRecord) registrados en el stream por defecto alrededor del lanzamiento, con cudaEventSynchronize antes de leer el resultado. Los eventos se crean y destruyen en cada medición y **nunca** envuelven reservas de memoria ni transferencias.

- **Medición del tiempo total:** los mismos eventos rodean ahora _AH→D_ , _BH→D_ , kernel y _CD→H_ . Las copias usan el mismo espacio de host alineado a 64 bytes que la Unidad 1 y son **paginables** (no se usó memoria fijada ni _streams_ ).

- **Calentamiento y repeticiones:** 1 ejecución de calentamiento no medida (para que las páginas de la GPU estén residentes) y 5 repeticiones medidas por combinación (implementación, _N_ ). Se reporta la **mediana** , junto con mínimo, media, desviación estándar y coeficiente de variación.

- **Protocolo por ejecución:** en cada repetición se mide primero el kernel solo y después el ciclo completo. El checksum de _C_ (suma en doble precisión, la misma función de la Unidad 1) se calcula en todas las ejecuciones y debe ser idéntico entre repeticiones.

- **Speedup:** _S_ kernel = _T_ CPU _/t_ kernel y _S_ total = _T_ CPU _/t_ total, con _T_ CPU = mediana de cpu_ikj **medida en este servidor** . Se re-midió la CPU porque los tiempos de la Unidad 1 provienen de otra máquina (portátil con WSL2) y usarlos como denominador sesgaría el speedup.

- **Validación:** el resultado del dispositivo se compara con cpu_ikj calculado en el servidor. Se reportan el error relativo de Frobenius _∥C − C_ ref _∥F /∥C_ ref _∥F_ y el error absoluto máximo, ambos calculados en double; se acepta si el error relativo es _≤_ 10<sup>_−_4</sup> . Si la validación falla, el proceso termina con código de salida 3 y la corrida se descarta.

- **Reproducibilidad:** todo el flujo (check _→_ build _→_ CPU _→_ GPU _→_ agregación _→_ tablas _→_ figuras) está automatizado en run_all.sh, con sandbox aislado, marcadores de etapa y registro de las versiones de nvcc, g++, driver y flags en build_<run>.txt.

## 5. Resultados

Los valores provienen de results/unidad2/summary_<run>.csv y baseline/cpu_summary_<run>.csv Todos los tiempos son medianas de 5 repeticiones medidas con 1 calentamiento previo.

### 5.1. Línea base en CPU del servidor

La CPU se re-midió antes de la GPU, con el mismo binario de la Unidad 1 compilado con -O3 -march=native (AVX-512 disponible en el Xeon Gold 6134).

Tabla 2: Línea base en CPU medida en el servidor ( _T_ CPU de los speedups).

|**Implementación**|**N**|_t_ **mediana (s)**|**GFLOPS**|**Reps**|
|---|---|---|---|---|
|cpu_ijk|1024|3.2237|0.67|5|
|cpu_ikj|1024|0.1613|13.31|5|
|cpu_ijk|2048|35.5246|0.48|5|
|cpu_ikj|2048|1.3665|12.57|5|
|cpu_ijk|4096|—|—|—|
|cpu_ikj|4096|17.1723|8.00|5|

cpu_ijk se omite en _N_ = 4096 porque el plan de la Unidad 1 lo excluye para tamaños grandes (tarda _≈_ 35 s solo en _N_ = 2048). El rendimiento de cpu_ikj cae de 13.31 a 8.00 GFLOPS al pasar de 1024 a 4096 porque a partir de los 64 MiB por matriz la reutilización en caché L2 (16 MiB) desaparece.

### 5.2. Kernel naive en la GPU

Tabla 3: Kernel naive: tiempo de cómputo puro, tiempo total y rendimiento.

|**N**|_t_**kernel (s)**|_σ_ **(s)**|_t_**total (s)**|**GFLOPS**|**Fracción PCIe**|
|---|---|---|---|---|---|
|1024|0.023950|0.000166|0.026685|89.66|10.2 %|
|2048|0.046263|0.002617|0.057315|371.35|19.3 %|
|4096|0.397015|0.000165|0.441634|346.18|10.1 %|

El esfuerzo se reparte entre _N_<sup>2</sup> hilos: 1 048 576 hilos para _N_ = 1024 y 16 777 216 para _N_ = 4096, ejecutados en bloques de 256. El kernel ocupa el 80–90 % del tiempo total; el resto son las copias por PCIe.

### 5.3. Speedup frente a la CPU

Tabla 4: Speedup del kernel naive respecto de cpu_ikj del mismo servidor.

|**N**|_T_**CPU (s)**|_S_**kernel**|_S_**total**|**GFLOPSCPU / GFLOPSGPU**|
|---|---|---|---|---|
|1024|0.1613|6.74_×_|6.05_×_|13.31 / 89.66|
|2048|1.3665|29.54_×_|23.84_×_|12.57 / 371.35|
|4096|17.1723|43.25_×_|38.88_×_|8.00 / 346.18|

El speedup **crece con** _N_ : 6.74 _×_ , 29.54 _×_ y 43.25 _×_ . La razón es doble. Por un lado, la GPU mantiene su rendimiento casi constante a partir de _N_ = 2048 (371 y 346 GFLOPS) mientras la CPU cae de 12.57 a 8.00 GFLOPS porque sus matrices dejan de caber en caché. Por otro, a _N_ = 1024 el kernel naive desperdicia el paralelismo disponible, como se discute a continuación.

### 5.4. Análisis

**El kernel naive es débil, y de forma no monótona.** El rendimiento pasa de 89.66 GFLOPS en _N_ = 1024 a 371.35 GFLOPS en _N_ = 2048 y baja levemente a 346.18 GFLOPS en _N_ = 4096. Las dos primeras cifras corresponden a la misma carga por hilo, así que un factor 4 _,_ 1 _×_ entre ellas no puede atribuirse al tamaño: la explicación más plausible es el patrón de acceso a _B_ . La columna de _B_ se recorre con un salto de _N ·_ 4 bytes (4 KiB, 8 KiB y 16 KiB), y con 4 KiB el conjunto de líneas que toca un warp se concentra en pocos conjuntos de la caché L1/L2, generando conflictos; con 8 KiB y 16 KiB el conflicto se reparte. Las tres mediciones son estables (coeficiente de variación de 0.69 %, 5.53 % y 0.04 %), de modo que la diferencia no es ruido. Queda como observación a confirmar con el perfilador en la Unidad 3.

**Distancia al techo del dispositivo.** Con 3584 _·_ 2 _·_ 1 _,_ 556 GHz _≈_ 11 _,_ 1 TFLOP/s de FP32, el kernel naive alcanza el 0.8 % ( _N_ = 1024), 3.3 % ( _N_ = 2048) y 3.1 % ( _N_ = 4096) del máximo teórico. Está lejos porque cada iteración de _k_ trae de memoria global 2 _N_ elementos para producir una sola suma: la intensidad aritmética efectiva es de _O_ (1) FLOP por byte y el kernel queda **limitado por ancho de banda** , exactamente el régimen que el modelo de _roofline_ identifica como no optimizable sin reutilizar datos [4].

**Costo de la comunicación (PCIe).** El volumen transferido es 3 _N_<sup>2</sup> _·_ 4 bytes: 12.6 MB, 50.3 MB y 201.3 MB. El tiempo de transferencia (2.73 ms, 11.05 ms y 44.62 ms) implica un ancho de banda efectivo de **4.60, 4.55 y 4.51 GB/s** , muy estable y muy por debajo de los _≈_ 12 GB/s prácticos de un enlace PCIe Gen3 _×_ 16. La causa es que las copias usan memoria **paginable** y no se solapan con el cómputo: no hay memoria fijada ( _pinned_ ) ni _streams_ . Aun así, como el kernel naive es lento, la transferencia solo pesa 10–19 % del total; en la Unidad 3, con un kernel 5 a 19 veces más rápido, ese porcentaje se vuelve dominante.

Tabla 5: Checksums y validación del kernel naive contra cpu_ikj.

|**N**|**Error relativo**|**Error máx. absoluto**|**¿Válido?**|
|---|---|---|---|
|1024|0.00|0.00|sí|
|2048|0.00|0.00|sí|
|4096|0.00|0.00|sí|

El error es exactamente cero en los tres tamaños: la GPU acumula en el registro _acc_ en el mismo orden que la CPU ( _k_ = 0 _,_ 1 _, . . . , N −_ 1) y ambas plataformas fusionan multiplicación y suma (FMA), de modo que el resultado es bit a bit idéntico. El checksum de _C_ coincide con el del binario de CPU, lo que confirma que ambas herramientas operan sobre las mismas matrices de entrada.

### 5.5. Figuras

Figura 1: Tiempo de cómputo puro del kernel naive frente a _N_ (escala logarítmica).

Figura 2: Rendimiento del kernel naive en GFLOPS frente a _N_ .

Figura 3: Speedup del kernel naive frente a cpu_ikj del servidor: _S_ kernel y _S_ total.

## 6. Observaciones y limitaciones

- **Servidor compartido sin relojes fijados:** la GPU no estaba en uso por otros procesos durante la corrida (conteo de procesos de cómputo ajenos igual a cero antes y después), pero no se fijaron las frecuencias porque es un recurso global. La dispersión quedó registrada: el coeficiente de variación del tiempo de kernel es de 0.69 %, 5.53 % y 0.04 % para _N_ = 1024, 2048 y 4096.

- **Transferencias sin optimizar:** las copias usan memoria paginable y no se solapan con el cómputo, por lo que el ancho de banda efectivo (4.5–4.6 GB/s) es aproximadamente un tercio del práctico del enlace. Es una decisión de alcance: la Unidad 2 mide el costo real del esquema más simple.

- **Error de validación nulo:** el resultado de la GPU es bit a bit igual al de la CPU porque el orden de acumulación y el uso de FMA coinciden. Esto valida el kernel, pero no permite observar diferencias de redondeo; si se comparara contra cuBLAS (que reordena las sumas) el error sería del orden de 10<sup>_−_7</sup> relativo, como en la Unidad 1 frente a OpenBLAS.

- **Re-muestreo obligatorio de la CPU:** los tiempos de la Unidad 1 se tomaron en otra máquina, y el servidor rinde entre 3 y 7 veces menos en cpu_ikj (13.31 vs 34.95 GFLOPS en _N_ = 1024). Los speedups de este reporte usan únicamente medidas del servidor.

- **Anomalía en** _N_ = 1024 **:** el rendimiento del kernel naive es 4.1 veces menor que en _N_ = 2048 con la misma carga por hilo. Se atribuye al patrón _strided_ sobre la columna de _B_ y a conflictos de conjunto en caché; es una hipótesis pendiente de confirmar con el perfilador.

- **Alcance numérico:** solo FP32 y matrices cuadradas. Otras precisiones y formas quedan como extensión.

### 6.1. Siguientes pasos

1. Implementar el kernel con **memoria compartida (tiling)** : bloques de 32 _×_ 32 hilos que cargan cooperativamente submatrices de _A_ y _B_ , con __syncthreads() entre carga y consumo [1].

2. Comparar la variante con _padding_ en los tiles frente a la variante sin _padding_ , para medir el efecto de los conflictos de banco.

3. Medir cuBLAS en el mismo equipo como techo práctico de la GPU, igual que OpenBLAS en la Unidad 1, y expresar el resultado propio como porcentaje de ese techo.

4. Evaluar el costo de comunicación con el kernel rápido: al reducir el tiempo de cómputo en un orden de magnitud, la fracción de PCIe pasa a dominar el tiempo total y justifica discutir memoria fijada y _streams_ .

5. Interpretar los resultados con el modelo de _roofline_ completo (intensidad aritmética frente a ancho de banda HBM2) y con métricas del perfilador si están disponibles en el servidor.

## Referencias

- [1] Kirk, D. B. y Hwu, W.-m. W., Programming Massively Parallel Processors: A Hands-on Approach. Morgan Kaufmann, 3rd ed., 2017.

- [2] Corporation, N., “Cuda c++ programming guide”, 2025, https://docs.nvidia.com/cuda /cuda-c-programming-guide/.

- [3] Goto, K. y van de Geijn, R. A., “Anatomy of high-performance matrix multiplication”, ACM Transactions on Mathematical Software, vol. 34, no. 3, pp. 12:1–12:25, 2008, doi: 10.1145/1356052.1356053.

- [4] Williams, S., Waterman, A., y Patterson, D. A., “Roofline: An insightful visual performance model for multicore architectures”, Communications of the ACM, vol. 52, no. 4, pp. 65–76, 2009, doi:10.1145/1498765.1498785.

- [5] Hennessy, J. L. y Patterson, D. A., Computer Architecture: A Quantitative Approach. Morgan Kaufmann, 6th ed., 2019.
