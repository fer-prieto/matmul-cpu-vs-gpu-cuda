# Multiplicación de Matrices de Gran Escala: Comparación CPU vs GPU con CUDA

Tiling, conflictos de banco y techo de cuBLAS en FP32

Alumno: David Fernando Prieto Cardoso Profesor: Henry Samuel Dueñas de la Cruz Curso: Arquitecturas de Alto Rendimiento (IF557ZIN)

Fecha de realización: 18 de noviembre de 2026 Cusco, Perú

## Resumen

Este reporte cierra el proyecto con la tercera entrega del producto de matrices _C_ = _A × B_ en FP32. Se implementa desde cero un **kernel con memoria compartida (tiling)** de 32 _×_ 32 hilos, se compara con una variante con _padding_ en los tiles para aislar el efecto de los conflictos de banco y se mide cuBLAS como techo práctico del dispositivo. Para _N_ = 1024, 2048 y 4096 el kernel con tiling alcanza 1649, 2043 y 1964 GFLOPS, es decir 18 _,_ 5 _×_ , 5 _,_ 1 _×_ y 5 _,_ 6 _×_ más que el kernel naive de la Unidad 2, y entre el 21 % y el 27 % de cuBLAS. El _padding_ , contra lo previsto, **no mejora el rendimiento** (+0 _,_ 2 %, _−_ 1 _,_ 4 %, _−_ 2 _,_ 9 %): con bloques de 32 _×_ 32 cada warp lee el tile de _A_ con dirección uniforme ( _broadcast_ ), de modo que no hay conflicto de banco que eliminar. Con el kernel rápido, la transferencia por PCIe pasa a representar hasta el 65 % del tiempo total.

## Índice de Contenidos

|**1.**|**Intr**|**oducción y objetivos**|**1**|
|---|---|---|---|
||1.1.|Antecedentes . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>1|
|**2.**|**Des**|**cripción del algoritmo**|**1**|
||2.1.|Producto de matrices y métricas . . . . . . . . . . . . . . . . .|. . . . . . .<br>2|
||2.2.|Kernel con memoria compartida (tiling) . . . . . . . . . . . . .|. . . . . . .<br>2|
||2.3.|Variante con padding y análisis de bancos . . . . . . . . . . . .|. . . . . . .<br>2|
||2.4.|cuBLAS como referencia . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>3|
|**3.**|**Ent**|**orno de prueba**|**3**|
|**4.**|**Met**|**odología**|**4**|
|**5.**|**Res**|**ultados y análisis**|**5**|
||5.1.|Tiempos y rendimiento . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>6|
||5.2.|Speedup frente a la CPU y porcentaje de cuBLAS . . . . . . .|. . . . . . .<br>7|
||5.3.|Ablación de padding: resultado negativo<br>. . . . . . . . . . . .|. . . . . . .<br>7|
||5.4.|Costo de comunicación . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>8|
||5.5.|Análisis con el modelo de roofline . . . . . . . . . . . . . . . .|. . . . . . .<br>8|
||5.6.|Validación . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>9|
||5.7.|Figuras . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>9|
|**6.**|**Con**|**clusiones, limitaciones y trabajo futuro**|**11**|
||6.1.|Conclusiones . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>11|
||6.2.|Limitaciones . . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>12|
||6.3.|Trabajo futuro . . . . . . . . . . . . . . . . . . . . . . . . . . .|. . . . . . .<br>12|
|**Re**|**feren**|**cias**|**13**|

|**Ín**|**dice de Figuras**||
|---|---|---|
|1.|Rendimiento de las cuatro implementaciones frente a _N_ (escala logarítmica). .|9|
|2.|Speedup frente a cpu_ikj del servidor para cada implementación. . . . . . . .|10|
|3.|Ablación de _padding_: el tile sin _padding_ ya es libre de conflictos de banco.<br>. .|10|
|4.|Tiempo de cómputo puro de cada implementación frente a _N_.<br>. . . . . . . . .|11|

## Índice de Tablas

|1.<br>Entorno de prueba de la Unidad 3|(servidor del curso). . . . . . . . . . . . . .|
|---|---|

|2.|Tiempo de cómputo puro, tiempo total y rendimiento de las cuatro implemen-|
|---|---|
||taciones. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>6|
|3.|Speedup respecto de cpu_ikj del servidor y porcentaje del techo cuBLAS. . .<br>7|
|4.|Efecto del _padding_ en los tiles (_stride_ 33 frente a 32).<br>. . . . . . . . . . . . . .<br>7|
|5.|Fracción del tiempo total consumida por las transferencias y ancho de banda|
||efectivo. . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .<br>8|
|6.|Validación de las cuatro implementaciones contra cpu_ikj. . . . . . . . . . . .<br>9|

## 1. Introducción y objetivos

La Unidad 2 mostró que el kernel naive apenas alcanza el 3 % del techo del dispositivo porque cada hilo lee de memoria global 2 _N_ elementos para producir una suma. El objetivo de la Unidad 3 es atacar exactamente ese problema: **reutilizar los datos en memoria compartida** para que cada elemento de _A_ y de _B_ se lea de memoria global una sola vez por bloque y se consuma muchas veces desde el chip. Los objetivos concretos son:

1. **Implementar** un kernel con _tiling_ : bloques de 32 _×_ 32 hilos que cargan cooperativamente submatrices de _A_ y _B_ en memoria compartida y se sincronizan con __syncthreads() entre carga y consumo.

2. **Aislar el efecto de los conflictos de banco** comparando la variante sin _padding_ con una variante cuyo tile tiene una columna extra (stride 33).

3. **Medir cuBLAS** en el mismo dispositivo como techo práctico, del mismo modo que OpenBLAS se usó como techo de la CPU en la Unidad 1, y expresar el resultado propio como porcentaje de ese techo.

4. **Consolidar el análisis** del costo de comunicación: con un kernel un orden de magnitud más rápido, la fracción del tiempo total que consume el bus PCIe cambia por completo.

El alcance sigue siendo FP32, matrices cuadradas de 1024<sup>2</sup> , 2048<sup>2</sup> y 4096<sup>2</sup> , sin _streams_ ni memoria fijada, y sin extensiones de _register tiling_ ni vectorización float4, que quedan como trabajo futuro.

### 1.1. Antecedentes

El _tiling_ o bloqueo es la técnica clásica para GEMM en GPU: el bloque de hilos coopera para traer un tile de _A_ y uno de _B_ a memoria compartida, y cada hilo acumula su elemento de _C_ desde ahí, de modo que la memoria global se recorre _N/_ 32 veces menos [1]. La memoria compartida está organizada en 32 bancos de 4 bytes por SM; dos hilos que acceden a direcciones del mismo banco en la misma transacción serializan el acceso, y el remedio habitual es desplazar una columna el tile ( _padding_ ) para romper la periodicidad de direcciones [2]. Este reporte verifica experimentalmente cuándo ese remedio es necesario y cuándo no.

El estado del arte en GPU lo representan bibliotecas como cuBLAS y CUTLASS, con empaquetado de datos, _register tiling_ y micro-kernels específicos por arquitectura; el trabajo de Goto y van de Geijn describe la lógica equivalente en CPU [3]. Igual que en la Unidad 1, esas bibliotecas se usan **solo como referencia comparativa** a través de su API, no son objeto de reimplementación.

## 2. Descripción del algoritmo

### 2.1. Producto de matrices y métricas

Se mantiene la formulación de las unidades anteriores, _C_ [ _i_ ][ _j_ ] =<sup>�</sup><sup>_N_</sup> _k_ =0<sup>_−_1</sup><sup>_A_[</sup><sup>_i_][</sup><sup>_k_]</sup><sup>_B_[</sup><sup>_k_][</sup><sup>_j_],con</sup> FLOPs = 2 _N_<sup>3</sup> y

<!-- Start of picture text -->
2 N 3<br>GFLOPS = S kernel = T CPU , S total = T CPU (1)<br>t kernel · 10 9 , t kernel t total<br><!-- End of picture text -->

donde _T_ CPU es la mediana de cpu_ikj medida en el mismo servidor. Se reporta también el porcentaje frente a cuBLAS, GFLOPStiled _/_ GFLOPScublas _·_ 100, que es el análogo del « % de OpenBLAS» de la Unidad 1.

### 2.2. Kernel con memoria compartida (tiling)

Cada bloque calcula un tile de _C_ de 32 _×_ 32 elementos con 32 _×_ 32 = 1024 hilos, es decir, **un hilo por elemento** del tile. El bloque recorre el eje _k_ en pasos de 32 y en cada paso:

1. Carga en memoria compartida la banda de _A_ (filas del tile) y la banda de _B_ (columnas del tile): 32 _×_ 32 elementos cada una, con guardas de borde para _N_ no múltiplo de 32.

2. Se sincroniza con __syncthreads() para garantizar que los 1024 hilos vean el tile completo.

3. Acumula los 32 productos correspondientes en un registro local.

4. Se vuelve a sincronizar antes de sobrescribir el tile en el paso siguiente.

Los índices son los mismos de la Unidad 2, con el bloque de 32 en lugar de 16:

<!-- Start of picture text -->
fila = blockIdx.y · 32 + threadIdx.y , col = blockIdx.x · 32 + threadIdx.x (2)<br><!-- End of picture text -->

y las dos cargas cooperativas escriben en el tile compartido en la posición (threadIdx.y _,_ threadIdx.x), leyendo _A_ [fila][ _kt_ + threadIdx.x] y _B_ [ _kt_ + threadIdx.y][col]. El elemento de _C_ se escribe **una sola vez** , al salir del bucle de tiles. El bloque usa 2 _·_ 32 _·_ 32 _·_ 4 = 8 KiB de memoria compartida, muy por debajo de los 48 KiB por SM del dispositivo, así que la ocupación no queda limitada por este recurso.

El beneficio teórico es inmediato: sin tiling cada elemento de _A_ y de _B_ se lee _N_ veces de memoria global; con tiles de 32 _×_ 32 se lee _N/_ 32 veces, es decir, 32 veces menos tráfico para _N_ = 1024 y 128 veces menos para _N_ = 4096.

### 2.3. Variante con padding y análisis de bancos

La variante con _padding_ declara los tiles con stride 33 ( _As_ [32][33], _Bs_ [32][33]): cada fila del tile empieza un elemento más allá, desplazando el mapeo a bancos. La variante de referencia usa stride 32.

El análisis habitual dice que con stride 32 la lectura _As_ [ty][ _k_ ] cae siempre en el banco _k_ para todos los hilos y produce un conflicto de 32 vías. Ese razonamiento supone que el warp abarca varias filas del tile, lo que **no ocurre con bloques de** 32 _×_ 32: un warp agrupa 32 hilos consecutivos en el eje _x_ , es decir, con threadIdx.y **fijo** . En consecuencia:

- _As_ [ty][ _k_ ] es la **misma dirección** para los 32 hilos del warp: el hardware la difunde en una sola transacción ( _broadcast_ ), sin conflicto.

- _Bs_ [ _k_ ][tx] recorre bancos consecutivos, uno por hilo, también sin conflicto.

Por tanto, con esta forma de bloque el tile sin _padding_ ya es libre de conflictos y el _padding_ solo añade consumo de memoria compartida. El conflicto aparecería si el warp abarcara varias filas del tile, por ejemplo con bloques de 8 _×_ 32, donde _As_ [ty][ _k_ ] cae en bancos distintos por cada valor de ty. La comparación experimental de la Sección 5 confirma el razonamiento: es el resultado negativo del reporte.

### 2.4. cuBLAS como referencia

La implementación de referencia llama a cublasSgemm sin transponer y con los operandos **intercambiados** : como cuBLAS interpreta las matrices en orden por columnas, la identidad _C_ = _A B ⇐⇒ C_<sup>T</sup> = _B_<sup>T</sup> _A_<sup>T</sup> permite calcular el producto _row-major_ pasando _B_ como primer operando y _A_ como segundo. Se mide con los mismos eventos CUDA y con las mismas matrices que el código propio, y su papel es exclusivamente comparativo: fija el techo práctico del dispositivo tal como OpenBLAS fijó el de la CPU.

## 3. Entorno de prueba

Se usa el mismo servidor de la Unidad 2, de modo que los resultados de ambas entregas son comparables y comparten la línea base en CPU. Los valores provienen de results/unidad3/env/.

Tabla 1: Entorno de prueba de la Unidad 3 (servidor del curso).

|**Componente**|**Valor**|
|---|---|
|CPU|2 _×_ Intel Xeon Gold 6134 @ 3.20 GHz|
|Núcleos / hilos|16 núcleos / 32 hilos (2 sockets)|
|Cachés|L1d 512 KiB, L1i 512 KiB, L2 16 MiB, L3 49,5 MiB|
|Memoria RAM|125 GiB|
|GPU|NVIDIA Quadro GP100, _compute capability_ 6.0 (sm_60)|
|Memoria de la GPU|16 384 MiB (HBM2)|
|Memoria compartida por SM|48 KiB (configuración por defecto)|
|Bancos por SM|32 de 4 bytes|
|Enlace GPU–CPU|PCIe Gen3 _×_16|
|Reloj máximo del SM|1556 MHz|
|Driver / toolkit|580.178.04 _·_ nvcc 12.2.140 (CUDA 12.2)|
|Flags del kernel|-O3 -std=c++17 -arch=sm_60 (sin –use_fast_math)|
|Perfilador|Nsight Compute 2023.2.2.3 (sin permisos de contadores)|
|Sistema operativo|Linux 5.15 (Ubuntu 22.04)|

Con 3584 núcleos CUDA al reloj máximo registrado (1556 MHz), el techo teórico de FP32 es _≈_ 11 _,_ 1 TFLOP/s. El techo **práctico** que se usa como referencia es cuBLAS medido en este mismo equipo: 6.05, 8.74 y 8.85 TFLOP/s, es decir, entre el 54 % y el 79 % del máximo teórico.

Los relojes de la GPU **no** se fijaron (servidor compartido); se registraron relojes, temperatura, potencia y motivos de _throttling_ antes y después de la corrida, y la dispersión de cada medición queda en el resumen de resultados.

## 4. Metodología

- **Generación de datos:** matrices en [ _−_ 1 _,_ 1] con semilla fija 42 y streams independientes para _A_ (stream 0) y _B_ (stream 1), con el mismo generador SplitMix64 de la Unidad 1. Como el binario de GPU reutiliza la biblioteca C++ de la Unidad 1, las matrices y el checksum son **idénticos** en ambas herramientas.

- **Medición del kernel:** dos eventos CUDA (cudaEventRecord) registrados en el stream por defecto alrededor del lanzamiento, con cudaEventSynchronize antes de leer el resultado. Los eventos se crean y destruyen en cada medición y **nunca** envuelven reservas de memoria ni transferencias.

- **Medición del tiempo total:** los mismos eventos rodean ahora _AH→D_ , _BH→D_ , kernel y _CD→H_ . Las copias usan el mismo espacio de host alineado a 64 bytes que la Unidad 1 y son **paginables** (no se usó memoria fijada ni _streams_ ).

- **Calentamiento y repeticiones:** 1 ejecución de calentamiento no medida (para que

las páginas de la GPU estén residentes) y 5 repeticiones medidas por combinación (implementación, _N_ ). Se reporta la **mediana** , junto con mínimo, media, desviación estándar y coeficiente de variación.

- **Protocolo por ejecución:** en cada repetición se mide primero el kernel solo y después el ciclo completo. El checksum de _C_ (suma en doble precisión, la misma función de la Unidad 1) se calcula en todas las ejecuciones y debe ser idéntico entre repeticiones.

- **Speedup:** _S_ kernel = _T_ CPU _/t_ kernel y _S_ total = _T_ CPU _/t_ total, con _T_ CPU = mediana de cpu_ikj **medida en este servidor** . Se re-midió la CPU porque los tiempos de la Unidad 1 provienen de otra máquina (portátil con WSL2) y usarlos como denominador sesgaría el speedup.

- **Validación:** el resultado del dispositivo se compara con cpu_ikj calculado en el servidor. Se reportan el error relativo de Frobenius _∥C − C_ ref _∥F /∥C_ ref _∥F_ y el error absoluto máximo, ambos calculados en double; se acepta si el error relativo es _≤_ 10<sup>_−_4</sup> . Si la validación falla, el proceso termina con código de salida 3 y la corrida se descarta.

- **Reproducibilidad:** todo el flujo (check _→_ build _→_ CPU _→_ GPU _→_ agregación _→_ tablas _→_ figuras) está automatizado en run_all.sh, con sandbox aislado, marcadores de etapa y registro de las versiones de nvcc, g++, driver y flags en build_<run>.txt.

## 5. Resultados y análisis

Los valores provienen de results/unidad3/summary_<run>.csv; los tiempos son medianas de 5 repeticiones medidas con 1 calentamiento previo. Todas las implementaciones se validaron contra cpu_ikj del servidor.

### 5.1. Tiempos y rendimiento

Tabla 2: Tiempo de cómputo puro, tiempo total y rendimiento de las cuatro implementaciones.

|**Implementación**|**Tipo**|**N**|_t_**kernel (s)**|_t_**total (s)**|**GFLOPS**|
|---|---|---|---|---|---|
|gpu_naive|propia|1024|0.024131|0.026460|88.99|
|gpu_tiled|propia|1024|0.001302|0.003734|1649.43|
|gpu_tiled_pad|propia|1024|0.001299|0.003692|1652.85|
|gpu_cublas|referencia|1024|0.000355|0.002757|6047.48|
|gpu_naive|propia|2048|0.042868|0.051981|400.76|
|gpu_tiled|propia|2048|0.008410|0.018779|2042.84|
|gpu_tiled_pad|propia|2048|0.008527|0.018880|2014.71|
|gpu_cublas|referencia|2048|0.001966|0.012345|8740.69|
|gpu_naive|propia|4096|0.394831|0.437194|348.10|
|gpu_tiled|propia|4096|0.069973|0.114040|1964.18|
|gpu_tiled_pad|propia|4096|0.072038|0.114052|1907.87|
|gpu_cublas|referencia|4096|0.015529|0.057369|8850.38|

El kernel con tiling reduce el tiempo de cómputo entre 5 _,_ 6 _×_ y 18 _,_ 5 _×_ respecto del naive: 18 _,_ 48 _×_ en _N_ = 1024 (de 24.13 a 1.302 ms), 5 _,_ 10 _×_ en _N_ = 2048 y 5 _,_ 64 _×_ en _N_ = 4096. La mejora es mayor en el tamaño pequeño porque ahí el naive es especialmente ineficiente (ver Unidad 2) y el tiled mantiene un rendimiento alto y estable: 1649, 2043 y 1964 GFLOPS, con un coeficiente de variación del tiempo de kernel de 0.23 %, 0.29 % y 0.78 %.

### 5.2. Speedup frente a la CPU y porcentaje de cuBLAS

Tabla 3: Speedup respecto de cpu_ikj del servidor y porcentaje del techo cuBLAS.

|**Implemen**|**tación**<br>**N**|_S_**kernel**|_S_**total**|**% de cuBLAS**|
|---|---|---|---|---|
|gpu_naive|1024|6.69_×_|6.10_×_|1.47 %|
|gpu_tiled|1024|123.91_×_|43.20_×_|27.27 %|
|gpu_tiled_|pad<br>1024|124.16_×_|43.69_×_|27.33 %|
|gpu_naive|2048|31.88_×_|26.29_×_|4.59 %|
|gpu_tiled|2048|162.49_×_|72.77_×_|23.37 %|
|gpu_tiled_|pad<br>2048|160.25_×_|72.38_×_|23.05 %|
|gpu_naive|4096|43.49_×_|39.28_×_|3.93 %|
|gpu_tiled|4096|245.41_×_|150.58_×_|22.19 %|
|gpu_tiled_|pad<br>4096|238.38_×_|150.57_×_|21.56 %|

Frente a cpu_ikj del servidor, el kernel con tiling llega a _S_ kernel = 124 _×_ , 162 _×_ y 245 _×_ ; si se cuenta la comunicación completa, el speedup baja a 43 _×_ , 73 _×_ y 151 _×_ . Frente a cuBLAS, el código propio alcanza entre el 21 % y el 27 % del techo, una proporción comparable al 11–25 % que la versión _i_ - _k_ - _j_ lograba frente a OpenBLAS en la Unidad 1.

### 5.3. Ablación de padding: resultado negativo

Tabla 4: Efecto del _padding_ en los tiles ( _stride_ 33 frente a 32).

|**N**|**GFLOPS sin padding**|**GFLOPS con padding**<br>∆|
|---|---|---|
|1024|1649.43|1652.85<br>+0_._21 %|
|2048|2042.84|2014.71<br>_−_1_._38 %|
|4096|1964.18|1907.87<br>_−_2_._87 %|

El _padding_ **no aporta rendimiento** y en los tamaños grandes cuesta entre 1.4 % y 2.9 %, exactamente lo que predice el análisis de bancos de la Sección 2.3: con bloques de 32 _×_ 32 el warp mantiene threadIdx.y constante, la lectura _As_ [ty][ _k_ ] se difunde en una sola transacción y _Bs_ [ _k_ ][tx] recorre bancos consecutivos, de modo que **no existía conflicto que eliminar** . La penalización proviene de la memoria compartida adicional (33 en lugar de 32 columnas por fila) y de perder la alineación natural de los tiles.

Este es un resultado negativo con valor metodológico: confirma que el remedio clásico solo es necesario cuando el patrón de acceso lo justifica, y sugiere que la ablación debería repetirse con una forma de bloque cuyo warp abarque varias filas del tile (por ejemplo 8 _×_ 32) para observar el fenómeno que el _padding_ corrige.

### 5.4. Costo de comunicación

Tabla 5: Fracción del tiempo total consumida por las transferencias y ancho de banda efectivo.

|**Implementación**|**N**|_t_**transf (ms)**|**Fracción**|**Volumen (MB)**|**GB/s**|
|---|---|---|---|---|---|
|gpu_tiled|1024|2.43|65.1 %|12.6|5.18|
|gpu_tiled|2048|10.37|55.2 %|50.3|4.85|
|gpu_tiled|4096|44.07|38.6 %|201.3|4.57|
|gpu_cublas|4096|41.84|72.9 %|201.3|4.81|

Al acelerar el cómputo, la comunicación pasa a dominar: con el kernel con tiling el PCIe consume el 65 % del tiempo total en _N_ = 1024 y el 39 % en _N_ = 4096, mientras que con el naive eran el 10–19 %. Para cuBLAS en _N_ = 4096 la copia representa el 73 % del tiempo total: el límite del sistema dejó de ser la GPU y pasó a ser el bus. El ancho de banda efectivo (4.6–5.2 GB/s) es estable y no depende de la implementación, porque es una propiedad del esquema de copia: memoria paginable, sin memoria fijada ( _pinned_ ) y sin solapar las transferencias con el cómputo mediante _streams_ .

### 5.5. Análisis con el modelo de roofline

Para el kernel con tiling, cada bloque lee 2 _·_ 32 _·N_ elementos y realiza 2 _·_ 32<sup>2</sup> _·N_ operaciones, de modo que la intensidad aritmética crece con _N_ y, para los tamaños medidos, se sitúa muy por encima del punto de inflexión del dispositivo. El kernel está, por tanto, en el **régimen limitado por cómputo** y su distancia al techo no se explica por el tráfico de memoria, sino por la eficiencia con que cada hilo usa el dato cargado: un hilo del tile calcula un único elemento de _C_ y reutiliza cada valor de _As_ y _Bs_ solo 32 veces, frente a los esquemas con _register tiling_ que reutilizan cada valor cargado decenas de veces desde registros [4]. Esa es la brecha entre el 22 % y el 27 % del techo de cuBLAS que se mide aquí, y también el límite de la mejora obtenida: pasar del naive al tiling resuelve el tráfico de memoria global, pero no la eficiencia aritmética por hilo.

### 5.6. Validación

Tabla 6: Validación de las cuatro implementaciones contra cpu_ikj.

|**Implementación**|**N**|**Error relativo**|**¿Válido?**|
|---|---|---|---|
|gpu_naive|1024 / 2048|/ 4096<br>0.00|sí|
|gpu_tiled|1024 / 2048|/ 4096<br>0.00|sí|
|gpu_tiled_pad|1024 / 2048|/ 4096<br>0.00|sí|
|gpu_cublas|1024|5_._47_×_10<sup>_−_7</sup>|sí|
|gpu_cublas|2048 / 40|96<br>0.00|sí|

Los tres kernels propios reproducen cpu_ikj **bit a bit** : acumulan en el mismo orden ( _k_ creciente) y ambos usan FMA. cuBLAS reordena las sumas y difiere en 5 _._ 5 _×_ 10<sup>_−_7</sup> relativo en _N_ = 1024, cuatro órdenes por debajo de la tolerancia de 10<sup>_−_4</sup> ; en los otros dos tamaños el reordenamiento de cuBLAS coincide con el de la CPU para estas matrices. La validación confirma tanto la correctitud del tiling (incluidas las guardas de borde y las dos sincronizaciones) como la del envoltorio de cuBLAS con operandos intercambiados.

### 5.7. Figuras

Figura 1: Rendimiento de las cuatro implementaciones frente a _N_ (escala logarítmica).

Figura 2: Speedup frente a cpu_ikj del servidor para cada implementación.

Figura 3: Ablación de _padding_ : el tile sin _padding_ ya es libre de conflictos de banco.

Figura 4: Tiempo de cómputo puro de cada implementación frente a _N_ .

## 6. Conclusiones, limitaciones y trabajo futuro

### 6.1. Conclusiones

1. **La jerarquía de memoria es el factor decisivo.** Pasar de leer la memoria global en cada iteración (naive) a reutilizar tiles en memoria compartida mejora el tiempo de cómputo entre 5 _,_ 6 _×_ y 18 _,_ 5 _×_ con el mismo dispositivo y el mismo problema. El rendimiento pasa de 89–401 GFLOPS a 1649–2043 GFLOPS.

2. **El techo práctico se mide, no se supone.** cuBLAS alcanza 6.05–8.85 TFLOP/s en este dispositivo, y el código propio llega al 21–27 % de ese techo. Esa proporción es del mismo orden que la que la versión _i_ - _k_ - _j_ lograba frente a OpenBLAS en la Unidad 1 (11–25 %), lo que sugiere que ambas implementaciones propias comparten el mismo tipo de carencia: les falta reutilización en el nivel más alto de la jerarquía (registros y empaquetado de datos).

3. **El_padding_ no es una mejora universal.** Con bloques de 32 _×_ 32 el tile sin _padding_ ya está libre de conflictos de banco, porque el warp mantiene constante threadIdx.y y la lectura del tile de _A_ se difunde. El _padding_ cuesta entre 0.2 % (a favor) y 2.9 % (en contra), y la predicción teórica coincide con la medición.

4. **El bus se vuelve el cuello de botella.** Con el kernel naïve la transferencia pesaba 10–19 % del tiempo total; con el kernel con tiling pesa 39–65 %, y para cuBLAS en _N_ = 4096 llega al 73 %. Optimizar el cómputo sin optimizar la comunicación deja de tener sentido a partir de cierto punto.

5. **El speedup depende de qué se mida.** Contar solo el kernel da 124–245 _×_ ; contar la

comunicación completa da 43–151 _×_ . Reportar ambas cifras —como exige la consigna— evita conclusiones infladas.

### 6.2. Limitaciones

- **Relojes no fijados:** servidor compartido; se registraron relojes, temperatura y _throttling_ antes y después de cada corrida, y la dispersión del kernel con tiling es baja (CV de 0.23 % a 0.78 %).

- **Transferencias sin optimizar:** memoria paginable y sin solapamiento, lo que limita el ancho de banda efectivo a 4.6–5.2 GB/s frente a los _≈_ 12 GB/s prácticos del enlace PCIe Gen3 _×_ 16.

- **Sin perfilador:** Nsight Compute se intentó ejecutar y no produjo perfiles en este entorno (sin permisos de contadores para el usuario del servidor), de modo que la atribución de la brecha frente a cuBLAS se apoya en el modelo de _roofline_ y no en contadores de hardware.

- **Alcance del_tiling_ :** un solo nivel de tile y un elemento de _C_ por hilo. No se implementaron _register tiling_ , vectorización float4, doble buffer ni _software pipelining_ .

- **Formas de bloque:** la ablación de _padding_ se hizo con la forma que fijó el diseño (32 _×_ 32). El resultado negativo es correcto para esa forma y motiva repetir el experimento con un warp que abarque varias filas del tile.

### 6.3. Trabajo futuro

1. **_Register tiling_** : que cada hilo calcule un bloque de 4 _×_ 4 elementos de _C_ para reutilizar cada valor cargado desde memoria compartida 4 veces más, que es el paso que separa un kernel docente de uno competitivo [1].

2. **Vectorización float4** en las cargas desde memoria global y en los accesos a los tiles.

3. **Comunicación solapada** : memoria fijada ( _pinned_ ) y _streams_ para solapar las copias con el cómputo, y medir cuánto del 39–73 % de PCIe se recupera.

4. **Repetir la ablación de bancos** con bloques de 8 _×_ 32 o con acceso transpuesto a los tiles, para obtener el caso en que el _padding_ sí cambia el resultado.

5. **Comparación con CUTLASS** y con implementaciones basadas en Tensor Cores, como referencia del estado del arte actual.

6. **Proyección MIMD** (Unidad 4 del sílabo): repartir el producto entre varios nodos, evaluando si el costo de comunicación supera la ganancia, como se discute en el plan del proyecto.

## Referencias

- [1] Kirk, D. B. y Hwu, W.-m. W., Programming Massively Parallel Processors: A Hands-on Approach. Morgan Kaufmann, 3rd ed., 2017.

- [2] Corporation, N., “Cuda c++ programming guide”, 2025, https://docs.nvidia.com/cuda /cuda-c-programming-guide/.

- [3] Goto, K. y van de Geijn, R. A., “Anatomy of high-performance matrix multiplication”, ACM Transactions on Mathematical Software, vol. 34, no. 3, pp. 12:1–12:25, 2008, doi: 10.1145/1356052.1356053.

- [4] Williams, S., Waterman, A., y Patterson, D. A., “Roofline: An insightful visual performance model for multicore architectures”, Communications of the ACM, vol. 52, no. 4, pp. 65–76, 2009, doi:10.1145/1498765.1498785.

- [5] Hennessy, J. L. y Patterson, D. A., Computer Architecture: A Quantitative Approach. Morgan Kaufmann, 6th ed., 2019.
