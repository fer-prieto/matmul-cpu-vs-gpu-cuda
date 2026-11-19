| Implementación | Tipo | N | t_kernel (s) | t_total (s) | GFLOPS |
|---|---|---|---|---|---|
| gpu_naive | propia | 1024 | 0.0241 | 0.0265 | 88.99 |
| gpu_naive | propia | 2048 | 0.0429 | 0.0520 | 400.76 |
| gpu_naive | propia | 4096 | 0.3948 | 0.4372 | 348.10 |
| gpu_tiled | propia | 1024 | 0.0013 | 0.0037 | 1649.43 |
| gpu_tiled | propia | 2048 | 0.0084 | 0.0188 | 2042.84 |
| gpu_tiled | propia | 4096 | 0.0700 | 0.1140 | 1964.18 |
| gpu_tiled_pad | propia | 1024 | 0.0013 | 0.0037 | 1652.85 |
| gpu_tiled_pad | propia | 2048 | 0.0085 | 0.0189 | 2014.71 |
| gpu_tiled_pad | propia | 4096 | 0.0720 | 0.1141 | 1907.87 |
| gpu_cublas | referencia | 1024 | 0.0004 | 0.0028 | 6047.48 |
| gpu_cublas | referencia | 2048 | 0.0020 | 0.0123 | 8740.69 |
| gpu_cublas | referencia | 4096 | 0.0155 | 0.0574 | 8850.38 |
