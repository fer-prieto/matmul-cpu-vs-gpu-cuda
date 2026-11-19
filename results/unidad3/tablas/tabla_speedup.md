| Implementación | N | speedup_kernel | speedup_total |
|---|---|---|---|
| gpu_naive | 1024 | 6.69 | 6.10 |
| gpu_naive | 2048 | 31.88 | 26.29 |
| gpu_naive | 4096 | 43.49 | 39.28 |
| gpu_tiled | 1024 | 123.91 | 43.20 |
| gpu_tiled | 2048 | 162.49 | 72.77 |
| gpu_tiled | 4096 | 245.41 | 150.58 |
| gpu_tiled_pad | 1024 | 124.16 | 43.69 |
| gpu_tiled_pad | 2048 | 160.25 | 72.38 |
| gpu_tiled_pad | 4096 | 238.38 | 150.57 |
| gpu_cublas | 1024 | 454.29 | 58.52 |
| gpu_cublas | 2048 | 695.23 | 110.69 |
| gpu_cublas | 4096 | 1105.81 | 299.33 |
