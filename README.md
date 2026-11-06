# Multiplicación de matrices de gran escala: CPU vs GPU con CUDA

Proyecto del curso **Arquitecturas de Alto Rendimiento (IF557ZIN)**.
Autor: David Fernando Prieto Cardoso.

Objetivo: implementar y comparar `C = A × B` (FP32, N = 1024, 2048 y 4096) en CPU
secuencial, GPU naive y GPU con memoria compartida (tiling).

Estado: **Unidad 1, línea base en CPU** (en construcción).

## Requisitos

- Linux, `g++` con C++17 y GNU `make`.
- Opcional: Rust (`cargo`) para el arnés de benchmark `harness/`.
