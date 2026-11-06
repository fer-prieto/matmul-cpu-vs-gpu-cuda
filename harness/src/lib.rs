//! # matbench: arnés de benchmark de `C = A × B`
//!
//! Proyecto: Multiplicación de Matrices de Gran Escala, CPU vs GPU (IF557ZIN).
//! Unidad 1: línea base secuencial en CPU.
//!
//! El cómputo que se evalúa está escrito en C++ (`src/cpu/matmul_cpu.cpp`) y se
//! enlaza como biblioteca estática compilada por el Makefile. Rust solo
//! **orquesta**: reserva e inicializa las matrices, ejecuta warm-up y
//! repeticiones, valida los resultados, recoge el entorno de prueba y la
//! telemetría (frecuencia, temperatura, throttling) y exporta TODO a
//! `results/` en CSV y JSON.
//!
//! Módulos:
//! - [`ffi`]: declaraciones `extern "C"` y envoltorios seguros.
//! - [`matrix`]: tipo [`matrix::Matrix`] (N×N, row-major, alineado a 64 B).
//! - [`rng`]: generador SplitMix64, idéntico al de C++.
//! - [`registry`]: implementaciones disponibles (propias y de referencia).
//! - [`bench`]: bucle de medición y estadísticas.
//! - [`validate`]: comparación con la referencia y verificación de Freivalds.
//! - [`sysinfo`]: entorno de prueba (CPU, cachés, SO, compilación, git...).
//! - [`telemetry`]: sensores y muestreo periódico durante la corrida.
//! - [`report`]: escritura de resultados, tablas y consolidación.

pub mod bench;
pub mod ffi;
pub mod matrix;
pub mod registry;
pub mod report;
pub mod rng;
pub mod sysinfo;
pub mod telemetry;
pub mod validate;
