//! Registro de implementaciones que el arnés sabe medir.
//!
//! Cada entrada asocia un nombre estable (el que aparece en los CSV) con su
//! función cronometrada y sus metadatos: si es código **propio** (evaluable) o
//! de **referencia** (biblioteca externa), si es **lenta** (se omite en N
//! grande salvo `--include-slow`) y si es multihilo.
//!
//! Para añadir una implementación (p. ej. `gpu_naive`, `gpu_tiled` o `cublas`
//! en las Unidades 2 y 3) basta con declarar su función en `ffi.rs` y añadir
//! una entrada en [`available`].

use crate::ffi::{self, FfiError};
use crate::matrix::Matrix;
use serde::Serialize;

pub const CPU_IJK: &str = "cpu_ijk";
pub const CPU_IKJ: &str = "cpu_ikj";
pub const CPU_OPENBLAS_1T: &str = "cpu_openblas_1t";
pub const CPU_OPENBLAS_MT: &str = "cpu_openblas_mt";

/// Origen del código de una implementación.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Kind {
    /// Escrita desde cero en este proyecto: es lo que se evalúa.
    Propia,
    /// Biblioteca externa usada solo como comparación (techo de rendimiento).
    Referencia,
}

impl Kind {
    pub fn as_str(self) -> &'static str {
        match self {
            Kind::Propia => "propia",
            Kind::Referencia => "referencia",
        }
    }
}

/// Modelo de ejecución.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize)]
#[serde(rename_all = "lowercase")]
pub enum Threading {
    /// Un solo hilo (todas las implementaciones propias de la Unidad 1).
    Single,
    /// Varios hilos (solo informativo en la Unidad 1: OpenBLAS multihilo).
    Multi,
}

/// Ejecuta la implementación una vez y devuelve el tiempo de cómputo en
/// segundos, medido dentro de C++ con `std::chrono::steady_clock`. El último
/// argumento es el número de hilos (lo ignoran las de un solo hilo).
pub type TimedFn = fn(&Matrix, &Matrix, &mut Matrix, usize) -> Result<f64, FfiError>;

/// Una implementación medible.
#[derive(Debug, Clone, Copy)]
pub struct Implementation {
    pub name: &'static str,
    pub description: &'static str,
    pub kind: Kind,
    /// Demasiado lenta para N grande: se omite si N > `--slow-max-n`
    /// (2048 por defecto) salvo con `--include-slow`.
    pub slow: bool,
    pub threading: Threading,
    pub run_timed: TimedFn,
}

impl Implementation {
    /// Hilos que usará: 1 si es secuencial, `mt_threads` si es multihilo.
    pub fn threads(&self, mt_threads: usize) -> usize {
        match self.threading {
            Threading::Single => 1,
            Threading::Multi => mt_threads.max(1),
        }
    }

    /// `true` si debe ejecutarse para este N con la configuración dada.
    pub fn runs_at(&self, n: usize, include_slow: bool, slow_max_n: usize) -> bool {
        !self.slow || include_slow || n <= slow_max_n
    }
}

fn run_cpu_ijk(a: &Matrix, b: &Matrix, c: &mut Matrix, _threads: usize) -> Result<f64, FfiError> {
    ffi::matmul_ijk_timed(a, b, c)
}

fn run_cpu_ikj(a: &Matrix, b: &Matrix, c: &mut Matrix, _threads: usize) -> Result<f64, FfiError> {
    ffi::matmul_ikj_timed(a, b, c)
}

/// Implementaciones compiladas en este binario, en el orden en que se miden.
pub fn available() -> Vec<Implementation> {
    let list = vec![
        Implementation {
            name: CPU_IJK,
            description: "Propia: orden i-j-k (ingenua). Referencia de correctitud y de rendimiento.",
            kind: Kind::Propia,
            slow: true,
            threading: Threading::Single,
            run_timed: run_cpu_ijk,
        },
        Implementation {
            name: CPU_IKJ,
            description: "Propia: orden i-k-j (bucle interno contiguo y vectorizable). Línea base oficial.",
            kind: Kind::Propia,
            slow: false,
            threading: Threading::Single,
            run_timed: run_cpu_ikj,
        },
    ];
    list
}

/// Busca una implementación disponible por nombre.
pub fn find(name: &str) -> Option<Implementation> {
    available().into_iter().find(|imp| imp.name == name)
}

/// Referencia de correctitud preferida: OpenBLAS (1 hilo) si está compilado,
/// si no `cpu_ijk`.
pub fn preferred_reference() -> &'static str {
    if cfg!(feature = "openblas") {
        CPU_OPENBLAS_1T
    } else {
        CPU_IJK
    }
}

/// Implementaciones conocidas que NO están compiladas en este binario, con
/// la forma de activarlas (para los mensajes de error).
pub fn not_compiled() -> Vec<(&'static str, &'static str)> {
    if cfg!(feature = "openblas") {
        Vec::new()
    } else {
        vec![
            (CPU_OPENBLAS_1T, "make bench-ref (feature `openblas`)"),
            (CPU_OPENBLAS_MT, "make bench-ref (feature `openblas`)"),
        ]
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nombres_unicos_y_orden_estable() {
        let names: Vec<&str> = available().iter().map(|i| i.name).collect();
        assert_eq!(&names[..2], &[CPU_IJK, CPU_IKJ]);
        let mut sorted = names.clone();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), names.len());
    }

    #[test]
    fn la_referencia_esta_disponible() {
        assert!(find(preferred_reference()).is_some());
    }

    #[test]
    fn regla_de_implementaciones_lentas() {
        let ijk = find(CPU_IJK).unwrap();
        assert!(ijk.runs_at(2048, false, 2048));
        assert!(!ijk.runs_at(4096, false, 2048));
        assert!(ijk.runs_at(4096, true, 2048));
        assert!(find(CPU_IKJ).unwrap().runs_at(4096, false, 2048));
    }
}
