//! Interfaz con la biblioteca C++ `build/libmatmul_cpu.a`.
//!
//! Contiene las declaraciones `extern "C"` de `include/matmul_cpu.h` (código
//! propio), `include/matmul_util.h` (utilidades) y, con la feature `openblas`,
//! `include/matmul_ref.h` (referencia), además de envoltorios seguros que
//! reciben [`Matrix`], comprueban que los tamaños coinciden y encapsulan todo
//! el `unsafe` del arnés.

use crate::matrix::Matrix;
use std::ffi::CStr;
use std::fmt;
use std::os::raw::c_char;

/// Declaraciones crudas (sin comprobaciones). Usar los envoltorios de abajo.
mod sys {
    use std::os::raw::c_char;

    extern "C" {
        // include/matmul_cpu.h: implementaciones propias (evaluables).
        pub fn matmul_cpu_ijk(a: *const f32, b: *const f32, c: *mut f32, n: u64);
        pub fn matmul_cpu_ikj(a: *const f32, b: *const f32, c: *mut f32, n: u64);
        pub fn matmul_cpu_ijk_timed(a: *const f32, b: *const f32, c: *mut f32, n: u64) -> f64;
        pub fn matmul_cpu_ikj_timed(a: *const f32, b: *const f32, c: *mut f32, n: u64) -> f64;

        // include/matmul_util.h: utilidades auxiliares.
        pub fn matmul_util_fill_random(m: *mut f32, count: u64, seed: u64, stream: u64);
        pub fn matmul_util_checksum(m: *const f32, count: u64) -> f64;
        pub fn matmul_util_build_info() -> *const c_char;

        // include/matmul_ref.h: implementaciones de REFERENCIA (no evaluables).
        // Solo existen en la biblioteca si se compiló con `make lib OPENBLAS=1`.
        #[cfg(feature = "openblas")]
        pub fn matmul_ref_openblas_1t_timed(a: *const f32, b: *const f32, c: *mut f32, n: u64) -> f64;
        #[cfg(feature = "openblas")]
        pub fn matmul_ref_openblas_mt_timed(a: *const f32, b: *const f32, c: *mut f32, n: u64, threads: u64) -> f64;
        #[cfg(feature = "openblas")]
        pub fn matmul_ref_openblas_info(
            config: *mut *const c_char,
            corename: *mut *const c_char,
            parallel_mode: *mut libc::c_int,
            num_procs: *mut libc::c_int,
        );
    }
}

/// Errores que detectan los envoltorios antes de llamar a C++.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FfiError {
    /// A, B y C no tienen la misma dimensión.
    SizeMismatch { a: usize, b: usize, c: usize },
    /// N supera el máximo que admite la función llamada.
    TooLarge { n: usize, max: usize },
}

impl fmt::Display for FfiError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            FfiError::SizeMismatch { a, b, c } => {
                write!(f, "tamaños incompatibles: A es {a}x{a}, B es {b}x{b} y C es {c}x{c}")
            }
            FfiError::TooLarge { n, max } => {
                write!(f, "N = {n} supera el máximo admitido ({max})")
            }
        }
    }
}

impl std::error::Error for FfiError {}

/// Comprueba que A, B y C son N×N con el mismo N y devuelve N.
fn check_sizes(a: &Matrix, b: &Matrix, c: &Matrix) -> Result<u64, FfiError> {
    if a.n() == b.n() && b.n() == c.n() {
        Ok(a.n() as u64)
    } else {
        Err(FfiError::SizeMismatch {
            a: a.n(),
            b: b.n(),
            c: c.n(),
        })
    }
}

/// Firma común de las funciones C de multiplicación cronometradas.
type RawTimedFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, u64) -> f64;
/// Firma común de las funciones C de multiplicación sin cronometrar.
type RawFn = unsafe extern "C" fn(*const f32, *const f32, *mut f32, u64);

fn call(f: RawFn, a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<(), FfiError> {
    let n = check_sizes(a, b, c)?;
    // SAFETY: las tres matrices tienen n·n elementos contiguos (invariante de
    // Matrix). `c` es un préstamo exclusivo, así que no se solapa con `a` ni con
    // `b`; `a` y `b` pueden ser la misma matriz, lo que el contrato de la API
    // permite porque ambas son de solo lectura.
    unsafe { f(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) };
    Ok(())
}

fn call_timed(f: RawTimedFn, a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<f64, FfiError> {
    let n = check_sizes(a, b, c)?;
    // SAFETY: ver `call`.
    Ok(unsafe { f(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n) })
}

/// C = A × B con el orden i-j-k (propia).
pub fn matmul_ijk(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<(), FfiError> {
    call(sys::matmul_cpu_ijk, a, b, c)
}

/// C = A × B con el orden i-k-j (propia, línea base).
pub fn matmul_ikj(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<(), FfiError> {
    call(sys::matmul_cpu_ikj, a, b, c)
}

/// Como [`matmul_ijk`], devolviendo el tiempo de cómputo medido en C++ (s).
pub fn matmul_ijk_timed(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<f64, FfiError> {
    call_timed(sys::matmul_cpu_ijk_timed, a, b, c)
}

/// Como [`matmul_ikj`], devolviendo el tiempo de cómputo medido en C++ (s).
pub fn matmul_ikj_timed(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<f64, FfiError> {
    call_timed(sys::matmul_cpu_ikj_timed, a, b, c)
}

/// Rellena `m` con el generador SplitMix64 implementado en C++.
pub fn fill_random_cpp(m: &mut Matrix, seed: u64, stream: u64) {
    let count = m.len() as u64;
    // SAFETY: `m` tiene `count` elementos contiguos y el préstamo es exclusivo.
    unsafe { sys::matmul_util_fill_random(m.as_mut_ptr(), count, seed, stream) };
}

/// Checksum (suma en double) calculado en C++.
pub fn checksum_cpp(m: &Matrix) -> f64 {
    // SAFETY: `m` tiene `m.len()` elementos contiguos.
    unsafe { sys::matmul_util_checksum(m.as_ptr(), m.len() as u64) }
}

/// Cadena con la información de compilación de la biblioteca C++.
pub fn build_info() -> String {
    // SAFETY: la función devuelve una cadena estática terminada en NUL.
    unsafe { c_str_to_string(sys::matmul_util_build_info()) }
}

// --- Referencia OpenBLAS (solo con la feature `openblas`) --------------------

/// C = A × B con `cblas_sgemm` de OpenBLAS usando exactamente 1 hilo.
///
/// Implementación de REFERENCIA (no evaluable): techo práctico de la CPU con
/// el que se compara el código propio y referencia de validación en N grande.
#[cfg(feature = "openblas")]
pub fn openblas_1t_timed(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<f64, FfiError> {
    call_timed(sys::matmul_ref_openblas_1t_timed, a, b, c)
}

/// C = A × B con `cblas_sgemm` de OpenBLAS multihilo (solo informativa: techo
/// absoluto de la CPU; NO comparable con el código secuencial).
#[cfg(feature = "openblas")]
pub fn openblas_mt_timed(a: &Matrix, b: &Matrix, c: &mut Matrix, threads: usize) -> Result<f64, FfiError> {
    let n = check_sizes(a, b, c)?;
    // SAFETY: ver `call`; el número de hilos es un u64 simple.
    Ok(unsafe { sys::matmul_ref_openblas_mt_timed(a.as_ptr(), b.as_ptr(), c.as_mut_ptr(), n, threads as u64) })
}

/// Información de la biblioteca OpenBLAS enlazada: (config, corename, modo de
/// paralelismo, unidades de ejecución). Para la sección "Entorno de prueba".
#[cfg(feature = "openblas")]
pub fn openblas_info() -> (String, String, i32, i32) {
    let mut config: *const c_char = std::ptr::null();
    let mut corename: *const c_char = std::ptr::null();
    let mut parallel_mode: i32 = 0;
    let mut num_procs: i32 = 0;
    // SAFETY: los cuatro punteros son locales válidos del tipo correcto y la
    // función solo los escribe; las cadenas que devuelve OpenBLAS son
    // estáticas y viven todo el programa (se copian a String de inmediato).
    unsafe { sys::matmul_ref_openblas_info(&mut config, &mut corename, &mut parallel_mode, &mut num_procs) };
    // SAFETY: OpenBLAS devuelve cadenas estáticas terminadas en NUL (nunca
    // nulas según su implementación, pero se tolera).
    let config = unsafe { c_str_to_string(config) };
    let corename = unsafe { c_str_to_string(corename) };
    (config, corename, parallel_mode, num_procs)
}

/// Convierte una cadena C en `String` (vacía si el puntero es nulo).
///
/// # Safety
/// `ptr` debe ser nulo o apuntar a una cadena terminada en NUL válida durante
/// la llamada.
unsafe fn c_str_to_string(ptr: *const c_char) -> String {
    if ptr.is_null() {
        String::new()
    } else {
        CStr::from_ptr(ptr).to_string_lossy().into_owned()
    }
}
