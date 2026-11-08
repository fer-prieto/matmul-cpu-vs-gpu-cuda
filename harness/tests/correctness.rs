//! Tests de correctitud de la biblioteca C++ a través de la FFI.
//!
//! Cubren la tabla de la Fase 5 del plan (caso 2×2 conocido, identidad,
//! matriz cero, ikj vs ijk aleatorio con N no potencia de 2, C con basura,
//! tamaños incorrectos y, con la feature `openblas`, ikj vs OpenBLAS), más
//! comprobaciones extra: referencia independiente en f64, caso 3×3 no
//! simétrico, A × A (A y B aliadas), N = 0 y N = 1, variantes cronometradas,
//! verificación de Freivalds y equivalencia exacta del generador y del
//! checksum entre C++ y Rust.
//!
//! Ejecutar con `make test` (o `make test-ref` para incluir OpenBLAS).

use matbench::ffi::{self, FfiError};
use matbench::matrix::Matrix;
use matbench::registry;
use matbench::rng::{STREAM_A, STREAM_B};
use matbench::validate;

/// Tolerancia de los tests (más estricta que la del benchmark, 1e-4).
const TEST_TOLERANCE: f64 = 1e-5;

type MulFn = fn(&Matrix, &Matrix, &mut Matrix) -> Result<(), FfiError>;

fn ijk_timed(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<(), FfiError> {
    let t = ffi::matmul_ijk_timed(a, b, c)?;
    assert!(t.is_finite() && t >= 0.0, "tiempo inválido: {t}");
    Ok(())
}

fn ikj_timed(a: &Matrix, b: &Matrix, c: &mut Matrix) -> Result<(), FfiError> {
    let t = ffi::matmul_ikj_timed(a, b, c)?;
    assert!(t.is_finite() && t >= 0.0, "tiempo inválido: {t}");
    Ok(())
}

/// Las cuatro funciones propias: normales y cronometradas.
const OWN: [(&str, MulFn); 4] = [
    ("ijk", ffi::matmul_ijk),
    ("ikj", ffi::matmul_ikj),
    ("ijk_timed", ijk_timed),
    ("ikj_timed", ikj_timed),
];

fn multiply(f: MulFn, a: &Matrix, b: &Matrix) -> Matrix {
    let mut c = Matrix::zeros(a.n());
    f(a, b, &mut c).expect("tamaños válidos");
    c
}

/// Producto de referencia independiente: triple bucle en f64, redondeado a f32.
fn reference_f64(a: &Matrix, b: &Matrix) -> Matrix {
    let n = a.n();
    let mut c = Matrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            let s: f64 = (0..n).map(|k| a.get(i, k) as f64 * b.get(k, j) as f64).sum();
            c.set(i, j, s as f32);
        }
    }
    c
}

#[test]
fn caso_2x2_conocido() {
    let a = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
    let b = Matrix::from_rows(&[&[5.0, 6.0], &[7.0, 8.0]]).unwrap();
    // Calculado a mano: [[1·5+2·7, 1·6+2·8], [3·5+4·7, 3·6+4·8]].
    let expected = Matrix::from_rows(&[&[19.0, 22.0], &[43.0, 50.0]]).unwrap();
    for (name, f) in OWN {
        assert_eq!(multiply(f, &a, &b), expected, "{name}");
    }
}

#[test]
fn caso_3x3_no_simetrico() {
    // Detecta operandos intercambiados (B·A) o índices transpuestos.
    let a = Matrix::from_rows(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0], &[7.0, 8.0, 9.0]]).unwrap();
    let b = Matrix::from_rows(&[&[9.0, 8.0, 7.0], &[6.0, 5.0, 4.0], &[3.0, 2.0, 1.0]]).unwrap();
    let expected = Matrix::from_rows(&[&[30.0, 24.0, 18.0], &[84.0, 69.0, 54.0], &[138.0, 114.0, 90.0]]).unwrap();
    for (name, f) in OWN {
        assert_eq!(multiply(f, &a, &b), expected, "{name}");
    }
}

#[test]
fn identidad_es_exacta() {
    for n in [1, 7, 64, 100] {
        let a = Matrix::random(n, 11, STREAM_A);
        let id = Matrix::identity(n);
        for (name, f) in OWN {
            assert_eq!(multiply(f, &a, &id), a, "{name}: A x I, n = {n}");
            assert_eq!(multiply(f, &id, &a), a, "{name}: I x A, n = {n}");
        }
    }
}

#[test]
fn matriz_cero() {
    for n in [1, 33, 128] {
        let a = Matrix::random(n, 5, STREAM_A);
        let zero = Matrix::zeros(n);
        for (name, f) in OWN {
            let c = multiply(f, &a, &zero);
            assert!(c.as_slice().iter().all(|&x| x == 0.0), "{name}: A x 0, n = {n}");
        }
    }
}

#[test]
fn ikj_vs_ijk_aleatorio() {
    // Incluye N = 300, que no es potencia de 2 (ni múltiplo del ancho SIMD
    // en todas las filas: ejercita los bucles de resto del vectorizador).
    for n in [64, 128, 256, 300] {
        let a = Matrix::random(n, 42, STREAM_A);
        let b = Matrix::random(n, 42, STREAM_B);
        let c_ijk = multiply(ffi::matmul_ijk, &a, &b);
        let c_ikj = multiply(ffi::matmul_ikj, &a, &b);
        let cmp = validate::compare(&c_ikj, &c_ijk);
        assert!(
            cmp.rel_error < TEST_TOLERANCE,
            "n = {n}: error relativo {:e} (máx. abs. {:e})",
            cmp.rel_error,
            cmp.max_abs_error
        );
    }
}

/// Solo con la feature `openblas` (make test-ref): ikj debe coincidir con
/// cblas_sgemm en los mismos tamaños, incluido uno que no es potencia de 2.
#[test]
#[cfg(feature = "openblas")]
fn ikj_vs_openblas() {
    for n in [64, 128, 256, 300] {
        let a = Matrix::random(n, 42, STREAM_A);
        let b = Matrix::random(n, 42, STREAM_B);
        let mut c_blas = Matrix::zeros(n);
        ffi::openblas_1t_timed(&a, &b, &mut c_blas).expect("tamaños válidos");
        let c_ikj = multiply(ffi::matmul_ikj, &a, &b);
        let cmp = validate::compare(&c_ikj, &c_blas);
        assert!(
            cmp.rel_error < TEST_TOLERANCE,
            "n = {n}: error relativo {:e} (máx. abs. {:e})",
            cmp.rel_error,
            cmp.max_abs_error
        );
    }
}

#[test]
fn propias_vs_referencia_f64_independiente() {
    for n in [17, 50, 96] {
        let a = Matrix::random(n, 9, STREAM_A);
        let b = Matrix::random(n, 9, STREAM_B);
        let r = reference_f64(&a, &b);
        for (name, f) in OWN {
            let cmp = validate::compare(&multiply(f, &a, &b), &r);
            assert!(cmp.rel_error < TEST_TOLERANCE, "{name}, n = {n}: {:e}", cmp.rel_error);
        }
    }
}

#[test]
fn c_con_basura_no_afecta() {
    let n = 65;
    let a = Matrix::random(n, 3, STREAM_A);
    let b = Matrix::random(n, 3, STREAM_B);
    for (name, f) in OWN {
        let clean = multiply(f, &a, &b);
        let mut dirty = Matrix::zeros(n);
        for (k, x) in dirty.as_mut_slice().iter_mut().enumerate() {
            *x = match k % 3 {
                0 => f32::NAN,
                1 => 1e30,
                _ => -7.5,
            };
        }
        f(&a, &b, &mut dirty).unwrap();
        assert_eq!(dirty, clean, "{name}: el contenido previo de C influyó en el resultado");
    }
}

#[test]
fn rechaza_tamanos_incorrectos() {
    let a = Matrix::random(4, 1, STREAM_A);
    let b5 = Matrix::random(5, 1, STREAM_B);
    let b4 = Matrix::random(4, 1, STREAM_B);
    for (name, f) in OWN {
        let mut c = Matrix::filled(4, 3.25);
        let before = c.clone();
        assert_eq!(
            f(&a, &b5, &mut c),
            Err(FfiError::SizeMismatch { a: 4, b: 5, c: 4 }),
            "{name}: B de otro tamaño"
        );
        assert_eq!(c, before, "{name}: C no debe modificarse si se rechaza la llamada");
        let mut c3 = Matrix::zeros(3);
        assert!(f(&a, &b4, &mut c3).is_err(), "{name}: C de otro tamaño");
    }
    let err = ffi::matmul_ikj(&a, &b5, &mut Matrix::zeros(4)).unwrap_err();
    assert!(err.to_string().contains("tamaños incompatibles"));
}

#[test]
fn a_por_a_con_operandos_aliados() {
    // A y B pueden ser la misma matriz (ambas son de solo lectura).
    let n = 40;
    let a = Matrix::random(n, 21, STREAM_A);
    let r = reference_f64(&a, &a);
    for (name, f) in OWN {
        let mut c = Matrix::zeros(n);
        f(&a, &a, &mut c).unwrap();
        assert!(validate::compare(&c, &r).rel_error < TEST_TOLERANCE, "{name}");
    }
}

#[test]
fn tamanos_minimos() {
    let empty = Matrix::zeros(0);
    let one_a = Matrix::from_rows(&[&[3.0]]).unwrap();
    let one_b = Matrix::from_rows(&[&[-2.5]]).unwrap();
    for (name, f) in OWN {
        let mut c0 = Matrix::zeros(0);
        f(&empty, &empty, &mut c0).unwrap_or_else(|e| panic!("{name}: N = 0 debe ser válido: {e}"));
        assert_eq!(multiply(f, &one_a, &one_b).get(0, 0), -7.5, "{name}: N = 1");
    }
}

#[test]
fn cronometradas_iguales_a_normales() {
    let n = 129;
    let a = Matrix::random(n, 8, STREAM_A);
    let b = Matrix::random(n, 8, STREAM_B);
    assert_eq!(multiply(ijk_timed, &a, &b), multiply(ffi::matmul_ijk, &a, &b));
    assert_eq!(multiply(ikj_timed, &a, &b), multiply(ffi::matmul_ikj, &a, &b));
}

#[test]
fn registro_ejecuta_implementaciones_correctas() {
    let n = 72;
    let a = Matrix::random(n, 13, STREAM_A);
    let b = Matrix::random(n, 13, STREAM_B);
    let r = reference_f64(&a, &b);
    for imp in registry::available() {
        let mut c = Matrix::zeros(n);
        let t = (imp.run_timed)(&a, &b, &mut c, imp.threads(2)).unwrap();
        assert!(t >= 0.0, "{}", imp.name);
        let cmp = validate::compare(&c, &r);
        assert!(cmp.rel_error < TEST_TOLERANCE, "{}: {:e}", imp.name, cmp.rel_error);
    }
}

#[test]
fn freivalds_acepta_ikj_y_detecta_un_error() {
    let n = 300;
    let a = Matrix::random(n, 42, STREAM_A);
    let b = Matrix::random(n, 42, STREAM_B);
    let mut c = multiply(ffi::matmul_ikj, &a, &b);
    assert!(validate::freivalds(&a, &b, &c, 3, 1) < TEST_TOLERANCE);
    let v = c.get(123, 45);
    c.set(123, 45, v + 0.5);
    assert!(validate::freivalds(&a, &b, &c, 3, 1) > 1e-4);
}

#[test]
fn generador_identico_a_cpp() {
    // Mismos bits en C++ (matmul_util_fill_random) y en Rust (rng.rs): el
    // binario cpu_baseline y el arnés multiplican las mismas matrices.
    for (n, seed, stream) in [(1, 0, 0), (17, 42, STREAM_A), (64, 42, STREAM_B), (257, 123_456_789, 7)] {
        let rust = Matrix::random(n, seed, stream);
        let mut cpp = Matrix::zeros(n);
        ffi::fill_random_cpp(&mut cpp, seed, stream);
        let same_bits = rust
            .as_slice()
            .iter()
            .zip(cpp.as_slice())
            .all(|(x, y)| x.to_bits() == y.to_bits());
        assert!(same_bits, "n = {n}, seed = {seed}, stream = {stream}");
    }
}

#[test]
fn checksum_identico_a_cpp() {
    for n in [0, 1, 100, 300] {
        let m = Matrix::random(n, 77, STREAM_A);
        assert_eq!(m.checksum().to_bits(), ffi::checksum_cpp(&m).to_bits(), "n = {n}");
    }
}

#[test]
fn checksum_coincide_con_cpu_baseline() {
    // Valor que imprime `./build/cpu_baseline 256` (semilla 42) para C = A·B:
    // si cambia, el binario C++ y el arnés dejaron de medir lo mismo.
    let a = Matrix::random(256, 42, STREAM_A);
    let b = Matrix::random(256, 42, STREAM_B);
    let c = multiply(ffi::matmul_ikj, &a, &b);
    assert_eq!(c.checksum(), ffi::checksum_cpp(&c));
    let c_ijk = multiply(ffi::matmul_ijk, &a, &b);
    // Con FMA (-march=native en una CPU con AVX2/FMA) los órdenes ikj/ijk
    // redondean distinto; se comparan con tolerancia, no bit a bit.
    let cmp = validate::compare(&c, &c_ijk);
    assert!(cmp.rel_error < TEST_TOLERANCE, "ikj vs ijk: {:e}", cmp.rel_error);
}

#[test]
fn info_de_compilacion_disponible() {
    let info = ffi::build_info();
    assert!(info.contains("compiler="), "{info}");
    assert!(info.contains("flags="), "{info}");
    assert!(
        info.contains("optimize=yes"),
        "la biblioteca debería compilarse con optimización: {info}"
    );
}
