//! Validación de correctitud.
//!
//! Dos comprobaciones complementarias, ambas calculadas en `f64`:
//!
//! 1. **Comparación con una referencia** (OpenBLAS si está compilado, si no
//!    `cpu_ijk`): error relativo de Frobenius ‖C − C_ref‖_F / ‖C_ref‖_F y error
//!    absoluto máximo max |C − C_ref|.
//! 2. **Verificación de Freivalds** (1977): con vectores aleatorios x se
//!    compara C·x con A·(B·x). Cuesta O(N²) en lugar de O(N³), así que sirve
//!    para cualquier N, incluso N = 4096 cuando `cpu_ijk` es demasiado lento y
//!    no hay OpenBLAS. Se informa el peor error relativo
//!    ‖C·x − A·(B·x)‖₂ / ‖A·(B·x)‖₂ entre los vectores usados.
//!
//! Nunca se compara con `==`: al cambiar el orden de las sumas en FP32 el
//! resultado cambia ligeramente. Tolerancia por defecto: 1e-4 (ver
//! [`DEFAULT_TOLERANCE`]).

use crate::matrix::Matrix;
use crate::rng::{SplitMix64, STREAM_FREIVALDS};

/// Tolerancia por defecto para el error relativo.
///
/// Justificación: el redondeo unitario de FP32 es u = 2⁻²⁴ ≈ 6·10⁻⁸. Al sumar
/// N productos con signos aleatorios, el error relativo típico de cada
/// elemento crece como ~√N·u (≈ 4·10⁻⁶ para N = 4096), y dos implementaciones
/// que suman en órdenes distintos difieren en ese orden de magnitud. 1e-4 deja
/// más de un orden de margen y sigue detectando cualquier error real (un
/// índice equivocado da errores relativos de orden 1).
pub const DEFAULT_TOLERANCE: f64 = 1e-4;

/// Resultado de comparar una matriz con la de referencia.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Comparison {
    /// ‖C − C_ref‖_F / ‖C_ref‖_F (0 si ambas son nulas; ∞ si solo C_ref lo es).
    pub rel_error: f64,
    /// max |C − C_ref| elemento a elemento.
    pub max_abs_error: f64,
    /// ‖C‖_F.
    pub norm: f64,
    /// ‖C_ref‖_F.
    pub norm_ref: f64,
}

impl Comparison {
    /// `true` si el error relativo está dentro de la tolerancia (falso con NaN).
    pub fn passes(&self, tolerance: f64) -> bool {
        self.rel_error <= tolerance
    }
}

/// Compara `c` con `c_ref` (deben tener el mismo tamaño).
pub fn compare(c: &Matrix, c_ref: &Matrix) -> Comparison {
    assert_eq!(c.n(), c_ref.n(), "las matrices a comparar deben tener el mismo tamaño");
    let mut diff2 = 0.0f64;
    let mut c2 = 0.0f64;
    let mut ref2 = 0.0f64;
    let mut max_abs = 0.0f64;
    for (&x, &y) in c.as_slice().iter().zip(c_ref.as_slice()) {
        let (x, y) = (x as f64, y as f64);
        let d = x - y;
        diff2 += d * d;
        c2 += x * x;
        ref2 += y * y;
        max_abs = max_propagating_nan(max_abs, d.abs());
    }
    Comparison {
        rel_error: relative(diff2.sqrt(), ref2.sqrt()),
        max_abs_error: max_abs,
        norm: c2.sqrt(),
        norm_ref: ref2.sqrt(),
    }
}

/// Error relativo `num / den` con los casos límite definidos.
fn relative(num: f64, den: f64) -> f64 {
    if den > 0.0 {
        num / den
    } else if num == 0.0 {
        0.0
    } else {
        // den = 0 con num ≠ 0 (o NaN): error infinito/indefinido.
        if num.is_nan() {
            f64::NAN
        } else {
            f64::INFINITY
        }
    }
}

/// y = M·x en `f64`.
fn matvec(m: &Matrix, x: &[f64], y: &mut [f64]) {
    let n = m.n();
    for (i, yi) in y.iter_mut().enumerate().take(n) {
        *yi = m.row(i).iter().zip(x).map(|(&mij, &xj)| mij as f64 * xj).sum();
    }
}

/// Verificación de Freivalds: peor error relativo ‖C·x − A·(B·x)‖₂ / ‖A·(B·x)‖₂
/// sobre `vectors` vectores aleatorios con componentes en [-1, 1).
/// Devuelve 0 si `vectors == 0` o N = 0.
pub fn freivalds(a: &Matrix, b: &Matrix, c: &Matrix, vectors: usize, seed: u64) -> f64 {
    let n = a.n();
    assert!(b.n() == n && c.n() == n, "A, B y C deben tener el mismo tamaño");
    if n == 0 {
        return 0.0;
    }
    let mut rng = SplitMix64::new(seed, STREAM_FREIVALDS);
    let mut x = vec![0.0f64; n];
    let mut bx = vec![0.0f64; n];
    let mut abx = vec![0.0f64; n];
    let mut cx = vec![0.0f64; n];
    let mut worst = 0.0f64;
    for _ in 0..vectors {
        for xi in x.iter_mut() {
            *xi = rng.next_f32_symmetric() as f64;
        }
        matvec(b, &x, &mut bx);
        matvec(a, &bx, &mut abx);
        matvec(c, &x, &mut cx);
        let num = cx.iter().zip(&abx).map(|(p, q)| (p - q) * (p - q)).sum::<f64>().sqrt();
        let den = abx.iter().map(|q| q * q).sum::<f64>().sqrt();
        worst = max_propagating_nan(worst, relative(num, den));
    }
    worst
}

/// Máximo que propaga NaN: si cualquiera de los dos es NaN, el resultado es
/// NaN (`f64::max` lo ignoraría y ocultaría un resultado inválido).
fn max_propagating_nan(acc: f64, x: f64) -> f64 {
    if acc.is_nan() || x.is_nan() {
        f64::NAN
    } else {
        acc.max(x)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn matrices_identicas_dan_error_cero() {
        let a = Matrix::random(17, 1, 0);
        let cmp = compare(&a, &a);
        assert_eq!(cmp.rel_error, 0.0);
        assert_eq!(cmp.max_abs_error, 0.0);
        assert!(cmp.passes(0.0));
    }

    #[test]
    fn error_conocido() {
        let r = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.0]]).unwrap(); // ‖r‖ = 5
        let c = Matrix::from_rows(&[&[3.0, 0.0], &[0.0, 4.5]]).unwrap(); // diferencia 0.5
        let cmp = compare(&c, &r);
        assert!((cmp.rel_error - 0.1).abs() < 1e-12);
        assert!((cmp.max_abs_error - 0.5).abs() < 1e-12);
        assert!(!cmp.passes(1e-4));
    }

    #[test]
    fn nan_nunca_pasa() {
        let r = Matrix::filled(3, 1.0);
        let mut c = Matrix::filled(3, 1.0);
        c.set(1, 1, f32::NAN);
        c.set(2, 2, 5.0); // un error mayor después del NaN no debe ocultarlo
        let cmp = compare(&c, &r);
        assert!(cmp.max_abs_error.is_nan());
        assert!(!cmp.passes(1.0));
    }

    #[test]
    fn referencia_nula() {
        let z = Matrix::zeros(4);
        assert_eq!(compare(&z, &z).rel_error, 0.0);
        assert_eq!(compare(&Matrix::filled(4, 1.0), &z).rel_error, f64::INFINITY);
    }

    #[test]
    fn freivalds_detecta_errores() {
        let n = 32;
        let a = Matrix::random(n, 3, 0);
        let b = Matrix::random(n, 3, 1);
        // C exacta (en f64 y redondeada a f32).
        let mut c = Matrix::zeros(n);
        for i in 0..n {
            for j in 0..n {
                let s: f64 = (0..n).map(|k| a.get(i, k) as f64 * b.get(k, j) as f64).sum();
                c.set(i, j, s as f32);
            }
        }
        assert!(freivalds(&a, &b, &c, 3, 42) < 1e-6);
        // Un solo elemento equivocado se detecta.
        let mut wrong = c.clone();
        wrong.set(5, 7, wrong.get(5, 7) + 1.0);
        assert!(freivalds(&a, &b, &wrong, 3, 42) > 1e-3);
    }
}
