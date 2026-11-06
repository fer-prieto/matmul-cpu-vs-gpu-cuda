//! Tipo [`Matrix`]: matriz cuadrada N×N de `f32` en orden por filas
//! (row-major, el elemento (i, j) está en `data[i*N + j]`), igual que en C++ y
//! en los futuros kernels CUDA.
//!
//! Los datos se guardan alineados a 64 bytes (una línea de caché), igual que en
//! el binario C++ (`std::aligned_alloc(64, ...)`). Así las dos herramientas
//! miden con la misma disposición de memoria y se elimina una fuente de
//! variabilidad: la alineación que devuelve `malloc` puede cambiar entre
//! ejecuciones y, en CPUs como Sandy Bridge, los accesos SIMD que cruzan
//! líneas de caché son más lentos.

use crate::rng::SplitMix64;
use std::fmt;
use std::ops::{Index, IndexMut};

/// Alineación de los datos en bytes.
pub const ALIGNMENT: usize = 64;
const FLOATS_PER_BLOCK: usize = ALIGNMENT / std::mem::size_of::<f32>();

/// Bloque de 64 bytes alineado a 64: la unidad de almacenamiento interna.
#[derive(Clone, Copy)]
#[repr(C, align(64))]
struct Block([f32; FLOATS_PER_BLOCK]);

// El bloque no tiene relleno: 16 floats contiguos que ocupan exactamente 64 bytes.
const _: () = assert!(std::mem::size_of::<Block>() == ALIGNMENT);
const _: () = assert!(std::mem::align_of::<Block>() == ALIGNMENT);

const ZERO_BLOCK: Block = Block([0.0; FLOATS_PER_BLOCK]);

/// Errores al construir una matriz a partir de datos externos.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum MatrixError {
    /// El número de elementos no es N·N.
    LengthMismatch { n: usize, expected: usize, got: usize },
    /// Alguna fila no tiene N elementos.
    NotSquare { rows: usize, row: usize, len: usize },
}

impl fmt::Display for MatrixError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            MatrixError::LengthMismatch { n, expected, got } => write!(
                f,
                "una matriz de {n}x{n} necesita {expected} elementos, se recibieron {got}"
            ),
            MatrixError::NotSquare { rows, row, len } => write!(
                f,
                "matriz no cuadrada: {rows} filas, pero la fila {row} tiene {len} elementos"
            ),
        }
    }
}

impl std::error::Error for MatrixError {}

/// Matriz cuadrada N×N de `f32`, row-major, alineada a 64 bytes.
///
/// Invariante: `blocks` contiene al menos `n·n` floats; los sobrantes del
/// último bloque valen 0 y nunca se exponen.
#[derive(Clone)]
pub struct Matrix {
    n: usize,
    len: usize,
    blocks: Vec<Block>,
}

impl Matrix {
    /// Matriz de ceros. Escribe toda la memoria, así que sus páginas quedan
    /// reservadas desde ya (no habrá fallos de página dentro de la medición).
    pub fn zeros(n: usize) -> Self {
        let len = n
            .checked_mul(n)
            .filter(|len| len.checked_mul(std::mem::size_of::<f32>()).is_some())
            .unwrap_or_else(|| panic!("matriz de {n}x{n} demasiado grande"));
        let nblocks = len.div_ceil(FLOATS_PER_BLOCK);
        Self {
            n,
            len,
            blocks: vec![ZERO_BLOCK; nblocks],
        }
    }

    /// Matriz con todos los elementos iguales a `value`.
    pub fn filled(n: usize, value: f32) -> Self {
        let mut m = Self::zeros(n);
        m.fill(value);
        m
    }

    /// Matriz identidad.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m[(i, i)] = 1.0;
        }
        m
    }

    /// Matriz con valores pseudoaleatorios en [-1, 1) generados con
    /// SplitMix64 (`seed`, `stream`). Por convención A usa
    /// [`crate::rng::STREAM_A`] y B [`crate::rng::STREAM_B`].
    pub fn random(n: usize, seed: u64, stream: u64) -> Self {
        let mut m = Self::zeros(n);
        let mut rng = SplitMix64::new(seed, stream);
        for x in m.as_mut_slice() {
            *x = rng.next_f32_symmetric();
        }
        m
    }

    /// Construye una matriz a partir de `n·n` valores en orden por filas.
    pub fn from_vec(n: usize, data: Vec<f32>) -> Result<Self, MatrixError> {
        let expected = n * n;
        if data.len() != expected {
            return Err(MatrixError::LengthMismatch {
                n,
                expected,
                got: data.len(),
            });
        }
        let mut m = Self::zeros(n);
        m.as_mut_slice().copy_from_slice(&data);
        Ok(m)
    }

    /// Construye una matriz a partir de sus filas (útil en tests).
    pub fn from_rows(rows: &[&[f32]]) -> Result<Self, MatrixError> {
        let n = rows.len();
        let mut m = Self::zeros(n);
        for (i, row) in rows.iter().enumerate() {
            if row.len() != n {
                return Err(MatrixError::NotSquare {
                    rows: n,
                    row: i,
                    len: row.len(),
                });
            }
            m.row_mut(i).copy_from_slice(row);
        }
        Ok(m)
    }

    /// Dimensión N.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Número de elementos (N·N).
    pub fn len(&self) -> usize {
        self.len
    }

    /// `true` si N = 0.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    /// Bytes que ocupan los datos (N·N·4).
    pub fn bytes(&self) -> usize {
        self.len * std::mem::size_of::<f32>()
    }

    /// Elementos en orden por filas.
    pub fn as_slice(&self) -> &[f32] {
        // SAFETY: `Block` es `repr(C)` con un único campo `[f32; 16]` y sin
        // relleno (comprobado arriba), así que el Vec es una secuencia contigua
        // de floats de al menos `len` elementos. Si `len == 0` el puntero es
        // el colgante (no nulo y alineado) del Vec vacío, válido para longitud 0.
        unsafe { std::slice::from_raw_parts(self.blocks.as_ptr().cast::<f32>(), self.len) }
    }

    /// Elementos en orden por filas (mutable).
    pub fn as_mut_slice(&mut self) -> &mut [f32] {
        // SAFETY: igual que en `as_slice`; `&mut self` garantiza exclusividad.
        unsafe { std::slice::from_raw_parts_mut(self.blocks.as_mut_ptr().cast::<f32>(), self.len) }
    }

    /// Puntero al primer elemento (para FFI).
    pub fn as_ptr(&self) -> *const f32 {
        self.blocks.as_ptr().cast::<f32>()
    }

    /// Puntero mutable al primer elemento (para FFI).
    pub fn as_mut_ptr(&mut self) -> *mut f32 {
        self.blocks.as_mut_ptr().cast::<f32>()
    }

    /// Fila `i`.
    pub fn row(&self, i: usize) -> &[f32] {
        &self.as_slice()[i * self.n..(i + 1) * self.n]
    }

    /// Fila `i` (mutable).
    pub fn row_mut(&mut self, i: usize) -> &mut [f32] {
        let n = self.n;
        &mut self.as_mut_slice()[i * n..(i + 1) * n]
    }

    /// Elemento (i, j).
    pub fn get(&self, i: usize, j: usize) -> f32 {
        self[(i, j)]
    }

    /// Asigna el elemento (i, j).
    pub fn set(&mut self, i: usize, j: usize, value: f32) {
        self[(i, j)] = value;
    }

    /// Asigna `value` a todos los elementos.
    pub fn fill(&mut self, value: f32) {
        self.as_mut_slice().fill(value);
    }

    /// Suma de todos los elementos acumulada en `f64`, en orden. Es el mismo
    /// cálculo que `matmul_util_checksum` en C++ (mismos bits).
    pub fn checksum(&self) -> f64 {
        let mut sum = 0.0f64;
        for &x in self.as_slice() {
            sum += x as f64;
        }
        sum
    }

    /// Norma de Frobenius calculada en `f64`.
    pub fn frobenius_norm(&self) -> f64 {
        self.as_slice()
            .iter()
            .map(|&x| (x as f64) * (x as f64))
            .sum::<f64>()
            .sqrt()
    }

    /// Desplazamiento de la dirección de los datos respecto a 64 bytes (0 si
    /// está alineada; siempre debería ser 0).
    pub fn alignment_offset(&self) -> usize {
        self.as_ptr() as usize % ALIGNMENT
    }
}

impl Index<(usize, usize)> for Matrix {
    type Output = f32;
    fn index(&self, (i, j): (usize, usize)) -> &f32 {
        assert!(
            i < self.n && j < self.n,
            "índice ({i}, {j}) fuera de una matriz {0}x{0}",
            self.n
        );
        &self.as_slice()[i * self.n + j]
    }
}

impl IndexMut<(usize, usize)> for Matrix {
    fn index_mut(&mut self, (i, j): (usize, usize)) -> &mut f32 {
        assert!(
            i < self.n && j < self.n,
            "índice ({i}, {j}) fuera de una matriz {0}x{0}",
            self.n
        );
        let n = self.n;
        &mut self.as_mut_slice()[i * n + j]
    }
}

impl PartialEq for Matrix {
    /// Igualdad exacta elemento a elemento (NaN ≠ NaN, como en IEEE 754).
    fn eq(&self, other: &Self) -> bool {
        self.n == other.n && self.as_slice() == other.as_slice()
    }
}

impl fmt::Debug for Matrix {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        const SHOW: usize = 4;
        write!(f, "Matrix {}x{} [", self.n, self.n)?;
        for i in 0..self.n.min(SHOW) {
            let row = self.row(i);
            let shown: Vec<String> = row.iter().take(SHOW).map(|v| format!("{v:.4}")).collect();
            write!(f, "[{}{}]", shown.join(", "), if self.n > SHOW { ", ..." } else { "" })?;
        }
        if self.n > SHOW {
            write!(f, ", ...")?;
        }
        write!(f, "]")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn alineada_a_64_bytes() {
        for n in [1, 3, 16, 100, 257] {
            assert_eq!(Matrix::zeros(n).alignment_offset(), 0, "n = {n}");
        }
    }

    #[test]
    fn indexado_row_major() {
        let m = Matrix::from_rows(&[&[1.0, 2.0], &[3.0, 4.0]]).unwrap();
        assert_eq!(m.as_slice(), &[1.0, 2.0, 3.0, 4.0]);
        assert_eq!(m[(1, 0)], 3.0);
        assert_eq!(m.row(1), &[3.0, 4.0]);
    }

    #[test]
    fn identidad_y_checksum() {
        let id = Matrix::identity(5);
        assert_eq!(id.checksum(), 5.0);
        assert_eq!(id.get(2, 2), 1.0);
        assert_eq!(id.get(2, 3), 0.0);
    }

    #[test]
    fn errores_de_construccion() {
        assert!(Matrix::from_vec(2, vec![1.0; 3]).is_err());
        assert!(Matrix::from_rows(&[&[1.0, 2.0], &[3.0]]).is_err());
    }

    #[test]
    fn matriz_vacia() {
        let m = Matrix::zeros(0);
        assert!(m.is_empty());
        assert_eq!(m.as_slice().len(), 0);
        assert_eq!(m.checksum(), 0.0);
    }

    #[test]
    fn aleatoria_reproducible() {
        let a = Matrix::random(33, 7, 0);
        let b = Matrix::random(33, 7, 0);
        let c = Matrix::random(33, 8, 0);
        assert_eq!(a, b);
        assert_ne!(a, c);
    }
}
