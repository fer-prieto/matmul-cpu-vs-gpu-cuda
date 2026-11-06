//! Generador pseudoaleatorio SplitMix64 (Steele, Lea y Flood, 2014).
//!
//! Es exactamente el mismo algoritmo que `matmul_util_fill_random` en
//! `src/cpu/matmul_util.cpp`. El test `generador_identico_a_cpp` de
//! `tests/correctness.rs` comprueba que ambos producen los mismos bits, de
//! modo que el binario C++ y el arnés multiplican las mismas matrices con la
//! misma semilla (y sus checksums son comparables).
//!
//! No se usa el crate `rand` precisamente para poder garantizar esa
//! equivalencia entre lenguajes.

const GOLDEN_GAMMA: u64 = 0x9E37_79B9_7F4A_7C15;
const MIX1: u64 = 0xBF58_476D_1CE4_E5B9;
const MIX2: u64 = 0x94D0_49BB_1331_11EB;
const STREAM_MIX: u64 = 0xD1B5_4A32_D192_ED03;

/// Secuencia usada para la matriz A.
pub const STREAM_A: u64 = 0;
/// Secuencia usada para la matriz B.
pub const STREAM_B: u64 = 1;
/// Secuencia para los vectores aleatorios de la verificación de Freivalds.
pub const STREAM_FREIVALDS: u64 = 0xF4EE;

/// Estado de un generador SplitMix64.
#[derive(Debug, Clone)]
pub struct SplitMix64 {
    state: u64,
}

impl SplitMix64 {
    /// Crea un generador a partir de una semilla y un identificador de
    /// secuencia (`stream`): estado inicial = `seed XOR stream·0xD1B54A32D192ED03`.
    pub fn new(seed: u64, stream: u64) -> Self {
        Self {
            state: seed ^ stream.wrapping_mul(STREAM_MIX),
        }
    }

    /// Siguiente valor de 64 bits.
    pub fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(GOLDEN_GAMMA);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(MIX1);
        z = (z ^ (z >> 27)).wrapping_mul(MIX2);
        z ^ (z >> 31)
    }

    /// Valor en [-1, 1) con 24 bits de resolución: `(k - 2^23) / 2^23` con
    /// `k` en [0, 2^24). Es exactamente representable en `f32`, por lo que no
    /// hay redondeo que pueda diferir entre C++ y Rust.
    pub fn next_f32_symmetric(&mut self) -> f32 {
        let k = (self.next_u64() >> 40) as i32 - (1 << 23);
        k as f32 * (1.0 / 8_388_608.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectores_de_referencia_splitmix64() {
        // Valores publicados para SplitMix64 con estado inicial 0.
        let mut rng = SplitMix64::new(0, 0);
        assert_eq!(rng.next_u64(), 0xE220_A839_7B1D_CDAF);
        assert_eq!(rng.next_u64(), 0x6E78_9E6A_A1B9_65F4);
        assert_eq!(rng.next_u64(), 0x06C4_5D18_8009_454F);
    }

    #[test]
    fn valores_en_rango_y_centrados() {
        let mut rng = SplitMix64::new(42, STREAM_A);
        let n = 100_000;
        let mut sum = 0.0f64;
        for _ in 0..n {
            let v = rng.next_f32_symmetric();
            assert!((-1.0..1.0).contains(&v), "valor fuera de rango: {v}");
            sum += v as f64;
        }
        let mean = sum / n as f64;
        assert!(mean.abs() < 0.01, "media demasiado lejos de 0: {mean}");
    }

    #[test]
    fn streams_distintos_dan_secuencias_distintas() {
        let mut a = SplitMix64::new(42, STREAM_A);
        let mut b = SplitMix64::new(42, STREAM_B);
        let same = (0..64).filter(|_| a.next_u64() == b.next_u64()).count();
        assert_eq!(same, 0);
    }
}
