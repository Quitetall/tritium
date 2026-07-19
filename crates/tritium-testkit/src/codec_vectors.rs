//! Codec (Conv1d / FSQ) conformance vectors (ADR 0030 Tier 4).
//!
//! Golden `input → expected` cases whose `expected` is computed from the `tritium-core` reference
//! oracles ([`reference_conv1d`] / [`reference_fsq`]). A candidate backend's conv/FSQ output is graded
//! **bit-exact** against `expected` (ternary conv is add/sub/skip and the FSQ grid is integer, so both
//! are exact — not `1e-4` like the float mpGEMM path). This is the harness a future CUDA or MCU conv/FSQ
//! kernel is measured against, and the format LamQuant's golden EEG windows drop into as extra cases.
//!
//! The oracle itself is validated bit-identical to the training ops in
//! `tritium-train/tests/reference_parity.rs`, so "matches the oracle" transitively means "matches what
//! the trainer emitted".

use tritium_core::{ConvShape, Trit, reference_conv1d, reference_fsq};

use crate::Tolerance;

/// One ternary Conv1d conformance case.
#[derive(Clone, Debug, PartialEq)]
pub struct Conv1dVector {
    /// Stable case id.
    pub id: String,
    /// Convolution geometry.
    pub shape: ConvShape,
    /// `[B·C_in·L_in]` activations.
    pub activation: Vec<f32>,
    /// `[C_out·K_g]` ternary weights in `{-1,0,1}`.
    pub weights: Vec<i8>,
    /// `[C_out]` per-output-channel scales.
    pub scales: Vec<f32>,
    /// `[B·C_out·L_out]` expected output (from the reference oracle).
    pub expected: Vec<f32>,
}

/// One FSQ conformance case (clamp deploy grid).
#[derive(Clone, Debug, PartialEq)]
pub struct FsqVector {
    /// Stable case id.
    pub id: String,
    /// Channel count (rows).
    pub channels: usize,
    /// Per-channel length (cols).
    pub len: usize,
    /// Per-channel level counts `L`.
    pub levels: Vec<u32>,
    /// `[channels·len]` input.
    pub input: Vec<f32>,
    /// `[channels·len]` expected quantized output (from the reference oracle).
    pub expected: Vec<f32>,
}

/// xorshift64 — the repo's dependency-free PRNG idiom (seed forced non-zero).
struct Rng(u64);
impl Rng {
    fn new(seed: u64) -> Self {
        Self(seed | 1)
    }
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }
    fn uniform(&mut self, lo: f32, hi: f32) -> f32 {
        lo + (self.next() % 1000) as f32 / 1000.0 * (hi - lo)
    }
    fn trit(&mut self) -> i8 {
        match self.next() % 3 {
            0 => -1,
            1 => 0,
            _ => 1,
        }
    }
}

/// The boundary geometries every generated conv set covers (pointwise, depthwise, grouped, dilated,
/// strided, even + wide odd kernels).
fn conv_geometries() -> Vec<(&'static str, ConvShape)> {
    let base = ConvShape {
        batch: 2,
        c_in: 6,
        c_out: 6,
        l_in: 13,
        k: 3,
        stride: 1,
        dilation: 1,
        pad_left: 1,
        pad_right: 1,
        groups: 1,
    };
    vec![
        ("dense", base),
        (
            "pointwise",
            ConvShape {
                k: 1,
                pad_left: 0,
                pad_right: 0,
                ..base
            },
        ),
        (
            "depthwise",
            ConvShape {
                groups: 6,
                k: 5,
                pad_left: 2,
                pad_right: 2,
                ..base
            },
        ),
        ("grouped", ConvShape { groups: 3, ..base }),
        (
            "dilated",
            ConvShape {
                dilation: 2,
                pad_left: 2,
                pad_right: 2,
                ..base
            },
        ),
        ("stride2", ConvShape { stride: 2, ..base }),
        (
            "k4-asym",
            ConvShape {
                k: 4,
                pad_left: 2,
                pad_right: 1,
                ..base
            },
        ),
        (
            "k7-causal",
            ConvShape {
                k: 7,
                pad_left: 6,
                pad_right: 0,
                ..base
            },
        ),
    ]
}

/// Generate `count` deterministic Conv1d conformance vectors (cycling the boundary geometries).
#[must_use]
pub fn generate_conv_vectors(seed: u64, count: usize) -> Vec<Conv1dVector> {
    let mut rng = Rng::new(seed);
    let geoms = conv_geometries();
    (0..count)
        .map(|i| {
            let (name, shape) = geoms[i % geoms.len()];
            let activation: Vec<f32> = (0..shape.batch * shape.c_in * shape.l_in)
                .map(|_| rng.uniform(-2.0, 2.0))
                .collect();
            let weights: Vec<i8> = (0..shape.c_out * shape.k_g()).map(|_| rng.trit()).collect();
            let scales: Vec<f32> = (0..shape.c_out).map(|_| rng.uniform(0.2, 1.7)).collect();
            let trits: Vec<Trit> = weights.iter().map(|&w| Trit::from_sign(w)).collect();
            let mut expected = vec![0.0f32; shape.batch * shape.c_out * shape.l_out()];
            reference_conv1d(&activation, &trits, &scales, shape, &mut expected)
                .expect("valid conv geometry");
            Conv1dVector {
                id: format!("conv-{name}-{i}"),
                shape,
                activation,
                weights,
                scales,
                expected,
            }
        })
        .collect()
}

/// Generate `count` deterministic FSQ conformance vectors (cycling level schedules {2,3,5,8,16,32}).
#[must_use]
pub fn generate_fsq_vectors(seed: u64, count: usize) -> Vec<FsqVector> {
    let mut rng = Rng::new(seed);
    let schedules: [&[u32]; 4] = [&[2, 3, 5, 8], &[16, 32], &[5], &[2, 2, 2, 2, 2, 2, 2, 2]];
    (0..count)
        .map(|i| {
            let levels = schedules[i % schedules.len()].to_vec();
            let channels = levels.len();
            let len = 7 + i % 5;
            let input: Vec<f32> = (0..channels * len)
                .map(|_| rng.uniform(-1.4, 1.4))
                .collect();
            let mut expected = vec![0.0f32; channels * len];
            reference_fsq(&input, &levels, channels, len, &mut expected).expect("valid fsq");
            FsqVector {
                id: format!("fsq-{i}"),
                channels,
                len,
                levels,
                input,
                expected,
            }
        })
        .collect()
}

/// Grade a candidate Conv1d output against the frozen `expected` (bit-exact — ternary add/sub/skip).
#[must_use]
pub fn grade_conv(v: &Conv1dVector, got: &[f32]) -> bool {
    let tol = Tolerance::bit_exact();
    got.len() == v.expected.len()
        && got
            .iter()
            .zip(&v.expected)
            .all(|(&g, &e)| tol.accepts(g, e))
}

/// Grade a candidate FSQ output against the frozen `expected` (bit-exact — integer grid).
#[must_use]
pub fn grade_fsq(v: &FsqVector, got: &[f32]) -> bool {
    let tol = Tolerance::bit_exact();
    got.len() == v.expected.len()
        && got
            .iter()
            .zip(&v.expected)
            .all(|(&g, &e)| tol.accepts(g, e))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        assert_eq!(
            generate_conv_vectors(0xC0DEC, 16),
            generate_conv_vectors(0xC0DEC, 16)
        );
        assert_eq!(
            generate_fsq_vectors(0xF5, 16),
            generate_fsq_vectors(0xF5, 16)
        );
    }

    #[test]
    fn reference_reproduces_conv_expected() {
        for v in generate_conv_vectors(0xC0DEC, 24) {
            let trits: Vec<Trit> = v.weights.iter().map(|&w| Trit::from_sign(w)).collect();
            let mut got = vec![0.0f32; v.expected.len()];
            reference_conv1d(&v.activation, &trits, &v.scales, v.shape, &mut got).unwrap();
            assert!(grade_conv(&v, &got), "case {} failed to reproduce", v.id);
        }
    }

    #[test]
    fn reference_reproduces_fsq_expected() {
        for v in generate_fsq_vectors(0xF5, 24) {
            let mut got = vec![0.0f32; v.expected.len()];
            reference_fsq(&v.input, &v.levels, v.channels, v.len, &mut got).unwrap();
            assert!(grade_fsq(&v, &got), "case {} failed to reproduce", v.id);
        }
    }

    #[test]
    fn grader_rejects_a_perturbation() {
        let v = generate_conv_vectors(0xC0DEC, 1).pop().unwrap();
        let mut bad = v.expected.clone();
        bad[0] += 1.0; // a single wrong element fails bit-exact grading
        assert!(!grade_conv(&v, &bad));
    }
}
