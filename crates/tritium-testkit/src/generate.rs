//! Deterministic generation of conformance vectors from the reference kernel.
//!
//! No `rand` dependency: a tiny xorshift64 PRNG, seeded per call and perturbed
//! per vector index, fills the random cases. A fixed BOUNDARY set covers the
//! degenerate shapes and weight patterns ADR 0002 calls out (all-zero, all-±1,
//! `M=1`, `N=1`, large `M`). Every vector's `expected` field is filled by
//! [`tritium_core::reference_mpgemm`], so the suite is by construction the
//! reference's own output.

use tritium_core::{GemmShape, Trit, reference_mpgemm};

use crate::vector::ConformanceVector;

/// A minimal, fast, deterministic xorshift64 PRNG. Not cryptographic — its only
/// job is reproducible pseudo-randomness for test vectors, with no external
/// dependency.
#[derive(Clone, Debug)]
struct XorShift64 {
    state: u64,
}

impl XorShift64 {
    /// Seed the generator. The seed is forced non-zero (xorshift fixes on zero).
    fn new(seed: u64) -> Self {
        XorShift64 { state: seed | 1 }
    }

    /// Next raw 64-bit value.
    fn next_u64(&mut self) -> u64 {
        let mut x = self.state;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.state = x;
        x
    }

    /// Uniform integer in `[0, n)` (`n > 0`).
    fn below(&mut self, n: u64) -> u64 {
        self.next_u64() % n
    }

    /// A small activation magnitude in roughly `[-10, 10)`, two decimals.
    fn activation(&mut self) -> f32 {
        (self.below(2000) as f32 - 1000.0) / 100.0
    }

    /// A positive per-channel scale in roughly `(0, 2]`.
    fn scale(&mut self) -> f32 {
        (self.below(200) as f32 + 1.0) / 100.0
    }

    /// A ternary weight in `{-1, 0, +1}`.
    fn trit(&mut self) -> Trit {
        Trit::from_sign(self.below(3) as i8 - 1)
    }
}

/// Build one vector from explicit data, filling `expected` via the reference.
///
/// `weights` are `Trit`s; they are stored as raw `i8` in the vector. Panics are
/// impossible here for well-formed inputs because the lengths are derived from
/// the same shape passed to the reference; the reference's only error path is a
/// length mismatch, which the construction precludes.
fn build(
    id: String,
    shape: GemmShape,
    activation: Vec<f32>,
    weights: Vec<Trit>,
    scales: Vec<f32>,
    format: &str,
) -> ConformanceVector {
    let mut expected = vec![0.0f32; shape.m * shape.n];
    // Lengths are constructed to match `shape`, so this never errors. We still
    // handle the Result rather than unwrap, to keep the function panic-free: on
    // the impossible error path we leave `expected` as zeros, which a downstream
    // conformance run would surface as a mismatch rather than a crash.
    let _ = reference_mpgemm(&activation, &weights, &scales, shape, &mut expected);
    ConformanceVector {
        id,
        m: shape.m,
        n: shape.n,
        k: shape.k,
        activation,
        weights: weights.iter().map(|t| t.get()).collect(),
        scales,
        format: format.to_string(),
        expected,
    }
}

/// Generate a random vector for the given shape, format, and label.
fn random_vector(
    rng: &mut XorShift64,
    id: String,
    shape: GemmShape,
    format: &str,
) -> ConformanceVector {
    let activation = (0..shape.m * shape.k).map(|_| rng.activation()).collect();
    let weights = (0..shape.n * shape.k).map(|_| rng.trit()).collect();
    let scales = (0..shape.n).map(|_| rng.scale()).collect();
    build(id, shape, activation, weights, scales, format)
}

/// The fixed boundary set: degenerate shapes and weight patterns that catch the
/// classic edge bugs (ADR 0002 §0.10 Degenerate). `K` is kept a multiple of 256
/// so packing is whole-block at this milestone.
fn boundary_vectors() -> Vec<ConformanceVector> {
    // A deterministic activation seed independent of the random stream, so the
    // boundary set is stable regardless of how the random count is chosen.
    let mut rng = XorShift64::new(0xB0117DA47_u64);
    let mut out = Vec::new();

    // Shapes exercising M in {1, 64}, N in {1, 32}, K in {256, 512}.
    let shapes = [
        (1usize, 1usize, 256usize),
        (1, 32, 256),
        (64, 1, 256),
        (64, 32, 512),
        (1, 1, 512),
    ];

    for (si, &(m, n, k)) in shapes.iter().enumerate() {
        let shape = GemmShape::new(m, n, k);
        let activation: Vec<f32> = (0..m * k).map(|_| rng.activation()).collect();
        let scales: Vec<f32> = (0..n).map(|_| rng.scale()).collect();

        // All-zero weights: output must be exactly zero.
        let zeros = vec![Trit::ZERO; n * k];
        out.push(build(
            format!("boundary-{si}-allzero"),
            shape,
            activation.clone(),
            zeros,
            scales.clone(),
            "tq2_0",
        ));

        // All-+1 weights.
        let allpos = vec![Trit::POS; n * k];
        out.push(build(
            format!("boundary-{si}-allpos"),
            shape,
            activation.clone(),
            allpos,
            scales.clone(),
            "tq2_0",
        ));

        // All--1 weights.
        let allneg = vec![Trit::NEG; n * k];
        out.push(build(
            format!("boundary-{si}-allneg"),
            shape,
            activation.clone(),
            allneg,
            scales.clone(),
            "tq1_0",
        ));

        // Mixed random weights at this boundary shape, one per format.
        let mixed: Vec<Trit> = (0..n * k).map(|_| rng.trit()).collect();
        out.push(build(
            format!("boundary-{si}-mixed-tq2"),
            shape,
            activation.clone(),
            mixed.clone(),
            scales.clone(),
            "tq2_0",
        ));
        out.push(build(
            format!("boundary-{si}-mixed-tq1"),
            shape,
            activation,
            mixed,
            scales,
            "tq1_0",
        ));
    }

    out
}

/// Generate `count` conformance vectors deterministically from `seed`, plus the
/// fixed boundary set.
///
/// The same `(seed, count)` always yields byte-identical vectors (ADR 0002 U6
/// determinism). Random shapes are small with `K` a multiple of 256 (whole-block
/// packing at 0.10); the boundary set is appended unconditionally so a caller who
/// asks for `count = 0` still gets the degenerate-case coverage.
///
/// Each random vector alternates between the `tq2_0` and `tq1_0` formats so both
/// packers are exercised.
///
/// ```
/// use tritium_testkit::generate_vectors;
/// let a = generate_vectors(42, 8);
/// let b = generate_vectors(42, 8);
/// assert_eq!(a, b); // reproducible from the seed
/// assert!(a.len() > 8); // random cases + the fixed boundary set
/// ```
#[must_use]
pub fn generate_vectors(seed: u64, count: usize) -> Vec<ConformanceVector> {
    let mut rng = XorShift64::new(seed);
    let mut out = Vec::with_capacity(count);

    for i in 0..count {
        // Vary the stream per index so consecutive vectors differ even at the
        // same shape, while staying a pure function of (seed, i).
        rng.state ^= (i as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1;

        // Small random shapes. K is a multiple of 256 (256 or 512) so packing is
        // whole-block at this milestone.
        let m = 1 + rng.below(64) as usize;
        let n = 1 + rng.below(32) as usize;
        let k = 256 * (1 + rng.below(2) as usize);
        let format = if i % 2 == 0 { "tq2_0" } else { "tq1_0" };

        out.push(random_vector(
            &mut rng,
            format!("rand-{i}"),
            GemmShape::new(m, n, k),
            format,
        ));
    }

    out.extend(boundary_vectors());
    out
}
