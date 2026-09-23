//! The normalized Sylvester–Walsh–Hadamard transform: the one Hadamard every Tritium basis uses.
//!
//! `H_n[i][j] = (-1)^popcount(i & j) / sqrt(n)` for `n` a power of two. It is symmetric and
//! orthonormal, so it is its own inverse: `H_n · H_n = I`. A rotated tensor stores `W·Hᵀ` and the
//! runtime feeds it `H·x`; because `H` is symmetric those are `W·H` and `H·x`, and the product
//! `W·x` is recovered exactly in exact arithmetic.
//!
//! This is the convention external ternary checkpoints use as well (PrismML's
//! `normalized-sylvester-walsh-hadamard`), so a basis read from such a file needs no translation.
//!
//! The transform runs as the standard in-place butterfly, `O(n log n)` additions and one final
//! scale, and needs no allocation — it is usable from `no_std` targets.

/// Apply `H_n` in place to `v`, where `n = v.len()`.
///
/// The scale `1/sqrt(n)` is computed without a square root: for `n = 2^k` it is `2^(-k/2)`,
/// which is exact for even `k` and a power of two times `FRAC_1_SQRT_2` for odd `k`, so it equals
/// the correctly rounded `1/sqrt(n)` in both cases.
///
/// # Panics
/// Panics if `v.len()` is not a power of two. A length of one is the identity.
pub fn fwht_normalized(v: &mut [f32]) {
    let n = v.len();
    assert!(
        n.is_power_of_two(),
        "the Walsh-Hadamard transform needs a power-of-two length, got {n}"
    );
    let mut half = 1;
    while half < n {
        let mut start = 0;
        while start < n {
            for i in start..start + half {
                let (a, b) = (v[i], v[i + half]);
                v[i] = a + b;
                v[i + half] = a - b;
            }
            start += half * 2;
        }
        half *= 2;
    }
    let scale = inverse_sqrt_power_of_two(n);
    for x in v.iter_mut() {
        *x *= scale;
    }
}

/// `1/sqrt(n)` for `n` a power of two, correctly rounded, without calling `sqrt`.
fn inverse_sqrt_power_of_two(n: usize) -> f32 {
    let k = n.trailing_zeros();
    let whole = 1.0 / (1u64 << (k / 2)) as f32;
    if k.is_multiple_of(2) {
        whole
    } else {
        whole * core::f32::consts::FRAC_1_SQRT_2
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The explicit matrix definition, entry by entry.
    fn sylvester_entry(row: usize, col: usize, n: usize) -> f64 {
        let sign = if (row & col).count_ones().is_multiple_of(2) {
            1.0
        } else {
            -1.0
        };
        sign / (n as f64).sqrt()
    }

    #[test]
    fn the_butterfly_is_the_sylvester_matrix() {
        for n in [1usize, 2, 4, 8, 64, 1024] {
            for col in [0, n / 3, n - 1] {
                let mut basis = [0.0f32; 1024];
                basis[col] = 1.0;
                fwht_normalized(&mut basis[..n]);
                for (row, got) in basis[..n].iter().enumerate() {
                    let want = sylvester_entry(row, col, n);
                    assert!(
                        (f64::from(*got) - want).abs() <= 1e-7,
                        "n={n} H[{row}][{col}] = {got}, want {want}"
                    );
                }
            }
        }
    }

    #[test]
    fn the_transform_is_its_own_inverse() {
        let original: [f32; 256] = core::array::from_fn(|i| ((i * 37 % 101) as f32 - 50.0) / 7.0);
        let mut v = original;
        fwht_normalized(&mut v);
        fwht_normalized(&mut v);
        for (a, b) in v.iter().zip(&original) {
            assert!((a - b).abs() <= 1e-5 * b.abs().max(1.0), "{a} vs {b}");
        }
    }

    #[test]
    fn the_scale_is_correctly_rounded() {
        for k in 0..20u32 {
            let n = 1usize << k;
            let want = (1.0f64 / (n as f64).sqrt()) as f32;
            assert_eq!(
                inverse_sqrt_power_of_two(n).to_bits(),
                want.to_bits(),
                "n = {n}"
            );
        }
    }

    #[test]
    #[should_panic(expected = "power-of-two")]
    fn a_non_power_of_two_length_is_rejected() {
        fwht_normalized(&mut [0.0; 6]);
    }
}
