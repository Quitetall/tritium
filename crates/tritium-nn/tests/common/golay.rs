//! Golay-coded trit-planes: the extended ternary Golay code [12, 6, 6] as a Construction-A lattice
//! on the finest trit plane (`T + ½` trits a weight, still an ordinary ternary plane), with exact
//! nearest-point decoding and an MSE step fit. Shared by `coded_planes` (SmolLM2) and
//! `qwen35_coded_planes` (the transfer check) so both measure the same structure.

use tritium_train::ops::ste;

/// Scale group and Hadamard block width.
pub const GROUP: usize = 128;
/// Step candidates for the ladder's step search.
pub const GRID: usize = 16;
/// Golay block length.
pub const BLOCK: usize = 12;

/// The 729 codewords of the extended ternary Golay code, balanced digits `{-1, 0, 1}`.
///
/// Generator `[I₆ | A]` with `A` the standard symmetric circulant-bordered form.
pub fn golay_codewords() -> Vec<[i8; BLOCK]> {
    const A: [[u8; 6]; 6] = [
        [0, 1, 1, 1, 1, 1],
        [1, 0, 1, 2, 2, 1],
        [1, 1, 0, 1, 2, 2],
        [1, 2, 1, 0, 1, 2],
        [1, 2, 2, 1, 0, 1],
        [1, 1, 2, 2, 1, 0],
    ];
    let mut out = Vec::with_capacity(729);
    for m in 0..729u32 {
        let msg: Vec<u8> = (0..6).map(|i| ((m / 3u32.pow(i)) % 3) as u8).collect();
        let mut word = [0i8; BLOCK];
        for i in 0..6 {
            word[i] = msg[i] as i8;
        }
        for j in 0..6 {
            let s: u32 = (0..6).map(|i| u32::from(msg[i]) * u32::from(A[i][j])).sum();
            word[6 + j] = (s % 3) as i8;
        }
        for d in &mut word {
            if *d == 2 {
                *d = -1;
            }
        }
        out.push(word);
    }
    out
}

pub fn kmax(t: usize) -> i64 {
    (3i64.pow(t as u32) - 1) / 2
}

pub fn hadamard_blocks(row: &mut [f32]) {
    for b in row.chunks_mut(GROUP) {
        if b.len().is_power_of_two() && b.len() > 1 {
            ste::fast_hadamard(b);
        }
    }
}

/// Nearest lattice point for ONE 12-block of target values `x` (weight units) starting at column
/// `b0`; returns the codes in step units and the squared error.
pub fn golay_decode_block(
    x: &[f64],
    b0: usize,
    steps: &[f64],
    k: f64,
    code: &[[i8; BLOCK]],
) -> ([i64; BLOCK], f64) {
    let mut best = (f64::INFINITY, 0usize);
    for (ci, c) in code.iter().enumerate() {
        let mut cost = 0.0;
        for i in 0..BLOCK {
            let d = steps[(b0 + i) / GROUP];
            let e = if d > 0.0 {
                let kk = ((x[i] / d - f64::from(c[i])) / 3.0).round().clamp(-k, k);
                x[i] - d * (3.0 * kk + f64::from(c[i]))
            } else {
                x[i]
            };
            cost += e * e;
        }
        if cost < best.0 {
            best = (cost, ci);
        }
    }
    let c = &code[best.1];
    let mut units = [0i64; BLOCK];
    for i in 0..BLOCK {
        let d = steps[(b0 + i) / GROUP];
        units[i] = if d > 0.0 {
            let kk = ((x[i] / d - f64::from(c[i])) / 3.0).round().clamp(-k, k);
            3 * kk as i64 + i64::from(c[i])
        } else {
            0
        };
    }
    (units, best.0)
}

/// Nearest point of the scaled lattice for every 12-block of `y`: codes in step units, and the
/// squared error. Per codeword the coarse digits round independently, so the minimum over the 729
/// codewords is exact (up to the coarse range clamp).
pub fn golay_decode(y: &[f32], steps: &[f64], k: f64, code: &[[i8; BLOCK]]) -> (Vec<i64>, f64) {
    let mut units = vec![0i64; y.len()];
    let mut total = 0.0;
    for b0 in (0..y.len()).step_by(BLOCK) {
        let x: Vec<f64> = y[b0..b0 + BLOCK].iter().map(|&v| f64::from(v)).collect();
        let (u, e) = golay_decode_block(&x, b0, steps, k, code);
        units[b0..b0 + BLOCK].copy_from_slice(&u);
        total += e;
    }
    (units, total)
}

/// Golay-coded `T + ½` quantization of one rotated row: returns the reconstruction.
///
/// Steps start from the best uniform step of the `T+1`-plane grid the lattice sits on, scaled by a
/// per-row multiplier searched over a small grid (the lattice is sparser than that grid, so its
/// best step is larger), then refit per group by least squares given the codes.
pub fn golay_row(y: &[f32], t: usize, code: &[[i8; BLOCK]]) -> Vec<f32> {
    let (steps, units) = golay_fit(y, t, code);
    units
        .iter()
        .enumerate()
        .map(|(j, &r)| (steps[j / GROUP] * r as f64) as f32)
        .collect()
}

/// The MSE fit behind [`golay_row`]: per-group steps and the codes in step units.
pub fn golay_fit(y: &[f32], t: usize, code: &[[i8; BLOCK]]) -> (Vec<f64>, Vec<i64>) {
    let cols = y.len();
    assert!(
        cols.is_multiple_of(BLOCK),
        "row width {cols} is not a whole number of 12-blocks"
    );
    let k = kmax(t) as f64;
    let base: Vec<f64> = y
        .chunks(GROUP)
        .map(|g| f64::from(ste::ladder_step(g, t + 1, GRID)))
        .collect();
    let (mut steps, mut units) = [1.0, 1.15, 1.3, 1.45, 1.6]
        .iter()
        .map(|&f| {
            let steps: Vec<f64> = base.iter().map(|d| d * f).collect();
            let (units, err) = golay_decode(y, &steps, k, code);
            (err, steps, units)
        })
        .min_by(|a, b| a.0.total_cmp(&b.0))
        .map(|(_, s, u)| (s, u))
        .expect("non-empty grid");
    for _ in 0..2 {
        let mut refit = steps.clone();
        for (g, step) in refit.iter_mut().enumerate() {
            let (lo, hi) = (g * GROUP, ((g + 1) * GROUP).min(cols));
            let num: f64 = (lo..hi).map(|j| f64::from(y[j]) * units[j] as f64).sum();
            let den: f64 = (lo..hi).map(|j| (units[j] as f64).powi(2)).sum();
            if den > 0.0 && num > 0.0 {
                *step = num / den;
            }
        }
        let (u, _) = golay_decode(y, &refit, k, code);
        steps = refit;
        units = u;
    }
    (steps, units)
}
