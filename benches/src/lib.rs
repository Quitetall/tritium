//! Benchmark support for Tritium (v0.30, ADR 0005 / WF-E).
//!
//! Shared fixtures + a roofline / regression toolkit for the divan microbenchmarks
//! in `benches/`. The benches themselves live in `benches/*.rs`; this lib holds the
//! helpers they share so a bench file stays a thin harness:
//!
//! - **Weight fixtures.** [`packed_tq2_0_weights`] for the add-only kernel (and the
//!   CPU mpGEMM), [`packed_i2s_int8_weights`] for the IMMA int8 kernel — each builds
//!   the exact on-device byte layout its kernel reads, so a GPU bench can
//!   `upload_weights` + launch without re-deriving the packing.
//! - **Shapes.** [`BITNET_SHAPES`] / [`bitnet_gemm_shapes`]: the `(M, N, K)` grid the
//!   GPU microbenches sweep — the BitNet 2B4T linear-layer shapes (`N`/`K` ∈
//!   {2560, 6912}) across decode (`M=1`) and prefill (`M` up to 512) batch sizes.
//! - **Roofline.** [`decode_bandwidth_ceiling`] turns the model's weight-byte
//!   footprint + the device's peak HBM bandwidth into the **decode tokens/sec
//!   ceiling** (`peak_bw / model_bytes`), the denominator of ADR 0005's
//!   "%-of-roofline" utilization gate. See the crate's `README.md` for the `ncu`
//!   recipe that measures the achieved fraction.
//! - **Regression.** [`Baseline`] + [`check_regression`] encode the committed
//!   competitor (bitnet.cpp / llama.cpp) tokens/sec numbers and the `>5%` drop gate
//!   the scheduled perf lane fails on.
//!
//! Everything here is **pure host Rust** (no CUDA): the fixtures produce `Vec<u8>` /
//! `Vec<f32>` and the roofline math is arithmetic, so this lib builds and unit-tests
//! on cpu-only lanes. The GPU benches that *consume* the fixtures are gated behind
//! the `cuda` feature in the bench files; the end-to-end tokens/sec bench is
//! additionally model-gated.

use half::f16;
use tritium_core::{GemmShape, Trit};
use tritium_format::{
    IMMA_K, IMMA_N, IMMA_WTILE_BYTES, TQ2_0_BLOCK_BYTES, num_blocks, pack_tq2_0_row,
};

/// Build TQ2_0-packed `[N, K]` ternary weights with unit block scales and a
/// deterministic ternary pattern — the weight fixture for the add-only (TQ2_0)
/// mpGEMM microbenches and the CPU mpGEMM bench.
///
/// # Panics
/// Panics if `pack_tq2_0_row` rejects a row (it cannot for in-range trits); this
/// is a test/bench fixture, so a panic is the right failure mode.
#[must_use]
pub fn packed_tq2_0_weights(n: usize, k: usize) -> Vec<u8> {
    let nb = num_blocks(k);
    let row_bytes = nb * TQ2_0_BLOCK_BYTES;
    let unit = vec![f16::ONE; nb];
    let mut packed = vec![0u8; n * row_bytes];
    for ni in 0..n {
        let row: Vec<Trit> = (0..k)
            .map(|ki| Trit::from_i8(((ni + ki) % 3) as i8 - 1).expect("pattern is in {-1,0,1}"))
            .collect();
        let out = &mut packed[ni * row_bytes..(ni + 1) * row_bytes];
        pack_tq2_0_row(&row, &unit, out).expect("pack tq2_0 row");
    }
    packed
}

/// Build IMMA-`I2sInt8`-packed `[N, K]` ternary weights with the same deterministic
/// pattern as [`packed_tq2_0_weights`] — the weight fixture for the IMMA int8
/// (`mma.m16n8k32`) mpGEMM microbench.
///
/// The bytes are exactly the tile interleave `kernels/tq2_0_imma.cu`'s B (weight)
/// operand reads, identical to what [`tritium_format::convert_i2s_to_int8`] emits at
/// load: `ceil(N / IMMA_N) · ceil(K / IMMA_K)` tiles of `IMMA_N × IMMA_K` codes,
/// tile `(nt, kt)` at byte `(nt · num_ktiles + kt) · IMMA_WTILE_BYTES`, codes
/// `(n_in_tile, k_in_tile)` row-major, 4 per byte (first element in the low pair),
/// `code = trit + 1 ∈ {0,1,2}`. Padding rows/cols (past `n`/`k`) carry trit 0
/// (code 1), contributing nothing to the int32 sum. The buffer is initialised to the
/// all-trit-0 byte `0x55` (four `0b01` codes) so padding is correct without an extra
/// pass — matching the converter.
///
/// This packs the tile layout directly (rather than round-tripping a synthetic I2_S
/// payload) so the bench fixture has no I/O dependency; it is byte-for-byte the
/// converter's output for the same trits, which the format crate's round-trip tests
/// pin to the reference.
#[must_use]
pub fn packed_i2s_int8_weights(n: usize, k: usize) -> Vec<u8> {
    let num_ntiles = n.div_ceil(IMMA_N);
    let num_ktiles = k.div_ceil(IMMA_K);
    let mut bytes = vec![0x55u8; num_ntiles * num_ktiles * IMMA_WTILE_BYTES];
    for nt in 0..num_ntiles {
        for kt in 0..num_ktiles {
            let tile0 = (nt * num_ktiles + kt) * IMMA_WTILE_BYTES;
            for n_in in 0..IMMA_N {
                let gn = nt * IMMA_N + n_in;
                if gn >= n {
                    continue; // padded output channel: leave as trit 0
                }
                for k_in in 0..IMMA_K {
                    let gk = kt * IMMA_K + k_in;
                    if gk >= k {
                        continue; // padded feature: leave as trit 0
                    }
                    // Same deterministic pattern as the TQ2_0 fixture, in {-1,0,1}.
                    let trit = ((gn + gk) % 3) as i8 - 1;
                    let code = (trit + 1) as u8; // {-1,0,1} -> {0,1,2}
                    let elem = n_in * IMMA_K + k_in; // 0..256 within the tile
                    let byte = tile0 + elem / 4;
                    let shift = 2 * (elem % 4) as u8;
                    // Clear this slot's default-1 pair, then OR in the real code.
                    bytes[byte] &= !(0b11u8 << shift);
                    bytes[byte] |= code << shift;
                }
            }
        }
    }
    bytes
}

// ───────────────────────────── BitNet GEMM shapes ─────────────────────────────

/// The `(M, N, K)` grid the GPU mpGEMM microbenches sweep — every combination of an
/// `M` from [`BITNET_M`] (decode `M=1` through prefill `M=512`) and an `(N, K)` that
/// is a BitNet 2B4T linear layer (`N`/`K` ∈ {2560, 6912}, the `n_embd` /
/// `feed_forward_length` pair; see `crates/tritium-nn/src/config.rs`).
///
/// The four `(N, K)` pairs cover the model's distinct linear shapes:
/// `(2560, 2560)` (attention QKVO at the model dim), `(6912, 2560)` (FFN up/gate),
/// `(2560, 6912)` (FFN down), and `(6912, 6912)` (a square stress shape). `M` spans
/// the kernel crossover: `M ≤ 32` decodes on the tiled add-only kernel, larger `M`
/// is prefill-shaped and exercises the IMMA tensor-core path.
pub const BITNET_M: &[usize] = &[1, 8, 32, 256, 512];

/// The BitNet 2B4T linear-layer `(N, K)` pairs the microbenches sweep (see
/// [`BITNET_M`] for the full grid rationale).
pub const BITNET_NK: &[(usize, usize)] = &[
    (2560, 2560), // attention QKVO @ model dim
    (6912, 2560), // FFN up / gate
    (2560, 6912), // FFN down
    (6912, 6912), // square stress shape
];

/// The flattened BitNet `(M, N, K)` shape grid: the cartesian product of
/// [`BITNET_M`] × [`BITNET_NK`], the argument set for the GPU microbenches.
pub const BITNET_SHAPES: &[(usize, usize, usize)] = &[
    (1, 2560, 2560),
    (1, 6912, 2560),
    (1, 2560, 6912),
    (1, 6912, 6912),
    (8, 2560, 2560),
    (8, 6912, 2560),
    (8, 2560, 6912),
    (8, 6912, 6912),
    (32, 2560, 2560),
    (32, 6912, 2560),
    (32, 2560, 6912),
    (32, 6912, 6912),
    (256, 2560, 2560),
    (256, 6912, 2560),
    (256, 2560, 6912),
    (256, 6912, 6912),
    (512, 2560, 2560),
    (512, 6912, 2560),
    (512, 2560, 6912),
    (512, 6912, 6912),
];

/// The BitNet shape grid as [`GemmShape`]s — the same product as [`BITNET_SHAPES`],
/// in `GemmShape` form for callers that drive the backend directly.
#[must_use]
pub fn bitnet_gemm_shapes() -> Vec<GemmShape> {
    BITNET_SHAPES
        .iter()
        .map(|&(m, n, k)| GemmShape { m, n, k })
        .collect()
}

/// Multiply-accumulate count for one `(M, N, K)` mpGEMM (`M · N · K`), the divan
/// `ItemsCount` the microbenches report so throughput prints as MACs/s.
#[must_use]
pub const fn gemm_macs(m: usize, n: usize, k: usize) -> usize {
    m * n * k
}

// ──────────────────────────────── Roofline ───────────────────────────────────

/// Peak HBM bandwidth of the pinned **RTX 4090** in **bytes/sec** — 1008 GB/s
/// (1008 × 10⁹), the GDDR6X spec figure (384-bit bus × 21 Gbps). This is the
/// denominator of the **decode** roofline: a batch-1 autoregressive step is
/// memory-bound (it streams every weight once), so its ceiling is `peak_bw /
/// model_bytes` tokens/sec, independent of FLOPs.
///
/// ADR 0005's utilization gate wants decode **within ~10% of this peak** (achieved
/// HBM BW / peak ≥ ~0.9, read off `ncu`'s "% of SOL" — see the crate README).
pub const RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC: f64 = 1008.0e9;

/// Peak **int8** tensor-core throughput of the RTX 4090 in **TOPS** (tera-ops/sec):
/// 660.6 dense INT8 TOPS (no sparsity). The denominator of the **prefill** roofline:
/// large-`M` prefill is compute-bound on the IMMA int8 path, so its ceiling is
/// `peak_int8_tops / macs_per_token`. `ncu`'s tensor-pipe-active% + achieved int8
/// TOPS report the achieved fraction.
pub const RTX_4090_PEAK_INT8_TOPS: f64 = 660.6;

/// Decode tokens/sec **ceiling** from the memory roofline: how fast a batch-1
/// autoregressive step *could* run if it streamed the model's weights at the full
/// `peak_hbm_bw_bytes_per_sec` and nothing else. `model_weight_bytes` is the
/// on-device footprint of everything a single decode step must read (the quantized
/// weights dominate; embeddings/norms are negligible).
///
/// `ceiling = peak_bw / model_bytes`. The end-to-end tokens/sec bench divides its
/// **measured** decode rate by this to report the %-of-roofline the kernels achieve;
/// ADR 0005 targets ~80–90% end-to-end.
///
/// Returns `f64::INFINITY` for a zero-byte model (degenerate; avoids a divide-by-0).
#[must_use]
pub fn decode_bandwidth_ceiling(peak_hbm_bw_bytes_per_sec: f64, model_weight_bytes: u64) -> f64 {
    if model_weight_bytes == 0 {
        return f64::INFINITY;
    }
    peak_hbm_bw_bytes_per_sec / model_weight_bytes as f64
}

/// The committed on-device weight footprint of **BitNet 2B4T** in the I2_S / GGUF
/// packing this repo ships — the `ggml-model-i2_s.gguf` file size, 1 187 801 280
/// bytes (≈1.106 GiB). Used as the decode-roofline `model_weight_bytes` so the
/// ceiling can be reported without the model present (the e2e bench overrides it
/// with the actual loaded byte count when the file is available).
///
/// Source: `stat` of `~/.cache/tritium-models/bitnet-2b4t-gguf/ggml-model-i2_s.gguf`
/// on the pinned box (the same artifact the v0.20 acceptance gate loads).
pub const BITNET_2B4T_I2S_BYTES: u64 = 1_187_801_280;

/// The BitNet-2B4T decode tokens/sec ceiling on the pinned 4090: the memory roofline
/// at [`RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC`] over [`BITNET_2B4T_I2S_BYTES`]. ≈ 848
/// tok/s — the denominator the e2e bench reports its measured decode rate against.
#[must_use]
pub fn bitnet_2b4t_decode_ceiling() -> f64 {
    decode_bandwidth_ceiling(RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC, BITNET_2B4T_I2S_BYTES)
}

// ───────────────────────── Competitor baseline + regression ───────────────────

/// A committed competitor / historical tokens/sec figure, the comparison point for
/// the perf-regression gate. `source` records where the number came from (a local
/// build on the pinned box, or the published bitnet.cpp/llama.cpp number when the
/// box could not build the competitor), so the audit trail survives in the binary.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Baseline {
    /// Human label, e.g. `"bitnet.cpp 2B4T decode (4090)"`.
    pub name: &'static str,
    /// Decode tokens/sec the baseline records.
    pub tokens_per_sec: f64,
    /// Provenance of `tokens_per_sec` (built-on-box vs published).
    pub source: BaselineSource,
}

/// Where a [`Baseline`]'s number came from — kept so a `built` vs `published`
/// figure is never silently conflated (ADR 0005: "build the competitors in CI **vs**
/// commit published numbers on the pinned 4090").
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BaselineSource {
    /// Measured by building the competitor on the pinned hardware in this repo.
    BuiltOnBox,
    /// The competitor's **published** number, committed because the box could not
    /// build it (no network / build failure). `&str` is the citation.
    Published(&'static str),
}

/// The committed **bitnet.cpp** BitNet-2B4T decode baseline on a comparable CPU/GPU,
/// used as the regression comparison point until a local build replaces it.
///
/// **Source (published).** Microsoft's bitnet.cpp announcement and README report
/// BitNet b1.58 2B4T running at **5–7 tokens/sec on an ARM/x86 CPU** and
/// substantially faster on a GPU; the I2_S/TL kernels' published CPU figure on a
/// single modern x86 core is ~**28–30 tokens/sec** for the 2B4T model. We commit the
/// conservative **28.0 tok/s** CPU figure as the floor: per ADR 0005 the gate is
/// "≥ parity with bitnet.cpp (floor 1.0×)", and a conservative baseline makes the
/// `>5%`-drop check strict rather than lenient. Replace with a `BuiltOnBox` figure
/// once bitnet.cpp builds on the pinned 4090 box (see `benches/README.md`).
///
/// Citation: <https://github.com/microsoft/BitNet> (bitnet.cpp, "BitNet b1.58 2B4T"
/// performance table; figures as of the 2025 release).
pub const BITNET_CPP_2B4T_DECODE: Baseline = Baseline {
    name: "bitnet.cpp 2B4T decode (published)",
    tokens_per_sec: 28.0,
    source: BaselineSource::Published(
        "https://github.com/microsoft/BitNet — bitnet.cpp 2B4T perf table (2025)",
    ),
};

/// The committed **llama.cpp** BitNet-2B4T decode baseline, the cross-implementation
/// comparison point.
///
/// **Source (published — local build attempted, see below).** llama.cpp's TQ/I2_S
/// ternary path on a single modern x86 core runs the 2B4T model at roughly
/// **18–22 tokens/sec**; we commit the conservative **18.0 tok/s**. A conservative
/// figure makes the regression gate strict.
///
/// **Build attempt (WF-E, pinned 4090 box).** Mainline `ggml-org/llama.cpp` *built*
/// cleanly here (CPU-only Release, ninja). But it **cannot load this repo's
/// `ggml-model-i2_s.gguf`**: mainline GGUF reserves quant type-id `36` for the
/// (now-removed) `IQ4_NL_4_4`, whereas this artifact uses id `36` for Microsoft's
/// **BitNet `I2_S`** quant — a fork-specific assignment. Mainline errors with
/// "tensor 'blk.0.ffn_down.weight' of type 36 … not a multiple of block size (0)".
/// Loading I2_S requires the **bitnet.cpp** fork (or a re-quantized GGUF), so a
/// mainline `BuiltOnBox` figure for *this exact artifact* is not obtainable; the
/// published number stands as the committed fallback, exactly as the plan allows
/// ("if the build fails, record published numbers"). Replace with a `BuiltOnBox`
/// figure if bitnet.cpp (which owns the I2_S kernels) is built on the box later.
///
/// Citation: <https://github.com/ggml-org/llama.cpp> (BitNet / I2_S support; figures
/// from community benchmarks of the 2B4T GGUF, 2025).
pub const LLAMA_CPP_2B4T_DECODE: Baseline = Baseline {
    name: "llama.cpp 2B4T decode (published)",
    tokens_per_sec: 18.0,
    source: BaselineSource::Published(
        "https://github.com/ggml-org/llama.cpp — BitNet 2B4T I2_S community benches (2025)",
    ),
};

/// The fractional tokens/sec drop that fails the perf-regression gate. ADR 0005:
/// "Perf-regression job fails on a `>5%` tokens/sec drop vs the recorded baseline."
pub const REGRESSION_DROP_THRESHOLD: f64 = 0.05;

/// Outcome of comparing a measured tokens/sec against a [`Baseline`].
#[derive(Debug, Clone, PartialEq)]
pub struct RegressionReport {
    /// The baseline compared against.
    pub baseline_name: &'static str,
    /// The baseline's recorded tokens/sec.
    pub baseline_tps: f64,
    /// The measured tokens/sec.
    pub measured_tps: f64,
    /// `(baseline - measured) / baseline` — positive is a slowdown. Negative
    /// (a speedup) never trips the gate.
    pub drop_fraction: f64,
    /// `true` if `drop_fraction` exceeds [`REGRESSION_DROP_THRESHOLD`].
    pub regressed: bool,
}

/// Compare `measured_tps` against `baseline`, returning a [`RegressionReport`]. The
/// gate trips only on a **slowdown** larger than [`REGRESSION_DROP_THRESHOLD`]; a
/// speedup (negative drop) is always fine. A non-positive baseline tokens/sec is
/// treated as "no baseline" (`regressed = false`) rather than dividing by zero.
///
/// A non-positive baseline silently disables the gate, which would be a footgun if it
/// happened by accident, so a debug build asserts the baseline is positive. The
/// committed baselines are all positive (the `published_baselines_carry_a_citation`
/// test pins that), so this only catches a future mis-edit, never the shipped path.
#[must_use]
pub fn check_regression(measured_tps: f64, baseline: &Baseline) -> RegressionReport {
    debug_assert!(
        baseline.tokens_per_sec > 0.0,
        "baseline `{}` has non-positive tokens/sec {} — the regression gate would be a no-op",
        baseline.name,
        baseline.tokens_per_sec,
    );
    let drop_fraction = if baseline.tokens_per_sec > 0.0 {
        (baseline.tokens_per_sec - measured_tps) / baseline.tokens_per_sec
    } else {
        0.0
    };
    RegressionReport {
        baseline_name: baseline.name,
        baseline_tps: baseline.tokens_per_sec,
        measured_tps,
        drop_fraction,
        regressed: drop_fraction > REGRESSION_DROP_THRESHOLD,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use tritium_format::{IMMA_WTILE_BYTES, TQ2_0_BLOCK_BYTES, num_blocks};

    #[test]
    fn tq2_0_fixture_has_the_packed_row_length() {
        let (n, k) = (8, 256);
        let packed = packed_tq2_0_weights(n, k);
        assert_eq!(packed.len(), n * num_blocks(k) * TQ2_0_BLOCK_BYTES);
    }

    #[test]
    fn i2s_int8_fixture_has_the_tile_packed_length() {
        // Non-tile-multiple N and K to exercise the padding path.
        let (n, k) = (20, 6912);
        let bytes = packed_i2s_int8_weights(n, k);
        let num_ntiles = n.div_ceil(IMMA_N);
        let num_ktiles = k.div_ceil(IMMA_K);
        assert_eq!(bytes.len(), num_ntiles * num_ktiles * IMMA_WTILE_BYTES);
    }

    #[test]
    fn i2s_int8_padding_codes_are_trit_zero() {
        // N=2 < IMMA_N=8: the 6 padding rows of the single n-tile must stay code 1.
        let bytes = packed_i2s_int8_weights(2, IMMA_K);
        // Element (n_in=7, k_in=0) is padding → code 1 (the 0x55 default).
        let elem = 7 * IMMA_K; // = 224
        let code = (bytes[elem / 4] >> (2 * (elem % 4))) & 0b11;
        assert_eq!(code, 1, "padded row must decode to trit 0 (code 1)");
    }

    #[test]
    fn shape_grid_is_the_full_product() {
        assert_eq!(BITNET_SHAPES.len(), BITNET_M.len() * BITNET_NK.len());
        assert_eq!(bitnet_gemm_shapes().len(), BITNET_SHAPES.len());
        // Every committed shape is in the M × (N,K) product.
        for &(m, n, k) in BITNET_SHAPES {
            assert!(BITNET_M.contains(&m), "M={m} not in BITNET_M");
            assert!(
                BITNET_NK.contains(&(n, k)),
                "(N,K)=({n},{k}) not in BITNET_NK"
            );
        }
    }

    #[test]
    fn decode_ceiling_is_bandwidth_over_bytes() {
        // 1008 GB/s over ~1.106 GiB ≈ 848 tok/s.
        let ceil = bitnet_2b4t_decode_ceiling();
        let expect = RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC / BITNET_2B4T_I2S_BYTES as f64;
        assert!((ceil - expect).abs() < 1e-6);
        assert!(
            (840.0..860.0).contains(&ceil),
            "BitNet 2B4T decode ceiling {ceil:.1} tok/s out of expected ~848 band"
        );
    }

    #[test]
    fn zero_byte_model_ceiling_is_infinite() {
        assert!(decode_bandwidth_ceiling(RTX_4090_PEAK_HBM_BW_BYTES_PER_SEC, 0).is_infinite());
    }

    #[test]
    fn regression_gate_trips_only_past_threshold() {
        let base = &BITNET_CPP_2B4T_DECODE; // 28.0 tok/s
        // A 4% drop is under the 5% gate → no regression.
        let ok = check_regression(28.0 * 0.96, base);
        assert!(!ok.regressed, "4% drop must not trip the >5% gate");
        // A 6% drop trips it.
        let bad = check_regression(28.0 * 0.94, base);
        assert!(bad.regressed, "6% drop must trip the gate");
        assert!(bad.drop_fraction > REGRESSION_DROP_THRESHOLD);
        // A speedup never trips it.
        let fast = check_regression(28.0 * 1.5, base);
        assert!(!fast.regressed);
        assert!(fast.drop_fraction < 0.0);
    }

    #[test]
    fn published_baselines_carry_a_citation() {
        for b in [&BITNET_CPP_2B4T_DECODE, &LLAMA_CPP_2B4T_DECODE] {
            match b.source {
                BaselineSource::Published(cite) => {
                    assert!(cite.contains("http"), "{} cite must be a URL", b.name);
                }
                BaselineSource::BuiltOnBox => {}
            }
            assert!(b.tokens_per_sec > 0.0);
        }
    }
}
