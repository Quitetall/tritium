//! Autotune tile config + on-disk cache keying for the IMMA kernel (v0.30,
//! ADR 0005 / WF-B — skeleton).
//!
//! The IMMA (`mma.m16n8k32`) kernel is templated over its tile shape (M/N/K,
//! warps, pipeline stages); the winning [`TileConfig`] per (GPU arch, weight
//! dtype, shape bucket) is searched once and cached on disk so later runs reuse
//! it. This module fixes the **cache key** and the **config type** now (both pure
//! Rust, testable on cpu-only lanes); WF-B adds the nvrtc codegen
//! ([`super::codegen`]), the tile search, and serde-backed persistence under the
//! [`cache_dir`].
//!
//! Autotuning must never change numerics: every candidate tile is held to the
//! vs-reference gate, so a tuned config is bit-equivalent to the untuned kernel
//! (cold-cache == warm-cache). The key includes the CUDA/driver version (WF-B) so
//! a toolkit bump invalidates stale entries rather than loading an ABI-mismatched
//! cubin.
#![allow(dead_code)] // skeleton: consumed by the WF-B kernel selection, not yet wired.

use std::path::PathBuf;

use tritium_core::GemmShape;

/// Tuned tile parameters for one IMMA launch configuration.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) struct TileConfig {
    /// Output rows per block tile (multiple of the `mma` M = 16).
    pub(crate) tile_m: u16,
    /// Output cols per block tile (multiple of the `mma` N = 8).
    pub(crate) tile_n: u16,
    /// Contraction depth per main-loop step (multiple of the `mma` K = 32).
    pub(crate) tile_k: u16,
    /// Warps per block.
    pub(crate) warps: u8,
    /// Software-pipeline stages (double-buffer = 2).
    pub(crate) stages: u8,
}

impl TileConfig {
    /// A conservative default that is valid on every supported arch (one warp,
    /// double-buffered, minimal `mma`-aligned tile). WF-B's search starts here.
    pub(crate) const BASELINE: TileConfig = TileConfig {
        tile_m: 16,
        tile_n: 8,
        tile_k: 32,
        warps: 4,
        stages: 2,
    };
}

/// Coarse shape bucket for cache keying. `N`/`K` are fixed per weight tensor so
/// they key exactly; `M` (token count) varies per call, so it is bucketed by
/// `floor(log2)` — this is also where the decode-vs-prefill crossover (~M=32)
/// lives, so adjacent `M` reuse the same tuned config.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub(crate) struct ShapeBucket {
    /// `floor(log2(max(1, M)))` — 0 for M∈{0,1}, 5 at M=32, etc.
    pub(crate) m_log2: u8,
    /// Output channels.
    pub(crate) n: u32,
    /// Input features.
    pub(crate) k: u32,
}

impl ShapeBucket {
    pub(crate) fn from_shape(shape: GemmShape) -> Self {
        let m = shape.m.max(1);
        // floor(log2): bit width minus one. usize::BITS - leading_zeros - 1.
        let m_log2 = (usize::BITS - 1 - m.leading_zeros()) as u8;
        // N/K key the cache as u32; real weight dims are <<4B, but make the
        // assumption explicit rather than silently truncating.
        debug_assert!(
            shape.n <= u32::MAX as usize && shape.k <= u32::MAX as usize,
            "weight dims {}x{} exceed the u32 cache-key range",
            shape.n,
            shape.k
        );
        ShapeBucket {
            m_log2,
            n: shape.n as u32,
            k: shape.k as u32,
        }
    }
}

/// The full cache key: GPU arch (`sm_89`), weight dtype tag (`i2sint8` /
/// `tq2_0`), and the shape bucket. WF-B appends the CUDA/driver version.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    pub(crate) arch: String,
    pub(crate) dtype: &'static str,
    pub(crate) bucket: ShapeBucket,
}

impl CacheKey {
    /// Stable filesystem-safe string form, e.g. `sm_89-i2sint8-m5-n2560-k2560`.
    pub(crate) fn to_key_string(&self) -> String {
        format!(
            "{}-{}-m{}-n{}-k{}",
            self.arch, self.dtype, self.bucket.m_log2, self.bucket.n, self.bucket.k
        )
    }
}

/// On-disk autotune cache directory: `$XDG_CACHE_HOME/tritium` if set, else
/// `$HOME/.cache/tritium`, else a relative `.cache/tritium` fallback. WF-B
/// persists tuned [`TileConfig`]s here keyed by [`CacheKey::to_key_string`].
pub(crate) fn cache_dir() -> PathBuf {
    if let Some(xdg) = std::env::var_os("XDG_CACHE_HOME") {
        return PathBuf::from(xdg).join("tritium");
    }
    if let Some(home) = std::env::var_os("HOME") {
        return PathBuf::from(home).join(".cache").join("tritium");
    }
    PathBuf::from(".cache").join("tritium")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_buckets_by_floor_log2() {
        assert_eq!(ShapeBucket::from_shape(GemmShape { m: 0, n: 1, k: 1 }).m_log2, 0);
        assert_eq!(ShapeBucket::from_shape(GemmShape { m: 1, n: 1, k: 1 }).m_log2, 0);
        assert_eq!(ShapeBucket::from_shape(GemmShape { m: 2, n: 1, k: 1 }).m_log2, 1);
        assert_eq!(ShapeBucket::from_shape(GemmShape { m: 31, n: 1, k: 1 }).m_log2, 4);
        assert_eq!(ShapeBucket::from_shape(GemmShape { m: 32, n: 1, k: 1 }).m_log2, 5);
        assert_eq!(ShapeBucket::from_shape(GemmShape { m: 4096, n: 1, k: 1 }).m_log2, 12);
    }

    #[test]
    fn key_string_is_stable() {
        let key = CacheKey {
            arch: "sm_89".to_owned(),
            dtype: "i2sint8",
            bucket: ShapeBucket::from_shape(GemmShape { m: 40, n: 2560, k: 2560 }),
        };
        // M=40 → floor(log2)=5.
        assert_eq!(key.to_key_string(), "sm_89-i2sint8-m5-n2560-k2560");
    }

    #[test]
    fn cache_dir_ends_in_tritium() {
        // Process env is global + shared across the test binary, so we don't mutate
        // XDG_CACHE_HOME/HOME here (that would race other tests); just assert the
        // resolved path ends in `tritium` for whichever branch the env selects.
        let dir = cache_dir();
        assert_eq!(dir.file_name().unwrap(), "tritium");
    }

    #[test]
    fn baseline_tile_is_mma_aligned() {
        let t = TileConfig::BASELINE;
        assert_eq!(t.tile_m % 16, 0);
        assert_eq!(t.tile_n % 8, 0);
        assert_eq!(t.tile_k % 32, 0);
    }
}
