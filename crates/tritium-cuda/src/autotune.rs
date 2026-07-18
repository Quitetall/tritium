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
//!
//! ## Determinism (WF-B)
//!
//! The on-device contraction is an **exact int32 `mma` accumulate**; the only
//! float step is one per-output scale fold. Reordering the K loop or splitting work
//! across warps/sub-tiles cannot change the accumulated integer, and the fold is a
//! fixed-order `float` multiply chain emitted identically for every tile. So *every*
//! candidate this module searches produces bit-identical output — the tuner only
//! ever picks the fastest, never a numerically different, kernel. The search still
//! validates each candidate against the reference (a tile whose launch geometry is
//! mis-shaped would produce a *wrong* result, not merely a slow one), and the
//! cold-cache == warm-cache gate pins JIT == AOT bit-for-bit for a fixed tile.
//!
//! Most items here are consumed only by the `cuda`-gated launcher (`super::cuda`)
//! and the tuner tests; in the default (cpu-only) build the tile/key helpers are
//! exercised solely by unit tests, so the module carries a crate-private
//! `allow(dead_code)` rather than threading `cfg(test)`/`cfg(feature = "cuda")`
//! through every helper.
#![allow(dead_code)]

#[cfg(feature = "cuda")]
use std::path::Path;
use std::path::PathBuf;

use tritium_core::GemmShape;

/// Tuned tile parameters for one IMMA launch configuration.
///
/// Serialised into the on-disk cache (`serde`) under the `cuda` feature, so the
/// field names are part of the cache file format — renaming one invalidates older
/// cache files (which is fine: a parse miss falls back to a re-tune). The serde
/// derives are `cuda`-gated because `serde` is a `cuda`-feature dep (the on-disk
/// cache is only wired up when the GPU backend is compiled); the pure tile/key logic
/// stays buildable + testable on cpu-only lanes without it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "cuda", derive(serde::Serialize, serde::Deserialize))]
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

    /// The config whose rendered source matches the committed AOT kernel
    /// (`kernels/tq2_0_imma.cu`) launch shape **exactly**: one 16×8 `mma` sub-tile
    /// per block, one warp, double-buffered, one k-tile per step. This is the config
    /// the cold-cache (JIT) == warm-cache (AOT) bit-identity test pins, and the
    /// guaranteed-correct fallback when the cache is empty and tuning is disabled.
    pub(crate) const AOT_EQUIVALENT: TileConfig = TileConfig {
        tile_m: 16,
        tile_n: 8,
        tile_k: 32,
        warps: 1,
        stages: 2,
    };

    /// Whether this config is a launchable IMMA tile: every tile dim a positive
    /// multiple of the corresponding `mma` dimension (16/8/32), at least one
    /// warp, and at least two pipeline stages (the rev-4 `cp.async` pipeline's
    /// `wait_group(STAGES-2)` immediate needs a double buffer minimum). Codegen
    /// and the launcher both require this.
    pub(crate) const fn is_valid(&self) -> bool {
        self.tile_m != 0
            && self.tile_m.is_multiple_of(16)
            && self.tile_n != 0
            && self.tile_n.is_multiple_of(8)
            && self.tile_k != 0
            && self.tile_k.is_multiple_of(32)
            && self.warps >= 1
            && self.stages >= 2
    }

    /// Threads per block this config launches (`warps · 32`).
    pub(crate) const fn block_threads(&self) -> u32 {
        self.warps as u32 * 32
    }

    /// The contiguous `(warps_m, warps_n)` warp-grid partition of the block's
    /// `M_SUBTILES × N_SUBTILES` sub-tile grid (rev-4 codegen). Deterministic:
    /// the largest divisor of `n_subtiles` that is `<= warps` becomes `warps_n`
    /// (splitting across N first maximises B-fragment locality), then the
    /// largest divisor of `m_subtiles` that fits the remaining warps becomes
    /// `warps_m`. Always well-formed (both `>= 1`, product `<= warps`); warps
    /// beyond the grid stage but do not compute.
    pub(crate) const fn warp_grid(&self) -> (u16, u16) {
        let m_sub = self.tile_m / 16;
        let n_sub = self.tile_n / 8;
        let w = self.warps as u16;
        let mut wn = if w < n_sub { w } else { n_sub };
        while !n_sub.is_multiple_of(wn) {
            wn -= 1;
        }
        let cap = w / wn;
        let mut wm = if cap < m_sub { cap } else { m_sub };
        while !m_sub.is_multiple_of(wm) {
            wm -= 1;
        }
        (wm, wn)
    }

    /// Static shared bytes the rendered kernel stages: `stages` deep, each holding
    /// the unpacked int8 A (`tile_m·tile_k`) and the PACKED I2sInt8 B tiles
    /// (`tile_n·tile_k/4` — rev-4 codegen keeps B 2-bit-packed in shared and
    /// expands at fragment-load time). Used to prune candidates that would exceed
    /// the per-block shared budget.
    pub(crate) const fn shared_bytes(&self) -> u32 {
        let per_stage =
            self.tile_m as u32 * self.tile_k as u32 + self.tile_n as u32 * self.tile_k as u32 / 4;
        per_stage * self.stages as u32
    }
}

/// The conservative per-block shared-memory ceiling used to prune candidates. The
/// kernel's staging is `__shared__` (static), and 48 KiB is the portable static
/// shared limit (no opt-in to the larger dynamic budget), so a candidate over this
/// would fail to launch — prune it before it reaches the device.
const SHARED_BUDGET_BYTES: u32 = 48 * 1024;

/// The candidate tile configs the search considers, in a **stable** order (the
/// search is a deterministic sweep so the same hardware + shape always evaluates the
/// same set; ties break toward the earlier, smaller tile). Pruned to those that are
/// valid and fit the shared budget.
///
/// The set is intentionally small and correctness-first (WF-B's mandate is "never
/// change numerics", not "win every shape"): the AOT-equivalent single-warp tile,
/// plus a handful of wider M/N tiles and deeper K steps that the int8 tensor cores
/// reward on prefill-shaped problems.
pub(crate) fn candidate_tiles() -> Vec<TileConfig> {
    const RAW: &[TileConfig] = &[
        // The guaranteed-correct anchor (== AOT kernel).
        TileConfig::AOT_EQUIVALENT,
        // Single warp, deeper K step (more mma per sync).
        TileConfig {
            tile_m: 16,
            tile_n: 8,
            tile_k: 64,
            warps: 1,
            stages: 2,
        },
        TileConfig {
            tile_m: 16,
            tile_n: 8,
            tile_k: 128,
            warps: 1,
            stages: 2,
        },
        // Wider N (more output cols per block), 2 warps.
        TileConfig {
            tile_m: 16,
            tile_n: 16,
            tile_k: 32,
            warps: 2,
            stages: 2,
        },
        TileConfig {
            tile_m: 16,
            tile_n: 16,
            tile_k: 64,
            warps: 2,
            stages: 2,
        },
        // Wider M (more output rows per block), 2 warps.
        TileConfig {
            tile_m: 32,
            tile_n: 8,
            tile_k: 32,
            warps: 2,
            stages: 2,
        },
        TileConfig {
            tile_m: 32,
            tile_n: 8,
            tile_k: 64,
            warps: 2,
            stages: 2,
        },
        // Square-ish 32x16, 4 warps — prefill workhorse.
        TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 32,
            warps: 4,
            stages: 2,
        },
        TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 64,
            warps: 4,
            stages: 2,
        },
        // Larger 64x16, 8 warps, triple-buffered.
        TileConfig {
            tile_m: 64,
            tile_n: 16,
            tile_k: 32,
            warps: 8,
            stages: 3,
        },
        // v1.x (post fragment-u32/B-reuse codegen): genuinely large block tiles.
        // The old per-byte fragment packing made big tiles pointless (SIMT pack
        // cost dwarfed the mma); with single-u32 fragment loads and B-fragment
        // reuse the tensor cores finally see back-to-back work, and these are
        // where compute-bound prefill shapes want to be.
        TileConfig {
            tile_m: 64,
            tile_n: 32,
            tile_k: 64,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 32,
            tile_k: 32,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 32,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 64,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 256,
            tile_n: 64,
            tile_k: 32,
            warps: 8,
            stages: 2,
        },
        // rev-4 (cp.async pipeline + packed-B shared): deeper pipelines and the
        // genuinely large tiles the quartered B footprint now affords. These are
        // where a compute-bound M>=256 prefill wants to land.
        TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 64,
            warps: 8,
            stages: 3,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 64,
            warps: 8,
            stages: 4,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 128,
            tile_k: 64,
            warps: 8,
            stages: 3,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 64,
            tile_k: 128,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 256,
            tile_n: 64,
            tile_k: 64,
            warps: 8,
            stages: 2,
        },
        TileConfig {
            tile_m: 64,
            tile_n: 64,
            tile_k: 64,
            warps: 4,
            stages: 3,
        },
        TileConfig {
            tile_m: 128,
            tile_n: 128,
            tile_k: 32,
            warps: 8,
            stages: 4,
        },
    ];
    RAW.iter()
        .copied()
        .filter(|c| c.is_valid() && c.shared_bytes() <= SHARED_BUDGET_BYTES)
        .collect()
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
/// `tq2_0`), the shape bucket, and the CUDA/driver version. The version is
/// load-bearing for invalidation — a toolkit/driver bump can change the JIT'd SASS,
/// so a stale tuned cubin must not be reused across it. It is part of the on-disk
/// **filename** (not just a field), so a version change resolves to a different file
/// and the old one is simply ignored.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub(crate) struct CacheKey {
    pub(crate) arch: String,
    pub(crate) dtype: &'static str,
    pub(crate) bucket: ShapeBucket,
    /// The CUDA driver version (`cuDriverGetVersion`, e.g. `13030` for 13.3), the
    /// invalidation axis. Captured by the launcher in `cuda.rs`.
    pub(crate) cuda_version: u32,
}

/// Codegen revision, part of the cache key: bump when `codegen.rs` changes the
/// rendered kernel's PERFORMANCE characteristics (fragment loads, loop order,
/// staging) so previously tuned winners re-tune instead of persisting a choice
/// made for a different kernel. rev 2 = u32 fragment loads + B-fragment reuse.
/// rev 3 = epilogue association unified with the dp4a family
/// ((float)acc * weight_scale * act_scale) — the ADR 0026 Track P
/// bit-identity contract; outputs shift at ULP level vs rev 2.
/// rev 4 = cp.async staging pipeline + packed-B shared + contiguous warp-grid
/// ownership + full unrolls (outputs bit-identical to rev 3; perf only).
pub(crate) const CODEGEN_REV: u32 = 4;

impl CacheKey {
    /// Stable filesystem-safe string form, e.g.
    /// `sm_89-i2sint8-m5-n2560-k2560-cuda13030-r4`. The CUDA version and codegen
    /// revision suffixes mean a driver bump or a codegen change keys a *different*
    /// file, transparently invalidating stale entries.
    pub(crate) fn to_key_string(&self) -> String {
        format!(
            "{}-{}-m{}-n{}-k{}-cuda{}-r{}",
            self.arch,
            self.dtype,
            self.bucket.m_log2,
            self.bucket.n,
            self.bucket.k,
            self.cuda_version,
            CODEGEN_REV
        )
    }
}

/// The on-disk tuned-config record. Carries the winning [`TileConfig`] plus the key
/// fields it was tuned for, so a loaded entry can be cross-checked against the key
/// it is being looked up under (a defence against a corrupt or mis-named file
/// silently feeding a wrong-shape config to the launcher). Only built under the
/// `cuda` feature (it is the serde shape of the cache file).
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, PartialEq, Eq, serde::Serialize, serde::Deserialize)]
pub(crate) struct TunedEntry {
    /// Cache-format version. Bumped if the record schema changes; a mismatch is
    /// treated as a cache miss (re-tune) rather than a hard error.
    pub(crate) schema: u32,
    pub(crate) arch: String,
    pub(crate) dtype: String,
    pub(crate) m_log2: u8,
    pub(crate) n: u32,
    pub(crate) k: u32,
    pub(crate) cuda_version: u32,
    /// The winning tile.
    pub(crate) tile: TileConfig,
}

#[cfg(feature = "cuda")]
impl TunedEntry {
    /// Current on-disk schema version. Bump on any breaking record-shape change.
    pub(crate) const SCHEMA: u32 = 1;

    /// Build a record for `key`'s winning `tile`.
    pub(crate) fn new(key: &CacheKey, tile: TileConfig) -> Self {
        TunedEntry {
            schema: Self::SCHEMA,
            arch: key.arch.clone(),
            dtype: key.dtype.to_owned(),
            m_log2: key.bucket.m_log2,
            n: key.bucket.n,
            k: key.bucket.k,
            cuda_version: key.cuda_version,
            tile,
        }
    }

    /// Whether this record was tuned for exactly `key` (schema + every key field).
    /// A loaded record that fails this is ignored (treated as a miss).
    pub(crate) fn matches(&self, key: &CacheKey) -> bool {
        self.schema == Self::SCHEMA
            && self.arch == key.arch
            && self.dtype == key.dtype
            && self.m_log2 == key.bucket.m_log2
            && self.n == key.bucket.n
            && self.k == key.bucket.k
            && self.cuda_version == key.cuda_version
    }
}

/// Path of the on-disk cache file for `key` under `dir`: `<key_string>.json`.
#[cfg(feature = "cuda")]
pub(crate) fn cache_file(dir: &Path, key: &CacheKey) -> PathBuf {
    dir.join(format!("{}.json", key.to_key_string()))
}

/// Load the cached winning tile for `key`, if a valid matching record exists.
///
/// Returns `None` on any miss — file absent, unreadable, malformed JSON, schema
/// mismatch, or a record whose embedded key fields disagree with `key`. A miss is
/// never an error here: the caller re-tunes (or falls back to the AOT-equivalent),
/// so a corrupt cache file degrades to a re-tune, not a launch failure.
#[cfg(feature = "cuda")]
pub(crate) fn load_cached(dir: &Path, key: &CacheKey) -> Option<TileConfig> {
    let path = cache_file(dir, key);
    let bytes = std::fs::read(&path).ok()?;
    let entry: TunedEntry = serde_json::from_slice(&bytes).ok()?;
    if entry.matches(key) && entry.tile.is_valid() {
        Some(entry.tile)
    } else {
        None
    }
}

/// Persist the winning `tile` for `key` under `dir`, creating `dir` if needed.
///
/// The write is atomic-ish: serialise to a temp file in the same directory, then
/// `rename` over the target, so a concurrent reader never observes a half-written
/// file (rename is atomic on the same filesystem). A persistence failure is returned
/// to the caller, which logs-and-continues (an un-cached but correct launch is fine).
///
/// # Errors
/// [`std::io::Error`] if the directory cannot be created or the file cannot be
/// written/renamed.
#[cfg(feature = "cuda")]
pub(crate) fn store_cached(dir: &Path, key: &CacheKey, tile: TileConfig) -> std::io::Result<()> {
    std::fs::create_dir_all(dir)?;
    let entry = TunedEntry::new(key, tile);
    let json = serde_json::to_vec_pretty(&entry)
        .map_err(|e| std::io::Error::new(std::io::ErrorKind::InvalidData, e))?;
    // Temp file in the same dir so the rename stays on one filesystem. Include the
    // process id so two concurrent tuners writing the same key do not clobber each
    // other's temp file before their respective renames.
    let tmp = dir.join(format!(
        ".{}.{}.tmp",
        key.to_key_string(),
        std::process::id()
    ));
    std::fs::write(&tmp, &json)?;
    std::fs::rename(&tmp, cache_file(dir, key))
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

/// Outcome of evaluating one candidate tile on the device.
#[cfg(feature = "cuda")]
#[derive(Debug, Clone, Copy)]
pub(crate) struct CandidateResult {
    /// Whether the candidate produced a result within the vs-reference tolerance.
    /// A `false` here means the launch geometry is mis-shaped (a correctness bug in
    /// the candidate), so the tile is rejected outright regardless of speed.
    pub(crate) correct: bool,
    /// Median wall-clock seconds over the timing repetitions (lower is better). Only
    /// meaningful when `correct`.
    pub(crate) seconds: f64,
}

/// Run the autotune policy for `key`: consult the on-disk cache, and on a miss run
/// `evaluate` over every [`candidate_tiles`] entry, keep the fastest **correct**
/// one, persist it, and return it.
///
/// `evaluate` is the device-specific half (supplied by `cuda.rs`): given a candidate
/// tile it JIT-compiles + launches it on the tuning shape and reports correctness +
/// timing. Keeping it a callback lets all the *policy* (cache I/O, candidate set,
/// winner selection, determinism guarantees) live here and stay unit-testable
/// without a GPU.
///
/// Selection is deterministic: candidates are evaluated in [`candidate_tiles`]
/// order, and ties (equal median time) keep the **earlier** candidate, so the same
/// hardware + shape always yields the same winner. If no candidate is correct (which
/// would indicate a codegen/launch bug, not a tuning outcome), returns
/// [`TileConfig::AOT_EQUIVALENT`] — the guaranteed-correct anchor — without caching
/// it, so the next run re-attempts the search.
///
/// `dir` is the cache directory ([`cache_dir`]). A cache write failure is reported
/// through `on_store_err` (so the caller can log it) but never aborts the launch.
#[cfg(feature = "cuda")]
pub(crate) fn tune_or_load(
    dir: &Path,
    key: &CacheKey,
    mut evaluate: impl FnMut(TileConfig) -> CandidateResult,
    mut on_store_err: impl FnMut(std::io::Error),
) -> TileConfig {
    if let Some(cached) = load_cached(dir, key) {
        return cached;
    }

    let mut best: Option<(TileConfig, f64)> = None;
    for cand in candidate_tiles() {
        let r = evaluate(cand);
        if !r.correct {
            continue;
        }
        // Strictly-less keeps the earlier candidate on ties (deterministic).
        match best {
            Some((_, t)) if t <= r.seconds => {}
            _ => best = Some((cand, r.seconds)),
        }
    }

    match best {
        Some((winner, _)) => {
            if let Err(e) = store_cached(dir, key, winner) {
                on_store_err(e);
            }
            winner
        }
        // No candidate validated: fall back to the AOT-equivalent anchor without
        // caching, so a transient failure does not poison the cache.
        None => TileConfig::AOT_EQUIVALENT,
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn m_buckets_by_floor_log2() {
        assert_eq!(
            ShapeBucket::from_shape(GemmShape { m: 0, n: 1, k: 1 }).m_log2,
            0
        );
        assert_eq!(
            ShapeBucket::from_shape(GemmShape { m: 1, n: 1, k: 1 }).m_log2,
            0
        );
        assert_eq!(
            ShapeBucket::from_shape(GemmShape { m: 2, n: 1, k: 1 }).m_log2,
            1
        );
        assert_eq!(
            ShapeBucket::from_shape(GemmShape { m: 31, n: 1, k: 1 }).m_log2,
            4
        );
        assert_eq!(
            ShapeBucket::from_shape(GemmShape { m: 32, n: 1, k: 1 }).m_log2,
            5
        );
        assert_eq!(
            ShapeBucket::from_shape(GemmShape {
                m: 4096,
                n: 1,
                k: 1
            })
            .m_log2,
            12
        );
    }

    /// A representative key for the cache tests.
    fn sample_key() -> CacheKey {
        CacheKey {
            arch: "sm_89".to_owned(),
            dtype: "i2sint8",
            bucket: ShapeBucket::from_shape(GemmShape {
                m: 40,
                n: 2560,
                k: 2560,
            }),
            cuda_version: 13030,
        }
    }

    #[test]
    fn key_string_is_stable() {
        // M=40 → floor(log2)=5; the CUDA version is part of the key string so a
        // driver bump invalidates the on-disk entry.
        assert_eq!(
            sample_key().to_key_string(),
            "sm_89-i2sint8-m5-n2560-k2560-cuda13030-r4"
        );
    }

    #[test]
    fn cuda_version_separates_keys() {
        let mut a = sample_key();
        let mut b = sample_key();
        a.cuda_version = 13030;
        b.cuda_version = 14000;
        assert_ne!(a.to_key_string(), b.to_key_string());
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

    #[test]
    fn aot_equivalent_is_single_warp_single_subtile() {
        let t = TileConfig::AOT_EQUIVALENT;
        assert!(t.is_valid());
        assert_eq!(t.tile_m, 16);
        assert_eq!(t.tile_n, 8);
        assert_eq!(t.tile_k, 32);
        assert_eq!(t.warps, 1);
        assert_eq!(t.block_threads(), 32);
    }

    #[test]
    fn is_valid_rejects_misaligned_or_empty() {
        let ok = TileConfig::AOT_EQUIVALENT;
        assert!(ok.is_valid());
        // Non-mma-multiple tile dims.
        assert!(!TileConfig { tile_m: 24, ..ok }.is_valid());
        assert!(!TileConfig { tile_n: 12, ..ok }.is_valid());
        assert!(!TileConfig { tile_k: 48, ..ok }.is_valid());
        // Zero dims / no warps / not enough pipeline stages for the cp.async
        // wait_group immediate (rev 4 requires a double buffer minimum).
        assert!(!TileConfig { tile_m: 0, ..ok }.is_valid());
        assert!(!TileConfig { warps: 0, ..ok }.is_valid());
        assert!(!TileConfig { stages: 0, ..ok }.is_valid());
        assert!(!TileConfig { stages: 1, ..ok }.is_valid());
    }

    #[test]
    fn candidate_set_is_nonempty_valid_and_fits_budget() {
        let cands = candidate_tiles();
        assert!(!cands.is_empty(), "candidate set must not be empty");
        assert_eq!(
            cands[0],
            TileConfig::AOT_EQUIVALENT,
            "the AOT-equivalent anchor must be first (deterministic tie-break floor)"
        );
        for c in &cands {
            assert!(c.is_valid(), "candidate {c:?} is not a valid tile");
            assert!(
                c.shared_bytes() <= SHARED_BUDGET_BYTES,
                "candidate {c:?} exceeds the shared budget"
            );
        }
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn cache_roundtrips_and_versions_invalidate() {
        // Use a unique temp dir so the test never touches the user's real cache and
        // does not race other tests.
        let dir = std::env::temp_dir().join(format!(
            "tritium-autotune-test-{}-{}",
            std::process::id(),
            // Distinct per call within the process.
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);

        let key = sample_key();
        // Cold: nothing cached yet.
        assert!(load_cached(&dir, &key).is_none());

        let tile = TileConfig {
            tile_m: 32,
            tile_n: 16,
            tile_k: 64,
            warps: 4,
            stages: 2,
        };
        store_cached(&dir, &key, tile).expect("store cached tile");

        // Warm: the same key reads back the exact tile.
        assert_eq!(load_cached(&dir, &key), Some(tile));

        // A different CUDA version is a different key string → a miss (invalidation).
        let mut bumped = key.clone();
        bumped.cuda_version += 10;
        assert!(
            load_cached(&dir, &bumped).is_none(),
            "a driver-version bump must invalidate the cached entry"
        );

        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn corrupt_cache_file_is_a_miss_not_an_error() {
        let dir = std::env::temp_dir().join(format!(
            "tritium-autotune-corrupt-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        let key = sample_key();
        std::fs::write(cache_file(&dir, &key), b"not valid json {{{").unwrap();
        // A garbage file degrades to a re-tune, never a panic / error.
        assert!(load_cached(&dir, &key).is_none());
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tune_or_load_picks_fastest_correct_and_caches() {
        let dir = std::env::temp_dir().join(format!(
            "tritium-autotune-tune-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let key = sample_key();

        let cands = candidate_tiles();
        assert!(cands.len() >= 3, "need a few candidates for this test");
        // Make the *third* candidate the fastest, the first incorrect (rejected).
        let fast = cands[2];
        let bad = cands[0];
        let winner = tune_or_load(
            &dir,
            &key,
            |c| {
                if c == bad {
                    CandidateResult {
                        correct: false,
                        seconds: 0.0,
                    }
                } else if c == fast {
                    CandidateResult {
                        correct: true,
                        seconds: 0.1,
                    }
                } else {
                    CandidateResult {
                        correct: true,
                        seconds: 1.0,
                    }
                }
            },
            |e| panic!("unexpected store error: {e}"),
        );
        assert_eq!(
            winner, fast,
            "tuner must pick the fastest correct candidate"
        );
        // It must have been persisted, so a second call reads it back without
        // evaluating anything (the closure would panic if called).
        let again = tune_or_load(
            &dir,
            &key,
            |_| panic!("warm cache must not re-evaluate candidates"),
            |e| panic!("unexpected store error: {e}"),
        );
        assert_eq!(again, fast);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tune_breaks_ties_toward_earlier_candidate() {
        let dir = std::env::temp_dir().join(format!(
            "tritium-autotune-tie-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let key = sample_key();
        // Every candidate equally fast and correct: the FIRST one wins (the
        // AOT-equivalent anchor), making the search deterministic.
        let winner = tune_or_load(
            &dir,
            &key,
            |_| CandidateResult {
                correct: true,
                seconds: 0.5,
            },
            |_| {},
        );
        assert_eq!(winner, candidate_tiles()[0]);
        assert_eq!(winner, TileConfig::AOT_EQUIVALENT);
        std::fs::remove_dir_all(&dir).ok();
    }

    #[cfg(feature = "cuda")]
    #[test]
    fn tune_falls_back_when_nothing_correct() {
        let dir = std::env::temp_dir().join(format!(
            "tritium-autotune-nofit-{}-{}",
            std::process::id(),
            line!()
        ));
        let _ = std::fs::remove_dir_all(&dir);
        let key = sample_key();
        let winner = tune_or_load(
            &dir,
            &key,
            |_| CandidateResult {
                correct: false,
                seconds: 0.0,
            },
            |_| {},
        );
        // No candidate validated → the guaranteed-correct anchor, and nothing cached.
        assert_eq!(winner, TileConfig::AOT_EQUIVALENT);
        assert!(
            load_cached(&dir, &key).is_none(),
            "a no-correct-candidate run must not poison the cache"
        );
        std::fs::remove_dir_all(&dir).ok();
    }
}
