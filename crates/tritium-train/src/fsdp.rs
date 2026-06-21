//! ZeRO-3 / FSDP flat-parameter sharding (plan 0015).
//!
//! Fully-Sharded Data Parallel shards a model's parameters, gradients, and optimizer state across
//! the ranks of a [`ProcessGroup`](crate::dist::ProcessGroup): each rank holds only its slice between
//! steps, [`all_gather`](crate::dist::ProcessGroup::all_gather)s the full parameters just-in-time for
//! the forward, and [`reduce_scatter`](crate::dist::ProcessGroup::reduce_scatter)s the gradients back
//! to slices after the backward. This module ships the load-bearing *data-layout* piece —
//! [`FlatShardPlan`], the "FlatParameter" descriptor — and leaves the training-loop orchestration
//! (gather → forward/backward → reduce_scatter → optimizer step) to the caller, which is the minimal
//! clean split: the plan is reused (0016's distributed-checkpoint manifest reshards over it), while a
//! step helper would today have exactly one consumer and is a cheap retrofit when a real loop wants one.
//!
//! **Flat-parameter model.** The trainable leaves are concatenated *in order* into one flat buffer of
//! length `total`, padded up to `padded_len = chunk * world` (with `chunk = ceil(total / world)`), and
//! split into `world` equal contiguous shards. Rank `r` owns `flat[r*chunk .. (r+1)*chunk]`. The
//! padding region `[total, padded_len)` exists only so the flat buffer divides evenly for the
//! collectives; [`FlatShardPlan::unflatten`] returns only the real `[0, total)` region, so the padding
//! values never reach a leaf and are irrelevant to the model (a sharded optimizer may step them
//! against a zero gradient — harmless, never gathered back).
//!
//! **Why this is the bridge to the loss-parity gate.** Gathering shards reconstructs the flat buffer
//! by *concatenation* (a copy, bit-exact), and the collectives fold in a fixed rank order (0014), so
//! FSDP changes only *where* parameters and grads live, not the math — modulo `f32` non-associativity
//! in cross-rank gradient summation, which is exactly what plan 0015's tolerance gate measures.

/// A plan for sharding a set of trainable leaves as one flat parameter buffer (the FSDP
/// "FlatParameter") across `world` ranks.
///
/// Construct once from the leaf lengths; then [`flatten`](Self::flatten) /
/// [`unflatten`](Self::unflatten) convert between the per-leaf view and the padded flat view, and
/// [`shard_range`](Self::shard_range) gives each rank's slice. The plan is pure layout — it holds no
/// data — so it is cheap to clone into each rank's thread.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct FlatShardPlan {
    /// `(offset, len)` of each leaf within the flat buffer, in leaf order.
    segments: Vec<(usize, usize)>,
    /// Unpadded total length `Σ len` — the real parameter count.
    total: usize,
    /// Number of ranks the flat buffer is split across.
    world: usize,
    /// Per-rank shard length `ceil(total / world)`; `chunk * world == padded`.
    chunk: usize,
    /// Padded flat-buffer length `chunk * world` (a multiple of `world`, `>= total`). Stored so the
    /// `chunk * world` multiply is overflow-checked exactly once, in [`new`](Self::new).
    padded: usize,
}

impl FlatShardPlan {
    /// Build a plan for leaves of the given lengths (in order), sharded across `world` ranks.
    ///
    /// # Panics
    /// If `world == 0`, if the concatenated length (`Σ len`) overflows `usize`, or if the padded
    /// length (`chunk * world`) overflows `usize` (only reachable at ~`usize::MAX` total elements).
    #[must_use]
    pub fn new(leaf_lens: &[usize], world: usize) -> Self {
        assert!(world > 0, "world size must be > 0");
        let mut segments = Vec::with_capacity(leaf_lens.len());
        let mut off = 0usize;
        for &len in leaf_lens {
            segments.push((off, len));
            off = off
                .checked_add(len)
                .expect("flat parameter length overflows usize");
        }
        let total = off;
        // `chunk * world` is the smallest multiple of `world` that is >= total, so the flat buffer
        // divides evenly into `world` shards for reduce_scatter / all_gather. `div_ceil` itself is
        // exact (total fits usize; chunk <= total), but `chunk * world` can in principle overflow at
        // ~usize::MAX total — guard it here once, mirroring dist.rs's `LengthOverflow` discipline.
        let chunk = total.div_ceil(world);
        let padded = chunk
            .checked_mul(world)
            .expect("padded flat length (chunk * world) overflows usize");
        Self {
            segments,
            total,
            world,
            chunk,
            padded,
        }
    }

    /// The per-rank shard length (`ceil(total / world)`).
    #[must_use]
    pub fn chunk(&self) -> usize {
        self.chunk
    }

    /// The unpadded real parameter count (`Σ leaf_len`).
    #[must_use]
    pub fn total(&self) -> usize {
        self.total
    }

    /// The padded flat-buffer length (`chunk * world`), a multiple of `world` and `>= total`.
    #[must_use]
    pub fn padded_len(&self) -> usize {
        self.padded
    }

    /// Rank `r`'s shard slice `[r*chunk, (r+1)*chunk)` within the padded flat buffer.
    ///
    /// # Panics
    /// If `rank >= world`.
    #[must_use]
    pub fn shard_range(&self, rank: usize) -> (usize, usize) {
        assert!(
            rank < self.world,
            "rank {rank} out of range for world {}",
            self.world
        );
        let lo = rank * self.chunk;
        (lo, lo + self.chunk)
    }

    /// Pack per-leaf buffers into the padded flat buffer (length [`padded_len`](Self::padded_len)),
    /// with the padding region `[total, padded_len)` zeroed.
    ///
    /// # Panics
    /// If `leaves.len()` or any leaf length does not match the plan.
    #[must_use]
    pub fn flatten(&self, leaves: &[Vec<f32>]) -> Vec<f32> {
        assert_eq!(
            leaves.len(),
            self.segments.len(),
            "leaf count {} != plan leaf count {}",
            leaves.len(),
            self.segments.len()
        );
        let mut flat = vec![0.0f32; self.padded_len()];
        for (leaf, &(off, len)) in leaves.iter().zip(&self.segments) {
            assert_eq!(
                leaf.len(),
                len,
                "leaf length {} != plan segment length {len}",
                leaf.len()
            );
            flat[off..off + len].copy_from_slice(leaf);
        }
        flat
    }

    /// Unpack a padded flat buffer back into per-leaf buffers, dropping the padding.
    ///
    /// # Panics
    /// If `flat.len() != padded_len()`.
    #[must_use]
    pub fn unflatten(&self, flat: &[f32]) -> Vec<Vec<f32>> {
        assert_eq!(
            flat.len(),
            self.padded_len(),
            "flat length {} != padded_len {}",
            flat.len(),
            self.padded_len()
        );
        self.segments
            .iter()
            .map(|&(off, len)| flat[off..off + len].to_vec())
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_unflatten_roundtrip_is_identity() {
        let leaves = vec![vec![1.0, 2.0, 3.0], vec![4.0], vec![5.0, 6.0]];
        let lens: Vec<usize> = leaves.iter().map(Vec::len).collect();
        for world in [1usize, 2, 3, 4, 8] {
            let plan = FlatShardPlan::new(&lens, world);
            let flat = plan.flatten(&leaves);
            assert_eq!(flat.len(), plan.padded_len());
            assert_eq!(
                plan.unflatten(&flat),
                leaves,
                "roundtrip failed (world {world})"
            );
        }
    }

    #[test]
    fn padded_len_is_multiple_of_world_and_ge_total() {
        // total = 6 + 1 + 2 = 9.
        let lens = [6usize, 1, 2];
        for world in [1usize, 2, 4, 5, 9, 16] {
            let plan = FlatShardPlan::new(&lens, world);
            assert_eq!(plan.total(), 9);
            assert!(plan.padded_len() >= plan.total());
            assert_eq!(plan.padded_len() % world, 0);
            assert_eq!(plan.padded_len(), plan.chunk() * world);
            // padded_len is the *smallest* such multiple: dropping one chunk would fall below total.
            assert!(plan.chunk() * world - world < plan.total() || plan.chunk() == 0);
        }
    }

    #[test]
    fn shard_ranges_partition_the_padded_buffer() {
        let lens = [10usize, 11]; // total 21
        let world = 4;
        let plan = FlatShardPlan::new(&lens, world);
        assert_eq!(plan.padded_len(), plan.chunk() * world); // 6*4 = 24, pad 3
        let mut covered = 0;
        let mut prev_hi = 0;
        for r in 0..world {
            let (lo, hi) = plan.shard_range(r);
            assert_eq!(lo, prev_hi, "shard {r} not contiguous with previous");
            assert_eq!(hi - lo, plan.chunk());
            covered += hi - lo;
            prev_hi = hi;
        }
        assert_eq!(prev_hi, plan.padded_len());
        assert_eq!(covered, plan.padded_len());
    }

    #[test]
    fn padding_region_is_zeroed() {
        let lens = [5usize]; // total 5
        let plan = FlatShardPlan::new(&lens, 4); // chunk 2, padded 8 → 3 padding slots
        let flat = plan.flatten(&[vec![1.0, 2.0, 3.0, 4.0, 5.0]]);
        assert_eq!(&flat[..5], &[1.0, 2.0, 3.0, 4.0, 5.0]);
        assert!(
            flat[5..].iter().all(|&x| x == 0.0),
            "padding not zeroed: {:?}",
            &flat[5..]
        );
    }

    #[test]
    fn no_padding_when_total_divisible_by_world() {
        let lens = [4usize, 8]; // total 12
        let plan = FlatShardPlan::new(&lens, 4);
        assert_eq!(plan.chunk(), 3);
        assert_eq!(plan.padded_len(), 12);
        assert_eq!(plan.padded_len(), plan.total());
    }

    #[test]
    #[should_panic(expected = "world size must be > 0")]
    fn world_zero_panics() {
        let _ = FlatShardPlan::new(&[1, 2], 0);
    }
}
