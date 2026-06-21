//! Deterministic, resumable, sharded data sampler for distributed pretraining (plan 0012).
//!
//! Given a global sample count `N`, a `seed`, and a `(rank, n_ranks)` partition, the sampler
//! yields *this rank's* global sample indices for each epoch such that:
//! - **deterministic:** the per-epoch order is a pure function of `(seed, epoch, n_samples)` —
//!   integer math, no float, no hash-iteration order. A saved [`Cursor`] reproduces the same order
//!   only against the *same* `n_samples`; resizing the corpus reshuffles *every* index (the
//!   Fisher–Yates draw order shifts), not just the added/removed tail;
//! - **no duplication / no loss:** across all ranks, one epoch covers `0..N` exactly once
//!   (minus the dropped ragged tail when `drop_last`);
//! - **resumable:** the [`Cursor`] `(seed, epoch, consumed)` restores the exact remaining order.
//!
//! Per epoch the permutation is a Fisher–Yates shuffle of `0..N` driven by a `splitmix64`-seeded
//! `xorshift64` stream (a biased-modulo bounded draw — the ~`N/2⁶⁴` non-uniformity is negligible
//! for a data shuffle); rank `r` takes the **strided** positions `r, r+n_ranks, r+2·n_ranks, …`.

/// A resumable position in the sample stream. Saved alongside the optimizer step so a
/// training checkpoint resumes data + weights together (wired in 0013).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Cursor {
    /// Global shuffle seed (fixed for the run).
    pub seed: u64,
    /// Current epoch (0-based).
    pub epoch: u64,
    /// Samples this rank has already consumed *within* `epoch`.
    pub consumed: usize,
}

/// A deterministic sharded sampler over `n_samples` global samples for one `rank` of `n_ranks`.
#[derive(Clone, Debug)]
pub struct DataSampler {
    n_samples: usize,
    n_ranks: usize,
    rank: usize,
    /// Drop the ragged tail so every rank gets exactly `n_samples / n_ranks` samples per epoch
    /// (the synchronized-training default). When `false`, ranks get uneven counts and the union
    /// is all of `0..N`.
    drop_last: bool,
    cursor: Cursor,
    /// Memoized permutation for `(cached_seed, cached_epoch)`, lazily built by [`Self::next_index`]
    /// and rebuilt only when the epoch rolls or the seed changes — so streaming a whole epoch is
    /// O(N) total, not O(N²). `None` until the first `next_index` call.
    cache: Option<(u64, u64, Vec<usize>)>,
}

/// splitmix64 finalizer — turns `(seed, epoch)` into a well-mixed per-epoch seed.
fn epoch_seed(seed: u64, epoch: u64) -> u64 {
    let mut z = seed.wrapping_add(epoch.wrapping_mul(0x9E37_79B9_7F4A_7C15));
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^ (z >> 31)
}

impl DataSampler {
    /// Construct a sampler. `rank < n_ranks`, `n_ranks > 0`, and — when `drop_last` — `n_ranks <=
    /// n_samples` (else every rank's `floor(n_samples / n_ranks)` is 0 and the stream is silently
    /// empty; see the panic below).
    ///
    /// # Panics
    /// If `n_ranks == 0`, `rank >= n_ranks`, or `drop_last && n_ranks > n_samples` (a config that
    /// would consume zero data per epoch — fail fast rather than run a dead loop).
    #[must_use]
    pub fn new(n_samples: usize, n_ranks: usize, rank: usize, drop_last: bool, seed: u64) -> Self {
        assert!(n_ranks > 0, "n_ranks must be > 0");
        assert!(
            rank < n_ranks,
            "rank {rank} out of range for n_ranks {n_ranks}"
        );
        assert!(
            !drop_last || n_ranks <= n_samples,
            "drop_last=true with n_ranks ({n_ranks}) > n_samples ({n_samples}) gives every rank 0 \
             samples per epoch — the training loop would consume zero data; reduce n_ranks, add \
             data, or use drop_last=false"
        );
        Self {
            n_samples,
            n_ranks,
            rank,
            drop_last,
            cursor: Cursor {
                seed,
                epoch: 0,
                consumed: 0,
            },
            cache: None,
        }
    }

    /// The current resume cursor.
    #[must_use]
    pub fn cursor(&self) -> Cursor {
        self.cursor
    }

    /// Restore a previously-saved cursor (resume mid-epoch).
    pub fn set_cursor(&mut self, cursor: Cursor) {
        self.cursor = cursor;
    }

    /// Samples each rank gets per epoch: `floor(N/n_ranks)` when `drop_last`, else this rank's
    /// strided count `ceil((N − rank)/n_ranks)`.
    #[must_use]
    pub fn samples_per_epoch(&self) -> usize {
        if self.drop_last {
            self.n_samples / self.n_ranks
        } else if self.rank >= self.n_samples {
            0
        } else {
            (self.n_samples - self.rank).div_ceil(self.n_ranks)
        }
    }

    /// The full Fisher–Yates permutation of `0..n_samples` for `epoch` (deterministic).
    #[must_use]
    pub fn epoch_permutation(&self, epoch: u64) -> Vec<usize> {
        let mut perm: Vec<usize> = (0..self.n_samples).collect();
        let mut s = epoch_seed(self.cursor.seed, epoch) | 1;
        let mut next = || {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            s
        };
        // Fisher–Yates: for i from N-1 down to 1, swap i with j = rng % (i+1) in [0, i] (a
        // biased-modulo draw; the ~i/2⁶⁴ non-uniformity is irrelevant for a data shuffle).
        for i in (1..self.n_samples).rev() {
            let j = (next() % (i as u64 + 1)) as usize;
            perm.swap(i, j);
        }
        perm
    }

    /// This rank's sample indices for `epoch`, in consumption order (strided over the permutation).
    #[must_use]
    pub fn epoch_indices(&self, epoch: u64) -> Vec<usize> {
        let perm = self.epoch_permutation(epoch);
        let count = self.samples_per_epoch();
        (0..count)
            .map(|i| perm[self.rank + i * self.n_ranks])
            .collect()
    }

    /// Yield the next global sample index, advancing the cursor. Rolls to the next epoch (a fresh
    /// deterministic shuffle) when the current one is exhausted. Returns `None` only for a
    /// degenerate empty rank — which, given `new`'s `drop_last` guard, can happen only when
    /// `drop_last == false` and `rank >= n_samples` (the strided stripe is empty); that rank never
    /// advances.
    ///
    /// The per-epoch permutation is memoized, so streaming a whole epoch is O(N) total — the shuffle
    /// is rebuilt only when the epoch rolls or [`Self::set_cursor`] moves to a different epoch/seed.
    pub fn next_index(&mut self) -> Option<usize> {
        let per_epoch = self.samples_per_epoch();
        if per_epoch == 0 {
            return None;
        }
        if self.cursor.consumed >= per_epoch {
            self.cursor.epoch += 1;
            self.cursor.consumed = 0;
        }
        // Rebuild the memoized permutation only when it no longer matches the current (seed, epoch).
        let stale = !matches!(
            &self.cache,
            Some((s, e, _)) if *s == self.cursor.seed && *e == self.cursor.epoch
        );
        if stale {
            let perm = self.epoch_permutation(self.cursor.epoch);
            self.cache = Some((self.cursor.seed, self.cursor.epoch, perm));
        }
        let idx = match &self.cache {
            Some((_, _, perm)) => perm[self.rank + self.cursor.consumed * self.n_ranks],
            None => return None, // unreachable: just populated above
        };
        self.cursor.consumed += 1;
        Some(idx)
    }
}
