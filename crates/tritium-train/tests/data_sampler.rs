//! Data-pipeline gate (ADR 0008 / plan 0012): the sharded sampler is deterministic per
//! seed, loses/duplicates no sample across ranks, and resumes mid-epoch exactly.

use proptest::prelude::*;
use std::collections::BTreeSet;
use tritium_train::DataSampler;

proptest! {
    /// Same (seed, epoch) ⇒ identical permutation, and it is a true permutation of 0..n.
    #[test]
    fn permutation_is_deterministic(n in 1usize..=64, seed in any::<u64>(), epoch in 0u64..5) {
        let a = DataSampler::new(n, 1, 0, false, seed);
        let b = DataSampler::new(n, 1, 0, false, seed);
        let pa = a.epoch_permutation(epoch);
        prop_assert_eq!(&pa, &b.epoch_permutation(epoch));
        let set: BTreeSet<usize> = pa.iter().copied().collect();
        prop_assert_eq!(set.len(), n);          // no duplicates
        prop_assert!(pa.iter().all(|&x| x < n)); // in range
    }

    /// drop_last = false: the union of all ranks' epoch indices is exactly 0..n (no dup, no loss).
    #[test]
    fn coverage_union_is_exact(n in 1usize..=64, nr in 1usize..=8, seed in any::<u64>()) {
        let mut all: Vec<usize> = Vec::new();
        for r in 0..nr {
            all.extend(DataSampler::new(n, nr, r, false, seed).epoch_indices(0));
        }
        all.sort_unstable();
        prop_assert_eq!(all, (0..n).collect::<Vec<_>>());
    }

    /// drop_last = true: every rank gets exactly floor(n/nr) samples, all distinct across ranks
    /// (the ragged tail of n mod nr is dropped, never duplicated).
    #[test]
    fn drop_last_equal_counts_no_dup(n in 1usize..=64, nr in 1usize..=8, seed in any::<u64>()) {
        // `new` rejects drop_last with nr > n (every rank would get 0 samples — see the
        // `drop_last_more_ranks_than_samples_panics` regression); excluding it also keeps this
        // gate non-vacuous (per >= 1 ⇒ every rank draws real work, not an empty set).
        prop_assume!(n >= nr);
        let per = n / nr;
        prop_assert!(per >= 1);
        let mut set = BTreeSet::new();
        for r in 0..nr {
            let idx = DataSampler::new(n, nr, r, true, seed).epoch_indices(0);
            prop_assert_eq!(idx.len(), per);
            for i in idx {
                prop_assert!(set.insert(i), "duplicate sample {i} across ranks");
            }
        }
        prop_assert_eq!(set.len(), per * nr);
    }

    /// Resuming from a mid-epoch cursor yields the exact remaining sequence (== uninterrupted tail),
    /// across an epoch boundary. Covers both `drop_last` modes, and (via `rank` up to 3 with `nr`
    /// down to 1) the `drop_last=false` empty-strided-rank path (`rank >= n`).
    #[test]
    fn resume_continues_exactly(
        n in 2usize..=40, nr in 1usize..=4, rank_sel in 0usize..4,
        drop_last in any::<bool>(),
        seed in any::<u64>(), m in 0usize..30, extra in 0usize..30,
    ) {
        prop_assume!(!drop_last || n >= nr); // `new` rejects drop_last with nr > n
        let rank = rank_sel % nr;
        let drain = |s: &mut DataSampler, k: usize| -> Vec<usize> {
            (0..k).filter_map(|_| s.next_index()).collect()
        };
        let mut uninterrupted = DataSampler::new(n, nr, rank, drop_last, seed);
        let full = drain(&mut uninterrupted, m + extra);

        let mut a = DataSampler::new(n, nr, rank, drop_last, seed);
        let mut head = drain(&mut a, m);
        let cursor = a.cursor();
        let mut b = DataSampler::new(n, nr, rank, drop_last, seed);
        b.set_cursor(cursor);
        head.extend(drain(&mut b, extra));
        prop_assert_eq!(head, full);
    }
}

/// Regression for the silent-zero-data footgun (review 0012 [0]): `drop_last=true` with more ranks
/// than samples gives every rank `floor(n/nr) == 0` — `new` must reject it loudly, not hand back a
/// sampler whose `next_index` returns `None` forever.
#[test]
#[should_panic(expected = "consume zero data")]
fn drop_last_more_ranks_than_samples_panics() {
    let _ = DataSampler::new(10, 16, 0, true, 0xABCD);
}
