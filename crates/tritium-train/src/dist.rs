//! Distributed collectives for data-/model-parallel training (plan 0014).
//!
//! [`ProcessGroup`] is the collective abstraction a training loop reduces gradients and gathers
//! parameters through. The real backend (`cudarc::nccl`, the 0017 wall) implements it from
//! `tritium-cuda`; this module ships the **thread-simulated** [`SimProcessGroup`] — N logical ranks
//! in N threads sharing a host buffer — so the collective-correctness gate (all-reduced grads == a
//! single-process summed reference) goes green on one machine.
//!
//! **Determinism.** Every reduction folds the per-rank contributions in a *fixed* rank order
//! `0..world`, so the result is independent of thread scheduling and bit-identical to a
//! single-process reference folded the same way (`f32` addition is non-associative — a fixed order
//! is the whole point, and the bridge to 0015's loss-parity and 0017's wire-correctness gates).
//!
//! **Contract.** As with MPI/NCCL, every rank must invoke the *same* collective sequence with
//! consistently-sized buffers; the paired barriers assume lockstep. Each collective is structured
//! "publish first, validate after barrier #1, ALWAYS reach barrier #2": every rank crosses barrier
//! #1 unconditionally (so a local size-relation or root violation can no longer strand peers), does
//! all validation while holding the slots lock, and then crosses barrier #2 unconditionally. The
//! invariant is that **every collective performs exactly two `barrier.wait()` calls on every code
//! path, Ok or Err** — without it a rank that fails a local check would do zero waits and deadlock
//! its peers blocked at barrier #1. Size-relation, per-rank length disagreement, and out-of-range
//! root all return [`DistError`] rather than panicking; ranks calling a *different* collective at
//! the same step are caught by an op-tag check and return [`DistError::CollectiveMismatch`].
//!
//! **Documented limitations of the CI sim (findings 3/8/12).** `std::sync::Barrier` has no
//! break/poison: a rank that *panics* inside a collective (a contract violation under our
//! never-panic discipline, not an expected path) leaves its peers blocked forever at a barrier —
//! there is no way to break the barrier from a panicking rank. Likewise, ranks calling a
//! *different number* of collectives is an uncatchable contract violation: the op-tag guard only
//! catches a type/order mismatch at the *same* step, not a count mismatch (a rank that stops
//! calling collectives early simply leaves the others waiting at the next barrier). These are
//! limitations of the thread sim, accepted because the real NCCL backend has its own
//! timeout/abort machinery; the sim's job is the determinism + size-contract gate.

use std::fmt;
use std::sync::{Arc, Barrier, Mutex};

/// The reduction applied by [`ProcessGroup::all_reduce`] / [`ProcessGroup::reduce_scatter`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ReduceOp {
    /// Element-wise sum across ranks.
    Sum,
    /// Element-wise mean across ranks (`Sum / world_size`).
    Avg,
}

/// Per-collective op tags. Published alongside each rank's data so that ranks invoking *different*
/// collectives at the same step are detected after barrier #1 (see [`DistError::CollectiveMismatch`]).
const TAG_ALL_REDUCE: u8 = 0;
const TAG_REDUCE_SCATTER: u8 = 1;
const TAG_ALL_GATHER: u8 = 2;
const TAG_BROADCAST: u8 = 3;

/// Human-readable name for an op tag (for diagnostics).
fn tag_name(tag: u8) -> &'static str {
    match tag {
        TAG_ALL_REDUCE => "all_reduce",
        TAG_REDUCE_SCATTER => "reduce_scatter",
        TAG_ALL_GATHER => "all_gather",
        TAG_BROADCAST => "broadcast",
        _ => "unknown",
    }
}

/// A collective error. Returned (never panicked) so the never-panic discipline holds at the
/// collective boundary too.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum DistError {
    /// A buffer was the wrong size for the collective (the size relation, or a per-rank disagreement).
    LengthMismatch {
        /// Required length.
        expected: usize,
        /// Length supplied.
        got: usize,
    },
    /// The required buffer length (`world * chunk`) overflowed `usize`. Distinct from
    /// [`DistError::LengthMismatch`] because there is no honest "expected" value to report — the
    /// required length is unrepresentable.
    LengthOverflow {
        /// The world size that was multiplied.
        world: usize,
        /// The per-rank chunk length that was multiplied.
        chunk: usize,
    },
    /// A `broadcast` root was `>= world_size`.
    InvalidRoot {
        /// The out-of-range root.
        root: usize,
        /// The world size.
        world_size: usize,
    },
    /// Ranks invoked *different* collectives at the same step (e.g. one rank called `all_gather`
    /// while another called `broadcast`). Detected after barrier #1 by comparing op tags; every
    /// rank still reaches barrier #2, so this is a clean symmetric error, not a hang.
    CollectiveMismatch {
        /// The op tag this rank invoked.
        expected: u8,
        /// A differing op tag a peer invoked.
        got: u8,
    },
    /// A real backend (e.g. NCCL) failed; carries its message. Unused by the simulated backend.
    Backend(String),
}

impl fmt::Display for DistError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DistError::LengthMismatch { expected, got } => {
                write!(
                    f,
                    "collective length mismatch: expected {expected}, got {got}"
                )
            }
            DistError::LengthOverflow { world, chunk } => {
                write!(
                    f,
                    "collective length overflow: world {world} * chunk {chunk} exceeds usize::MAX"
                )
            }
            DistError::InvalidRoot { root, world_size } => {
                write!(
                    f,
                    "broadcast root {root} out of range for world size {world_size}"
                )
            }
            DistError::CollectiveMismatch { expected, got } => {
                write!(
                    f,
                    "collective mismatch: this rank invoked {} ({expected}) but a peer invoked {} ({got})",
                    tag_name(*expected),
                    tag_name(*got)
                )
            }
            DistError::Backend(m) => write!(f, "collective backend error: {m}"),
        }
    }
}

impl std::error::Error for DistError {}

/// A group of ranks that participate in collectives together. Object-safe so a loop can hold a
/// `Box<dyn ProcessGroup>` (the simulated backend now, NCCL later).
pub trait ProcessGroup {
    /// This rank's index in `0..world_size()`.
    fn rank(&self) -> usize;
    /// The number of ranks in the group.
    fn world_size(&self) -> usize;

    /// In-place all-reduce: every rank ends with the reduction over all ranks' `buf`.
    ///
    /// # Errors
    /// [`DistError::LengthMismatch`] if ranks supplied different-length buffers;
    /// [`DistError::CollectiveMismatch`] if ranks invoked different collectives at this step.
    fn all_reduce(&self, buf: &mut [f32], op: ReduceOp) -> Result<(), DistError>;

    /// Reduce-scatter: `input.len() == world·output.len()`; rank `r` ends with the reduction of the
    /// `r`-th `output.len()`-sized chunk across ranks.
    ///
    /// # Errors
    /// [`DistError::LengthMismatch`] if `input.len() != world·output.len()` or ranks disagree;
    /// [`DistError::LengthOverflow`] if `world·output.len()` overflows `usize`;
    /// [`DistError::CollectiveMismatch`] if ranks invoked different collectives at this step.
    fn reduce_scatter(
        &self,
        input: &[f32],
        output: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), DistError>;

    /// All-gather: `output.len() == world·input.len()`; every rank ends with all ranks' inputs
    /// concatenated in rank order.
    ///
    /// # Errors
    /// [`DistError::LengthMismatch`] if `output.len() != world·input.len()` or ranks disagree;
    /// [`DistError::LengthOverflow`] if `world·input.len()` overflows `usize`;
    /// [`DistError::CollectiveMismatch`] if ranks invoked different collectives at this step.
    fn all_gather(&self, input: &[f32], output: &mut [f32]) -> Result<(), DistError>;

    /// Broadcast: every rank ends with `root`'s `buf`.
    ///
    /// # Errors
    /// [`DistError::InvalidRoot`] if `root >= world`; [`DistError::LengthMismatch`] if a rank's
    /// buffer differs in length from `root`'s; [`DistError::CollectiveMismatch`] if ranks invoked
    /// different collectives at this step.
    fn broadcast(&self, buf: &mut [f32], root: usize) -> Result<(), DistError>;
}

/// State shared by all ranks of a [`SimProcessGroup`]: per-rank staging slots (each tagged with the
/// collective that wrote it) and a cyclic barrier. Each collective publishes `(tag, data)` to
/// `slots[rank]`, crosses barrier #1, reads all slots in rank order while validating, then crosses
/// barrier #2 to fence slot reuse for the next collective.
#[derive(Debug)]
struct SimShared {
    world: usize,
    slots: Mutex<Vec<(u8, Vec<f32>)>>,
    barrier: Barrier,
}

/// One rank's handle to a thread-simulated process group. Build a full set with
/// [`SimProcessGroup::world`] and move one handle into each rank's thread.
#[derive(Debug)]
pub struct SimProcessGroup {
    rank: usize,
    shared: Arc<SimShared>,
}

impl SimProcessGroup {
    /// Build `n` rank handles sharing one simulated group. Move handle `r` into the thread that
    /// plays rank `r`; all `n` must run concurrently (collectives barrier across them).
    ///
    /// # Panics
    /// If `n == 0`.
    #[must_use]
    pub fn world(n: usize) -> Vec<SimProcessGroup> {
        assert!(n > 0, "world size must be > 0");
        let shared = Arc::new(SimShared {
            world: n,
            slots: Mutex::new(vec![(0u8, Vec::new()); n]),
            barrier: Barrier::new(n),
        });
        (0..n)
            .map(|rank| SimProcessGroup {
                rank,
                shared: Arc::clone(&shared),
            })
            .collect()
    }

    /// Publish this rank's `(tag, data)` to its slot, then cross barrier #1 — after this returns,
    /// every rank's slot is written and stable for reading. Called *unconditionally* by every
    /// collective with nothing size-dependent computed beforehand, so every rank always reaches
    /// barrier #1 regardless of whether its buffers are valid.
    fn publish(&self, tag: u8, data: &[f32]) {
        {
            let mut slots = self.shared.slots.lock().expect("sim slots mutex poisoned");
            let slot = &mut slots[self.rank];
            slot.0 = tag;
            slot.1.clear();
            slot.1.extend_from_slice(data);
        }
        self.shared.barrier.wait();
    }

    /// After barrier #1, scan all slots for an op-tag differing from this rank's `tag`. Returns the
    /// first mismatch as a [`DistError::CollectiveMismatch`]. Called inside the locked post-barrier
    /// block before any size validation, so a cross-collective desync is reported cleanly (every
    /// rank still reaches barrier #2).
    fn check_tags(slots: &[(u8, Vec<f32>)], tag: u8) -> Result<(), DistError> {
        match slots.iter().find(|s| s.0 != tag) {
            Some(s) => Err(DistError::CollectiveMismatch {
                expected: tag,
                got: s.0,
            }),
            None => Ok(()),
        }
    }
}

impl ProcessGroup for SimProcessGroup {
    fn rank(&self) -> usize {
        self.rank
    }

    fn world_size(&self) -> usize {
        self.shared.world
    }

    fn all_reduce(&self, buf: &mut [f32], op: ReduceOp) -> Result<(), DistError> {
        let world = self.shared.world;
        let n = buf.len();
        // Publish unconditionally: every rank crosses barrier #1 (nothing size-dependent first).
        self.publish(TAG_ALL_REDUCE, buf);
        // All validation happens after barrier #1, inside the locked block. No early return /
        // `?` / panic between the two barriers — on any violation we set `res` and fall through.
        let res = {
            let slots = self.shared.slots.lock().expect("sim slots mutex poisoned");
            Self::check_tags(&slots, TAG_ALL_REDUCE).and_then(|()| {
                match slots.iter().find(|s| s.1.len() != n) {
                    Some(s) => Err(DistError::LengthMismatch {
                        expected: n,
                        got: s.1.len(),
                    }),
                    None => {
                        // In-bounds: every slot verified `len == n` above, and `i < n`.
                        debug_assert!(slots.iter().all(|s| s.1.len() == n));
                        for (i, out) in buf.iter_mut().enumerate() {
                            let mut acc = 0.0f32;
                            for r in 0..world {
                                acc += slots[r].1[i];
                            }
                            *out = acc;
                        }
                        Ok(())
                    }
                }
            })
        };
        if res.is_ok() && op == ReduceOp::Avg {
            let w = world as f32;
            buf.iter_mut().for_each(|x| *x /= w);
        }
        // Barrier #2: fence slot reuse — always reached (exactly 2 waits on every Ok/Err path).
        self.shared.barrier.wait();
        res
    }

    fn reduce_scatter(
        &self,
        input: &[f32],
        output: &mut [f32],
        op: ReduceOp,
    ) -> Result<(), DistError> {
        let world = self.shared.world;
        let chunk = output.len();
        // Publish unconditionally: every rank crosses barrier #1. The size relation
        // (`input.len() == world*chunk`) and overflow are deferred to the post-barrier block so a
        // rank that fails them does not strand peers blocked at barrier #1.
        self.publish(TAG_REDUCE_SCATTER, input);
        // All validation after barrier #1; no early return / `?` / panic between the barriers.
        let res = {
            let slots = self.shared.slots.lock().expect("sim slots mutex poisoned");
            Self::check_tags(&slots, TAG_REDUCE_SCATTER).and_then(|()| {
                match world.checked_mul(chunk) {
                    None => Err(DistError::LengthOverflow { world, chunk }),
                    Some(need) if input.len() != need => Err(DistError::LengthMismatch {
                        expected: need,
                        got: input.len(),
                    }),
                    Some(need) => match slots.iter().find(|s| s.1.len() != need) {
                        Some(s) => Err(DistError::LengthMismatch {
                            expected: need,
                            got: s.1.len(),
                        }),
                        None => {
                            let base = self.rank * chunk;
                            // In-bounds: every slot verified `len == need == world*chunk` above,
                            // and the max index read is `base + (chunk-1) = rank*chunk + chunk-1`,
                            // which for `rank < world` is `<= world*chunk - 1 = need - 1`.
                            debug_assert!(slots.iter().all(|s| s.1.len() == need));
                            debug_assert!(base + chunk <= need);
                            for (i, out) in output.iter_mut().enumerate() {
                                let mut acc = 0.0f32;
                                for r in 0..world {
                                    acc += slots[r].1[base + i];
                                }
                                *out = acc;
                            }
                            Ok(())
                        }
                    },
                }
            })
        };
        if res.is_ok() && op == ReduceOp::Avg {
            let w = world as f32;
            output.iter_mut().for_each(|x| *x /= w);
        }
        // Barrier #2: always reached (exactly 2 waits on every Ok/Err path).
        self.shared.barrier.wait();
        res
    }

    fn all_gather(&self, input: &[f32], output: &mut [f32]) -> Result<(), DistError> {
        let world = self.shared.world;
        let n = input.len();
        // Publish unconditionally: every rank crosses barrier #1. The size relation
        // (`output.len() == world*input.len()`) and overflow are deferred to the post-barrier block.
        self.publish(TAG_ALL_GATHER, input);
        // All validation after barrier #1; no early return / `?` / panic between the barriers.
        let res = {
            let slots = self.shared.slots.lock().expect("sim slots mutex poisoned");
            Self::check_tags(&slots, TAG_ALL_GATHER).and_then(|()| {
                match world.checked_mul(n) {
                    None => Err(DistError::LengthOverflow { world, chunk: n }),
                    Some(need) if output.len() != need => Err(DistError::LengthMismatch {
                        expected: need,
                        got: output.len(),
                    }),
                    Some(_) => match slots.iter().find(|s| s.1.len() != n) {
                        Some(s) => Err(DistError::LengthMismatch {
                            expected: n,
                            got: s.1.len(),
                        }),
                        None => {
                            // In-bounds: every slot verified `len == n`, `output.len() == world*n`,
                            // and `r < world`, so the write to `output[r*n .. r*n+n]` stays within
                            // `[0, world*n)`.
                            debug_assert!(slots.iter().all(|s| s.1.len() == n));
                            debug_assert_eq!(output.len(), world * n);
                            for (r, slot) in slots.iter().enumerate() {
                                output[r * n..r * n + n].copy_from_slice(&slot.1);
                            }
                            Ok(())
                        }
                    },
                }
            })
        };
        // Barrier #2: always reached (exactly 2 waits on every Ok/Err path).
        self.shared.barrier.wait();
        res
    }

    fn broadcast(&self, buf: &mut [f32], root: usize) -> Result<(), DistError> {
        let world = self.shared.world;
        // Publish unconditionally: every rank crosses barrier #1. The root-range check is deferred
        // to the post-barrier block so a rank that passes an out-of-range root does not strand peers.
        self.publish(TAG_BROADCAST, buf);
        // All validation after barrier #1; no early return / `?` / panic between the barriers.
        let res = {
            let slots = self.shared.slots.lock().expect("sim slots mutex poisoned");
            Self::check_tags(&slots, TAG_BROADCAST).and_then(|()| {
                if root >= world {
                    Err(DistError::InvalidRoot {
                        root,
                        world_size: world,
                    })
                } else {
                    // In-bounds: `root < world == slots.len()`.
                    debug_assert!(root < slots.len());
                    let root_slot = &slots[root].1;
                    if buf.len() != root_slot.len() {
                        Err(DistError::LengthMismatch {
                            expected: root_slot.len(),
                            got: buf.len(),
                        })
                    } else {
                        buf.copy_from_slice(root_slot);
                        Ok(())
                    }
                }
            })
        };
        // Barrier #2: always reached (exactly 2 waits on every Ok/Err path).
        self.shared.barrier.wait();
        res
    }
}
