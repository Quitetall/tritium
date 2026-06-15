//! Per-layer key/value cache for incremental decode.
//!
//! Keys and values for every attended token are kept in fp32 so the next decode
//! step attends over the whole context without recomputing past projections.
//! One [`KvCache`] is held per transformer block. WF-3 makes this paged and
//! proves incremental decode == full recompute across a page boundary; today the
//! methods are documented stubs.

use crate::error::NnError;

/// Cached keys and values for one attention layer.
///
/// `k` and `v` are row-major `[len, n_head_kv · head_dim]`, growing by `seq` rows
/// per [`append`](KvCache::append). `len` is the number of valid (cached) tokens;
/// `capacity` is the maximum context this cache can hold.
#[derive(Debug)]
pub struct KvCache {
    /// Flattened cached keys, `[capacity, n_head_kv · head_dim]` (only the first
    /// `len` rows are valid).
    pub k: Vec<f32>,
    /// Flattened cached values, same layout as [`k`](KvCache::k).
    pub v: Vec<f32>,
    /// Number of tokens currently cached.
    pub len: usize,
    /// Maximum number of tokens (rows) this cache holds.
    pub capacity: usize,
    /// Width of one cached row, `n_head_kv · head_dim`.
    pub row_width: usize,
}

impl KvCache {
    /// Allocate a cache for up to `capacity` tokens, each a `row_width`-wide
    /// (`n_head_kv · head_dim`) key and value row. Starts empty (`len == 0`).
    #[must_use]
    pub fn new(capacity: usize, row_width: usize) -> Self {
        Self {
            k: vec![0.0; capacity * row_width],
            v: vec![0.0; capacity * row_width],
            len: 0,
            capacity,
            row_width,
        }
    }

    /// Append `seq` new key/value rows (`k_new`/`v_new` are `[seq, row_width]`),
    /// advancing [`len`](KvCache::len) by `seq`.
    ///
    /// # Errors
    /// [`NnError::Shape`] if the inputs are not `seq · row_width` long, or if the
    /// append would exceed [`capacity`](KvCache::capacity).
    pub fn append(&mut self, k_new: &[f32], v_new: &[f32], seq: usize) -> Result<(), NnError> {
        let _ = (k_new, v_new, seq);
        todo!("WF-3: append new KV rows, paged")
    }

    /// Borrow the valid prefix of the cache as `(&k[..len·row_width], &v[..])`.
    #[must_use]
    pub fn view(&self) -> (&[f32], &[f32]) {
        let n = self.len * self.row_width;
        (&self.k[..n], &self.v[..n])
    }

    /// Drop all cached tokens (set `len` to 0) without freeing the allocation, so
    /// the cache can be reused for a fresh sequence.
    pub fn reset(&mut self) {
        self.len = 0;
    }
}
