//! Per-layer key/value cache for incremental decode.
//!
//! Keys and values for every attended token are kept in fp32 so the next decode
//! step attends over the whole context without recomputing past projections.
//! One [`KvCache`] is held per transformer block. WF-3 proves incremental decode
//! == full recompute across a page/chunk boundary (ADR 0002, v0.20); the cache
//! is a flat, lazily grown `[context, n_head_kv · head_dim]` ring-free arena with
//! a moving `len` watermark, so [`view`](KvCache::view) hands attention a
//! contiguous prefix and [`append`](KvCache::append) is a bounded `memcpy`.

use crate::error::NnError;

const GROWTH_ROWS: usize = 64;

/// Cached keys and values for one attention layer.
///
/// `k` and `v` are row-major `[max_ctx, n_head_kv · head_dim]`, growing by `seq`
/// rows per [`append`](KvCache::append). `len` is the number of valid (cached)
/// tokens; `max_ctx` is the maximum context this cache can hold. A row layout of
/// `n_head_kv · head_dim` matches the `[ctx, n_head_kv, head_dim]` operand
/// [`gqa_attention`](crate::gqa_attention) expects, so [`view`](KvCache::view)
/// feeds it directly with no repacking.
#[derive(Debug)]
pub struct KvCache {
    /// Flattened cached keys, `[max_ctx, n_head_kv · head_dim]` (only the first
    /// `len` rows are valid).
    pub k: Vec<f32>,
    /// Flattened cached values, same layout as [`k`](KvCache::k).
    pub v: Vec<f32>,
    /// Number of tokens currently cached.
    pub len: usize,
    /// Maximum number of tokens (rows) this cache holds.
    pub max_ctx: usize,
    /// Width of one cached row, `n_head_kv · head_dim`.
    pub row_width: usize,
}

impl KvCache {
    /// Create a cache for up to `max_ctx` tokens, each a `n_head_kv · head_dim`
    /// -wide key and value row. Starts empty (`len == 0`) without allocating the
    /// configured maximum context.
    ///
    /// # Panics
    /// Panics if the row-width multiplication overflows. File-backed model loaders
    /// use [`try_new`](Self::try_new) and return a typed error instead.
    #[must_use]
    pub fn new(max_ctx: usize, n_head_kv: usize, head_dim: usize) -> Self {
        Self::try_new(max_ctx, n_head_kv, head_dim).expect("KV-cache row width overflow")
    }

    /// Fallibly create an empty, lazy cache.
    ///
    /// No context-sized allocation happens until [`append`](Self::append). Growth
    /// is fallible and occurs in small row chunks, so a model's declared maximum
    /// context does not become an eager startup allocation.
    ///
    /// # Errors
    /// [`NnError::Backend`] if `n_head_kv · head_dim` overflows `usize`.
    pub fn try_new(max_ctx: usize, n_head_kv: usize, head_dim: usize) -> Result<Self, NnError> {
        let row_width = n_head_kv.checked_mul(head_dim).ok_or_else(|| {
            NnError::Backend("KV-cache row-width multiplication overflow".to_owned())
        })?;
        Ok(Self {
            k: Vec::new(),
            v: Vec::new(),
            len: 0,
            max_ctx,
            row_width,
        })
    }

    /// Append `n_new_tokens` new key/value rows (`k_new`/`v_new` are
    /// `[n_new_tokens, row_width]`), advancing [`len`](KvCache::len) by
    /// `n_new_tokens`.
    ///
    /// The rows are copied to `[len, len + n_new_tokens)` and `len` advances, so a
    /// subsequent [`view`](KvCache::view) covers them. Appending zero rows is a
    /// no-op.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `k_new`/`v_new` are not exactly
    /// `n_new_tokens · row_width` long, or if the append would push `len` past
    /// [`max_ctx`](KvCache::max_ctx).
    pub fn append(
        &mut self,
        k_new: &[f32],
        v_new: &[f32],
        n_new_tokens: usize,
    ) -> Result<(), NnError> {
        let need = n_new_tokens.checked_mul(self.row_width).ok_or_else(|| {
            NnError::Backend("KV-cache append length multiplication overflow".to_owned())
        })?;
        if k_new.len() != need {
            return Err(NnError::Shape {
                expected: need,
                got: k_new.len(),
            });
        }
        if v_new.len() != need {
            return Err(NnError::Shape {
                expected: need,
                got: v_new.len(),
            });
        }
        // Over-capacity guard: the new watermark must fit the arena. Compared in
        // token (row) units so it cannot overflow on the byte count.
        let new_len = self.len.checked_add(n_new_tokens).ok_or(NnError::Shape {
            expected: self.max_ctx,
            got: usize::MAX,
        })?;
        if new_len > self.max_ctx {
            return Err(NnError::Shape {
                expected: self.max_ctx,
                got: new_len,
            });
        }

        let required_elements = new_len.checked_mul(self.row_width).ok_or_else(|| {
            NnError::Backend("KV-cache backing length multiplication overflow".to_owned())
        })?;
        let growth_rows = new_len
            .div_ceil(GROWTH_ROWS)
            .checked_mul(GROWTH_ROWS)
            .unwrap_or(self.max_ctx)
            .min(self.max_ctx);
        let target_elements = growth_rows.checked_mul(self.row_width).ok_or_else(|| {
            NnError::Backend("KV-cache growth length multiplication overflow".to_owned())
        })?;
        reserve_to(&mut self.k, target_elements, "key")?;
        reserve_to(&mut self.v, target_elements, "value")?;
        self.k.resize(required_elements, 0.0);
        self.v.resize(required_elements, 0.0);

        let start = self.len.checked_mul(self.row_width).ok_or_else(|| {
            NnError::Backend("KV-cache write offset multiplication overflow".to_owned())
        })?;
        self.k[start..start + need].copy_from_slice(k_new);
        self.v[start..start + need].copy_from_slice(v_new);
        self.len = new_len;
        Ok(())
    }

    /// Borrow the valid prefix of the cache as `(k, v, len)`, where `k`/`v` are
    /// `[len · row_width]` and `len` is the cached-token count — exactly the
    /// `[len, n_head_kv, head_dim]` keys/values
    /// [`gqa_attention`](crate::gqa_attention) reads (pass `len` as its `ctx`).
    #[must_use]
    pub fn view(&self) -> (&[f32], &[f32], usize) {
        let n = self.len * self.row_width;
        (&self.k[..n], &self.v[..n], self.len)
    }

    /// Drop all cached tokens (set `len` to 0) without freeing the allocation, so
    /// the cache can be reused for a fresh sequence. Stale rows past the new
    /// `len` are never read, so they are left untouched.
    pub fn reset(&mut self) {
        self.rollback_to(0);
    }

    pub(crate) fn rollback_to(&mut self, len: usize) {
        debug_assert!(len <= self.len);
        let elements = len
            .checked_mul(self.row_width)
            .expect("previously validated KV-cache length");
        self.k.truncate(elements);
        self.v.truncate(elements);
        self.len = len;
    }
}

fn reserve_to(buffer: &mut Vec<f32>, target: usize, kind: &str) -> Result<(), NnError> {
    if buffer.capacity() < target {
        buffer
            .try_reserve_exact(target.saturating_sub(buffer.len()))
            .map_err(|error| {
                NnError::Backend(format!(
                    "allocate {kind} KV cache for {target} f32 values: {error}"
                ))
            })?;
    }
    Ok(())
}
