//! WF-3 KV-cache gate (ADR 0002, v0.20): incremental decode == full recompute.
//!
//! The KV cache must let an autoregressive decoder attend over its whole history
//! one token at a time and get *bit-for-bit-ish* (≤ `REL_TOL`) the same answer as
//! a single full-sequence prefill. We build a cache by appending tokens one at a
//! time, run [`gqa_attention`](tritium_nn::gqa_attention) against
//! [`KvCache::view`] after each step, and assert each step's output equals the
//! matching row of a one-shot prefill over the entire sequence — *including* a
//! step that crosses a page/chunk boundary. We also cover [`KvCache::reset`] and
//! the over-capacity append error.

use tritium_nn::{KvCache, gqa_attention};

/// ADR-0004 non-ternary fp32 tolerance.
const REL_TOL: f32 = 2e-3;

// BitNet 2B4T GQA shape (n_head=20, n_head_kv=5, head_dim=128) is large; the
// gate is shape-agnostic, so we use a small but representative GQA config
// (n_head=4, n_head_kv=2) that still exercises the `h / n_rep` head sharing.
const N_HEAD: usize = 4;
const N_HEAD_KV: usize = 2;
const HEAD_DIM: usize = 8;
const ROW_WIDTH: usize = N_HEAD_KV * HEAD_DIM; // KV row width
const Q_ROW: usize = N_HEAD * HEAD_DIM; // query/output row width

/// Deterministic, reproducible pseudo-random f32 in roughly `[-1, 1)` from a
/// splitmix64-style hash of `seed`. Keeps the test free of an `rand` dependency
/// while giving non-degenerate (asymmetric) attention scores.
fn rng(seed: u64) -> f32 {
    let mut z = seed.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    // Map the top 24 bits to [0, 1), then to [-1, 1).
    let unit = (z >> 40) as f32 / (1u64 << 24) as f32;
    unit * 2.0 - 1.0
}

/// Build full per-token query / key / value tensors for `seq` tokens.
///
/// Returns `(q, k, v)` flattened row-major: `q`/`out` rows are `Q_ROW` wide,
/// `k`/`v` rows are `ROW_WIDTH` wide. Each token `t` gets a distinct,
/// reproducible content so attention is genuinely position-dependent.
fn make_tensors(seq: usize) -> (Vec<f32>, Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; seq * Q_ROW];
    let mut k = vec![0.0f32; seq * ROW_WIDTH];
    let mut v = vec![0.0f32; seq * ROW_WIDTH];
    for t in 0..seq {
        for d in 0..Q_ROW {
            q[t * Q_ROW + d] = rng((t as u64) << 32 | (0x1_0000 + d as u64));
        }
        for d in 0..ROW_WIDTH {
            k[t * ROW_WIDTH + d] = rng((t as u64) << 32 | (0x2_0000 + d as u64));
            v[t * ROW_WIDTH + d] = rng((t as u64) << 32 | (0x3_0000 + d as u64));
        }
    }
    (q, k, v)
}

/// Full-sequence prefill: attend all `seq` query rows over all `seq` keys at
/// once. Returns `[seq, N_HEAD, HEAD_DIM]` row-major.
fn full_prefill(q: &[f32], k: &[f32], v: &[f32], seq: usize) -> Vec<f32> {
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut out = vec![0.0f32; seq * Q_ROW];
    gqa_attention(
        q, k, v, seq, seq, N_HEAD, N_HEAD_KV, HEAD_DIM, scale, 0, &mut out,
    )
    .expect("full prefill gqa");
    out
}

/// Assert `got ~= want` element-wise within `REL_TOL`.
fn assert_close(label: &str, got: &[f32], want: &[f32]) {
    assert_eq!(got.len(), want.len(), "{label}: length mismatch");
    for (idx, (&g, &w)) in got.iter().zip(want.iter()).enumerate() {
        assert!(g.is_finite(), "{label}[{idx}]: got non-finite {g}");
        let denom = w.abs().max(1.0);
        let rel = (g - w).abs() / denom;
        assert!(
            rel <= REL_TOL,
            "{label}[{idx}]: got {g}, want {w}, rel err {rel} > {REL_TOL}"
        );
    }
}

/// Run the gate for a given `(max_ctx, seq)`: decode `seq` tokens one at a time
/// through the cache and compare every step against the full prefill.
fn run_gate(max_ctx: usize, seq: usize) {
    assert!(seq <= max_ctx, "test misconfigured: seq must fit max_ctx");
    let (q, k, v) = make_tensors(seq);
    let full = full_prefill(&q, &k, &v, seq);

    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut cache = KvCache::new(max_ctx, N_HEAD_KV, HEAD_DIM);

    for t in 0..seq {
        // Append token `t`'s key/value row, then decode its query against the
        // whole cached prefix (positions 0..=t).
        let k_row = &k[t * ROW_WIDTH..(t + 1) * ROW_WIDTH];
        let v_row = &v[t * ROW_WIDTH..(t + 1) * ROW_WIDTH];
        cache.append(k_row, v_row, 1).expect("append one token");

        let (ck, cv, len) = cache.view();
        assert_eq!(len, t + 1, "cache len after appending token {t}");
        assert_eq!(ck.len(), len * ROW_WIDTH, "viewed k length");
        assert_eq!(cv.len(), len * ROW_WIDTH, "viewed v length");

        // Single-query decode step: seq=1, ctx=len, this token sits at absolute
        // position `t`, so `causal_offset = t` makes it see keys 0..=t.
        let q_row = &q[t * Q_ROW..(t + 1) * Q_ROW];
        let mut step = vec![0.0f32; Q_ROW];
        gqa_attention(
            q_row, ck, cv, 1, len, N_HEAD, N_HEAD_KV, HEAD_DIM, scale, t, &mut step,
        )
        .expect("incremental decode step");

        let want = &full[t * Q_ROW..(t + 1) * Q_ROW];
        assert_close(&format!("step/{t} (max_ctx={max_ctx})"), &step, want);
    }
}

/// The core gate, well within a single page: incremental decode of a short
/// sequence matches the full prefill row-for-row.
#[test]
fn incremental_decode_matches_full_prefill() {
    run_gate(16, 12);
}

/// THE GATE across a chunk/page boundary: `max_ctx = 40`, appending in 1s past
/// 32. Decoding tokens whose absolute position spans the 32-row boundary must
/// still match the full prefill exactly, proving the cache is not silently
/// truncating or wrapping at a page edge.
#[test]
fn incremental_decode_crosses_chunk_boundary() {
    run_gate(40, 40);
}

/// A fresh sequence reuses the same allocation: after [`KvCache::reset`] the
/// cache decodes a second, *different* sequence and still matches that
/// sequence's prefill — proving reset fully rewinds the watermark and stale rows
/// never leak in.
#[test]
fn reset_rewinds_and_allows_reuse() {
    let max_ctx = 16;
    let scale = 1.0 / (HEAD_DIM as f32).sqrt();
    let mut cache = KvCache::new(max_ctx, N_HEAD_KV, HEAD_DIM);

    // First sequence: fill a chunk of the cache.
    let seq_a = 10;
    let (qa, ka, va) = make_tensors(seq_a);
    for t in 0..seq_a {
        cache
            .append(
                &ka[t * ROW_WIDTH..(t + 1) * ROW_WIDTH],
                &va[t * ROW_WIDTH..(t + 1) * ROW_WIDTH],
                1,
            )
            .expect("append A");
    }
    assert_eq!(cache.view().2, seq_a, "len before reset");
    let _ = &qa; // first sequence's queries are not needed past the fill.

    cache.reset();
    let (rk, rv, rlen) = cache.view();
    assert_eq!(rlen, 0, "reset must zero len");
    assert!(rk.is_empty() && rv.is_empty(), "reset must empty the view");

    // Second sequence: different token positions/content. Use position offset so
    // the data differs from sequence A even at matching indices.
    let seq_b = 7;
    let mut q = vec![0.0f32; seq_b * Q_ROW];
    let mut k = vec![0.0f32; seq_b * ROW_WIDTH];
    let mut v = vec![0.0f32; seq_b * ROW_WIDTH];
    for t in 0..seq_b {
        let tag = t as u64 + 100; // shift the seed so it differs from make_tensors
        for d in 0..Q_ROW {
            q[t * Q_ROW + d] = rng(tag << 32 | (0x1_0000 + d as u64));
        }
        for d in 0..ROW_WIDTH {
            k[t * ROW_WIDTH + d] = rng(tag << 32 | (0x2_0000 + d as u64));
            v[t * ROW_WIDTH + d] = rng(tag << 32 | (0x3_0000 + d as u64));
        }
    }
    let full = {
        let mut out = vec![0.0f32; seq_b * Q_ROW];
        gqa_attention(
            &q, &k, &v, seq_b, seq_b, N_HEAD, N_HEAD_KV, HEAD_DIM, scale, 0, &mut out,
        )
        .expect("prefill B");
        out
    };

    for t in 0..seq_b {
        cache
            .append(
                &k[t * ROW_WIDTH..(t + 1) * ROW_WIDTH],
                &v[t * ROW_WIDTH..(t + 1) * ROW_WIDTH],
                1,
            )
            .expect("append B");
        let (ck, cv, len) = cache.view();
        assert_eq!(len, t + 1, "len during B");
        let mut step = vec![0.0f32; Q_ROW];
        gqa_attention(
            &q[t * Q_ROW..(t + 1) * Q_ROW],
            ck,
            cv,
            1,
            len,
            N_HEAD,
            N_HEAD_KV,
            HEAD_DIM,
            scale,
            t,
            &mut step,
        )
        .expect("decode B");
        assert_close(
            &format!("reuse/{t}"),
            &step,
            &full[t * Q_ROW..(t + 1) * Q_ROW],
        );
    }
}

/// Appending past `max_ctx` is a [`NnError::Shape`] error, and the failed append
/// must not corrupt the cache (`len` unchanged, prior rows intact, retry of a
/// fitting append still works).
#[test]
fn append_past_capacity_errors_without_corruption() {
    let max_ctx = 4;
    let mut cache = KvCache::new(max_ctx, N_HEAD_KV, HEAD_DIM);
    let (_, k, v) = make_tensors(8);

    // Fill exactly to capacity in one chunked append (3 rows) + 1 row.
    cache
        .append(&k[..3 * ROW_WIDTH], &v[..3 * ROW_WIDTH], 3)
        .expect("append 3");
    assert_eq!(cache.view().2, 3, "len after first append");

    // One more single row brings us to exactly max_ctx — must succeed.
    cache
        .append(
            &k[3 * ROW_WIDTH..4 * ROW_WIDTH],
            &v[3 * ROW_WIDTH..4 * ROW_WIDTH],
            1,
        )
        .expect("append to exactly max_ctx");
    assert_eq!(cache.view().2, max_ctx, "len at capacity");

    // Snapshot the full cache so we can prove the rejected append leaves it
    // byte-for-byte unchanged.
    let before_k = cache.view().0.to_vec();

    // Now any further append overflows and must error.
    let err = cache.append(
        &k[4 * ROW_WIDTH..5 * ROW_WIDTH],
        &v[4 * ROW_WIDTH..5 * ROW_WIDTH],
        1,
    );
    assert!(err.is_err(), "append past max_ctx must error");
    assert_eq!(cache.view().2, max_ctx, "len unchanged after failed append");
    assert_eq!(
        cache.view().0,
        &before_k[..],
        "data intact after failed append"
    );
}

/// A chunked append that *individually* fits but whose total crosses `max_ctx`
/// is rejected as a single unit (no partial write).
#[test]
fn chunked_append_overflow_is_atomic() {
    let max_ctx = 4;
    let mut cache = KvCache::new(max_ctx, N_HEAD_KV, HEAD_DIM);
    let (_, k, v) = make_tensors(8);

    cache
        .append(&k[..2 * ROW_WIDTH], &v[..2 * ROW_WIDTH], 2)
        .expect("append 2");

    // 3 more rows would make len=5 > max_ctx=4: reject, leave len=2.
    let err = cache.append(
        &k[2 * ROW_WIDTH..5 * ROW_WIDTH],
        &v[2 * ROW_WIDTH..5 * ROW_WIDTH],
        3,
    );
    assert!(err.is_err(), "overflowing chunk must error");
    assert_eq!(cache.view().2, 2, "len unchanged after rejected chunk");
}

/// Length-mismatched inputs (slice length disagrees with `n_new_tokens`) are
/// rejected before any write.
#[test]
fn append_rejects_mismatched_lengths() {
    let mut cache = KvCache::new(8, N_HEAD_KV, HEAD_DIM);
    let (_, k, v) = make_tensors(2);
    // Claim 2 tokens but pass 1 row of k.
    assert!(
        cache
            .append(&k[..ROW_WIDTH], &v[..2 * ROW_WIDTH], 2)
            .is_err(),
        "short k must error"
    );
    // Claim 1 token but pass 2 rows of v.
    assert!(
        cache
            .append(&k[..ROW_WIDTH], &v[..2 * ROW_WIDTH], 1)
            .is_err(),
        "long v must error"
    );
    assert_eq!(cache.view().2, 0, "no rows written on rejected append");
}

#[test]
fn declared_max_context_is_lazy_and_row_geometry_is_fallible() {
    let mut cache = KvCache::try_new(usize::MAX, 1, 1).expect("representable row width");
    assert!(cache.k.is_empty());
    assert!(cache.v.is_empty());
    assert_eq!(cache.k.capacity(), 0);
    assert_eq!(cache.v.capacity(), 0);

    cache.append(&[1.0], &[2.0], 1).expect("first row");
    assert_eq!(cache.view(), (&[1.0][..], &[2.0][..], 1));

    assert!(KvCache::try_new(1, usize::MAX, 2).is_err());
}
