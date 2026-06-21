//! Benchmarks for CPU inference hot paths. Run before/after each optimization.
//!
//! Each bench function prints median/p10/p90 over 500 iterations.
//! Use `cargo test -p tritium-nn --release --test bench_cpu_hotpaths -- --nocapture`

use tritium_nn::softmax_rows;

/// Deterministic xorshift fill.
fn seeded_f32(seed: u64, len: usize) -> Vec<f32> {
    let mut s = seed | 1;
    (0..len)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            ((s >> 11) as f32 / (1u64 << 53) as f32) * 4.0 - 2.0
        })
        .collect()
}

fn bench_stats(label: &str, times: &mut [std::time::Duration]) {
    times.sort();
    let n = times.len();
    let p10 = times[n / 10];
    let median = times[n / 2];
    let p90 = times[n * 9 / 10];
    let mean: std::time::Duration = times.iter().sum::<std::time::Duration>() / n as u32;
    println!("\n  {label}:");
    println!("    mean:   {mean:?}");
    println!("    p10:    {p10:?}");
    println!("    median: {median:?}");
    println!("    p90:    {p90:?}");
}

// ── RoPE ──────────────────────────────────────────────────────────────────

/// Current RoPE: recomputes cos/sin per (token, head) pair.
fn rope_current(x: &mut [f32], positions: &[usize], n_head: usize, head_dim: usize, theta: f32) {
    let half = head_dim / 2;
    let theta = f64::from(theta);
    let inv_head_dim = 1.0 / head_dim as f64;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta.powf(-2.0 * j as f64 * inv_head_dim))
        .collect();
    for (token, &pos) in positions.iter().enumerate() {
        let pos = pos as f64;
        let token_base = token * n_head * head_dim;
        for head in 0..n_head {
            let head_base = token_base + head * head_dim;
            for j in 0..half {
                let angle = pos * inv_freq[j];
                let (sin, cos) = angle.sin_cos();
                let cos = cos as f32;
                let sin = sin as f32;
                let a = x[head_base + j];
                let b = x[head_base + j + half];
                x[head_base + j] = a * cos - b * sin;
                x[head_base + j + half] = b * cos + a * sin;
            }
        }
    }
}

/// Optimized RoPE: precompute cos/sin table [positions × half], then index.
fn rope_precomputed(
    x: &mut [f32],
    positions: &[usize],
    n_head: usize,
    head_dim: usize,
    theta: f32,
) {
    let half = head_dim / 2;
    let theta_f64 = f64::from(theta);
    let inv_head_dim = 1.0 / head_dim as f64;
    let inv_freq: Vec<f64> = (0..half)
        .map(|j| theta_f64.powf(-2.0 * j as f64 * inv_head_dim))
        .collect();

    // Precompute cos/sin table: [positions.len() × half].
    let n_pos = positions.len();
    let mut cos_table = vec![0.0f32; n_pos * half];
    let mut sin_table = vec![0.0f32; n_pos * half];
    for (token, &pos) in positions.iter().enumerate() {
        let pos = pos as f64;
        for j in 0..half {
            let angle = pos * inv_freq[j];
            let (s, c) = angle.sin_cos();
            cos_table[token * half + j] = c as f32;
            sin_table[token * half + j] = s as f32;
        }
    }

    // Apply using precomputed tables.
    for (token, _) in positions.iter().enumerate() {
        let token_base = token * n_head * head_dim;
        let ct = &cos_table[token * half..token * half + half];
        let st = &sin_table[token * half..token * half + half];
        for head in 0..n_head {
            let head_base = token_base + head * head_dim;
            for j in 0..half {
                let a = x[head_base + j];
                let b = x[head_base + j + half];
                x[head_base + j] = a * ct[j] - b * st[j];
                x[head_base + j + half] = b * ct[j] + a * st[j];
            }
        }
    }
}

#[test]
fn bench_rope() {
    // BitNet 2B4T geometry: n_head=20+5=25, head_dim=128, theta=10000.0
    let n_head = 25usize;
    let head_dim = 128usize;
    let theta = 10000.0f32;
    let seq = 16usize; // typical prefill length
    let positions: Vec<usize> = (0..seq).collect();
    let n_tokens = seq * n_head * head_dim;
    let iters = 500;

    println!("\n=== RoPE bench (n_head={n_head}, head_dim={head_dim}, seq={seq}) ===");

    // Current.
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut x = seeded_f32(42, n_tokens);
            let start = std::time::Instant::now();
            rope_current(&mut x, &positions, n_head, head_dim, theta);
            times.push(start.elapsed());
            std::hint::black_box(&x);
        }
        bench_stats("current (sin_cos per head)", &mut times);
    }

    // Precomputed.
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut x = seeded_f32(42, n_tokens);
            let start = std::time::Instant::now();
            rope_precomputed(&mut x, &positions, n_head, head_dim, theta);
            times.push(start.elapsed());
            std::hint::black_box(&x);
        }
        bench_stats("precomputed table", &mut times);
    }
}

// ── softmax_rows ──────────────────────────────────────────────────────────

/// Current: 3 passes (max, exp+sum, normalize).
fn softmax_3pass(x: &mut [f32], row_len: usize) {
    for row in x.chunks_mut(row_len) {
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        let inv = 1.0f32 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

/// Optimized: 2 passes (max, exp+sum+normalize).
fn softmax_2pass(x: &mut [f32], row_len: usize) {
    for row in x.chunks_mut(row_len) {
        let mut max = f32::NEG_INFINITY;
        for &v in row.iter() {
            if v > max {
                max = v;
            }
        }
        let mut sum = 0.0f32;
        for v in row.iter_mut() {
            let e = (*v - max).exp();
            *v = e;
            sum += e;
        }
        let inv = 1.0f32 / sum;
        for v in row.iter_mut() {
            *v *= inv;
        }
    }
}

#[test]
fn bench_softmax() {
    // Typical attention scores: n_head=20, ctx=256, seq=1.
    let row_len = 256usize;
    let n_rows = 20usize;
    let n_elems = n_rows * row_len;
    let iters = 500;

    println!("\n=== softmax_rows bench (rows={n_rows}, row_len={row_len}) ===");

    // Current (3-pass via the public API).
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut x = seeded_f32(99, n_elems);
            let start = std::time::Instant::now();
            softmax_rows(&mut x, row_len).unwrap();
            times.push(start.elapsed());
            std::hint::black_box(&x);
        }
        bench_stats("current (3-pass, public API)", &mut times);
    }

    // Inline 2-pass.
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut x = seeded_f32(99, n_elems);
            let start = std::time::Instant::now();
            softmax_2pass(&mut x, row_len);
            times.push(start.elapsed());
            std::hint::black_box(&x);
        }
        bench_stats("optimized (2-pass, inline)", &mut times);
    }
}

// ── RMSNorm prefill skip ─────────────────────────────────────────────────

fn rmsnorm(x: &[f32], w: &[f32], eps: f32, out: &mut [f32]) {
    let n = x.len();
    if n == 0 {
        return;
    }
    let mean_sq = x.iter().map(|v| v * v).sum::<f32>() / n as f32;
    let inv = 1.0 / (mean_sq + eps).sqrt();
    for ((o, &xi), &wi) in out.iter_mut().zip(x).zip(w) {
        *o = xi * inv * wi;
    }
}

#[test]
fn bench_rmsnorm_prefill() {
    // BitNet 2B4T: n_embd=2560, seq=128 prefill.
    let n_embd = 2560usize;
    let seq = 128usize;
    let eps = 1e-5f32;
    let w: Vec<f32> = vec![1.0; n_embd]; // unit weights
    let hidden = seeded_f32(77, seq * n_embd);
    let iters = 500;

    println!("\n=== RMSNorm prefill bench (n_embd={n_embd}, seq={seq}) ===");

    // Current: compute all seq tokens.
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut out = vec![0.0f32; seq * n_embd];
            let start = std::time::Instant::now();
            for t in 0..seq {
                let src = &hidden[t * n_embd..t * n_embd + n_embd];
                let dst = &mut out[t * n_embd..t * n_embd + n_embd];
                rmsnorm(src, &w, eps, dst);
            }
            times.push(start.elapsed());
            std::hint::black_box(&out);
        }
        bench_stats("current (all seq tokens)", &mut times);
    }

    // Optimized: only last token.
    {
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut last_norm = vec![0.0f32; n_embd];
            let start = std::time::Instant::now();
            let last = seq - 1;
            let src = &hidden[last * n_embd..last * n_embd + n_embd];
            rmsnorm(src, &w, eps, &mut last_norm);
            times.push(start.elapsed());
            std::hint::black_box(&last_norm);
        }
        bench_stats("optimized (last token only)", &mut times);
    }
}

// ── Quantize + scratch allocation ────────────────────────────────────────

#[test]
fn bench_quantize_alloc() {
    // Typical: K=2560 (n_embd), M=1 (decode).
    let k = 2560usize;
    let m = 1usize;
    let iters = 500;

    println!("\n=== quantize_activation_int8 scratch bench (m={m}, k={k}) ===");

    // Current: allocate every call.
    {
        let act = seeded_f32(55, m * k);
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let mut q_act = vec![0.0f32; m * k];
            let mut act_scale = vec![0.0f32; m];
            let start = std::time::Instant::now();
            tritium_nn::quantize_activation_int8(&act, m, k, &mut q_act, &mut act_scale).unwrap();
            times.push(start.elapsed());
        }
        bench_stats("current (alloc per call)", &mut times);
    }

    // Optimized: pre-allocated scratch.
    {
        let act = seeded_f32(55, m * k);
        let mut q_act = vec![0.0f32; m * k];
        let mut act_scale = vec![0.0f32; m];
        let mut times = Vec::with_capacity(iters);
        for _ in 0..iters {
            let start = std::time::Instant::now();
            tritium_nn::quantize_activation_int8(&act, m, k, &mut q_act, &mut act_scale).unwrap();
            times.push(start.elapsed());
        }
        bench_stats("optimized (pre-allocated scratch)", &mut times);
    }
}
