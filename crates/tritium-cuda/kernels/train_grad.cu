// v0.50 training: f32 backward kernels for the ternary matmul
//   forward:  Y[m,n] = s[n] * sum_k A[m,k] * W[n,k]      (A:[M,K], W:[N,K], s:[N])
//
// These mirror tritium_train::ops::matmul::vjp exactly, element for element and in
// the SAME reduction order, so the GPU result matches the CPU vjp oracle. One thread
// per output element; the inner reduction is a sequential f32 loop with NO atomics, so
// the result is deterministic (work distribution can never reorder a sum). Compiled
// with --fmad=false (see build.rs) so `a*b*c`/`+=` are not fused into a single rounded
// fma — they reproduce the host's separate multiply/add rounding.
//
// All buffers row-major. gy = grad_out [M,N].

// Y[m,n] = s[n] * sum_k A[m,k] * W[n,k]      (A:[M,K], W:[N,K], s:[N], Y:[M,N])
// The forward companion to the grad kernels above: same f32 layout, same sequential
// reduction order, same --fmad=false rounding, so it reproduces tritium_train::ops::matmul::forward
// element for element (W is the STE-quantized {-1,0,+1} weight, passed as f32).
extern "C" __global__ void ternary_matmul_forward(
    const float* __restrict__ a,    // [M, K]
    const float* __restrict__ w,    // [N, K]
    const float* __restrict__ s,    // [N]
    float* __restrict__ y,          // [M, N]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; // over M*N
    if (idx >= (long)m * n) return;
    int mi = idx / n;
    int ni = idx % n;
    float acc = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
        acc += a[mi * k + ki] * w[ni * k + ki];
    }
    y[idx] = s[ni] * acc;
}

// gA[m,k] = sum_n gy[m,n] * s[n] * W[n,k]      (shape [M,K])
extern "C" __global__ void ternary_matmul_grad_a(
    const float* __restrict__ gy,   // [M, N]
    const float* __restrict__ w,    // [N, K]
    const float* __restrict__ s,    // [N]
    float* __restrict__ ga,         // [M, K]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; // over M*K
    if (idx >= (long)m * k) return;
    int mi = idx / k;
    int ki = idx % k;
    float acc = 0.0f;
    for (int ni = 0; ni < n; ++ni) {
        acc += gy[mi * n + ni] * s[ni] * w[ni * k + ki];
    }
    ga[idx] = acc;
}

// gW[n,k] = sum_m gy[m,n] * s[n] * A[m,k]      (shape [N,K]; STE'd to Wf upstream)
extern "C" __global__ void ternary_matmul_grad_w(
    const float* __restrict__ gy,   // [M, N]
    const float* __restrict__ a,    // [M, K]
    const float* __restrict__ s,    // [N]
    float* __restrict__ gw,         // [N, K]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x; // over N*K
    if (idx >= (long)n * k) return;
    int ni = idx / k;
    int ki = idx % k;
    float acc = 0.0f;
    for (int mi = 0; mi < m; ++mi) {
        acc += gy[mi * n + ni] * s[ni] * a[mi * k + ki];
    }
    gw[idx] = acc;
}

// gs[n] = sum_m gy[m,n] * P[m,n],  P[m,n] = sum_k A[m,k] * W[n,k]  (unscaled)   (shape [N])
extern "C" __global__ void ternary_matmul_grad_s(
    const float* __restrict__ gy,   // [M, N]
    const float* __restrict__ a,    // [M, K]
    const float* __restrict__ w,    // [N, K]
    float* __restrict__ gs,         // [N]
    int m, int n, int k)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x; // over N
    if (ni >= n) return;
    float acc = 0.0f;
    for (int mi = 0; mi < m; ++mi) {
        float p = 0.0f;
        for (int ki = 0; ki < k; ++ki) {
            p += a[mi * k + ki] * w[ni * k + ki];
        }
        acc += gy[mi * n + ni] * p;
    }
    gs[ni] = acc;
}

// ADR 0027 Track A: greedy per-row multi-plane SALT reconstruction.
// Mirrors tritium_train::ops::ste::salt_quantize_forward exactly: each row's
// AbsMean is accumulated in ascending-column order, then its residual is
// updated elementwise before fitting the next plane. One thread owns one row,
// so no reduction can reorder. `residual` is caller-owned reusable scratch.
extern "C" __global__ void salt_quantize_forward(
    const float* __restrict__ master,       // [rows, cols]
    float* __restrict__ residual,           // [rows, cols] scratch
    float* __restrict__ quantized,          // [rows, cols]
    int rows, int cols, int planes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    long base = (long)row * cols;

    for (int col = 0; col < cols; ++col) {
        residual[base + col] = master[base + col];
        quantized[base + col] = 0.0f;
    }

    for (int plane = 0; plane < planes; ++plane) {
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += fabsf(residual[base + col]);
        }
        float scale = sum / (float)cols;
        if (scale == 0.0f) continue;

        for (int col = 0; col < cols; ++col) {
            long idx = base + col;
            float trit = roundf(residual[idx] / scale);
            trit = fminf(1.0f, fmaxf(-1.0f, trit));
            float contribution = scale * trit;
            quantized[idx] += contribution;
            residual[idx] -= contribution;
        }
    }
}

// ADR 0027 Track A: fused elementwise AdamW update over resident buffers.
// Host code computes bias correction with the CPU oracle's `powi` contract;
// this kernel preserves optim::AdamW::step's per-element operation order.
extern "C" __global__ void adamw_step(
    float* __restrict__ param,
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int n,
    float lr,
    float beta1,
    float beta2,
    float one_minus_beta1,
    float one_minus_beta2,
    float bc1,
    float bc2,
    float eps,
    float shrink)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + one_minus_beta1 * g;
    float vi = beta2 * v[i] + one_minus_beta2 * g * g;
    m[i] = mi;
    v[i] = vi;
    param[i] = param[i] * shrink - lr * (mi / bc1 / (sqrtf(vi / bc2) + eps));
}

// ───────────────────────── device-resident glue ops (plan 0043 P2.2) ─────────────────────────
// Elementwise forward/backward for the tape's non-matmul ops, run on the resident activation
// buffers so a whole block chains fwd→bwd without host round-trips. One thread per element.
//
// add/mul use only +/* and (--fmad=false) reproduce the host ops::elementwise rounding BIT-FOR-BIT.
// silu uses expf, whose device implementation may differ from host libm by ~1 ULP — so silu is
// gated device==CPU within rel 1e-4 (not bit-exact), unlike the pure +/* kernels.

// σ(x) = 1/(1+e^{-x})  — matches ops::act::sigmoid (saturates cleanly, no NaN at large |x|).
__device__ __forceinline__ float sigmoidf(float x) {
    return 1.0f / (1.0f + expf(-x));
}

// SiLU forward: y[i] = x[i]·σ(x[i]).   (ops::act::silu_forward)
extern "C" __global__ void silu_forward(
    const float* __restrict__ x, float* __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * sigmoidf(x[i]);
}

// SiLU backward: gx[i] = gy[i]·(s + x·s·(1−s)),  s = σ(x[i]).   (ops::act::silu_vjp)
// Same operation order as the host: s + x*s*(1-s).
extern "C" __global__ void silu_backward(
    const float* __restrict__ x, const float* __restrict__ gy,
    float* __restrict__ gx, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float s = sigmoidf(x[i]);
    gx[i] = gy[i] * (s + x[i] * s * (1.0f - s));
}

// Elementwise multiply forward: y[i] = a[i]·b[i].   (ops::elementwise::mul_forward)
extern "C" __global__ void ew_mul_forward(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] * b[i];
}

// Elementwise multiply backward, one factor: g_out[i] = gy[i]·other[i].
// mul_vjp gives gA = gY⊙B and gB = gY⊙A — call twice with other = b, then other = a.
extern "C" __global__ void ew_mul_backward(
    const float* __restrict__ gy, const float* __restrict__ other,
    float* __restrict__ g_out, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    g_out[i] = gy[i] * other[i];
}

// Elementwise add forward: y[i] = a[i] + b[i].   (ops::elementwise::add_forward)
extern "C" __global__ void ew_add_forward(
    const float* __restrict__ a, const float* __restrict__ b,
    float* __restrict__ y, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = a[i] + b[i];
}

// In-place gradient accumulate: dst[i] += src[i]. The device tape zeroes each leaf/activation grad
// buffer, then every vjp accumulates its contribution here — so a value consumed by multiple ops
// (residual adds, the tied embedding) sums its grads exactly like the CPU tape's `grads[id] += v`.
// (add's backward is just this: accumulate gy into each input's grad.)
extern "C" __global__ void accumulate(
    float* __restrict__ dst, const float* __restrict__ src, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    dst[i] += src[i];
}

// ───────────────────────── device-resident RMSNorm (plan 0043 P2.3) ─────────────────────────
// The TRAINING RMSNorm (mirrors tritium_train::ops::norm — a plain SEQUENTIAL sum, NOT the
// inference tree-order rmsnorm in decode.cu). RMSNorm uses only +,*,/,sqrt — all IEEE
// correctly-rounded — so with --fmad=false these reproduce the host ops::norm forward/vjp
// BIT-FOR-BIT (unlike silu's expf). x:[rows,cols], w:[cols] (shared), y/gx:[rows,cols], gw:[cols],
// inv:[rows]. Per-row work → one thread per row (forward, inv, grad_x); grad_w reduces down rows
// per column → one thread per column.
//
//   inv_r  = 1/sqrt( (Σ_i x[r,i]²)/cols + eps )
//   y[r,i] = x[r,i]·inv_r·w[i]
//   c_r    = Σ_i g[r,i]·w[i]·x[r,i]
//   gx[r,k]= inv_r·g[r,k]·w[k] − (inv_r³·c_r/cols)·x[r,k]
//   gw[i]  = Σ_r g[r,i]·x[r,i]·inv_r

// Forward: one thread per row (mean_sq then y). Op order matches ops::norm::forward exactly.
extern "C" __global__ void rmsnorm_train_forward(
    const float* __restrict__ x, const float* __restrict__ w,
    float* __restrict__ y, int rows, int cols, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * cols;
    float acc = 0.0f;
    for (int i = 0; i < cols; ++i) acc += xr[i] * xr[i];
    float inv = 1.0f / sqrtf(acc / (float)cols + eps);
    float* yr = y + (long)r * cols;
    for (int i = 0; i < cols; ++i) yr[i] = xr[i] * inv * w[i];
}

// Per-row inverse-RMS `inv[r]` — shared by grad_x and grad_w so neither recomputes it.
extern "C" __global__ void rmsnorm_train_inv(
    const float* __restrict__ x, float* __restrict__ inv, int rows, int cols, float eps)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * cols;
    float acc = 0.0f;
    for (int i = 0; i < cols; ++i) acc += xr[i] * xr[i];
    inv[r] = 1.0f / sqrtf(acc / (float)cols + eps);
}

// grad_x: one thread per row. c_r then gx[r,·]. Uses precomputed inv[r].
extern "C" __global__ void rmsnorm_train_grad_x(
    const float* __restrict__ x, const float* __restrict__ w, const float* __restrict__ gy,
    const float* __restrict__ inv, float* __restrict__ gx, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * cols;
    const float* gr = gy + (long)r * cols;
    float invr = inv[r];
    float c = 0.0f;
    for (int i = 0; i < cols; ++i) c += gr[i] * w[i] * xr[i];
    float inv3_c_over_n = invr * invr * invr * c / (float)cols;
    float* gxr = gx + (long)r * cols;
    for (int k = 0; k < cols; ++k)
        gxr[k] = invr * gr[k] * w[k] - inv3_c_over_n * xr[k];
}

// grad_w: one thread per column i. gw[i] = Σ_r g[r,i]·x[r,i]·inv[r] (ascending r, matches the host
// accumulation order). Uses precomputed inv[r].
extern "C" __global__ void rmsnorm_train_grad_w(
    const float* __restrict__ x, const float* __restrict__ gy,
    const float* __restrict__ inv, float* __restrict__ gw, int rows, int cols)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= cols) return;
    float acc = 0.0f;
    for (int r = 0; r < rows; ++r)
        acc += gy[(long)r * cols + i] * x[(long)r * cols + i] * inv[r];
    gw[i] = acc;
}

// ───────────────────────── device-resident attention glue (plan 0043 P2.4) ─────────────────────────
// Softmax + causal mask + RoPE + reshape (slice/insert/transpose) + embedding gather + softmax-xent,
// mirroring the tritium_train::ops::{softmax,rope,shape,embed,loss} vjps. The pure-copy/select ops
// (mask, slice, insert, transpose, gather) are BIT-EXACT; softmax/xent (expf/logf) and RoPE
// (sin/cos) are device==CPU within 1e-4.

// Multiply by a constant scalar: y[i] = x[i]·c (attention's 1/sqrt(head_dim); its own vjp form,
// gx = gy·c). ops::* scale_const. Bit-exact (single multiply, --fmad=false).
extern "C" __global__ void scale_const(
    const float* __restrict__ x, float* __restrict__ y, float c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    y[i] = x[i] * c;
}

// Row softmax forward — one thread per row (stable: subtract row max). ops::softmax::forward.
extern "C" __global__ void softmax_forward(
    const float* __restrict__ x, float* __restrict__ y, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* xr = x + (long)r * cols;
    float* yr = y + (long)r * cols;
    float m = -INFINITY;
    for (int i = 0; i < cols; ++i) m = fmaxf(m, xr[i]);
    float sum = 0.0f;
    for (int i = 0; i < cols; ++i) { float e = expf(xr[i] - m); yr[i] = e; sum += e; }
    for (int i = 0; i < cols; ++i) yr[i] /= sum;
}

// Softmax backward from the saved probs p: gx[r,i] = p[r,i]·(g[r,i] − Σ_j p·g). ops::softmax::vjp.
extern "C" __global__ void softmax_backward(
    const float* __restrict__ p, const float* __restrict__ gy,
    float* __restrict__ gx, int rows, int cols)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* pr = p + (long)r * cols;
    const float* gr = gy + (long)r * cols;
    float* gxr = gx + (long)r * cols;
    float dot = 0.0f;
    for (int j = 0; j < cols; ++j) dot += pr[j] * gr[j];
    for (int i = 0; i < cols; ++i) gxr[i] = pr[i] * (gr[i] - dot);
}

// Additive causal mask: key j visible to query i iff j<=i, else MASK_NEG (-1e30). ops::softmax.
extern "C" __global__ void causal_mask_forward(
    const float* __restrict__ x, float* __restrict__ y, int rows, int cols)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * cols) return;
    int i = idx / cols, j = idx % cols;
    y[idx] = (j <= i) ? x[idx] : -1e30f;
}

// Causal-mask backward: gx[i,j] = (j<=i) ? gy[i,j] : 0. ops::softmax::causal_mask_vjp.
extern "C" __global__ void causal_mask_backward(
    const float* __restrict__ gy, float* __restrict__ gx, int rows, int cols)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * cols) return;
    int i = idx / cols, j = idx % cols;
    gx[idx] = (j <= i) ? gy[idx] : 0.0f;
}

// RoPE apply over [n_token, n_head, head_dim]. sign=+1 forward, sign=-1 vjp (inverse rotation).
// Angles in double to track ops::rope (which uses f64 sin_cos/powf then casts). One thread per
// rotation pair (token, head, j<half). ops::rope::{forward,vjp}.
extern "C" __global__ void rope_apply(
    const float* __restrict__ x, float* __restrict__ y, const int* __restrict__ positions,
    int n_head, int head_dim, float theta, int n_token, float sign)
{
    int half = head_dim / 2;
    long total = (long)n_token * n_head * half;
    long t = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= total) return;
    int j = t % half;
    int head = (t / half) % n_head;
    int token = t / ((long)n_head * half);
    long base = ((long)token * n_head + head) * head_dim;
    double inv_freq = pow((double)theta, -2.0 * (double)j / (double)head_dim);
    double angle = (double)positions[token] * inv_freq;
    double sd, cd;
    sincos(angle, &sd, &cd);
    float cos = (float)cd;
    float sin = sign * (float)sd;
    float a = x[base + j];
    float b = x[base + j + half];
    y[base + j] = a * cos - b * sin;
    y[base + j + half] = b * cos + a * sin;
}

// Extract a contiguous column range: y[r,c] = x[r, start+c], c in [0,len). ops::shape::slice_cols_forward.
extern "C" __global__ void slice_cols_forward(
    const float* __restrict__ x, float* __restrict__ y,
    int rows, int cols, int start, int len)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * len) return;
    int r = idx / len, c = idx % len;
    y[idx] = x[(long)r * cols + start + c];
}

// Insert a [rows,len] block into columns [start,start+len) of a [rows,total] buffer:
// dst[r, start+c] = src[r,c]. Builds concat (N inserts) and slice's vjp (insert into a zeroed buf).
extern "C" __global__ void copy_into_cols(
    const float* __restrict__ src, float* __restrict__ dst,
    int rows, int total, int start, int len)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * len) return;
    int r = idx / len, c = idx % len;
    dst[(long)r * total + start + c] = src[idx];
}

// Transpose [rows,cols] → [cols,rows]: y[c,r] = x[r,c]. Its own vjp (transpose again).
// ops::dense::transpose_{forward,vjp}.
extern "C" __global__ void transpose_forward(
    const float* __restrict__ x, float* __restrict__ y, int rows, int cols)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)rows * cols) return;
    int r = idx / cols, c = idx % cols;
    y[(long)c * rows + r] = x[idx];
}

// Embedding gather forward: y[t,:] = w[tokens[t], :]. ops::embed::gather_forward.
extern "C" __global__ void embed_gather_forward(
    const float* __restrict__ w, const int* __restrict__ tokens,
    float* __restrict__ y, int seq, int dim)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)seq * dim) return;
    int t = idx / dim, d = idx % dim;
    y[idx] = w[(long)tokens[t] * dim + d];
}

// Embedding gather backward: gw[v,d] = Σ_{t: tokens[t]==v} gy[t,d], summed in ASCENDING t (matches
// the host's sequential scatter-add). One thread per gw element → bit-exact (no atomics/reorder).
// ops::embed::gather_vjp.
extern "C" __global__ void embed_gather_backward(
    const float* __restrict__ gy, const int* __restrict__ tokens,
    float* __restrict__ gw, int seq, int dim, int vocab)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)vocab * dim) return;
    int v = idx / dim, d = idx % dim;
    float acc = 0.0f;
    for (int t = 0; t < seq; ++t)
        if (tokens[t] == v) acc += gy[(long)t * dim + d];
    gw[idx] = acc;
}

// Softmax cross-entropy backward: g_logits[r,c] = (gscale)·(p[r,c]·Σ_c target − target[r,c]),
// gscale = grad_out/rows. One thread per row (recompute stable softmax). ops::loss::softmax_xent_vjp.
extern "C" __global__ void softmax_xent_backward(
    const float* __restrict__ logits, const float* __restrict__ target,
    float* __restrict__ g_logits, int rows, int cols, float gscale)
{
    int r = blockIdx.x * blockDim.x + threadIdx.x;
    if (r >= rows) return;
    const float* lr = logits + (long)r * cols;
    const float* tr = target + (long)r * cols;
    float* gr = g_logits + (long)r * cols;
    float m = -INFINITY;
    for (int c = 0; c < cols; ++c) m = fmaxf(m, lr[c]);
    float sum = 0.0f;
    for (int c = 0; c < cols; ++c) sum += expf(lr[c] - m);
    float sum_t = 0.0f;
    for (int c = 0; c < cols; ++c) sum_t += tr[c];
    for (int c = 0; c < cols; ++c) {
        float p = expf(lr[c] - m) / sum;
        gr[c] = gscale * (p * sum_t - tr[c]);
    }
}
