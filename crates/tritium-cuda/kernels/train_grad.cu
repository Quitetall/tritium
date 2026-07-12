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
