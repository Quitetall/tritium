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
