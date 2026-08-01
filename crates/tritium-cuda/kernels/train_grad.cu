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

#include <cuda_fp16.h>

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

// Packed TQ2_0 activation VJP for the backend-neutral projected-weight contract:
// gA[m,k] = sum_n gy[m,n] * scale[n] * trit[n,k].
//
// Unlike `ternary_matmul_grad_a`, this consumes the inference packing already
// resident in `CudaBuffer`; no dense f32 weight shadow is materialized.
#define PROJECTED_VJP_QK 256
#define PROJECTED_VJP_TQ2_BLOCK_BYTES 66

// Framework adapter: pack one dense fp32 master row into inference-native
// TQ2_0. One thread owns a row, so no atomics or cross-thread reduction can
// change projection semantics. Block scales stay 1.0 because row scales live
// in the separate fp32 scale tensor.
extern "C" __global__ void tq2_pack_from_master_scale(
    const float* __restrict__ master,          // [N, K]
    const float* __restrict__ scales,          // [N]
    unsigned char* __restrict__ weights,       // [N, row_bytes]
    int n, int k, int row_bytes)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n) return;
    const float scale = scales[ni];
    const float denominator = fmaxf(scale, 1.1754943508222875e-38f);
    unsigned char* row = weights + (long)ni * row_bytes;
    const int blocks = (k + PROJECTED_VJP_QK - 1) / PROJECTED_VJP_QK;

    for (int block = 0; block < blocks; ++block) {
        unsigned char* packed = row + block * PROJECTED_VJP_TQ2_BLOCK_BYTES;
        for (int byte = 0; byte < 64; ++byte) packed[byte] = 0x55u;
        packed[64] = 0x00u; // IEEE-f16 1.0, little endian.
        packed[65] = 0x3cu;
    }

    if (!(scale > 0.0f) || !isfinite(scale)) return;
    for (int ki = 0; ki < k; ++ki) {
        float quantized = nearbyintf(master[(long)ni * k + ki] / denominator);
        quantized = fminf(1.0f, fmaxf(-1.0f, quantized));
        unsigned int code = quantized < 0.0f ? 0u : (quantized > 0.0f ? 2u : 1u);
        int block = ki / PROJECTED_VJP_QK;
        int e = ki - block * PROJECTED_VJP_QK;
        int c = e >> 7;
        int mm = e & 31;
        int l = (e & 127) >> 5;
        unsigned char* byte =
            row + block * PROJECTED_VJP_TQ2_BLOCK_BYTES + c * 32 + mm;
        unsigned char mask = (unsigned char)(3u << (2 * l));
        *byte = (unsigned char)((*byte & ~mask) | (code << (2 * l)));
    }
}

// Autocast companion: fp16 master with fp32 row scales and packed output.
extern "C" __global__ void tq2_pack_from_master_scale_f16(
    const __half* __restrict__ master,         // [N, K]
    const float* __restrict__ scales,          // [N]
    unsigned char* __restrict__ weights,       // [N, row_bytes]
    int n, int k, int row_bytes)
{
    int ni = blockIdx.x * blockDim.x + threadIdx.x;
    if (ni >= n) return;
    const float scale = scales[ni];
    const float denominator = fmaxf(scale, 6.103515625e-5f);
    unsigned char* row = weights + (long)ni * row_bytes;
    const int blocks = (k + PROJECTED_VJP_QK - 1) / PROJECTED_VJP_QK;

    for (int block = 0; block < blocks; ++block) {
        unsigned char* packed = row + block * PROJECTED_VJP_TQ2_BLOCK_BYTES;
        for (int byte = 0; byte < 64; ++byte) packed[byte] = 0x55u;
        packed[64] = 0x00u;
        packed[65] = 0x3cu;
    }

    if (!(scale > 0.0f) || !isfinite(scale)) return;
    for (int ki = 0; ki < k; ++ki) {
        float value = __half2float(master[(long)ni * k + ki]);
        float quantized = nearbyintf(value / denominator);
        quantized = fminf(1.0f, fmaxf(-1.0f, quantized));
        unsigned int code = quantized < 0.0f ? 0u : (quantized > 0.0f ? 2u : 1u);
        int block = ki / PROJECTED_VJP_QK;
        int e = ki - block * PROJECTED_VJP_QK;
        int c = e >> 7;
        int mm = e & 31;
        int l = (e & 127) >> 5;
        unsigned char* byte =
            row + block * PROJECTED_VJP_TQ2_BLOCK_BYTES + c * 32 + mm;
        unsigned char mask = (unsigned char)(3u << (2 * l));
        *byte = (unsigned char)((*byte & ~mask) | (code << (2 * l)));
    }
}

// Stream-native framework forward. `bias` may be null. Packed weights and
// scales remain resident across calls; output storage belongs to framework.
extern "C" __global__ void tq2_projected_linear_forward(
    const float* __restrict__ a,               // [M, K]
    const unsigned char* __restrict__ weights, // [N, row_bytes]
    const float* __restrict__ scales,           // [N]
    const float* __restrict__ bias,             // [N] or null
    float* __restrict__ y,                      // [M, N]
    int m, int n, int k, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * n) return;
    int mi = idx / n;
    int ni = idx % n;
    const unsigned char* row = weights + (long)ni * row_bytes;
    if (!isfinite(scales[ni])) {
        y[idx] = NAN;
        return;
    }
    float acc = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
        int block = ki / PROJECTED_VJP_QK;
        int e = ki - block * PROJECTED_VJP_QK;
        int c = e >> 7;
        int mm = e & 31;
        int l = (e & 127) >> 5;
        int byte_off = block * PROJECTED_VJP_TQ2_BLOCK_BYTES + c * 32 + mm;
        unsigned int code = ((unsigned int)row[byte_off] >> (2 * l)) & 3u;
        float av = a[(long)mi * k + ki];
        if (!isfinite(av)) {
            acc = NAN;
            break;
        }
        if (code == 2u) acc += av;
        else if (code == 0u) acc -= av;
    }
    float value = scales[ni] * acc;
    y[idx] = bias ? value + bias[ni] : value;
}

// Native CUDA autocast forward. Products accumulate in fp32; framework-visible
// input, bias, and output remain fp16.
extern "C" __global__ void tq2_projected_linear_forward_f16(
    const __half* __restrict__ a,              // [M, K]
    const unsigned char* __restrict__ weights, // [N, row_bytes]
    const float* __restrict__ scales,          // [N]
    const __half* __restrict__ bias,            // [N] or null
    __half* __restrict__ y,                     // [M, N]
    int m, int n, int k, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * n) return;
    int mi = idx / n;
    int ni = idx % n;
    const unsigned char* row = weights + (long)ni * row_bytes;
    if (!isfinite(scales[ni])) {
        y[idx] = __float2half(NAN);
        return;
    }
    float acc = 0.0f;
    for (int ki = 0; ki < k; ++ki) {
        int block = ki / PROJECTED_VJP_QK;
        int e = ki - block * PROJECTED_VJP_QK;
        int c = e >> 7;
        int mm = e & 31;
        int l = (e & 127) >> 5;
        int byte_off = block * PROJECTED_VJP_TQ2_BLOCK_BYTES + c * 32 + mm;
        unsigned int code = ((unsigned int)row[byte_off] >> (2 * l)) & 3u;
        float av = __half2float(a[(long)mi * k + ki]);
        if (!isfinite(av)) {
            acc = NAN;
            break;
        }
        if (code == 2u) acc += av;
        else if (code == 0u) acc -= av;
    }
    float value = scales[ni] * acc;
    if (bias) value += __half2float(bias[ni]);
    y[idx] = __float2half(value);
}

extern "C" __global__ void tq2_projected_grad_a(
    const float* __restrict__ gy,              // [M, N]
    const unsigned char* __restrict__ weights, // [N, row_bytes]
    const float* __restrict__ scales,          // [N]
    float* __restrict__ ga,                    // [M, K]
    int m, int n, int k, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * k) return;
    int mi = idx / k;
    int ki = idx % k;
    int block = ki / PROJECTED_VJP_QK;
    int e = ki - block * PROJECTED_VJP_QK;
    int c = e >> 7;
    int mm = e & 31;
    int l = (e & 127) >> 5;
    int byte_off = block * PROJECTED_VJP_TQ2_BLOCK_BYTES + c * 32 + mm;

    float acc = 0.0f;
    for (int ni = 0; ni < n; ++ni) {
        const unsigned char* wrow = weights + (long)ni * row_bytes;
        unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        float upstream = gy[(long)mi * n + ni] * scales[ni];
        if (!isfinite(gy[(long)mi * n + ni]) || !isfinite(scales[ni])) {
            acc = NAN;
            break;
        }
        if (code == 2u) {
            acc += upstream;
        } else if (code == 0u) {
            acc -= upstream;
        }
    }
    ga[idx] = acc;
}

extern "C" __global__ void tq2_projected_grad_a_f16(
    const __half* __restrict__ gy,             // [M, N]
    const unsigned char* __restrict__ weights, // [N, row_bytes]
    const float* __restrict__ scales,          // [N]
    __half* __restrict__ ga,                    // [M, K]
    int m, int n, int k, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * k) return;
    int mi = idx / k;
    int ki = idx % k;
    int block = ki / PROJECTED_VJP_QK;
    int e = ki - block * PROJECTED_VJP_QK;
    int c = e >> 7;
    int mm = e & 31;
    int l = (e & 127) >> 5;
    int byte_off = block * PROJECTED_VJP_TQ2_BLOCK_BYTES + c * 32 + mm;

    float acc = 0.0f;
    for (int ni = 0; ni < n; ++ni) {
        const unsigned char* wrow = weights + (long)ni * row_bytes;
        unsigned int code = ((unsigned int)wrow[byte_off] >> (2 * l)) & 3u;
        float upstream_value = __half2float(gy[(long)mi * n + ni]);
        float upstream = upstream_value * scales[ni];
        if (!isfinite(upstream_value) || !isfinite(scales[ni])) {
            acc = NAN;
            break;
        }
        if (code == 2u) acc += upstream;
        else if (code == 0u) acc -= upstream;
    }
    ga[idx] = __float2half(acc);
}

// VJP with respect to the dense projected weight P (not latent trits/scales):
// gP[n,k] = sum_m gy[m,n] * A[m,k].
extern "C" __global__ void linear_grad_projected_weight(
    const float* __restrict__ gy, // [M, N]
    const float* __restrict__ a,  // [M, K]
    float* __restrict__ gp,       // [N, K]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * k) return;
    int ni = idx / k;
    int ki = idx % k;
    float acc = 0.0f;
    for (int mi = 0; mi < m; ++mi) {
        acc += gy[(long)mi * n + ni] * a[(long)mi * k + ki];
    }
    gp[idx] = acc;
}

// Dense-master VJP with frozen AbsMean scale and strict clipped STE mask:
// gMaster[n,k] = sum_m gy[m,n] * A[m,k] when
// abs(master[n,k] / max(scale[n], FLT_MIN)) < 1 and scale[n] > 0.
extern "C" __global__ void linear_grad_master_ste(
    const float* __restrict__ gy,      // [M, N]
    const float* __restrict__ a,       // [M, K]
    const float* __restrict__ master,  // [N, K]
    const float* __restrict__ scales,  // [N]
    float* __restrict__ gm,            // [N, K]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * k) return;
    int ni = idx / k;
    int ki = idx % k;
    float scale = scales[ni];
    float denominator = fmaxf(scale, 1.1754943508222875e-38f);
    float mask =
        scale > 0.0f && fabsf(master[idx] / denominator) < 1.0f ? 1.0f : 0.0f;
    float acc = 0.0f;
    for (int mi = 0; mi < m; ++mi) {
        acc += gy[(long)mi * n + ni] * a[(long)mi * k + ki];
    }
    gm[idx] = acc * mask;
}

extern "C" __global__ void linear_grad_master_ste_f16(
    const __half* __restrict__ gy,      // [M, N]
    const __half* __restrict__ a,       // [M, K]
    const __half* __restrict__ master,  // [N, K]
    const float* __restrict__ scales,   // [N]
    __half* __restrict__ gm,            // [N, K]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * k) return;
    int ni = idx / k;
    int ki = idx % k;
    float scale = scales[ni];
    float denominator = fmaxf(scale, 6.103515625e-5f);
    float master_value = __half2float(master[idx]);
    float mask =
        scale > 0.0f && fabsf(master_value / denominator) < 1.0f ? 1.0f : 0.0f;
    float acc = 0.0f;
    for (int mi = 0; mi < m; ++mi) {
        acc += __half2float(gy[(long)mi * n + ni])
            * __half2float(a[(long)mi * k + ki]);
    }
    gm[idx] = __float2half(acc * mask);
}

// Autocast keeps fp32 master weights/optimizer gradients while activations and
// upstream gradients use fp16.
extern "C" __global__ void linear_grad_master_ste_autocast(
    const __half* __restrict__ gy,     // [M, N]
    const __half* __restrict__ a,      // [M, K]
    const float* __restrict__ master,  // [N, K]
    const float* __restrict__ scales,  // [N]
    float* __restrict__ gm,            // [N, K]
    int m, int n, int k)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)n * k) return;
    int ni = idx / k;
    int ki = idx % k;
    float scale = scales[ni];
    float denominator = fmaxf(scale, 1.1754943508222875e-38f);
    float mask =
        scale > 0.0f && fabsf(master[idx] / denominator) < 1.0f ? 1.0f : 0.0f;
    float acc = 0.0f;
    for (int mi = 0; mi < m; ++mi) {
        acc += __half2float(gy[(long)mi * n + ni])
            * __half2float(a[(long)mi * k + ki]);
    }
    gm[idx] = acc * mask;
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
    int rows, int cols, int planes,
    float sherry_alpha)                     // 0 = pure ternary (the committed behaviour)
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

    // Sherry (annealed fp residual): blend a decaying fraction of the untouched fp master back in,
    // `(1-a)*Q + a*master`, mirroring tritium_train::ops::ste::salt_quantize_forward_sherry exactly
    // (same single-rounding `q += a*(m-q)` form). Guarded so a==0 leaves `quantized` BIT-IDENTICAL to
    // the pure-ternary path — including the -0.0 case, where `q + 0.0f*(...)` would flip the sign bit.
    if (sherry_alpha != 0.0f) {
        for (int col = 0; col < cols; ++col) {
            long idx = base + col;
            quantized[idx] += sherry_alpha * (master[idx] - quantized[idx]);
        }
    }
}

// Portable-training SALT variant with one-row scratch. A single thread walks
// rows in contract order so the peak scratch is cols*sizeof(float), independent
// of row count, while preserving the reference reduction order exactly.
// Grouped SALT with an optional per-group Hadamard rotation — the fitter configuration that took
// PTQ from 10786x to 1.74x fp on SmolLM2-135M. Mirrors
// tritium_train::ops::ste::salt_quantize_forward_grouped.
//
// ONE BLOCK PER GROUP. The group is staged in shared memory so the Hadamard butterfly and the AbsMean
// reductions are block-cooperative. `rotate_mask[g] != 0` selects rotation for group g; the mask is
// decided ONCE on the host (a group's tail weight is a property of its distribution, not of the step),
// so the kernel never has to fit a group twice.
//
// H is an involution at this normalization, so the same transform runs before and after quantizing and
// the reconstruction lands back in the original basis. Rotation only ever touches power-of-two groups;
// a ragged tail is left unrotated, matching the host.
#define SALT_GROUP_MAX 256
extern "C" __global__ void salt_quantize_forward_grouped(
    const float* __restrict__ master,        // [rows, cols]
    float* __restrict__ quantized,           // [rows, cols]
    const unsigned char* __restrict__ rotate_mask, // [total_groups] 0/1, may be null
    int rows, int cols, int planes, int group, int iters)
{
    extern __shared__ float sh[];            // group floats, then blockDim.x for the reduction
    const int groups_per_row = (cols + group - 1) / group;
    const int gid = blockIdx.x;              // one block per group
    if (gid >= rows * groups_per_row) return;
    const int row = gid / groups_per_row;
    const int gcol = (gid % groups_per_row) * group;
    const int n = min(group, cols - gcol);   // ragged tail
    const long base = (long)row * cols + gcol;
    float* red = sh + SALT_GROUP_MAX;        // reduction scratch

    // Rotation only for power-of-two groups, matching the host.
    const bool pow2 = (n > 1) && ((n & (n - 1)) == 0);
    const bool rot = pow2 && rotate_mask != nullptr && rotate_mask[gid] != 0;

    for (int i = threadIdx.x; i < n; i += blockDim.x) sh[i] = master[base + i];
    __syncthreads();

    // Forward Hadamard (normalized), in shared memory.
    if (rot) {
        for (int len = 1; len < n; len <<= 1) {
            for (int idx = threadIdx.x; idx < (n >> 1); idx += blockDim.x) {
                int blk = idx / len, off = idx % len;
                int i = blk * (len << 1) + off;
                float a = sh[i], b = sh[i + len];
                sh[i] = a + b; sh[i + len] = a - b;
            }
            __syncthreads();
        }
        float s = rsqrtf((float)n);
        for (int i = threadIdx.x; i < n; i += blockDim.x) sh[i] *= s;
        __syncthreads();
    }

    // Residual expansion in place: sh holds the running residual, `acc` the reconstruction.
    // Each thread owns a strided slice, so per-thread partial sums are deterministic.
    float acc[8];                             // up to SALT_GROUP_MAX/32 elements per thread
    int owned = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { acc[owned] = 0.0f; owned++; }

    // Every reduction below is block-wide, so `red[0]` is identical in all threads and every
    // `break` driven by it is taken uniformly — which is what keeps the __syncthreads() inside the
    // ITF loop legal.
#define SALT_BLOCK_SUM(expr_partial)                                                  \
    do {                                                                              \
        red[threadIdx.x] = (expr_partial);                                            \
        __syncthreads();                                                              \
        for (int _s = blockDim.x >> 1; _s > 0; _s >>= 1) {                            \
            if (threadIdx.x < _s) red[threadIdx.x] += red[threadIdx.x + _s];          \
            __syncthreads();                                                          \
        }                                                                             \
    } while (0)

    float trit[8];
    float cand_trit[8];
    for (int p = 0; p < planes; ++p) {
        // Plane 0 of the fit is always AbsMean, so iters == 0 reproduces the greedy expansion
        // bit-for-bit and the ITF path is a pure refinement on top (mirrors host fit_plane_itf).
        float partial = 0.0f;
        for (int i = threadIdx.x; i < n; i += blockDim.x) partial += fabsf(sh[i]);
        SALT_BLOCK_SUM(partial);
        float scale = red[0] / (float)n;
        __syncthreads();
        if (!(scale > 0.0f) || !isfinite(scale)) {
            // Dead group: the host returns (0, zeros), i.e. this plane contributes nothing.
            __syncthreads();
            continue;
        }

        { int k = 0;
          for (int i = threadIdx.x; i < n; i += blockDim.x) {
              trit[k] = fminf(1.0f, fmaxf(-1.0f, roundf(sh[i] / scale))); k++; } }
        __syncthreads();

        // best_sse = SUM (r - scale*t)^2
        partial = 0.0f;
        { int k = 0;
          for (int i = threadIdx.x; i < n; i += blockDim.x) {
              float d = sh[i] - scale * trit[k]; partial += d * d; k++; } }
        SALT_BLOCK_SUM(partial);
        float best_sse = red[0];
        __syncthreads();

        for (int it = 0; it < iters; ++it) {
            // s* = <r,t> / <t,t>, the least-squares optimal scale for the current trits.
            partial = 0.0f;
            { int k = 0;
              for (int i = threadIdx.x; i < n; i += blockDim.x) { partial += sh[i] * trit[k]; k++; } }
            SALT_BLOCK_SUM(partial);
            float num = red[0];
            __syncthreads();

            partial = 0.0f;
            { int k = 0;
              for (int i = threadIdx.x; i < n; i += blockDim.x) { partial += trit[k] * trit[k]; k++; } }
            SALT_BLOCK_SUM(partial);
            float den = red[0];
            __syncthreads();

            if (!(den > 0.0f)) break;              // host: ls_optimal_scale -> None
            float cand = num / den;
            if (!(cand > 0.0f) || !isfinite(cand)) break;

            { int k = 0;
              for (int i = threadIdx.x; i < n; i += blockDim.x) {
                  cand_trit[k] = fminf(1.0f, fmaxf(-1.0f, roundf(sh[i] / cand))); k++; } }
            __syncthreads();

            partial = 0.0f;
            { int k = 0;
              for (int i = threadIdx.x; i < n; i += blockDim.x) {
                  float d = sh[i] - cand * cand_trit[k]; partial += d * d; k++; } }
            SALT_BLOCK_SUM(partial);
            float cand_sse = red[0];
            __syncthreads();

            // Accept only on a strict improvement, so the result is never worse than AbsMean.
            if (cand_sse >= best_sse) break;
            scale = cand;
            best_sse = cand_sse;
            for (int k = 0; k < owned; ++k) trit[k] = cand_trit[k];
        }

        { int k = 0;
          for (int i = threadIdx.x; i < n; i += blockDim.x) {
              float contrib = scale * trit[k];
              acc[k] += contrib;
              sh[i] -= contrib;
              k++; } }
        __syncthreads();
    }
#undef SALT_BLOCK_SUM

    // Put the reconstruction back into shared, un-rotate, and store.
    { int k = 0;
      for (int i = threadIdx.x; i < n; i += blockDim.x) { sh[i] = acc[k]; k++; } }
    __syncthreads();
    if (rot) {
        for (int len = 1; len < n; len <<= 1) {
            for (int idx = threadIdx.x; idx < (n >> 1); idx += blockDim.x) {
                int blk = idx / len, off = idx % len;
                int i = blk * (len << 1) + off;
                float a = sh[i], b = sh[i + len];
                sh[i] = a + b; sh[i + len] = a - b;
            }
            __syncthreads();
        }
        float s = rsqrtf((float)n);
        for (int i = threadIdx.x; i < n; i += blockDim.x) sh[i] *= s;
        __syncthreads();
    }
    for (int i = threadIdx.x; i < n; i += blockDim.x) quantized[base + i] = sh[i];
}

extern "C" __global__ void salt_quantize_forward_bounded(
    const float* __restrict__ master, float* __restrict__ residual,
    float* __restrict__ quantized, int rows, int cols, int planes)
{
    if (blockIdx.x || threadIdx.x) return;
    for (int row = 0; row < rows; ++row) {
        long base = (long)row * cols;
        for (int col = 0; col < cols; ++col) {
            residual[col] = master[base + col];
            quantized[base + col] = 0.0f;
        }
        for (int plane = 0; plane < planes; ++plane) {
            float sum = 0.0f;
            for (int col = 0; col < cols; ++col) sum += fabsf(residual[col]);
            float scale = sum / (float)cols;
            if (scale == 0.0f) continue;
            for (int col = 0; col < cols; ++col) {
                float trit = roundf(residual[col] / scale);
                trit = fminf(1.0f, fmaxf(-1.0f, trit));
                float contribution = scale * trit;
                quantized[base + col] += contribution;
                residual[col] -= contribution;
            }
        }
    }
}

// ADR 0027 Track D: compact training-only SALT representation. Each plane keeps
// the canonical TQ2 2-bit address mapping but omits the inference format's f16
// block scale. Track A has one f32 AbsMean per (plane,row), so storing those
// scales externally is both exact and smaller: ceil(K/256)*64 code bytes/row.
#define TRAIN_SALT_QK 256
#define TRAIN_SALT_QS_BYTES 64

__device__ __forceinline__ long train_salt_code_offset(
    int plane, int row, int col, int rows, int row_bytes)
{
    int block = col / TRAIN_SALT_QK;
    int e = col - block * TRAIN_SALT_QK;
    int c = e >> 7;
    int mm = e & 31;
    return ((long)plane * rows + row) * row_bytes
        + block * TRAIN_SALT_QS_BYTES + c * 32 + mm;
}

__device__ __forceinline__ unsigned int train_salt_code(
    const unsigned char* __restrict__ codes,
    int plane, int row, int col, int rows, int row_bytes)
{
    int e = col & (TRAIN_SALT_QK - 1);
    int l = (e & 127) >> 5;
    long off = train_salt_code_offset(plane, row, col, rows, row_bytes);
    return ((unsigned int)codes[off] >> (2 * l)) & 3u;
}

// Reconstruct one compact SALT weight in the same plane order as
// salt_quantize_forward's dense quantized buffer. Keeping this helper scalar
// and compiling the module with --fmad=false makes the exact contractions
// below reproduce the dense materialize-then-contract arithmetic order without
// allocating the dense [N,K] weight.
__device__ __forceinline__ float train_salt_reconstruct_weight(
    const unsigned char* __restrict__ codes,
    const float* __restrict__ scales,
    int row, int col, int rows, int planes, int row_bytes)
{
    float weight = 0.0f;
    for (int plane = 0; plane < planes; ++plane) {
        unsigned int code = train_salt_code(
            codes, plane, row, col, rows, row_bytes);
        float scale = scales[(long)plane * rows + row];
        if (code == 2u) {
            weight += scale;
        } else if (code == 0u) {
            weight -= scale;
        }
    }
    return weight;
}

// Pack the resident latent master directly into compact planes. One thread owns
// a row, preserving Track A's ascending-column f32 AbsMean and residual order.
extern "C" __global__ void salt_pack_training(
    const float* __restrict__ master,       // [rows, cols]
    float* __restrict__ residual,           // [rows, cols] scratch
    unsigned char* __restrict__ codes,      // [planes, rows, row_bytes]
    float* __restrict__ scales,             // [planes, rows]
    int rows, int cols, int planes, int row_bytes)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    long base = (long)row * cols;

    for (int col = 0; col < cols; ++col) {
        residual[base + col] = master[base + col];
    }

    for (int plane = 0; plane < planes; ++plane) {
        long plane_row = ((long)plane * rows + row) * row_bytes;
        // TQ2 code 1 is zero; 0x55 initializes four zero trits per byte,
        // including padding in the final partial 256-trit block.
        for (int byte = 0; byte < row_bytes; ++byte) {
            codes[plane_row + byte] = 0x55u;
        }

        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) {
            sum += fabsf(residual[base + col]);
        }
        float scale = sum / (float)cols;
        scales[(long)plane * rows + row] = scale;
        if (scale == 0.0f) continue;

        for (int col = 0; col < cols; ++col) {
            long idx = base + col;
            float trit = roundf(residual[idx] / scale);
            trit = fminf(1.0f, fmaxf(-1.0f, trit));
            float contribution = scale * trit;
            residual[idx] -= contribution;

            int e = col & (TRAIN_SALT_QK - 1);
            int l = (e & 127) >> 5;
            long off = train_salt_code_offset(
                plane, row, col, rows, row_bytes);
            unsigned int shift = 2u * (unsigned int)l;
            unsigned int code = (unsigned int)((int)trit + 1);
            unsigned int old = codes[off];
            codes[off] = (unsigned char)(
                (old & ~(3u << shift)) | (code << shift));
        }
    }
}

// Y[M,N] = sum_p scale[p,n] * (A[M,K] dot trit[p,n,K]). The
// contraction is add/sub/skip; only one scale multiply is paid per plane/output.
extern "C" __global__ void salt_training_forward(
    const float* __restrict__ a,             // [M, K]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ y,                   // [M, N]
    int m, int n, int k, int planes, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * n) return;
    int mi = idx / n;
    int ni = idx % n;
    const float* arow = a + (long)mi * k;
    float out = 0.0f;

    for (int plane = 0; plane < planes; ++plane) {
        float acc = 0.0f;
        for (int ki = 0; ki < k; ++ki) {
            unsigned int code = train_salt_code(
                codes, plane, ni, ki, n, row_bytes);
            if (code == 2u) {
                acc += arow[ki];
            } else if (code == 0u) {
                acc -= arow[ki];
            }
        }
        out += acc * scales[(long)plane * n + ni];
    }
    y[idx] = out;
}

// gA[M,K] = sum_n,p gy[M,N] * scale[p,n] * trit[p,n,K]. A row scale
// is multiplied into gy once per plane/output-row, then the packed trit selects
// add/sub/skip; no dense quantized weight is materialized.
extern "C" __global__ void salt_training_grad_a(
    const float* __restrict__ gy,            // [M, N]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ ga,                  // [M, K]
    int m, int n, int k, int planes, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * k) return;
    int mi = idx / k;
    int ki = idx % k;
    float acc = 0.0f;

    for (int ni = 0; ni < n; ++ni) {
        float g = gy[(long)mi * n + ni];
        for (int plane = 0; plane < planes; ++plane) {
            float scaled = g * scales[(long)plane * n + ni];
            unsigned int code = train_salt_code(
                codes, plane, ni, ki, n, row_bytes);
            if (code == 2u) {
                acc += scaled;
            } else if (code == 0u) {
                acc -= scaled;
            }
        }
    }
    ga[idx] = acc;
}

// Numerically exact compact twin of dense SALT materialization followed by
// ternary_matmul_forward with an all-ones external scale. Each weight is first
// reconstructed in ascending plane order, then contracted in ascending K
// order. No dense [N,K] buffer is materialized.
extern "C" __global__ void salt_training_forward_exact(
    const float* __restrict__ a,             // [M, K]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ y,                   // [M, N]
    int m, int n, int k, int planes, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * n) return;
    int mi = idx / n;
    int ni = idx % n;
    const float* arow = a + (long)mi * k;
    float acc = 0.0f;

    for (int ki = 0; ki < k; ++ki) {
        float weight = train_salt_reconstruct_weight(
            codes, scales, ni, ki, n, planes, row_bytes);
        acc += arow[ki] * weight;
    }
    y[idx] = acc;
}

// Numerically exact compact twin of ternary_matmul_grad_a over a dense SALT
// reconstruction with an all-ones external scale. Reconstruction follows
// ascending plane order and the contraction follows ascending N order.
extern "C" __global__ void salt_training_grad_a_exact(
    const float* __restrict__ gy,            // [M, N]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ ga,                  // [M, K]
    int m, int n, int k, int planes, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)m * k) return;
    int mi = idx / k;
    int ki = idx % k;
    float acc = 0.0f;

    for (int ni = 0; ni < n; ++ni) {
        float weight = train_salt_reconstruct_weight(
            codes, scales, ni, ki, n, planes, row_bytes);
        acc += gy[(long)mi * n + ni] * weight;
    }
    ga[idx] = acc;
}

// Exact tiled twins of the scalar dense-order kernels above. A 32x16 block
// reconstructs one 64x32 compact weight tile into shared memory, then reuses
// it across sixteen activation rows. Each shared weight is reconstructed in
// ascending plane order, and each output-owning thread contracts its reduction
// dimension in strictly ascending order. Shared-memory staging changes where
// operands are read from, but never reassociates the f32 arithmetic.
#define TRAIN_SALT_EXACT_TILE_X 32
#define TRAIN_SALT_EXACT_TILE_M 16
#define TRAIN_SALT_EXACT_REDUCTION_TILE 64
#define TRAIN_SALT_EXACT_TILE_STRIDE (TRAIN_SALT_EXACT_TILE_X + 1)

extern "C" __global__ void salt_training_forward_exact_tiled(
    const float* __restrict__ a,             // [M, K]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ y,                   // [M, N]
    int m, int n, int k, int planes, int row_bytes)
{
    // [K,N+1] lets neighboring output columns read neighboring banks while
    // the padding avoids a 32-way alias between successive K rows.
    __shared__ float weight_tile
        [TRAIN_SALT_EXACT_REDUCTION_TILE][TRAIN_SALT_EXACT_TILE_STRIDE];
    __shared__ float activation_tile
        [TRAIN_SALT_EXACT_TILE_M][TRAIN_SALT_EXACT_REDUCTION_TILE];

    int ni = (int)blockIdx.x * TRAIN_SALT_EXACT_TILE_X + (int)threadIdx.x;
    int mi = (int)blockIdx.y * TRAIN_SALT_EXACT_TILE_M + (int)threadIdx.y;
    int lane = (int)threadIdx.y * TRAIN_SALT_EXACT_TILE_X + (int)threadIdx.x;
    int threads = TRAIN_SALT_EXACT_TILE_X * TRAIN_SALT_EXACT_TILE_M;
    float acc = 0.0f;

    for (int k_base = 0; k_base < k;
         k_base += TRAIN_SALT_EXACT_REDUCTION_TILE) {
        for (int load = lane;
             load < TRAIN_SALT_EXACT_TILE_M * TRAIN_SALT_EXACT_REDUCTION_TILE;
             load += threads) {
            int tile_m = load / TRAIN_SALT_EXACT_REDUCTION_TILE;
            int tile_k = load - tile_m * TRAIN_SALT_EXACT_REDUCTION_TILE;
            int global_m = (int)blockIdx.y * TRAIN_SALT_EXACT_TILE_M + tile_m;
            int global_k = k_base + tile_k;
            activation_tile[tile_m][tile_k] =
                global_m < m && global_k < k
                    ? a[(long)global_m * k + global_k]
                    : 0.0f;
        }
        for (int load = lane;
             load < TRAIN_SALT_EXACT_TILE_X * TRAIN_SALT_EXACT_REDUCTION_TILE;
             load += threads) {
            int tile_n = load / TRAIN_SALT_EXACT_REDUCTION_TILE;
            int tile_k = load - tile_n * TRAIN_SALT_EXACT_REDUCTION_TILE;
            int global_n = (int)blockIdx.x * TRAIN_SALT_EXACT_TILE_X + tile_n;
            int global_k = k_base + tile_k;
            float weight = 0.0f;
            if (global_n < n && global_k < k) {
                weight = train_salt_reconstruct_weight(
                    codes, scales, global_n, global_k, n, planes, row_bytes);
            }
            weight_tile[tile_k][tile_n] = weight;
        }
        __syncthreads();

        if (mi < m && ni < n) {
            int tile_k_count = k - k_base;
            if (tile_k_count > TRAIN_SALT_EXACT_REDUCTION_TILE) {
                tile_k_count = TRAIN_SALT_EXACT_REDUCTION_TILE;
            }
            for (int tile_k = 0; tile_k < tile_k_count; ++tile_k) {
                acc += activation_tile[threadIdx.y][tile_k]
                    * weight_tile[tile_k][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (mi < m && ni < n) {
        y[(long)mi * n + ni] = acc;
    }
}

extern "C" __global__ void salt_training_grad_a_exact_tiled(
    const float* __restrict__ gy,            // [M, N]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ ga,                  // [M, K]
    int m, int n, int k, int planes, int row_bytes)
{
    // [N,K+1] gives each warp a conflict-free row when neighboring K output
    // columns consume the same reconstructed N row.
    __shared__ float weight_tile
        [TRAIN_SALT_EXACT_REDUCTION_TILE][TRAIN_SALT_EXACT_TILE_STRIDE];
    __shared__ float gy_tile
        [TRAIN_SALT_EXACT_TILE_M][TRAIN_SALT_EXACT_REDUCTION_TILE];

    int ki = (int)blockIdx.x * TRAIN_SALT_EXACT_TILE_X + (int)threadIdx.x;
    int mi = (int)blockIdx.y * TRAIN_SALT_EXACT_TILE_M + (int)threadIdx.y;
    int lane = (int)threadIdx.y * TRAIN_SALT_EXACT_TILE_X + (int)threadIdx.x;
    int threads = TRAIN_SALT_EXACT_TILE_X * TRAIN_SALT_EXACT_TILE_M;
    float acc = 0.0f;

    for (int n_base = 0; n_base < n;
         n_base += TRAIN_SALT_EXACT_REDUCTION_TILE) {
        for (int load = lane;
             load < TRAIN_SALT_EXACT_TILE_M * TRAIN_SALT_EXACT_REDUCTION_TILE;
             load += threads) {
            int tile_m = load / TRAIN_SALT_EXACT_REDUCTION_TILE;
            int tile_n = load - tile_m * TRAIN_SALT_EXACT_REDUCTION_TILE;
            int global_m = (int)blockIdx.y * TRAIN_SALT_EXACT_TILE_M + tile_m;
            int global_n = n_base + tile_n;
            gy_tile[tile_m][tile_n] =
                global_m < m && global_n < n
                    ? gy[(long)global_m * n + global_n]
                    : 0.0f;
        }
        for (int load = lane;
             load < TRAIN_SALT_EXACT_TILE_X * TRAIN_SALT_EXACT_REDUCTION_TILE;
             load += threads) {
            int tile_n = load / TRAIN_SALT_EXACT_TILE_X;
            int tile_k = load - tile_n * TRAIN_SALT_EXACT_TILE_X;
            int global_n = n_base + tile_n;
            int global_k = (int)blockIdx.x * TRAIN_SALT_EXACT_TILE_X + tile_k;
            float weight = 0.0f;
            if (global_n < n && global_k < k) {
                weight = train_salt_reconstruct_weight(
                    codes, scales, global_n, global_k, n, planes, row_bytes);
            }
            weight_tile[tile_n][tile_k] = weight;
        }
        __syncthreads();

        if (mi < m && ki < k) {
            int tile_n_count = n - n_base;
            if (tile_n_count > TRAIN_SALT_EXACT_REDUCTION_TILE) {
                tile_n_count = TRAIN_SALT_EXACT_REDUCTION_TILE;
            }
            for (int tile_n = 0; tile_n < tile_n_count; ++tile_n) {
                acc += gy_tile[threadIdx.y][tile_n]
                    * weight_tile[tile_n][threadIdx.x];
            }
        }
        __syncthreads();
    }

    if (mi < m && ki < k) {
        ga[(long)mi * k + ki] = acc;
    }
}

// Tiled Track D contractions. One thread still owns one output and visits its
// reduction dimension in strictly ascending order; tiling only shares repeated
// activation/gy reads across neighboring outputs. This preserves the scalar
// kernels' f32 accumulation contract while mapping threadIdx.x to contiguous
// output columns. Forward stores and grad-A packed-code reads are coalesced;
// the activation/gy tiles remove the corresponding repeated global reads.
#define TRAIN_SALT_TILE_X 32
#define TRAIN_SALT_TILE_M 4
#define TRAIN_SALT_FORWARD_K_TILE TRAIN_SALT_QK
#define TRAIN_SALT_GRAD_N_TILE 64
#define TRAIN_SALT_CODE_N_STRIDE (TRAIN_SALT_TILE_X + 1)

extern "C" __global__ void salt_training_forward_tiled(
    const float* __restrict__ a,             // [M, K]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ y,                   // [M, N]
    int m, int n, int k, int planes, int row_bytes)
{
    __shared__ float activation_tile
        [TRAIN_SALT_TILE_M][TRAIN_SALT_FORWARD_K_TILE];
    // Transpose [N,byte] global rows to [byte,N+1] shared rows. Cooperative
    // loads are contiguous in each packed row; the +1 avoids shared-bank
    // aliasing when output threads read the same byte from adjacent N rows.
    __shared__ unsigned char code_tile
        [3][TRAIN_SALT_QS_BYTES][TRAIN_SALT_CODE_N_STRIDE];

    int ni = (int)blockIdx.x * TRAIN_SALT_TILE_X + (int)threadIdx.x;
    int mi = (int)blockIdx.y * TRAIN_SALT_TILE_M + (int)threadIdx.y;
    int lane = (int)threadIdx.y * TRAIN_SALT_TILE_X + (int)threadIdx.x;
    int threads = TRAIN_SALT_TILE_X * TRAIN_SALT_TILE_M;
    float plane_acc[3] = {0.0f, 0.0f, 0.0f};

    for (int k_base = 0; k_base < k; k_base += TRAIN_SALT_FORWARD_K_TILE) {
        for (int load = lane;
             load < TRAIN_SALT_TILE_M * TRAIN_SALT_FORWARD_K_TILE;
             load += threads) {
            int tile_m = load / TRAIN_SALT_FORWARD_K_TILE;
            int tile_k = load - tile_m * TRAIN_SALT_FORWARD_K_TILE;
            int global_m = (int)blockIdx.y * TRAIN_SALT_TILE_M + tile_m;
            int global_k = k_base + tile_k;
            activation_tile[tile_m][tile_k] =
                global_m < m && global_k < k
                    ? a[(long)global_m * k + global_k]
                    : 0.0f;
        }
        int code_values = planes * TRAIN_SALT_TILE_X * TRAIN_SALT_QS_BYTES;
        for (int load = lane; load < code_values; load += threads) {
            int plane_stride = TRAIN_SALT_TILE_X * TRAIN_SALT_QS_BYTES;
            int plane = load / plane_stride;
            int plane_offset = load - plane * plane_stride;
            int tile_n = plane_offset / TRAIN_SALT_QS_BYTES;
            int byte = plane_offset - tile_n * TRAIN_SALT_QS_BYTES;
            int global_n = (int)blockIdx.x * TRAIN_SALT_TILE_X + tile_n;
            long global_offset = ((long)plane * n + global_n) * row_bytes
                + (long)(k_base / TRAIN_SALT_QK) * TRAIN_SALT_QS_BYTES + byte;
            code_tile[plane][byte][tile_n] = global_n < n
                ? codes[global_offset]
                : 0x55u;
        }
        __syncthreads();

        if (mi < m && ni < n) {
            int tile_k_count = k - k_base;
            if (tile_k_count > TRAIN_SALT_FORWARD_K_TILE) {
                tile_k_count = TRAIN_SALT_FORWARD_K_TILE;
            }
            for (int tile_k = 0; tile_k < tile_k_count; ++tile_k) {
                float value = activation_tile[threadIdx.y][tile_k];
                int c = tile_k >> 7;
                int mm = tile_k & 31;
                int l = (tile_k & 127) >> 5;
                int byte = c * 32 + mm;
                for (int plane = 0; plane < planes; ++plane) {
                    unsigned int code =
                        ((unsigned int)code_tile[plane][byte][threadIdx.x]
                            >> (2 * l)) & 3u;
                    if (code == 2u) {
                        plane_acc[plane] += value;
                    } else if (code == 0u) {
                        plane_acc[plane] -= value;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (mi < m && ni < n) {
        float out = 0.0f;
        for (int plane = 0; plane < planes; ++plane) {
            out += plane_acc[plane] * scales[(long)plane * n + ni];
        }
        y[(long)mi * n + ni] = out;
    }
}

extern "C" __global__ void salt_training_grad_a_tiled(
    const float* __restrict__ gy,            // [M, N]
    const unsigned char* __restrict__ codes, // [planes, N, row_bytes]
    const float* __restrict__ scales,        // [planes, N]
    float* __restrict__ ga,                  // [M, K]
    int m, int n, int k, int planes, int row_bytes)
{
    __shared__ float scaled_gy_tile
        [3][TRAIN_SALT_TILE_M][TRAIN_SALT_GRAD_N_TILE];

    int ki = (int)blockIdx.x * TRAIN_SALT_TILE_X + (int)threadIdx.x;
    int mi = (int)blockIdx.y * TRAIN_SALT_TILE_M + (int)threadIdx.y;
    int lane = (int)threadIdx.y * TRAIN_SALT_TILE_X + (int)threadIdx.x;
    int threads = TRAIN_SALT_TILE_X * TRAIN_SALT_TILE_M;
    float acc = 0.0f;

    for (int n_base = 0; n_base < n; n_base += TRAIN_SALT_GRAD_N_TILE) {
        for (int load = lane;
             load < planes * TRAIN_SALT_TILE_M * TRAIN_SALT_GRAD_N_TILE;
             load += threads) {
            int plane_stride = TRAIN_SALT_TILE_M * TRAIN_SALT_GRAD_N_TILE;
            int plane = load / plane_stride;
            int plane_offset = load - plane * plane_stride;
            int tile_m = plane_offset / TRAIN_SALT_GRAD_N_TILE;
            int tile_n = plane_offset - tile_m * TRAIN_SALT_GRAD_N_TILE;
            int global_m = (int)blockIdx.y * TRAIN_SALT_TILE_M + tile_m;
            int global_n = n_base + tile_n;
            scaled_gy_tile[plane][tile_m][tile_n] =
                global_m < m && global_n < n
                    ? gy[(long)global_m * n + global_n]
                        * scales[(long)plane * n + global_n]
                    : 0.0f;
        }
        __syncthreads();

        if (mi < m && ki < k) {
            int tile_n_count = n - n_base;
            if (tile_n_count > TRAIN_SALT_GRAD_N_TILE) {
                tile_n_count = TRAIN_SALT_GRAD_N_TILE;
            }
            for (int tile_n = 0; tile_n < tile_n_count; ++tile_n) {
                int ni = n_base + tile_n;
                for (int plane = 0; plane < planes; ++plane) {
                    float scaled = scaled_gy_tile[plane][threadIdx.y][tile_n];
                    unsigned int code = train_salt_code(
                        codes, plane, ni, ki, n, row_bytes);
                    if (code == 2u) {
                        acc += scaled;
                    } else if (code == 0u) {
                        acc -= scaled;
                    }
                }
            }
        }
        __syncthreads();
    }

    if (mi < m && ki < k) {
        ga[(long)mi * k + ki] = acc;
    }
}

// Tied-token embedding gather directly from the same compact Track D planes
// used by salt_training_forward. One thread reconstructs one requested table
// cell; code ±1 adds/subtracts the external f32 row scale, code 0 skips.
extern "C" __global__ void salt_training_embed_gather(
    const unsigned char* __restrict__ codes, // [planes, vocab, row_bytes]
    const float* __restrict__ scales,        // [planes, vocab]
    const int* __restrict__ tokens,          // [seq]
    float* __restrict__ out,                 // [seq, dim]
    int seq, int vocab, int dim, int planes, int row_bytes)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)seq * dim) return;
    int position = idx / dim;
    int col = idx % dim;
    int row = tokens[position];
    // Public host entry points validate token ids before upload. Keep the
    // device guard as defence in depth against any future resident-token path.
    if (row < 0 || row >= vocab) {
        out[idx] = 0.0f;
        return;
    }

    float value = 0.0f;
    for (int plane = 0; plane < planes; ++plane) {
        unsigned int code = train_salt_code(
            codes, plane, row, col, vocab, row_bytes);
        float scale = scales[(long)plane * vocab + row];
        if (code == 2u) {
            value += scale;
        } else if (code == 0u) {
            value -= scale;
        }
    }
    out[idx] = value;
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

// Block-wise int8 AdamW (Lever 5): one CUDA block per ADAMW8_BLOCK-element optimizer block. Mirrors
// the CPU oracle tritium_train::optim::Int8AdamW BIT-FOR-BIT — m signed int8 (per-block absmax/127);
// the second moment stored in SQRT-SPACE unsigned int8 (v_q = round(sqrt(v)/scale), v = (v_q·scale)²)
// so its wide dynamic range does not underflow while m keeps its history. The param buffer stays f32
// and the update uses the full-precision mi,vi — only the persisted STATE is quantized. All ops are
// +,−,×,÷,sqrtf,roundf,fmaxf (IEEE correctly-rounded) and the kernel builds --fmad=false, so it is
// bit-identical to the host oracle; the requant reduction (max) is order-independent. Launch with
// exactly ADAMW8_BLOCK threads per block and ceil(n/ADAMW8_BLOCK) blocks.
#define ADAMW8_BLOCK 256
extern "C" __global__ void adamw_step_8bit(
    float* __restrict__ param,
    const float* __restrict__ grad,
    signed char* __restrict__ m_q,
    unsigned char* __restrict__ v_q,
    float* __restrict__ m_scale, // [nblocks]
    float* __restrict__ v_scale, // [nblocks]
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
    int blk = blockIdx.x;
    int tid = threadIdx.x;
    int i = blk * ADAMW8_BLOCK + tid;
    __shared__ float s_mabs[ADAMW8_BLOCK];
    __shared__ float s_rabs[ADAMW8_BLOCK];
    __shared__ float new_ms;
    __shared__ float new_vs;

    float mi = 0.0f;
    float root_i = 0.0f; // sqrt(vi) — the value requantized in sqrt-space
    bool active = (i < n);
    if (active) {
        float g = grad[i];
        float ms = m_scale[blk];
        float vs = v_scale[blk];
        float m_old = (float)m_q[i] * ms;
        float root_old = (float)v_q[i] * vs;
        float v_old = root_old * root_old;
        mi = beta1 * m_old + one_minus_beta1 * g;
        float vi = beta2 * v_old + one_minus_beta2 * g * g;
        param[i] = param[i] * shrink - lr * (mi / bc1 / (sqrtf(vi / bc2) + eps));
        root_i = sqrtf(vi);
    }
    // Per-block absmax of m and of sqrt(v). Inactive tail threads contribute 0.
    s_mabs[tid] = active ? fabsf(mi) : 0.0f;
    s_rabs[tid] = active ? root_i : 0.0f;
    __syncthreads();
    for (int stride = ADAMW8_BLOCK / 2; stride > 0; stride >>= 1) {
        if (tid < stride) {
            s_mabs[tid] = fmaxf(s_mabs[tid], s_mabs[tid + stride]);
            s_rabs[tid] = fmaxf(s_rabs[tid], s_rabs[tid + stride]);
        }
        __syncthreads();
    }
    if (tid == 0) {
        new_ms = s_mabs[0] > 0.0f ? s_mabs[0] / 127.0f : 0.0f;
        new_vs = s_rabs[0] > 0.0f ? s_rabs[0] / 255.0f : 0.0f;
        m_scale[blk] = new_ms;
        v_scale[blk] = new_vs;
    }
    __syncthreads();
    if (active) {
        // roundf is round-half-away-from-zero, matching Rust f32::round; a zero-absmax block keeps
        // scale 0 and code 0 (dequantizes back to 0 — no division by zero).
        if (new_ms > 0.0f) {
            m_q[i] = (signed char)fminf(fmaxf(roundf(mi / new_ms), -127.0f), 127.0f);
        } else {
            m_q[i] = 0;
        }
        // Floor a nonzero sqrt(v) at code 1: it must never dequantize to 0, or a later quiet step
        // (g~0, residual m) collapses the AdamW denominator to eps and the step explodes.
        if (new_vs > 0.0f && root_i > 0.0f) {
            v_q[i] = (unsigned char)fminf(fmaxf(roundf(root_i / new_vs), 1.0f), 255.0f);
        } else {
            v_q[i] = 0;
        }
    }
}

// ── Lever 5: bf16 master (stochastic rounding). Mirrors tritium_train::bf16 bit-for-bit. ──
// bf16 = the high 16 bits of an f32, so widen with a shift and stochastic-round by adding a 16-bit
// dither to the low bits and truncating (the carry out IS the round-up). The dither stream is the
// same xorshift64(seed, idx) as the FSQ STE and the Rust `dither16`, so host and device produce
// identical codes for the same (seed, index).
__device__ __forceinline__ float bf16_to_f32(unsigned short bits) {
    return __uint_as_float(((unsigned int)bits) << 16);
}
__device__ __forceinline__ unsigned short f32_to_bf16_stochastic(float x, unsigned short dither) {
    unsigned int bits = __float_as_uint(x);
    if (!isfinite(x)) {
        // from_f32_nearest fallback: preserve inf; keep a NaN a NaN.
        unsigned short hi = (unsigned short)(bits >> 16);
        return (x != x) ? (unsigned short)(hi | 0x0040) : hi;
    }
    return (unsigned short)((bits + (unsigned int)dither) >> 16);
}
__device__ __forceinline__ unsigned short bf16_dither16(unsigned long long seed,
                                                        unsigned long long idx) {
    unsigned long long s = (seed ^ (idx * 0x9E3779B97F4A7C15ULL)) | 1ULL;
    s ^= s << 13;
    s ^= s >> 7;
    s ^= s << 17;
    return (unsigned short)(s >> 32);
}

// AdamW on a bf16 master (Lever 5): dequantize the bf16 weight, apply the exact f32 AdamW update, and
// stochastic-round back to bf16 so a sub-ULP update survives in expectation. Moments stay f32. One
// thread per element; `dither_seed` should already fold in the step index (host passes seed ^ step).
extern "C" __global__ void adamw_step_bf16_master(
    unsigned short* __restrict__ master, // bf16 in/out
    const float* __restrict__ grad,
    float* __restrict__ m,
    float* __restrict__ v,
    int n,
    unsigned long long dither_seed,
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
    float w = bf16_to_f32(master[i]);
    float g = grad[i];
    float mi = beta1 * m[i] + one_minus_beta1 * g;
    float vi = beta2 * v[i] + one_minus_beta2 * g * g;
    m[i] = mi;
    v[i] = vi;
    float w_new = w * shrink - lr * (mi / bc1 / (sqrtf(vi / bc2) + eps));
    master[i] = f32_to_bf16_stochastic(w_new, bf16_dither16(dither_seed, (unsigned long long)i));
}

// Stochastically round an f32 buffer onto the bf16 grid in place (Lever 5 bf16-master validation).
// Confining the f32 master to bf16-representable values with SR is numerically identical to holding
// the master in a u16 bf16 buffer and dequantizing it for the SALT reconstruction — so this validates
// the recovery impact of a bf16 master without swapping the storage type through the reconstruction
// path. One thread per element; `dither_seed` should fold in the step.
extern "C" __global__ void sr_round_to_bf16grid(
    float* __restrict__ buf, int n, unsigned long long dither_seed)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    unsigned short d = bf16_dither16(dither_seed, (unsigned long long)i);
    buf[i] = bf16_to_f32(f32_to_bf16_stochastic(buf[i], d));
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
    const float* __restrict__ x, float* __restrict__ y, const unsigned int* __restrict__ positions,
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

// Segmented embedding backward. Host metadata groups equal token ids while
// preserving ascending original position inside each group. One thread owns
// one (unique vocab row, dimension), so its additions exactly match the CPU
// scatter order without atomics. Untouched gw rows are zeroed by the caller.
extern "C" __global__ void embed_gather_backward_segmented(
    const float* __restrict__ gy, const int* __restrict__ metadata,
    float* __restrict__ gw, int unique_rows, int dim)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= (long)unique_rows * dim) return;
    int u = idx / dim, d = idx % dim;
    const int* rows = metadata;
    const int* offsets = rows + unique_rows;
    const int* positions = offsets + unique_rows + 1;
    float acc = 0.0f;
    for (int j = offsets[u]; j < offsets[u + 1]; ++j)
        acc += gy[(long)positions[j] * dim + d];
    gw[(long)rows[u] * dim + d] = acc;
}

// Row-broadcast bias. Backward bias reduction stays ascending-row and bit-matches CPU order.
extern "C" __global__ void bias_forward(
    const float* __restrict__ x, const float* __restrict__ bias,
    float* __restrict__ y, int rows, int cols)
{
    long idx = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < (long)rows * cols) y[idx] = x[idx] + bias[idx % cols];
}

extern "C" __global__ void bias_backward(
    const float* __restrict__ gy, float* __restrict__ gb, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float acc = 0.0f;
    for (int row = 0; row < rows; ++row) acc += gy[(long)row * cols + col];
    gb[col] = acc;
}

extern "C" __global__ void bias_backward_f16(
    const __half* __restrict__ gy, __half* __restrict__ gb, int rows, int cols)
{
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    float acc = 0.0f;
    for (int row = 0; row < rows; ++row)
        acc += __half2float(gy[(long)row * cols + col]);
    gb[col] = __float2half(acc);
}

extern "C" __global__ void relu2_forward(
    const float* __restrict__ x, float* __restrict__ y, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; y[i] = v > 0.0f ? v * v : 0.0f; }
}

extern "C" __global__ void relu2_backward(
    const float* __restrict__ x, const float* __restrict__ gy,
    float* __restrict__ gx, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) { float v = x[i]; gx[i] = v > 0.0f ? gy[i] * (2.0f * v) : 0.0f; }
}

extern "C" __global__ void mse_forward(
    const float* __restrict__ prediction, const float* __restrict__ target,
    float* __restrict__ loss, long n)
{
    if (blockIdx.x || threadIdx.x) return;
    float sum = 0.0f;
    for (long i = 0; i < n; ++i) { float d = prediction[i] - target[i]; sum += d * d; }
    loss[0] = sum / (float)n;
}

extern "C" __global__ void mse_backward(
    const float* __restrict__ prediction, const float* __restrict__ target,
    const float* __restrict__ upstream, float* __restrict__ gradient, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) gradient[i] = upstream[0] * 2.0f * (prediction[i] - target[i]) / (float)n;
}

extern "C" __global__ void softmax_xent_forward(
    const float* __restrict__ logits, const float* __restrict__ target,
    float* __restrict__ loss, int rows, int cols)
{
    if (blockIdx.x || threadIdx.x) return;
    float total = 0.0f;
    for (int row = 0; row < rows; ++row) {
        const float* lr = logits + (long)row * cols;
        const float* tr = target + (long)row * cols;
        float maximum = -INFINITY;
        for (int col = 0; col < cols; ++col) maximum = fmaxf(maximum, lr[col]);
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) sum += expf(lr[col] - maximum);
        for (int col = 0; col < cols; ++col) {
            float probability = expf(lr[col] - maximum) / sum;
            total -= tr[col] * logf(fmaxf(probability, 1.17549435e-38f));
        }
    }
    loss[0] = total / (float)rows;
}

extern "C" __global__ void topk_kd_forward(
    const float* __restrict__ logits, const unsigned int* __restrict__ indices,
    const float* __restrict__ probabilities, float* __restrict__ loss,
    int rows, int cols, int k)
{
    if (blockIdx.x || threadIdx.x) return;
    float total = 0.0f;
    for (int row = 0; row < rows; ++row) {
        const float* lr = logits + (long)row * cols;
        const unsigned int* ir = indices + (long)row * k;
        const float* pr = probabilities + (long)row * k;
        float maximum = -INFINITY;
        for (int col = 0; col < cols; ++col) maximum = fmaxf(maximum, lr[col]);
        float sum = 0.0f;
        for (int col = 0; col < cols; ++col) sum += expf(lr[col] - maximum);
        for (int sparse_col = 0; sparse_col < k; ++sparse_col) {
            float probability = expf(lr[ir[sparse_col]] - maximum) / sum;
            total -= pr[sparse_col] * logf(fmaxf(probability, 1.17549435e-38f));
        }
    }
    loss[0] = total / (float)rows;
}

extern "C" __global__ void ste_surrogate_forward(
    const float* __restrict__ weight, const float* __restrict__ scale,
    float* __restrict__ result, int rows, int cols)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * cols) return;
    float s = scale[i / cols];
    result[i] = s == 0.0f ? 0.0f : fminf(1.0f, fmaxf(-1.0f, weight[i] / s));
}

extern "C" __global__ void ste_surrogate_backward(
    const float* __restrict__ weight, const float* __restrict__ scale,
    const float* __restrict__ upstream, float* __restrict__ grad_weight,
    int rows, int cols)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * cols) return;
    float s = scale[i / cols];
    grad_weight[i] = s != 0.0f && fabsf(weight[i] / s) < 1.0f ? upstream[i] / s : 0.0f;
}

extern "C" __global__ void lsq_forward(
    const float* __restrict__ weight, const float* __restrict__ alpha,
    float* __restrict__ result, int rows, int cols)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * cols) return;
    float a = alpha[i / cols];
    result[i] = a > 0.0f ? fminf(1.0f, fmaxf(-1.0f, roundf(weight[i] / a))) * a : 0.0f;
}

extern "C" __global__ void lsq_backward_weight(
    const float* __restrict__ weight, const float* __restrict__ alpha,
    const float* __restrict__ upstream, float* __restrict__ grad_weight,
    int rows, int cols)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)rows * cols) return;
    float a = alpha[i / cols];
    grad_weight[i] = a > 0.0f && fabsf(weight[i] / a) < 1.0f ? upstream[i] : 0.0f;
}

extern "C" __global__ void lsq_backward_alpha(
    const float* __restrict__ weight, const float* __restrict__ alpha,
    const float* __restrict__ upstream, float* __restrict__ grad_alpha,
    int rows, int cols)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    float a = alpha[row];
    float gradient = 0.0f;
    if (a > 0.0f) {
        for (int col = 0; col < cols; ++col) {
            long i = (long)row * cols + col;
            float normalized = weight[i] / a;
            float local = fabsf(normalized) < 1.0f
                ? roundf(normalized) - normalized
                : copysignf(1.0f, normalized);
            gradient += upstream[i] * local;
        }
        gradient *= 1.0f / sqrtf((float)cols);
    }
    grad_alpha[row] = gradient;
}

__device__ __forceinline__ float fsq_bound_value(float value, int bound)
{
    return bound == 0 ? fminf(1.0f, fmaxf(-1.0f, value)) : tanhf(value);
}

__device__ __forceinline__ float fsq_bound_gradient(float value, int bound)
{
    if (bound == 0) return fabsf(value) < 1.0f ? 1.0f : 0.0f;
    float bounded = tanhf(value);
    return 1.0f - bounded * bounded;
}

__device__ __forceinline__ float fsq_uniform(unsigned long long seed, unsigned long long index)
{
    unsigned long long state = (seed ^ index * 0x9E3779B97F4A7C15ULL) | 1ULL;
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return (float)(state % 1000000ULL) / 1000000.0f;
}

extern "C" __global__ void fsq_forward(
    const float* __restrict__ x, const unsigned int* __restrict__ levels,
    float* __restrict__ result, int channels, int len, int bound, int estimator,
    unsigned long long seed)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)channels * len) return;
    float maximum = (float)(levels[i / len] - 1U);
    float bounded = fsq_bound_value(x[i], bound);
    float position = (bounded + 1.0f) * 0.5f * maximum;
    float code;
    if (estimator == 2) {
        position = fminf(maximum, fmaxf(0.0f, position));
        float base = floorf(position);
        code = base + (fsq_uniform(seed, (unsigned long long)i) < position - base ? 1.0f : 0.0f);
    } else {
        code = floorf(position + 0.5f);
    }
    code = fminf(maximum, fmaxf(0.0f, code));
    result[i] = code / maximum * 2.0f - 1.0f;
}

extern "C" __global__ void fsq_backward(
    const float* __restrict__ x, const unsigned int* __restrict__ levels,
    const float* __restrict__ upstream, float* __restrict__ grad_x,
    int channels, int len, int bound, int estimator, float alpha)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (long)channels * len) return;
    float derivative = fsq_bound_gradient(x[i], bound);
    if (estimator == 1) {
        float maximum = (float)(levels[i / len] - 1U);
        float bounded = fsq_bound_value(x[i], bound);
        float position = (bounded + 1.0f) * 0.5f * maximum;
        derivative *= 1.0f - alpha * cosf(2.0f * 3.14159265358979323846f * position);
    }
    grad_x[i] = upstream[i] * derivative;
}

extern "C" __global__ void sgd_step(
    const float* __restrict__ parameter, const float* __restrict__ gradient,
    float* __restrict__ updated, float lr, long n)
{
    long i = (long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) updated[i] = parameter[i] - lr * gradient[i];
}

extern "C" __global__ void cautious_adamw_prepare(
    const float* __restrict__ grad, float* __restrict__ m,
    float* __restrict__ v, float* __restrict__ update,
    unsigned int* __restrict__ aligned, int n, float beta1, float beta2,
    float one_minus_beta1, float one_minus_beta2, float bc1, float bc2, float eps)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float g = grad[i];
    float mi = beta1 * m[i] + one_minus_beta1 * g;
    float vi = beta2 * v[i] + one_minus_beta2 * g * g;
    m[i] = mi;
    v[i] = vi;
    float u = (mi / bc1) / (sqrtf(vi / bc2) + eps);
    if (u * g > 0.0f) {
        update[i] = u;
        atomicAdd(aligned, 1U);
    } else {
        update[i] = 0.0f;
    }
}

extern "C" __global__ void cautious_adamw_apply(
    float* __restrict__ param, const float* __restrict__ update,
    const unsigned int* __restrict__ aligned, int n, float lr, float shrink)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    float rescale = (float)n / ((float)aligned[0] + 1.0f);
    param[i] = param[i] * shrink - lr * update[i] * rescale;
}

extern "C" __global__ void muon_step(
    float* __restrict__ parameter, const float* __restrict__ gradient,
    float* __restrict__ momentum, float* __restrict__ x,
    float* __restrict__ xt, float* __restrict__ bx,
    float* __restrict__ gram, float* __restrict__ gram2,
    float* __restrict__ bmat, int rows, int cols, int steps,
    float momentum_decay, float scale, float shrink)
{
    if (blockIdx.x || threadIdx.x) return;
    const float qa = 3.4445f, qb = -4.7750f, qc = 2.0315f;
    int n = rows * cols;
    float norm2 = 0.0f;
    for (int i = 0; i < n; ++i) {
        float value = momentum_decay * momentum[i] + gradient[i];
        momentum[i] = value;
        norm2 += value * value;
    }
    float fnorm = sqrtf(norm2) + 1.0e-7f;
    int r = rows <= cols ? rows : cols;
    int c = rows <= cols ? cols : rows;
    bool transposed = rows > cols;
    for (int row = 0; row < r; ++row) {
        for (int col = 0; col < c; ++col) {
            int source = transposed ? col * cols + row : row * cols + col;
            x[row * c + col] = momentum[source] / fnorm;
        }
    }
    for (int iteration = 0; iteration < steps; ++iteration) {
        for (int row = 0; row < r; ++row) {
            for (int col = 0; col < r; ++col) {
                float sum = 0.0f;
                for (int k = 0; k < c; ++k) sum += x[row * c + k] * x[col * c + k];
                gram[row * r + col] = sum;
            }
        }
        for (int row = 0; row < r; ++row) {
            for (int col = 0; col < r; ++col) {
                float sum = 0.0f;
                for (int k = 0; k < r; ++k) sum += gram[row * r + k] * gram[col * r + k];
                gram2[row * r + col] = sum;
            }
        }
        for (int i = 0; i < r * r; ++i) bmat[i] = qb * gram[i] + qc * gram2[i];
        for (int row = 0; row < c; ++row)
            for (int col = 0; col < r; ++col) xt[row * r + col] = x[col * c + row];
        for (int row = 0; row < r; ++row) {
            for (int col = 0; col < c; ++col) {
                float sum = 0.0f;
                for (int k = 0; k < r; ++k) sum += bmat[row * r + k] * xt[col * r + k];
                bx[row * c + col] = sum;
            }
        }
        for (int i = 0; i < r * c; ++i) x[i] = qa * x[i] + bx[i];
    }
    for (int row = 0; row < rows; ++row) {
        for (int col = 0; col < cols; ++col) {
            int i = row * cols + col;
            int ortho = transposed ? col * c + row : i;
            parameter[i] = parameter[i] * shrink - scale * x[ortho];
        }
    }
}

extern "C" __global__ void conv1d_forward_portable(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ scale, float* __restrict__ result,
    int batch, int c_in, int c_out, int l_in, int kernel, int stride,
    int dilation, int pad_left, int groups, int l_out)
{
    if (blockIdx.x || threadIdx.x) return;
    int c_in_pg = c_in / groups;
    int n_g = c_out / groups;
    int k_g = c_in_pg * kernel;
    for (int b = 0; b < batch; ++b) {
        for (int group = 0; group < groups; ++group) {
            for (int l = 0; l < l_out; ++l) {
                for (int n = 0; n < n_g; ++n) {
                    int co = group * n_g + n;
                    float acc = 0.0f;
                    for (int j = 0; j < k_g; ++j) {
                        int ci_local = j / kernel;
                        int kk = j - ci_local * kernel;
                        int position = l * stride + kk * dilation - pad_left;
                        float activation = 0.0f;
                        if (position >= 0 && position < l_in) {
                            int ci = group * c_in_pg + ci_local;
                            activation = x[(b * c_in + ci) * l_in + position];
                        }
                        acc += activation * weight[co * k_g + j];
                    }
                    result[(b * c_out + co) * l_out + l] = scale[co] * acc;
                }
            }
        }
    }
}

extern "C" __global__ void conv1d_backward_portable(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ scale, const float* __restrict__ grad_output,
    float* __restrict__ grad_x, float* __restrict__ grad_weight,
    float* __restrict__ grad_scale, float* __restrict__ columns,
    float* __restrict__ group_gradient, float* __restrict__ grad_columns,
    float* __restrict__ grad_weight_group, float* __restrict__ grad_scale_group,
    int batch, int c_in, int c_out, int l_in, int kernel, int stride,
    int dilation, int pad_left, int groups, int l_out)
{
    if (blockIdx.x || threadIdx.x) return;
    int c_in_pg = c_in / groups;
    int n_g = c_out / groups;
    int k_g = c_in_pg * kernel;
    for (int b = 0; b < batch; ++b) {
        for (int group = 0; group < groups; ++group) {
            for (int l = 0; l < l_out; ++l) {
                for (int ci_local = 0; ci_local < c_in_pg; ++ci_local) {
                    int ci = group * c_in_pg + ci_local;
                    for (int kk = 0; kk < kernel; ++kk) {
                        int position = l * stride + kk * dilation - pad_left;
                        int j = ci_local * kernel + kk;
                        columns[l * k_g + j] = position >= 0 && position < l_in
                            ? x[(b * c_in + ci) * l_in + position] : 0.0f;
                    }
                }
            }
            for (int n = 0; n < n_g; ++n) {
                int co = group * n_g + n;
                for (int l = 0; l < l_out; ++l)
                    group_gradient[l * n_g + n] = grad_output[(b * c_out + co) * l_out + l];
            }
            for (int i = 0; i < l_out * k_g; ++i) grad_columns[i] = 0.0f;
            for (int i = 0; i < n_g * k_g; ++i) grad_weight_group[i] = 0.0f;
            for (int i = 0; i < n_g; ++i) grad_scale_group[i] = 0.0f;
            for (int l = 0; l < l_out; ++l) {
                for (int n = 0; n < n_g; ++n) {
                    float gy = group_gradient[l * n_g + n];
                    float s = scale[group * n_g + n];
                    float product = 0.0f;
                    for (int j = 0; j < k_g; ++j) {
                        float activation = columns[l * k_g + j];
                        float w = weight[(group * n_g + n) * k_g + j];
                        product += activation * w;
                        grad_columns[l * k_g + j] += gy * s * w;
                        grad_weight_group[n * k_g + j] += gy * s * activation;
                    }
                    grad_scale_group[n] += gy * product;
                }
            }
            for (int n = 0; n < n_g; ++n) {
                int co = group * n_g + n;
                grad_scale[co] += grad_scale_group[n];
                for (int j = 0; j < k_g; ++j)
                    grad_weight[co * k_g + j] += grad_weight_group[n * k_g + j];
            }
            for (int l = 0; l < l_out; ++l) {
                for (int ci_local = 0; ci_local < c_in_pg; ++ci_local) {
                    int ci = group * c_in_pg + ci_local;
                    for (int kk = 0; kk < kernel; ++kk) {
                        int position = l * stride + kk * dilation - pad_left;
                        if (position >= 0 && position < l_in)
                            grad_x[(b * c_in + ci) * l_in + position]
                                += grad_columns[l * k_g + ci_local * kernel + kk];
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ float conv2d_input_value(
    const float* x, int batch, int channel, int oh, int ow, int kh, int kw,
    int c_in, int input_h, int input_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int pad_top, int pad_left)
{
    int ih = oh * stride_h + kh * dilation_h - pad_top;
    int iw = ow * stride_w + kw * dilation_w - pad_left;
    if (ih < 0 || ih >= input_h || iw < 0 || iw >= input_w) return 0.0f;
    return x[((batch * c_in + channel) * input_h + ih) * input_w + iw];
}

extern "C" __global__ void conv2d_forward_portable(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ scale, float* __restrict__ result,
    int batch, int c_in, int c_out, int input_h, int input_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int pad_top, int pad_left,
    int groups, int height_out, int width_out)
{
    if (blockIdx.x || threadIdx.x) return;
    int c_in_pg = c_in / groups;
    int n_g = c_out / groups;
    int patch_columns = c_in_pg * kernel_h * kernel_w;
    int patch_rows = height_out * width_out;
    for (int b = 0; b < batch; ++b) {
        for (int group = 0; group < groups; ++group) {
            for (int row_start = 0; row_start < patch_rows; row_start += 32) {
                int row_count = patch_rows - row_start;
                if (row_count > 32) row_count = 32;
                for (int local_row = 0; local_row < row_count; ++local_row) {
                    int row = row_start + local_row;
                    int oh = row / width_out, ow = row - oh * width_out;
                    for (int n = 0; n < n_g; ++n) {
                        int co = group * n_g + n;
                        float acc = 0.0f;
                        for (int column = 0; column < patch_columns; ++column) {
                            int kw = column % kernel_w;
                            int rest = column / kernel_w;
                            int kh = rest % kernel_h;
                            int ci_local = rest / kernel_h;
                            int channel = group * c_in_pg + ci_local;
                            float activation = conv2d_input_value(
                                x, b, channel, oh, ow, kh, kw, c_in, input_h, input_w,
                                stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left);
                            acc += activation * weight[co * patch_columns + column];
                        }
                        int out_index = ((b * c_out + co) * height_out + oh) * width_out + ow;
                        result[out_index] = scale[co] * acc;
                    }
                }
            }
        }
    }
}

extern "C" __global__ void conv2d_backward_portable(
    const float* __restrict__ x, const float* __restrict__ weight,
    const float* __restrict__ scale, const float* __restrict__ grad_output,
    float* __restrict__ grad_x, float* __restrict__ grad_weight,
    float* __restrict__ grad_scale, float* __restrict__ columns,
    float* __restrict__ group_gradient, float* __restrict__ grad_columns,
    float* __restrict__ grad_weight_group, float* __restrict__ grad_scale_group,
    int batch, int c_in, int c_out, int input_h, int input_w,
    int kernel_h, int kernel_w, int stride_h, int stride_w,
    int dilation_h, int dilation_w, int pad_top, int pad_left,
    int groups, int height_out, int width_out)
{
    if (blockIdx.x || threadIdx.x) return;
    int c_in_pg = c_in / groups;
    int n_g = c_out / groups;
    int patch_columns = c_in_pg * kernel_h * kernel_w;
    int patch_rows = height_out * width_out;
    for (int b = 0; b < batch; ++b) {
        for (int group = 0; group < groups; ++group) {
            for (int row_start = 0; row_start < patch_rows; row_start += 32) {
                int row_count = patch_rows - row_start;
                if (row_count > 32) row_count = 32;
                for (int local_row = 0; local_row < row_count; ++local_row) {
                    int row = row_start + local_row;
                    int oh = row / width_out, ow = row - oh * width_out;
                    for (int ci_local = 0; ci_local < c_in_pg; ++ci_local) {
                        int channel = group * c_in_pg + ci_local;
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int column = (ci_local * kernel_h + kh) * kernel_w + kw;
                                columns[local_row * patch_columns + column] = conv2d_input_value(
                                    x, b, channel, oh, ow, kh, kw, c_in, input_h, input_w,
                                    stride_h, stride_w, dilation_h, dilation_w, pad_top, pad_left);
                            }
                        }
                    }
                }
                for (int n = 0; n < n_g; ++n) {
                    int co = group * n_g + n;
                    for (int local_row = 0; local_row < row_count; ++local_row) {
                        int row = row_start + local_row;
                        int oh = row / width_out, ow = row - oh * width_out;
                        int out_index = ((b * c_out + co) * height_out + oh) * width_out + ow;
                        group_gradient[local_row * n_g + n] = grad_output[out_index];
                    }
                }
                for (int i = 0; i < row_count * patch_columns; ++i) grad_columns[i] = 0.0f;
                for (int i = 0; i < n_g * patch_columns; ++i) grad_weight_group[i] = 0.0f;
                for (int i = 0; i < n_g; ++i) grad_scale_group[i] = 0.0f;
                for (int local_row = 0; local_row < row_count; ++local_row) {
                    for (int n = 0; n < n_g; ++n) {
                        float gy = group_gradient[local_row * n_g + n];
                        float s = scale[group * n_g + n];
                        float product = 0.0f;
                        for (int column = 0; column < patch_columns; ++column) {
                            float activation = columns[local_row * patch_columns + column];
                            float w = weight[(group * n_g + n) * patch_columns + column];
                            product += activation * w;
                            grad_columns[local_row * patch_columns + column] += gy * s * w;
                            grad_weight_group[n * patch_columns + column] += gy * s * activation;
                        }
                        grad_scale_group[n] += gy * product;
                    }
                }
                for (int n = 0; n < n_g; ++n) {
                    int co = group * n_g + n;
                    grad_scale[co] += grad_scale_group[n];
                    for (int column = 0; column < patch_columns; ++column)
                        grad_weight[co * patch_columns + column]
                            += grad_weight_group[n * patch_columns + column];
                }
                for (int local_row = 0; local_row < row_count; ++local_row) {
                    int row = row_start + local_row;
                    int oh = row / width_out, ow = row - oh * width_out;
                    for (int ci_local = 0; ci_local < c_in_pg; ++ci_local) {
                        int channel = group * c_in_pg + ci_local;
                        for (int kh = 0; kh < kernel_h; ++kh) {
                            for (int kw = 0; kw < kernel_w; ++kw) {
                                int ih = oh * stride_h + kh * dilation_h - pad_top;
                                int iw = ow * stride_w + kw * dilation_w - pad_left;
                                if (ih >= 0 && ih < input_h && iw >= 0 && iw < input_w) {
                                    int input_index = ((b * c_in + channel) * input_h + ih) * input_w + iw;
                                    int column = (ci_local * kernel_h + kh) * kernel_w + kw;
                                    grad_x[input_index] += grad_columns[local_row * patch_columns + column];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

__device__ __forceinline__ long attention_vector_index(
    int token, int head, int head_count, int head_dim, int lane)
{
    return ((long)token * head_count + head) * head_dim + lane;
}

__device__ void attention_probabilities_portable(
    const float* q, const float* k, float* probabilities,
    int seq, int n_head, int n_kv_head, int head_dim, int causal,
    int head, int kv_head, float scale)
{
    for (int query = 0; query < seq; ++query) {
        long row = (long)query * seq;
        for (int key = 0; key < seq; ++key) {
            float score = 0.0f;
            for (int lane = 0; lane < head_dim; ++lane) {
                score += q[attention_vector_index(query, head, n_head, head_dim, lane)]
                       * k[attention_vector_index(key, kv_head, n_kv_head, head_dim, lane)];
            }
            probabilities[row + key] = causal && key > query ? -INFINITY : score * scale;
        }
        float maximum = -INFINITY;
        for (int key = 0; key < seq; ++key)
            maximum = fmaxf(maximum, probabilities[row + key]);
        float sum = 0.0f;
        for (int key = 0; key < seq; ++key) {
            float exponential = causal && key > query
                ? 0.0f : expf(probabilities[row + key] - maximum);
            probabilities[row + key] = exponential;
            sum += exponential;
        }
        for (int key = 0; key < seq; ++key) probabilities[row + key] /= sum;
    }
}

// Canonical portable GQA attention. A serial kernel intentionally preserves the frozen reference
// accumulation order; optimized twins may replace it only with receipt-distinguished tolerances.
extern "C" __global__ void attention_forward_portable(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, float* __restrict__ result,
    float* __restrict__ probabilities, int seq, int n_head,
    int n_kv_head, int head_dim, int causal)
{
    if (blockIdx.x || threadIdx.x) return;
    int group_size = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int head = 0; head < n_head; ++head) {
        int kv_head = head / group_size;
        attention_probabilities_portable(q, k, probabilities, seq, n_head, n_kv_head,
                                         head_dim, causal, head, kv_head, scale);
        for (int query = 0; query < seq; ++query) {
            for (int lane = 0; lane < head_dim; ++lane) {
                float accumulator = 0.0f;
                for (int key = 0; key < seq; ++key) {
                    accumulator += probabilities[(long)query * seq + key]
                        * v[attention_vector_index(key, kv_head, n_kv_head, head_dim, lane)];
                }
                result[attention_vector_index(query, head, n_head, head_dim, lane)] = accumulator;
            }
        }
    }
}

extern "C" __global__ void attention_backward_portable(
    const float* __restrict__ q, const float* __restrict__ k,
    const float* __restrict__ v, const float* __restrict__ grad_output,
    float* __restrict__ grad_q, float* __restrict__ grad_k,
    float* __restrict__ grad_v, float* __restrict__ probabilities,
    float* __restrict__ grad_probabilities, int seq, int n_head,
    int n_kv_head, int head_dim, int causal)
{
    if (blockIdx.x || threadIdx.x) return;
    int group_size = n_head / n_kv_head;
    float scale = 1.0f / sqrtf((float)head_dim);
    for (int head = n_head - 1; head >= 0; --head) {
        int kv_head = head / group_size;
        attention_probabilities_portable(q, k, probabilities, seq, n_head, n_kv_head,
                                         head_dim, causal, head, kv_head, scale);
        for (long i = 0; i < (long)seq * seq; ++i) grad_probabilities[i] = 0.0f;
        for (int query = 0; query < seq; ++query) {
            for (int lane = 0; lane < head_dim; ++lane) {
                float gradient = grad_output[
                    attention_vector_index(query, head, n_head, head_dim, lane)];
                for (int key = 0; key < seq; ++key) {
                    long probability_index = (long)query * seq + key;
                    long value_index = attention_vector_index(
                        key, kv_head, n_kv_head, head_dim, lane);
                    grad_probabilities[probability_index] += gradient * v[value_index];
                    grad_v[value_index] += gradient * probabilities[probability_index];
                }
            }
        }
        for (int query = 0; query < seq; ++query) {
            long row = (long)query * seq;
            float contraction = 0.0f;
            for (int key = 0; key < seq; ++key)
                contraction += probabilities[row + key] * grad_probabilities[row + key];
            for (int key = 0; key < seq; ++key) {
                long index = row + key;
                grad_probabilities[index] = causal && key > query ? 0.0f
                    : probabilities[index] * (grad_probabilities[index] - contraction) * scale;
            }
        }
        for (int query = 0; query < seq; ++query) {
            for (int key = 0; key < seq; ++key) {
                float gradient = grad_probabilities[(long)query * seq + key];
                for (int lane = 0; lane < head_dim; ++lane) {
                    long query_index = attention_vector_index(
                        query, head, n_head, head_dim, lane);
                    long key_index = attention_vector_index(
                        key, kv_head, n_kv_head, head_dim, lane);
                    grad_q[query_index] += gradient * k[key_index];
                    grad_k[key_index] += gradient * q[query_index];
                }
            }
        }
    }
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

extern "C" __global__ void topk_kd_backward(
    const float* __restrict__ logits, const unsigned int* __restrict__ indices,
    const float* __restrict__ probabilities, float* __restrict__ g_logits,
    int rows, int cols, int k, float gscale)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= rows) return;
    const float* lr = logits + (long)row * cols;
    const unsigned int* ir = indices + (long)row * k;
    const float* pr = probabilities + (long)row * k;
    float* gr = g_logits + (long)row * cols;
    float maximum = -INFINITY;
    for (int col = 0; col < cols; ++col) maximum = fmaxf(maximum, lr[col]);
    float sum = 0.0f;
    for (int col = 0; col < cols; ++col) sum += expf(lr[col] - maximum);
    float probability_sum = 0.0f;
    for (int sparse_col = 0; sparse_col < k; ++sparse_col) {
        probability_sum += pr[sparse_col];
    }
    for (int col = 0; col < cols; ++col) {
        float probability = expf(lr[col] - maximum) / sum;
        gr[col] = gscale * probability * probability_sum;
    }
    for (int sparse_col = 0; sparse_col < k; ++sparse_col) {
        gr[ir[sparse_col]] -= gscale * pr[sparse_col];
    }
}
