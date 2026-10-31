# tritium-cuda

CUDA backend: dp4a decode, IMMA tensor-core prefill, device-resident training kernels.

The explicit shared-tensor device SwiGLU region rejects nonfinite input,
intermediates and output atomically; see
[shared-tensor SwiGLU](../../docs/SHARED-TENSOR-SWIGLU.md) for numerical policy,
private scratch exclusions and explicit physical regression commands.

Experimental direct packed QTEA upload, ordered-f32 matmul and selected-row gather
are also available with explicit logical allocation limits and no dense shadow.
See [QTEA-CUDA](../../docs/QTEA-CUDA.md) for API, explicit device tests and evidence
boundaries and explicit host-orchestrated CUDA model profiles. These are not a
fully resident graph, whole-model speed or release qualification claim.

Part of [Tritium](https://github.com/Quitetall/tritium) — Apache-2.0 infrastructure for
quantizing, training, and serving additive-ternary ({-1, 0, +1}) neural networks with
exact byte accounting and receipt-backed benchmarks.

See the [repository README](https://github.com/Quitetall/tritium#readme) and the
[book](https://github.com/Quitetall/tritium/tree/main/docs/book) for usage.

The shared packed additive path retains scalar column/group/plane accumulation
order. Its cross-row B3 kernel stages canonical packed bytes, not dense weights,
using 1,408 bytes of on-chip shared memory per block and no additional retained
tensor payload. Dispatch is deliberately limited to the measured 17,408×5,120,
batch-one, uniform B3/free-F16/group-128, one- or three-plane family. Other
geometry, prefill batches, codecs, laws and compact allocations keep their
existing scalar/ordered-warp paths. This is not a whole-model performance claim.

Physical parity and balanced warm-cache timing diagnostics can be run explicitly:

```sh
cargo test --locked -p tritium-cuda --features cuda --lib b3_rows_ -- \
  --include-ignored --nocapture --test-threads=1
```

These check cancellation/ragged-tail bit parity and actual public view/stream
ingress, including basis transforms and the batch-two fallback. Timing remains
diagnostic: no exclusive-device, clock/cache control, full-graph speed, model
quality, cross-hardware or release qualification is implied.

Packed-kernel diagnostics retain raw per-call nanoseconds alongside summaries,
with host launch/synchronization and CUDA-event samples kept separate. Medians
use the midpoint of the sorted middle pair for even counts, rounded down to a
nanosecond without overflowing. Older upper-middle-only logs remain historical
observations; their missing raw samples cannot be reconstructed or reclassified.
