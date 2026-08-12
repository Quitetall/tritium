# Tritium Threat Model

**Date:** 2026-06-24
**Status:** Committed security document — satisfies the **ADR-0011 v0.90 security gate**.

This document is the consolidated threat model for Tritium, synthesized from five code-grounded surface audits (model-file parsers, the C ABI / FFI boundary, the OpenAI-compatible HTTP server, the compute kernels + backend dispatch, and the supply-chain + build pipeline). Its purpose is to state plainly what Tritium protects, against whom, where the trust boundaries lie, and — surface by surface — which threats exist, which mitigations are present **in committed code** (cited to file/line or CI lane), and which residual risks remain. Mitigations are reported only where they exist in the source; gaps are stated honestly as "no mitigation yet; tracked" rather than fabricated.

---

## 1. Scope & deployment model

Tritium is a **Rust library plus a local / trusted-deployment inference server**. It is not a multi-tenant, internet-facing service. The assumed deployment is one of:

- An embedding application linking the library directly (Rust or via the C ABI / Python ctypes).
- A local OpenAI-compatible HTTP server (`tritium-serve`) that **binds loopback only** (`127.0.0.1`) and is **off by default** (feature-gated behind `serve`).

The trusted parties are: the **local operator** who chooses what to run, the **embedding application** that links the library, and the **build/CI host**. The principal untrusted inputs are: **model files downloaded from third parties**, **network request bytes** to the local server, **raw pointers/lengths from foreign FFI callers**, and the **third-party crate graph** pulled at build time.

Because the entire host-side parsing/compute path that touches untrusted bytes is **safe Rust** (`#![forbid(unsafe_code)]` in `tritium-format`) or carefully-bounded `unsafe` (the SIMD/GPU kernels), the **realistic worst case is denial of service (panic/abort/OOM) or a logic-level mis-load — not remote code execution**. Severities throughout are calibrated to this trusted/local model; several would rise to high/critical under public network exposure (called out where relevant).

---

## 2. Assets

| Asset | What we protect |
|---|---|
| **Host process integrity / memory safety** | No untrusted input may corrupt memory, alias mutable state, or escalate to code execution in the host process or embedding application. |
| **Availability** | A malformed model file, hostile request, or pathological shape should not crash, hang, or OOM the process beyond a contained, recoverable error. |
| **The user's machine when loading third-party models** | Loading an attacker-controlled model file must not escape structural-parse safety: no OOB, no type confusion, no arbitrary file/path access beyond the operator's own choice. |
| **Build host integrity** | The developer/CI build machine must not execute attacker-controlled code beyond the inherent (and gated) Rust build-time exposure. |

---

## 3. Trust boundaries

```
        UNTRUSTED INPUTS                         TRUSTED CORE
  ┌──────────────────────────┐
  │ Third-party model files   │  parse →  ┌────────────────────────────────┐
  │ (.gguf/.safetensors/SALT) │──────────▶│ tritium-format                  │
  └──────────────────────────┘           │  #![forbid(unsafe_code)]         │
                                          │  bounds-checked LeCursor/Cursor  │
  ┌──────────────────────────┐           └──────────────┬─────────────────┘
  │ Network req bytes (JSON,   │  HTTP →                  │ post-parse: shapes,
  │ prompts, sampling params)  │──────┐                   │ weights, TensorInfo
  │ → loopback 127.0.0.1 only  │      │                   ▼  (trusted relative
  └──────────────────────────┘      │            ┌──────────────────────┐    to the parser)
                                     ├───────────▶│ tritium-nn / runtime  │
  ┌──────────────────────────┐      │            │ tritium-cpu / -cuda    │
  │ FFI raw ptr+len, model     │ C ABI│            │ -rocm / -metal kernels │
  │ path (foreign C/ctypes)    │──────┘            └──────────────────────┘
  └──────────────────────────┘
                                          ┌────────────────────────────────┐
  ┌──────────────────────────┐  resolve  │ Build host (dev/CI)             │
  │ crates.io graph, build.rs, │──────────▶│ cargo-deny, Cargo.lock, SBOM,   │
  │ proc-macros, nvcc/hipcc    │           │ CARGO_FEATURE_* gates           │
  └──────────────────────────┘           └────────────────────────────────┘

  Trusted: the local operator, the embedding application, shapes/weights AFTER
  the format parser has validated them, the build/CI host itself.
```

The boundary is the **function/handler/trait-call entry**: from there inward, all header counts, lengths, offsets, dims, type tags, pointers, JSON fields, and request params are treated as hostile. Once data has crossed the `tritium-format` parser, it is *relatively* trusted by the downstream kernels — the residual kernel-side attacker model is a buggy/malicious in-process caller passing inconsistent shapes, plus pathological dimensions.

---

## 4. Threats by surface

### 4.1 Model-file parsers (`tritium-format`)

The whole crate is `#![forbid(unsafe_code)]` (`crates/tritium-format/src/lib.rs:29`), so the worst outcome is a clean panic, never memory corruption.

| Threat | STRIDE | Severity | Mitigation (cited) | Residual risk |
|---|---|---|---|---|
| Integer overflow / OOB slice in GGUF offset+length & dim arithmetic | Tampering | low | `forbid(unsafe_code)` (lib.rs:29); `Cursor::take` checked_add + `slice::get` (gguf.rs:106-111); `checked_mul` dim product (gguf.rs:273-279, 452-455); `div_ceil`+`checked_mul` for n_bytes (gguf.rs:312-315); span check `checked_add` vs `data_section_len` via `saturating_sub` (gguf.rs:447,458-465); `MAX_COUNT`/`MAX_DIMS` caps (gguf.rs:401,430); tests for dims_overflow/offset_oob/truncation/2000 arbitrary inputs | None in `read_gguf` itself. Unknown ggml types get `n_bytes=0`; only their offset is bounds-checked (gguf.rs:463) — payload sizing is pushed to the consumer (see I2_S threat). |
| Unbounded allocation / OOM from attacker-declared counts | DoS | low | `MAX_PREALLOC=4096` speculative reserve, grow-on-demand (gguf.rs:48,426); array reserve `.min(1024)` (gguf.rs:240); tqbin validates `n_tokens*4 <= remaining` (tqbin.rs:60-73); tqidx caps reserve by `remaining/10` (tqidx.rs:117); salt_bundle caps by `len/18` & per-row by `blob.len()/HEADER` (salt_bundle.rs:148,173); sparse rejects `k>=2^31` + exact-length check (sparse.rs:301,324); `unpack_salt_row` exact `== bytes.len()` (salt.rs:120-126); safetensors `header_end <= buf.len()` (safetensors.rs:130). Tests for adversarial n_dims/tensor_count/n_tokens/shard_count. | No hard cap on total metadata/tensor memory beyond input length; a multi-GB array needs a multi-GB file. OOM bounded by file size — acceptable for local/trusted load. |
| `compute_zero_bitmap` / `compute_zero_bitmaps` panic or OOB-slice on under-length / absurd-size input | DoS | info | Both functions are total: under-length input, `row_bytes` smaller than a packed row (the `row_bytes = 0` + huge-`n` capacity-overflow class a review found surviving the first guard), and wrapped `nb * TQ2_0_BLOCK_BYTES` / `n * row_bytes` products all return `FormatError::WrongBlockLen` (saturating size math + row-size precondition, tq2.rs); every allocation is bounded by the input length once the preconditions pass; regression test pins the exact review repro; fuzz target `zero_bitmap` draws n/row_bytes at full u64 width so these branches are actually reachable (fuzz_targets/zero_bitmap.rs, scheduled CI lane). | Closed. Callers get a typed error, never a panic; fuzzed alongside every other untrusted-byte entry point (`unpack_i2s` / `unpack_tq_rows` targets cover the row decoders too). |
| Type confusion / unvalidated enum & tag fields | Tampering | info | `read_value` rejects unknown `value_type` (gguf.rs:249) and forbids nested arrays via depth guard (gguf.rs:229-232); version whitelist `{2,3}` (gguf.rs:395); magic check (gguf.rs:390); unknown ggml_type sized 0 not misread (gguf.rs:299); I2_S reserved code `0b11` rejected (i2s.rs:56-63); safetensors rejects unsupported dtype (safetensors.rs:202); every sidecar enforces magic+version; UTF-8 validated for all strings. | None material. `as_u64` widening returns None for negative ints; zero alignment falls back to `DEFAULT_ALIGNMENT`. No silent coercion found. |
| Truncated / overlapping / mis-sized tensor data regions | Tampering | info | safetensors `data.get(a..b)` (safetensors.rs:194-197), `checked_mul` shape product + `LengthMismatch` (safetensors.rs:212-226); GGUF validates every `[offset, offset+n_bytes)` vs `data_section_len` (gguf.rs:458-465); salt_gguf walks self-describing rows bounds-checked (salt_gguf.rs:149-169). | safetensors does **not** detect overlapping tensor regions — harmless because the reader only ever reads (never writes), each read independently bounds-checked. No security impact. |
| No authenticity/integrity verification of model files | Spoofing | info | None at this layer, **by design** — `tritium-format` is a structural parser, not a trust anchor. `forbid(unsafe_code)` + total parsers ensure a malicious file cannot escalate beyond mis-load or DoS. | Model-content trust is out of scope and unsolvable here (no signature scheme in GGUF/safetensors). Mitigation is operational: obtain models from trusted sources / verify hashes out of band. |

**Downstream consequence (just outside the crate boundary, reachable from the same untrusted file):** the `tritium-nn` loader re-sizes I2_S payloads and recomputes `n_out*k_in` with unchecked arithmetic — `(file.tensor_data_offset + info.offset) as usize` (weights.rs:124), `start + len` (weights.rs:134), `n_out * k_in` (weights.rs:174). `bytes.get(start..start+len)` keeps the slice safe (returns None, not OOB), so impact is at most a debug-build panic (DoS) on a hostile file; on 64-bit with realistic file sizes these cannot wrap to an in-bounds slice. **Tracked:** loader should reuse `info.element_count()` and checked adds.

---

### 4.2 C ABI / FFI boundary (`tritium-ffi`)

The boundary is trusted Rust ↔ a foreign caller supplying raw pointers, lengths, and a model path. All memory-safety obligations on inbound pointers (validity, alignment, capacity, exclusivity, liveness) are **contractual**; Tritium can only null-check.

| Threat | STRIDE | Severity | Mitigation (cited) | Residual risk |
|---|---|---|---|---|
| `catch_unwind` is a no-op under shipped `panic=abort` — `TritiumStatus::Panic` unreachable in release/dist | DoS | low | This is the **safe** outcome for a no-unwind FFI contract (unwinding across `extern "C"` would be UB). Behavior documented honestly (lib.rs:11-17,55-58); `catch_unwind` still protects dev/test (`panic="unwind"`). | Any reachable internal panic aborts the host process in release/dist; `Panic` status can never be returned there. Availability-only; acceptable for trusted/local. Documentation/expectation issue, not a code fix. |
| Inbound pointer validity/alignment/true-capacity unverifiable beyond null | Tampering | **medium** | Every pointer null-checked (lib.rs:106,166,173-178); out-buffer write gated — `*out_len=tokens.len()`, `BufferTooSmall` with nothing written when `> out_cap` (lib.rs:192-195), copy uses `tokens.len() <= out_cap` so Tritium never writes past declared capacity (lib.rs:196-199); null-iff-zero-length invariants enforced; per-fn safety contracts documented. | Inherent to any C ABI: validity/alignment/true capacity of non-null pointers cannot be checked in-language. Residual write-overflow risk = a caller who **lies** about `out_cap`. Acceptable for trusted/local; real risk for any embedder forwarding untrusted length/pointer values. |
| Concurrent `tritium_generate` calls on one handle race the KV cache | Tampering | **medium** | `TritiumModel` stores the runner behind `Mutex`; generation uses `try_lock` and returns `TritiumStatus::Busy` on overlap, so only one call mutates the cache at a time (lib.rs:105,341-353). A compile-time `Sync` assertion covers the shared-handle contract (tests/abi.rs:28-31). | The raw-handle lifetime contract still forbids freeing a handle while another call is active; stale-pointer use-after-free remains an inherent C ABI risk. |
| Use-after-free / double-free of the model handle | Tampering | **medium** | `tritium_model_free` is null-safe (lib.rs:212); ownership documented (free exactly once, no use-after-free, lib.rs:20-22,206-209); drop wrapped in `catch_unwind`. | No defense against UAF/double-free: a stale non-null pointer is indistinguishable from a live one (raw `Box::into_raw`, no generation counter/freed-flag). Inherent to opaque-handle C ABIs. Tracked: magic sentinel zeroed on free, or a live-handle registry, to fail fast. |
| Out-parameter state on every path of `tritium_generate` | Info Disclosure | info | `*out_len=0` written before any fallible work (lib.rs:169-172) so output state defined on every path where `out_len` is non-null; real count overwrites on Ok/BufferTooSmall (lib.rs:192); `out_status` only written when non-null (lib.rs:100); out-of-vocab token IDs bounds-checked in embedding gather → `MissingTensor`→`Generate` not OOB (runner.rs:241-245). | Defined-output behavior is genuinely correct; this entry documents the mitigation. Residual = the usual "caller ignores status code." No code fix needed. |
| Model file read from caller-controlled path with host privileges | Info Disclosure | low | Invalid-UTF-8 paths rejected (lib.rs:111-112); I/O failure → `TritiumStatus::Load` without leaking errno (lib.rs:113); runs with caller's own privileges (no escalation); parse failures contained to `Load`. | No path sandboxing/allowlisting; whole file read into memory before parse (memory-pressure DoS on a huge file). Acceptable where the caller chooses the path with its own rights; a concern only if an embedder forwards untrusted paths (embedder's responsibility). Worth a doc note. |

---

### 4.3 Network server (`tritium-serve`, OpenAI-compatible HTTP/SSE)

Binds **loopback by default** and is **off by default** (`serve` not in default features; bin `required-features=["serve"]`). An explicit `--host <ip>` flag can bind beyond loopback, but a non-loopback bind **refuses to start unless `TRITIUM_AUTH_TOKEN` or the rotating `TRITIUM_AUTH_TOKENS` set is configured** and prints a loud exposure warning. Severities below are calibrated to the default localhost posture; the hardening layers (auth, timeout, concurrency cap, body limit and principal admission) are what keep them bounded under exposure.

| Threat | STRIDE | Severity | Mitigation (cited) | Residual risk |
|---|---|---|---|---|
| Missing or bypassed authentication | Elevation of Privilege | low | Uniform middleware accepts a startup-validated rotation set of at most 32 bearer keys, stores only BLAKE3 digests, scans the full fixed digest set, returns 401 plus `WWW-Authenticate`, and protects health/readiness/metrics/404 as well as generation. Non-loopback bind requires at least one key. | Loopback may deliberately run anonymous, so every local user is one principal. Authenticated probes still need a static header until plan 0052's separate admin listener lands. Bearer transport requires TLS at the trusted proxy boundary. |
| No request timeout — slowloris / slow-send | DoS | low | Explicit 2 MiB `DefaultBodyLimit`; a local timeout middleware (default 600 s) bounds body extraction plus the non-streaming service future and returns a typed OpenAI 408 envelope; the lazy SSE body independently enforces the same absolute lifetime from queue admission, emits an OpenAI `request_timeout` error event, and drops its receiver to cancel generation; `ConcurrencyLimitLayer` (default 64) caps in-flight requests; generation uses a bounded queue. | Permit wait at the concurrency limiter is outside the timeout clock (mitigated by queue 429 backpressure). Header/read time before axum constructs the request is controlled by the deployment's HTTP edge. |
| Unbounded prompt length & `max_tokens` value (compute exhaustion within 2MB body) | DoS | low | `RequestLimits` rejects excessive message count, aggregate role/content bytes, tokenized prompt length, explicit completion length and checked combined token budget before queue admission; the 2 MiB body limit bounds JSON parsing; excessive `max_tokens` is rejected instead of silently clamped; bounded queue returns 429. | Tokenization still occurs before the token-count limit can be applied. Single decode thread implies head-of-line blocking, bounded by queue, principal rate and request lifetime. |
| Streaming (SSE) connection exhaustion via many concurrent streams | DoS | low | Bounded job queue returns 429 + Retry-After; expensive routes use fixed-point token buckets per authenticated key (or one anonymous bucket), with a fixed maximum of 32 configured principals and no attacker-created map entries; per-token `try_send` cancels stalled readers; global `ConcurrencyLimitLayer` caps in-flight requests; the absolute SSE lifetime cancels streams; graceful drain closes in-flight work. | 64-token-per-job buffer times active streams is bounded. An admitted SSE stream holds a concurrency permit until completion or deadline; operators must size burst/concurrency for the hardware and may still need an edge-wide distributed limiter across replicas. |
| Handler panic taking down the server | DoS | low | Generation under `catch_unwind` per job returns `GenEvent::Error` and the worker continues; `AliveGuard` plus `/healthz` and `/readyz` report 503 when the worker is dead; sampler absence is handled as a backend error. | **Under `panic=abort` release/dist builds `catch_unwind` is a no-op** (FFI panic-abort contract) — a panic-inducing request kills the whole process (process-kill DoS). Mitigation holds only for unwind dev/test builds. Residual depends on build profile. |
| Worker death or drain (closed job channel) | DoS | info | Closed queue admission returns 503. `/healthz` is worker liveness; `/readyz` additionally turns 503 before drain, while `/healthz` stays 200 so a scheduler does not restart a healthy draining process. Both endpoints are contract-tested and remain behind the configured auth middleware. | No automatic worker restart — external process supervision performs recovery. Startup artifact/self-test readiness remains plan 0052 work. |
| Unknown JSON fields silently ignored (no `deny_unknown_fields`) | Tampering | info | Intentional OpenAI-compatibility (dto.rs:5-6,45-69); security-relevant fields individually validated: temperature `[0,2]` (router.rs:150), top_p `(0,1]` & finite (router.rs:187-196), stop count `<=4` non-empty (router.rs:171-186), `max_tokens != 0` (router.rs:197), empty messages rejected (router.rs:142), model id 404-matched (router.rs:158-165). | Footgun (typo'd field silently dropped), no security impact. Not a real vulnerability. |

---

### 4.4 Kernels & backend dispatch (`tritium-cpu` SIMD/LUT, `tritium-spec` trait, CUDA/ROCm/Metal)

The boundary is the `TernaryBackend` trait call; callers are in-process Rust passing host slices + a `GemmShape`. Weights are **post-parse** (already crossed the `tritium-format` boundary). The UB-bearing code is the hand-written `unsafe` SIMD and the GPU device kernels (no memory protection).

| Threat | STRIDE | Severity | Mitigation (cited) | Residual risk |
|---|---|---|---|---|
| Shape-mismatch OOB in CPU SIMD kernels | Tampering | low | **4-deep defense:** (1) `CpuBackend::mpgemm` validates `act.len()==m*k`, `scales.len()==n`, `out.len()==m*n` → `ShapeMismatch` (lib.rs:251-270); (2) `unpack_weights` allocates `trits = vec![…; n*k]` exactly (lib.rs:146-173); (3) `kernel::dispatch_mpgemm` re-validates all four lengths (kernel.rs:55-61); (4) each SIMD kernel re-validates at entry, defers to scalar on mismatch (kernel.rs:208-212; avx512.rs:82-85; neon.rs:80-83); every raw load/store carries a proven `// SAFETY:` note. Tests `mpgemm_rejects_bad_operand_lengths` (lib.rs:535), ASan/MSan/TSan build-std lane (sanitizers.yml:84-170). | None material. Only residual is that the sanitizer lane is **weekly/manual, not a required per-push gate** — a newly-introduced OOB could merge and sit until the next scheduled run. |
| CUDA *inference* `mpgemm_kernel` dimension overflow | DoS | low | Shared `check_grad_launch_bounds` validates all `m*n`, `n*k`, and `m*k` products against `i32::MAX` before length checks, allocation, grid math, or narrowing casts in both dense and sparse inference paths (`crates/tritium-cuda/src/cuda/backend.rs`). Regression `mpgemm_launch_bounds_reject_index_overflow` covers each dimension. | No material shape-overflow residual. GPU OOB still lacks sanitizer coverage in default CI; physical CUDA conformance remains required on the GPU lane. |
| Index-arithmetic overflow in GPU device kernels | Tampering | info | Per-row bases widened to 64-bit: metal `ulong` (mpgemm.metal:57-60,93-97), hip/cuda `long long` (tq2_0_add.hip:58-59,66-67,89); host guards cover CUDA, Metal and ROCm; tail-thread `idx >= total` guard before any load (mpgemm.metal:51,88; tq2_0_add.hip:60); `row_bytes = nb*66` validated host-side at upload (backend.rs:270-277; rocm.rs:347-350). | GPU memory has **no sanitizer coverage** in default (non-self-hosted) CI; physical CUDA conformance remains required on the GPU lane. |
| rayon row fan-out data race | Tampering | info | `out.par_chunks_mut(...).zip(act.par_chunks(...))` yields provably non-overlapping `&mut` slices (rayon/std guarantee); per-task `chunk_shape` derived from actual chunk length (kernel.rs:69-77); weights/scales shared immutable `&[]`; output bit-identical regardless of thread count (determinism tests lib.rs:746,765); verified under TSan (sanitizers.yml:151-170); `CpuBackend` stateless `Send+Sync`. | None identified. Residual: TSan is weekly/on-demand, not per-push — a future chunking-math change could introduce a race uncaught until the scheduled run. |
| Backend no-panic contract not enforced against arithmetic overflow | DoS | low | Contract real & tested for bad-shape case: `fused_rejects_bad_shapes` (lib.rs:425), `mpgemm_rejects_bad_operand_lengths` (lib.rs:535); `GemmShape::macs()` widens to u64 (shape.rs:29-31); ROCm/Metal/CUDA-training use `checked_mul`/`u32::MAX` guards. | `m*k`/`m*n`/`n*k` use bare `*` (not `checked_mul`) in the length comparisons across spec/cpu (lib.rs:146, kernel.rs:58, etc.). Overflowing-usize shape ⇒ debug panic (which under `panic=abort` dist aborts the process — DoS) / release wrap. Realistic impact low (petabyte-scale tensors never arise from a real parsed model). Tracked: documented precondition or uniform `checked_mul`. |

---

### 4.5 Supply chain & build (workspace pins, `deny.toml`, build scripts, CI lanes)

The boundary is the developer/CI build machine. Realistic attacker = a compromised upstream/typosquat/malicious transitive dep, or a poisoned local toolchain — not a remote network attacker.

| Threat | STRIDE | Severity | Mitigation (cited) | Residual risk |
|---|---|---|---|---|
| Cargo lockfile drift in build/test/lint gates | Tampering | low | `Cargo.lock` is committed (repo root + `crates/tritium-format/fuzz/Cargo.lock`); required CI build/test/lint jobs use `--locked`, as do canonical local `verify-gates.sh` tiers and the pre-push scratch-worktree hook; cargo-deny advisories lane (ci.yml:45-52, deny.toml:5-12, `yanked="deny"`) flags known-advisory deps; `unknown-registry="deny"`/`unknown-git="deny"` (deny.toml:52-54). | Ancillary tool installers and SBOM generation do not resolve workspace dependencies; a compromised runner/toolchain remains outside Cargo lockfile protection. cargo-deny only catches *published* advisories. |
| Build scripts trust whatever `nvcc`/`hipcc` is on the local machine | Elevation of Privilege | low | Both scripts hard-gated behind `CARGO_FEATURE_CUDA`/`CARGO_FEATURE_ROCM` early-return (cuda build.rs:55-58, rocm build.rs:41-44) — default cpu-only build never touches a compiler; env-prefixed paths preferred over bare PATH; compiler invoked with only fixed, non-interpolated args (no shell). | PATH fallback (build.rs:178 / 142) trusts the ambient environment; no checksum/signature of the located compiler. Low given trusted-build model (a PATH-poisonable host is already compromised) + the default build is unaffected. Worth a comment acknowledging toolchain-trust. |
| Arbitrary code execution via dependency build scripts & proc-macros | Elevation of Privilege | medium | cargo-deny `sources` denies unknown registries/git (deny.toml:52-54); `wildcards="deny"` (deny.toml:49); network/native deps exact-pinned (wgpu `=23.0.1`, metal `=0.33.0`, ort `=2.0.0-rc.12`, Cargo.toml:78,84,92) and feature-gated **off by default**; advisories lane + `yanked="deny"`; `multiple-versions="warn"`. | cargo-deny does not detect a novel no-advisory-yet malicious crate and does not sandbox build scripts (no cargo-vet, no `-Zsandbox`). The `ort` `download-binaries` path fetches a native binary at build time (integrity outside Cargo's checksum model) — but only under `--features onnx`. Standard Rust build-time-RCE exposure, partially contained by feature gates. |
| Floating toolchain and unpinned tool installs in CI | Tampering | low | `rust-toolchain.toml` pins Rust `1.89.0`; GitHub Actions are SHA-pinned; `rust-version="1.89"` remains the compatibility floor (Cargo.toml:44); SBOM generated + uploaded as artifact (ci.yml:325-337); cargo-deny-action pins a recent cargo-deny deliberately. | External tool installers and SBOM output are not baseline-diffed; trusted action/tool inputs can still drift where their own versions are not frozen. Low impact for a library; weakens reproducibility/audit trail. |
| Advisory ignore + broad license allow-list widen the trusted set | Repudiation | info | Every ignore/allow documented with rationale/scope/revisit condition (RUSTSEC-2024-0436 `paste`, deny.toml:7-12; licenses 14-41); `yanked="deny"` + advisories lane still run for everything else; licenses are a default-deny **allow-list** (fails closed); NOTICE attribution tracked. | The `paste` ignore is by RUSTSEC id, so a hypothetical future *vulnerability* filed under the reused id would be suppressed (minor — RustSec generally issues new ids). License allow-list growth to review at v1.0 packaging. Well-documented, fails closed on the default path. |

---

## 5. Cross-cutting mitigations

- **`#![forbid(unsafe_code)]` where it holds.** `tritium-format` (lib.rs:29) is the **only** crate-level `forbid(unsafe_code)` — confirmed — so no malformed model byte can corrupt memory; the worst realistic outcome is a contained panic or logic-level mis-load. (The SIMD/GPU kernels are necessarily `unsafe` and are covered by layered shape validation + sanitizers instead.)
- **Fuzzing.** Committed cargo-fuzz harness with **8 targets** (`gguf`, `safetensors`, `salt_bundle`, `salt_gguf`, `salt_legacy`, `sparse_plane`, `tqbin`, `tqidx`) and **769 corpus files** checked into git. Gap: no `tq2` zero-bitmap target (see 4.1).
- **Sanitizers + miri.** `.github/workflows/sanitizers.yml` runs ASan/MSan/TSan via `-Zbuild-std` over `tritium-cpu` (and a miri lane over `tritium-core` + `tritium-format --lib`), targeting exactly the unsafe/parse surfaces. **Caveat:** weekly + `workflow_dispatch`, **not a required per-push gate** — regressions in unsafe paths could merge unnoticed until the scheduled run. GPU device-side OOB is observable only via compute-sanitizer on the self-hosted GPU lane.
- **`panic=abort`.** `[profile.release]` sets `panic="abort"` and `[profile.dist]` inherits it (Cargo.toml:132,136). This makes unwinding across `extern "C"` impossible (the safe FFI choice) but means `catch_unwind` is a **no-op in shipped artifacts** — an internal panic aborts the process (availability loss, not UB/RCE). This applies to both the FFI boundary and the server worker.
- **Backend no-panic contract.** `BackendError` ("backends never panic on bad input", tritium-spec/src/lib.rs:240-241); enforced & tested for bad shapes, but **not** against unchecked `usize` product overflow (see 4.4).
- **Supply chain.** Committed `Cargo.lock`; required CI and local build/test/lint gates use `--locked`; cargo-deny `check licenses bans sources advisories` with `yanked="deny"`, `wildcards="deny"`, `unknown-registry`/`unknown-git="deny"`, default-deny license allow-list; CycloneDX SBOM lane; `CARGO_FEATURE_*` gates keep GPU build scripts inert on the default cpu-only build (the single strongest build-surface mitigation). Remaining gaps are trusted toolchain/action drift and unsandboxed build scripts.
- **Bounds-checked cursors & caps.** `gguf.rs Cursor` / `le_cursor.rs LeCursor`; `MAX_COUNT`/`MAX_DIMS`/`MAX_PREALLOC`/`MAX_STRING_LEN` (gguf.rs:36-48); typed error enums (`GgufError`/`SafeTensorsError`/`FormatError`/`BackendError`/`TritiumStatus`) instead of panics throughout.

---

## 6. Explicitly out of scope

- **TLS / HTTPS termination** for the server — none implemented; expected behind a reverse proxy if ever exposed. Operator's responsibility.
- **Fine-grained authorization and identity management** for the server — the
  v1.1 server implements bounded rotating bearer authentication and per-principal
  rate admission, and requires authentication for non-loopback binds. It does not
  implement user accounts, roles, delegated identity, or tenant isolation;
  operators terminate TLS and provide those controls at the trusted edge.
- **A malicious *operator*** — the local operator who chooses what to load and run is trusted by definition.
- **Model-output content safety / prompt injection / harmful generations** — a model is trusted to be the model the user intended; content-level trust is unsolvable at the parser layer (no signature scheme in GGUF/safetensors).
- **Model-content provenance / backdoored weights** — structural parse safety is guaranteed; provenance is not. Mitigation is operational (trusted sources, out-of-band hash verification).
- **Side-channel attacks** (timing, cache, power) — not modeled.
- **Tokenizer correctness** — the server's id-passthrough MVP is not a security surface here.
- **Embedder-forwarded untrusted input** — if an embedding application forwards untrusted file paths or untrusted pointer/length values across the FFI, that is the embedder's responsibility (Tritium bounds its own writes to declared capacity but cannot validate a lying caller).

---

## 7. Residual risks & follow-ups

Honest list of gaps the audits flagged, ordered by actionability. None are critical under the documented trusted/local model; several rise under public exposure.

**Cheap, recommended:**
1. **Add `--locked` to all CI cargo invocations** (ci.yml) + a step that fails on lockfile drift. *(supply chain, medium)* — closes the silent-malicious-patch window the committed lockfile is supposed to prevent.
2. ~~**Lift `check_grad_launch_bounds` into CUDA inference `mpgemm_kernel`.**~~ **DONE** (`0aecae5`) — dense and sparse add paths reject checked-product overflow before allocation, grid math, and i32/u32 casts; regression covers each dimension.
3. ~~**Length-check discipline + fuzz target for `tq2.rs compute_zero_bitmap`/`compute_zero_bitmaps`.**~~ **DONE** — both functions total (typed errors on every absurd-size shape incl. the `row_bytes = 0` capacity-overflow class), `zero_bitmap` fuzz target in the scheduled lane (see §4.1 row).
4. **Defensive FFI hardening (optional):** an `AtomicBool` in-use guard on `TritiumModel` to reject concurrent `tritium_generate` (medium — concurrency UB); a magic sentinel zeroed on free to fail fast on UAF/double-free (medium).
5. **Loader hardening:** make `weights.rs` (124,134,174) reuse `info.element_count()` and checked adds instead of unchecked `+`/`*`. *(low)*

**Server hardening (only needed if the loopback/off-by-default assumption is ever relaxed):**
6. ~~**No auth gate.**~~ **DONE** — non-loopback bind requires uniform bearer authentication; loopback may opt in.
7. ~~**No request timeout / connection cap.**~~ **DONE** — typed service and lazy-SSE deadlines plus a global in-flight cap are enforced.
8. ~~**No explicit max-prompt-tokens / per-request decode budget.**~~ **DONE** — pre-admission message/byte/prompt/completion/combined ceilings reject rather than clamp; fixed-cardinality per-principal token buckets bound admitted generation requests.

**Assurance / process:**
9. **Sanitizer + miri lanes are weekly/manual, not required per-push gates** — a regression in an unsafe/parse path could merge and sit until the next scheduled run. Consider promoting at least a fast subset to a required gate.
10. **No exact toolchain pin, no SHA-pinned Actions, SBOM not diff-gated** — trusted build inputs can drift silently; the SBOM is an artifact, not a baseline check.

**Accepted / inherent (documented, no fix planned):**
11. `catch_unwind` is a no-op under `panic=abort` in release/dist — internal panics abort the process (availability, not UB). Applies to both FFI and the server worker. Understood and documented.
12. Non-null pointer validity/alignment/true-capacity is uncheckable at any C ABI — caller obligation.
13. No model-file authenticity/integrity verification — out of scope for a structural parser; operational mitigation only.
14. No build-script sandboxing / cargo-vet / crev trust review — standard Rust build-time exposure, partially contained by feature gates and `cargo-deny sources`.
15. Backend no-panic contract does not cover unchecked `usize` product overflow — realistic impact requires petabyte-scale shapes that never arise from a real parsed model.
