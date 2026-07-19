# ADR 0030 — Ternary codec substrate: Conv1d, FSQ, and PyTorch autograd bindings

Status: **PROPOSED** (2026-07-18) — onboarding contract, to be co-authored with **LamQuant**

- **Deciders:** Brian Lam + LamQuant
- **Relates:** extends the STE/QAT autograd substrate of ADR 0007 (tape) + ADR 0016 (ternary training)
  + [ADR 0028](./0028-salt-v2-additive-ternarization.md) (additive ternarization, distillation shape);
  the format work extends `tritium-format`; the MCU backend extends the ADR 0009 backend-breadth
  contract; the conformance work extends the [ADR 0018](./0018-canonical-tree-reduction-order.md)
  cross-backend determinism.

## Context
LamQuant is a clinical-grade neural biosignal codec (EEG/ECG/iEEG): a **ternary Conv1d CNN encoder +
FSQ-quantized latent + Vocos (ConvNeXt + iSTFT) decoder**, whose endgame is byte-exact deploy to a
$2 MCU (STM32N6 Cortex-M55 / RP2350 Cortex-M33), with **CPU↔CUDA↔MCU bit-identity** for FDA/PCCP. They
want Tritium as the ternary **compute + format + determinism** substrate. Tritium is a ternary
*transformer* engine; the codec needs **conv-and-quantizer analogues** of what Tritium already does for
linears. **Tier-0 three items (Conv1d + FSQ + PyTorch autograd bindings) are the whole gate** — landing
them lets LamQuant feed real codec layers immediately; everything above is incremental. LamQuant brings
golden bit-exact EEG conformance vectors and a real clinical determinism requirement.

Model, for concreteness. Encoder `TernaryMobileNetV5_Subband`: ternary Conv1d stack (kernels 3/3/5/5/7/7,
dilations, grouped + pointwise), a learned orthogonal 32×32 pre-rotation, an empirical-CDF LUT, int8
(A8) activations, latent `[32, 79]` per ~10 s window from input `[≤168 ch, 313]`. Quantizer `ScalarFSQ`:
levels `{2,3,5,8,16,32}`, round grid `step=2/L`, STE ∈ `{hard, annealed-soft, stochastic}`. Decoder
`VocosDecoder`: ConvNeXt blocks + an inverse-STFT head.

## Substrate audit (EXISTS to reuse vs MISSING) — from 3 exploration agents
| Need | Status | Where |
|---|---|---|
| STE weight-ternarization fwd+vjp (single + multi-plane) | **EXISTS**, 2-D per-row AbsMean | `tritium-train/src/ops/ste.rs` |
| Reverse-mode autograd tape + op convention (fwd+vjp module + `Tape` recording method) + finite-diff gradcheck | **EXISTS**, mechanical to extend | `tritium-train/src/{tape.rs,ops/mod.rs,gradcheck.rs}` |
| Optimizers (AdamW, Muon), **quantization-agnostic** (STE inside graph) | **EXISTS** | `tritium-train/src/optim.rs` |
| Distillation loss (token CE + cached-logit-KD; MSE + softmax-xent tape losses) | **EXISTS** (logit-level) | `salt_v2_recovery.rs`, `ops/loss.rs` |
| Ternary tensor format (TQ2_0/TQ1_0, SALT bundle, GGUF) + **128KB checksummed opaque metadata blob** | **EXISTS**, 2-D `rows×k` only | `tritium-format/src/{salt_bundle.rs,salt_v2_master.rs,tq2.rs}` |
| Backend trait + `linkme` BACKENDS registry (add backend = 1 distributed_slice, no central edit) | **EXISTS** | `tritium-spec/src/lib.rs`, `tritium-runtime/src/lib.rs` |
| `tritium-cpu` scalar path = `reference_mpgemm` (bit-exact) + int8 integer-accumulate kernel | **EXISTS** (MCU mirror candidate) | `tritium-cpu/src/{lib.rs,kernel.rs}` |
| `tritium-core` no_std-ready; conformance harness (frozen vectors, drift gate, Tolerance) | **EXISTS** | `tritium-core`, `tritium-testkit` |
| PyO3 wheel (`tritium-py`) — **inference only** (`Model.load/generate`, toy `ternary_matmul`) | **EXISTS but no torch/autograd/dlpack** | `tritium-py/src/lib.rs` |
| **Ternary Conv1d op (fwd+bwd)** | **MISSING** (only DeltaNet's inference-only depthwise fp32 conv; conv is `PreserveSource` in quantize) | — |
| **FSQ op** (round-to-L grid + STE) and **trainable activation-quant node** | **MISSING** (A8 exists only as inference path) | — |
| N-D / rank-3 ternary tensor in the format | **MISSING** (all 2-D) | — |
| torch `autograd.Function` bindings; C-ABI/ONNX per-op conv/FSQ entry points; no_std backend / MCU target / fixed-arena allocator; 3-way CPU↔CUDA↔MCU gate | **MISSING** (greenfield) | — |

## Tier 0 — the gate (three deliverables)

### 1. Ternary Conv1d — `crates/tritium-train/src/ops/conv1d.rs`
- `forward` + `vjp` (same signature style as `ops/dense.rs`), batched dense Conv1d
  `[batch, C_in, L] → [batch, C_out, L_out]`, supporting **pointwise (1×1), grouped/depthwise, dilated,
  arbitrary kernel** via explicit stride/dilation/groups/padding args (im2col forward, col2im backward).
- **Ternary weight path (recommended):** reshape `[C_out, C_in·K]` and reuse
  `ste::quantize_forward/surrogate/vjp` unchanged → **per-output-channel AbsMean scale** (the natural
  ternary-conv choice). Depthwise/grouped: per-group reshape.
- Add `Tape::conv1d` recording method (copy the `Tape::rmsnorm` template, `tape.rs:435`).
- Gradient-check both the dense-fp and ternary-STE paths with `gradcheck.rs`.

### 2. FSQ — `crates/tritium-train/src/ops/fsq.rs`
- `forward` = bound (tanh/clamp) + round to `L`-level grid `round(x/step)·step`, `step=2/L`, per-channel
  configurable levels `{2,3,5,8,16,32}`. `vjp` = STE variants **hard passthrough / annealed-soft-round /
  unbiased-stochastic (seedable)** — a near-clone of `ste::quantize_surrogate`/`quantize_vjp` with a
  different grid. This op also delivers the currently-missing **trainable activation-quant STE node**.
- Add `Tape::fsq`; gradient-check against the smooth surrogate (as `ste` does — `round` is 0-a.e.).

### 3. PyTorch autograd bindings — extend `crates/tritium-py`
- Expose the tape's `forward`+`vjp` for conv1d + fsq + ste as paired pyo3 functions; add a **tensor
  bridge** (dlpack preferred, else numpy/ndarray — today inputs are plain nested lists). Wrap each on
  the **Python side** in a `torch.autograd.Function` (`forward` calls the fwd fn, `backward` calls the
  vjp fn), so LamQuant swaps layers into `encoder.py`/`vocos_decoder.py` in-place with no rewrite.
  Preserve the `panic=abort` boundary contract (cf. `tritium-ffi`).

## Tiers 1–6 — roadmap (incremental after the gate)
- **Tier 1 (QAT/distill, LamQuant critical path):** LSQ learned step-size α (extend `ste.rs` — today the
  scale is stop-gradient AbsMean); one joint weight-ternary + activation-FSQ QAT recipe; **feature/
  reconstruction distillation** (`ops/loss.rs::mse` covers plain reconstruction; feature/sequence-KL is
  new); SOAP/ESOAP compatibility (already optimizer-agnostic — just keep STE optimizer-agnostic + expose
  a Cautious-WD toggle); seedable stochastic rounding for deterministic replay.
- **Tier 2 (format):** extend the SALT V2 `TSV2MTR` metadata blob (`salt_v2_master.rs`, 128KB
  checksummed) to carry **conv weights (`C_out×C_in×K`) + per-layer LSQ α + FSQ level schedule + CDF LUTs
  + rotation matrix** → a codec-complete ternary artifact; a container co-locating encoder + decoder +
  quantizer params + montage/coords (GGUF-extended or a sibling — one mmap-able file).
- **Tier 3 (MCU):** new `crates/tritium-mcu` backend mirroring `tritium-cpu` (scalar
  `mpgemm`→`reference_mpgemm`, bit-exact by construction), registered via one `distributed_slice`.
  **Prerequisite:** make `tritium-spec` no_std/alloc (today it pulls `std::error::Error`;
  `tritium-core` is already no_std). Add embedded targets (`thumbv8m.main-none-eabi*`), a fixed-arena
  allocator (static SRAM budget — their float build `.bss`-OOM'd at ~94KB), and a **fixed-point
  int-only** path built on the `tritium-cpu` int8 integer-accumulate kernel (`kernel.rs`, order-free
  i32 accumulate — the bit-exact-friendly primitive). Neural-ART NPU lowering is P3.
- **Tier 4 (conformance — clinical, load-bearing):** add Conv1d/FSQ/rotation **reference ops in
  `tritium-core`** + typed conformance vectors in `tritium-testkit` + a **three-way CPU↔CUDA↔MCU
  byte-identity gate** (today only CPU↔CUDA f32 parity exists via ADR 0018); re-freeze as a new vector
  version. LamQuant contributes golden EEG vectors.
- **Tier 5 (interop):** extend `tritium-onnx` (conv+FSQ graph import beyond the single mpGEMM op),
  candle/burn codec ops (their host autograd can supply gradients), a WASM decoder via `tritium-wasm`.
- **Tier 6 (research):** ternary SSM / Mamba selective-scan kernels (adjacent to DeltaNet linear
  attention) for their compression controller.

## Decisions / open questions (resolve with LamQuant during co-authoring)
- **Conv ternary scale granularity:** per-output-channel (reshape) — recommended default; grouped/
  depthwise per-group.
- **FSQ STE default:** hard passthrough; soft/stochastic opt-in.
- **Format:** extend the SALT V2 `TSV2MTR` blob vs a new codec format section — recommend *extend + add a
  typed codec section*; confirm with LamQuant's container needs (montage/coords, channel-agnostic 8–256ch).
- **Bindings:** pyo3-expose-tape + Python `autograd.Function` wrapper (recommended, lowest friction) vs
  binding torch custom ops in Rust.

## Verification
- `gradcheck` passes for conv1d + fsq (finite-diff vs smooth surrogate), fp and ternary/FSQ-STE paths.
- **Torch parity:** the Tritium conv/FSQ `autograd.Function` matches `torch.nn.Conv1d` / a manual FSQ
  within tolerance on a golden EEG window (forward and gradient).
- Conv1d/FSQ/rotation **conformance vectors** CPU↔CUDA bit-exact via `tritium-testkit`; MCU byte-identity
  when the backend lands (LamQuant's golden EEG vectors as the clinical gate).
- End-to-end adoption smoke: LamQuant swaps one ternary Conv1d + FSQ layer into `encoder.py`, trains a
  step, and the deployed artifact is byte-exact host↔MCU.
