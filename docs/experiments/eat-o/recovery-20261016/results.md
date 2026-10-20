# Recovering a ternarized fp model: backprop vs fused forward-only, dense vs sparse carry

2026-10-16..20. Branch `research/eat-residual-carry`, `examples/ternary_deep.rs`
(`--fp-steps`, `--target teacher`, `--sparse-carry`, `--carry-init fp`). Exploratory; quality
UNKNOWN; official MNIST test set untouched.

## Question

The deployment idea is to SALT an fp model to ternary, then recover its strength on device with
forward-only training. Can fused forward-only training (one ternary noise draw per example,
credited straight into the integer carry) recover a ternarized model, and at what cost against
backprop? Does a sparse carry help: only weights near a rounding boundary may move, chosen by a
1-bit mask the converter could ship?

## Method

- **fp model** (per seed):
  - same 784-128-10 architecture and initial values as the ternary net;
  - real-valued weights and activations;
  - f64 backprop, squared loss to the lab's 256-scaled one-hot, 165k examples, lr 1e-4 (best of
    a sweep over 1e-7..3e-4);
  - backprop checked against central finite differences (unit test).
- **Ternarization:**
  - per layer, a power-of-two scale 2^k minimizing ‖W − 2^k·round_clamp(W/2^k)‖², a SALT-style
    scale fit restricted to powers of two so it folds into the layer's shift and the forward stays
    shift-only integer;
  - trit = round_clamp(W/2^k).
  - The fit chose k = 0 for the input layer and k = 3 for the output layer (shifts 8, 3 → 8, 0).
  - Round-to-nearest at the fixed lab scales was rejected as an unfair stand-in: the fp weights
    exceed ±1, and clamping left the output scale broken (loss ≈ an untrained model).
- **Recovery:**
  - carry-sgd from the trit centres (no fp master on device), on labels or on the fp model's
    outputs ("teacher", which would ship precomputed);
  - exact STE gradient, or the fused noise rule (ternary antithetic, σ = 2 quantization steps,
    K = 1).
- **Variants:**
  - `--sparse-carry p`: only the p% of weights nearest a ternary rounding boundary (in fitted trit
    units) carry and move; the rest are frozen;
  - `--carry-init fp`: the carry starts at the fp residual (what shipping the master would buy).
- **Protocol:** lr shift tuned on seed 101 at half the budget (validation 0–999); five paired
  seeds (validation 1000–4999); every seed has its own fp model and ternarization.
  - Grid {2..11}. Every arm selecting the top edge was extended to {12, 13, 14}, and again to
    {15, 16, 17} where still at the edge.
  - Two fused arms had selected shift 4 on a noisy single-seed curve and were never extended.
    For parity they were tuned over the same extended grid, and both moved (to 13 and 14).
  - Superseded evaluations are kept (`eval-*.edge11.json`, `*.edge14.json`, `*.shift4.json`)
    and every extension is recorded in `runs/protocol.json`.
  - Shifts 2 and 3 are identical: the output layer's fitted scale makes the effective shift clamp
    at 0.

## Results (five seeds; mean ± SD accuracy, mean loss)

fp model: **97.54%**, loss 6,303. Ternarized start: **82.20%**, loss 36,795.

| Recovery | Examples | Labels | Teacher (fp outputs) |
|---|---:|---:|---:|
| backprop | 2k | 95.39 ± 0.66 (18,235) | 95.11 ± 0.65 (18,688) |
| backprop | 8k | 95.96 ± 0.20 (15,117) | 96.11 ± 0.07 (15,441) |
| backprop | 32k | **96.39 ± 0.07** (13,464) | 96.31 ± 0.16 (14,143) |
| fused forward-only | 2k | 93.82 ± 0.58 (24,667) | 93.11 ± 1.02 (24,894) |
| fused forward-only | 8k | 94.14 ± 0.63 (21,722) | 93.38 ± 2.03 (22,343) |
| fused forward-only | 32k | 94.69 ± 0.23 (19,656) | 94.45 ± 0.61 (18,682) |
| fused forward-only | 128k | 95.10 ± 0.28 (19,057) | 95.37 ± 0.57 (17,342) |
| fused forward-only | 512k | 95.09 ± 0.33 (17,453) | 94.59 ± 0.33 (17,830) |

| Variant (teacher) | Examples | Carry on | Accuracy | Loss |
|---|---:|---:|---:|---:|
| backprop, sparse 20% | 8k | 20% | 95.97 ± 0.33 | 13,958 |
| **fused, sparse 20%** | 128k | 20% | **95.25 ± 0.23** | **15,661** |
| backprop, sparse 5% | 8k | 5% | 94.12 ± 0.93 | 24,353 |
| fused, sparse 5% | 128k | 5% | 93.97 ± 1.43 | 24,669 |
| backprop, carry seeded with fp residual | 8k | 100% | 96.30 ± 0.19 | 14,765 |
| fused, carry seeded with fp residual | 128k | 100% | 95.53 ± 0.07 | 18,204 |

Paired contrasts (Δloss, first minus second, 95% t, df = 4):

| Contrast | Δloss | Reading |
|---|---:|---|
| fused sparse 20% − fused dense (128k, teacher) | **−1,681 [−2,472, −890]** | sparse 20% helps forward-only |
| fused sparse 20% (128k) − backprop dense (8k, teacher) | +220 [−1,261, 1,701] | **tie** |
| fused sparse 5% − fused dense (128k) | +7,326 [5,657, 8,996] | 5% is too few movable weights |
| fused sparse 5% − backprop sparse 5% | +316 [156, 477] | nearly equal at the same mask |
| backprop sparse 20% − backprop dense (8k) | −1,483 [−3,014, 48] | unresolved, not worse |
| fp-seeded − centre carry (backprop 8k) | −676 [−1,852, 500] | unresolved |
| fp-seeded − centre carry (fused 128k) | +862 [−48, 1,771] | unresolved, no benefit |
| teacher − labels (backprop 8k) | +324 [−1,334, 1,982] | unresolved |
| teacher − labels (fused 512k) | +378 [−1,445, 2,201] | unresolved |
| fused 512k − backprop 2k (labels/teacher) | −404 [−2,044, 1,235] | tie at 256× the examples |

## Reading

1. **Ternarization damage is easy to recover with backprop.** 2k examples lift 82.2% to 95.4%,
   and 32k reach 96.4%, about 1.1 points under the fp model. On this network the recoverable
   ceiling for a ternary 784-128-10 is about 96.4%.
2. **Fused forward-only recovery works, at the right learning rate.**
   - At 2k examples it recovers 82.2 → 93.8%, three quarters of what backprop recovers at the
     same data.
   - It plateaus at about 95.1–95.4% from 128k on (512k is no better), roughly 1 point under
     backprop.
   - At constant learning rate, the noise floor caps it. An lr decay schedule is the obvious
     untested lever.
   - It matches backprop's 2k-example result at 512k examples, a ~256× data multiplier for full
     parity. Most of the recovery, though, arrives in the first few thousand examples.
3. **Sparse carry at 20% is the best forward-only configuration.**
   - It beats dense fused recovery (−1,681 [−2,472, −890]) and ties dense backprop at 8k
     (+220 [−1,261, 1,701]).
   - It also cuts carry memory to a fifth: 1 byte/weight on 20% of weights.
   - The mask is 1 bit per weight from the converter, which knows each weight's residual at
     ternarization time.
   - At 5% the mask leaves too few movable weights. Both methods cap near 94.1%, and fused ≈
     backprop there. The best fraction lies between 5% and 100%, with 20% the best measured.
4. **Shipping the fp residual buys nothing** measurable (both contrasts unresolved). Ship the
   1-bit mask, not the master.
5. **Teacher outputs vs labels:** no resolved difference for either method on MNIST. The
   distillation case should be re-tested on a language model, where soft targets carry far more
   information than one-hot labels.
6. **Learning rate is the dominant hyperparameter for fused training.**
   - Fused arms need 2⁻⁷–2⁻¹¹ of backprop's step (shifts 12–15 vs 7–11).
   - A single-seed tuning curve can select a catastrophic setting: two arms first selected
     shift 4 and evaluated at 25% and 76%.
   - Any production use needs an lr schedule or a noise-aware step size, not a hand-picked shift.

Revision note: an interim reading (2026-10-20, before the grid extensions) said dense fused
recovery made the model worse than its start below 32k examples. That was an artefact of a
grid that stopped at too large a learning rate. Every fused arm improved once tuned at smaller
steps.

Bounds: one MNIST MLP, squared loss, five seeds, validation reused across tuning and
evaluation, power-of-two scale fit (not SALT's ladder or GPTQ), constant learning rate. No LM,
no MTP, no hardware or wall-clock claim.

## Files

`run.py` (campaign), `extend.py` (first grid extension), `runs/protocol.json` (arms, grid,
selections, all three extension passes, binary hash), `runs/tune-*.json`, `runs/eval-*.json`
(current), `runs/eval-*.{edge11,edge14,shift4}.json` (superseded). The fp models are cached
outside the repo (`/mnt/4tb/tmp/tritium-eat-ladder/fp-cache`); they are deterministic from
`--fp-steps 165000 --fp-lr 1e-4` and the seed.
