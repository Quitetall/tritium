# Width, depth and timing of fused forward-only training

2026-10-16. Branch `research/eat-residual-carry`, `examples/ternary_deep.rs` (integer ternary MLP
of any widths; carry-sgd from the exact STE gradient or the forward-only noise estimate; batched
draws; buffered carry writes; per-phase timing). Exploratory; quality UNKNOWN; official MNIST test
set untouched.

## Questions

1. **Cost per example.** How does fused forward-only training compare with backprop in wall-clock
   per example, phase by phase? Do batched noise draws ride a single weight read for free, and how
   much does buffering carry writes save?
2. **Width and depth.** How many more examples does the fused rule need to match exact-gradient
   quality, and how does that multiplier grow with width and depth?

## Identity

`--sizes 784,128,10` reproduces the lab network and `carry-sgd` end to end, bit for bit (seed 1:
0.9200 / 19,821.73 / 2,066,020 transitions, identical to `ternary_carry`).

Unit tests assert:
- equal initial trits, forward pass and gradient (`trunc(g/2¹¹)` = `Model::sensitivity`);
- the batched forward at zero noise equals the plain loss;
- the batched noise estimate approaches the exact unit errors on a 3-layer net (cosine > 0.8 per
  layer at 3,000 draws);
- exact training lowers the loss.

The first timing pass had two defects, fixed before the numbers below:
- the backward pass read weight columns with a stride (Wᵀδ at ~9× the forward's time), which
  biased the comparison against backprop. It now reads rows contiguously and scatters.
- the 64-example buffer never flushed inside the timed loop on 40-example runs.

## 1. Timing (single thread pinned to one core, minimum of 3 repeats, µs per example)

The host was shared with other sessions (load average ~90 on 32 threads), so these timings are
descriptive. The code is a scalar reference implementation (i8 trits, i64 sums, i128 carry
arithmetic), not Tritium's packed kernels.

**784-1024-1024-10 (1.85 M weights)**

| Arm | Buffer | Forward | Credit | Update | Total |
|---|---:|---:|---:|---:|---:|
| exact backprop | 1 | 386 | 324 | 3,154 | 3,864 |
| exact backprop | 16 | 351 | 326 | 2,373 | 3,050 |
| fused K=1 | 1 | 392 | 430 | 3,642 | 4,464 |
| fused K=1 | 16 | 351 | 436 | 2,571 | 3,358 |
| K=4 | 16 | 354 | 1,678 | 2,582 | 4,615 |
| K=16 | 16 | 362 | 6,708 | 2,578 | 9,648 |
| K=64 | 16 | 370 | 27,567 | 2,419 | 30,356 |

**784-4096-4096-10 (20.0 M weights; the 80 MB i32 carry exceeds the 36 MB L3)**

| Arm | Buffer | Forward | Credit | Update | Total |
|---|---:|---:|---:|---:|---:|
| exact backprop | 1 | 3,977 | 5,039 | 36,058 | 45,074 |
| exact backprop | 64 | 4,224 | 5,169 | 28,409 | 37,802 |
| fused K=1 | 1 | 3,933 | 6,633 | 54,060 | 64,626 |
| fused K=1 | 64 | 3,977 | 6,713 | 29,434 | 40,124 |
| K=4 | 64 | 4,145 | 27,876 | 30,292 | 62,314 |
| K=16 | 64 | 3,993 | 105,845 | 29,070 | 138,908 |
| K=64 | 64 | 3,991 | 441,404 | 28,209 | 473,604 |

Full rows: `timing/t-*.json` (three repeats each; K=64 on the 4096 net once).

## 2. Width and depth (five paired seeds, validation 1000–4999)

| Network | Params | Exact, 20k examples | Fused K=1, 20k | 80k | 320k | Extrapolated data multiplier |
|---|---:|---:|---:|---:|---:|---:|
| 784-128-10 | 101,632 | 92.05 ± 0.14 (loss 19,813) | 81.64 ± 1.97 | 87.32 ± 0.67 | 89.81 ± 0.24 (loss 23,748) | ~60× |
| 784-512-10 | 406,528 | 94.84 ± 0.12 (loss 13,501) | 81.11 ± 4.09 | 87.37 ± 0.65 | 90.48 ± 0.29 (loss 22,524) | ~250× |
| 784-256-256-10 | 268,800 | 95.46 ± 0.24 (loss 9,004) | 71.93 ± 5.05 | 85.37 ± 1.96 | 91.73 ± 0.52 (loss 17,926) | ~90× |

Every fused arm is significantly worse than its exact arm (paired Δloss intervals exclude zero;
`runs/eval-*.json`). The multiplier is the number of examples the fused rule would need to reach
the exact arm's 20k-example loss. It is extrapolated log-linearly in examples from the 80k and
320k points, because no fused arm reached parity within 16×. Treat it as an order of magnitude.
One arm (w512, 80k) selected the grid edge (shift 1); extending to shift 0 was worse
(29,898 vs 28,417), so the selection stands.

## Reading

1. **Per example, fused K=1 costs about what backprop costs:** 1.06–1.43× wall-clock.
   - Its credit (two antithetic partial forwards per perturbed layer) is about 1.1–1.7× the
     forward pass. Backprop's error step is about 0.85–1.3× the forward pass.
   - Both are dwarfed by the **weight update**: the δ ⊗ x outer product into the carry, 70–85% of
     every step. That cost is common to both methods.
   - The forward pass is not where training time goes, and neither is credit assignment.
2. **Batched draws do not ride free here.** Credit time is linear in K (K=4 ≈ 4×, K=16 ≈ 15.5×,
   K=64 ≈ 63–66×). The 4096-wide forward reads 20 MB in 4 ms, about 5 GB/s, far below DRAM
   bandwidth. So this scalar CPU path is compute-bound even at K=1, and there is no idle compute
   for extra draws to use. The "fat forward" advantage requires a kernel that is bandwidth-bound
   at batch 1 (packed ternary GEMV on GPU, or a ternary accelerator). That is untested here and
   stays UNKNOWN.
3. **Buffering carry writes helps 15–40%** (4096 net: K=1 64.6 → 40.1 ms; exact 45.1 → 37.8 ms).
   - The carry's memory traffic is real but not the whole update cost. The per-weight
     multiply-accumulate of the buffered outer products remains.
   - A production update kernel (int8 carry, SIMD, stochastic rounding) is the lever that matters
     for **both** methods.
4. **The data multiplier grows with width, as the dimension tax predicts.**
   - At 4× hidden width it rises about 4× (~60× → ~250×). The wider net gains 2.8 points from
     exact training but only 0.7 from fused training at 320k examples.
   - Depth at 2 × 256 costs less than width at 512 (~90×), roughly in line with its 256-unit
     layers.
   - Training **from scratch** forward-only is therefore 10²–10³× backprop's examples at these
     sizes, and that number grows with layer width. Fusion (rung F) made the per-step population
     unnecessary; it did not remove the tax.
5. **Combined:** wall-clock to equal quality from scratch ≈ (data multiplier) × (~1.1–1.4×
   per-example cost), i.e. about 70–350× backprop on these CPU nets. Forward-only from-scratch
   training is not competitive on compute. Its case rests on memory, hardware and the
   low-intrinsic-dimension regime (fine-tuning), which these runs do not measure.

Bounds: MNIST MLPs only, squared loss, five seeds, validation reused across tuning and evaluation,
extrapolated multipliers, a shared host for timing, scalar reference code. No LM, no GPU, no
fine-tuning measurement.

## Files

`run.py` (width/depth campaign), `runs/` (every tune and eval run, `protocol.json` with binary
hash, and the w512-80k shift-0 extension), `timing/t-*.json` (every timing run).
