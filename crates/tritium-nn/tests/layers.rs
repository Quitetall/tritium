//! WF-3 integration tests for the inference layers, driven through the real CPU
//! ternary backend.
//!
//! - [`ternary_linear_matches_reference`] — the `TernaryLinear` W1.58A8 forward
//!   (int8 activation quant → ternary mpGEMM → per-token/per-tensor dequant fold)
//!   against an inline reference, `≤ 1e-4` relative (the ternary+quant path is
//!   near-exact).
//! - [`relu2_mlp_matches_torch`] — the gated ReLU² MLP against a `transformers`
//!   `BitNetMLP`-shaped torch reference (gate/up/down ternary linears, `relu(z)²`
//!   gating, both following the same int8 act-quant path), `≤ 2e-3`.
//! - [`transformer_block_smoke`] — the decoder block assembles: correct output
//!   shape, finite values, and both residual paths observably wired.
//!
//! The torch numbers in [`relu2_mlp_matches_torch`] were produced by a snippet
//! that mirrors the Rust W1.58A8 path (per-token int8 absmax quant, ternary
//! matmul, weight-scale + act-scale fold) so the only residual difference is fp32
//! accumulation order — see the test's doc comment for the exact recipe.

use tritium_core::Trit;
use tritium_cpu::CpuBackend;
use tritium_nn::{Mlp, ModelConfig, Projection, Relu2Mlp, TernaryLinear, TransformerBlock};

/// Deterministic xorshift64 PRNG so the "random" weights/activations are
/// reproducible without a dependency.
struct Rng(u64);

impl Rng {
    fn new(seed: u64) -> Self {
        Rng(seed)
    }

    fn next_u64(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    /// Uniform `f32` in `[-range, range)`.
    fn next_f32(&mut self, range: f32) -> f32 {
        let u = (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32; // [0,1)
        (u * 2.0 - 1.0) * range
    }

    /// A ternary value in `{-1, 0, +1}` (uniform over the three states).
    fn next_trit(&mut self) -> Trit {
        match self.next_u64() % 3 {
            0 => Trit::NEG,
            1 => Trit::ZERO,
            _ => Trit::POS,
        }
    }
}

// --------------------------------------------------------------------------- //
// Shared int8 activation-quant reference (matches `quantize_activation_int8`).
// --------------------------------------------------------------------------- //

/// Per-token int8 absmax quant: returns the int8-as-f32 values and per-row
/// dequant scale `gamma / 127`. Mirrors `ops::quantize_activation_int8` (the
/// authority is the in-crate test of that op; here it is the reference the
/// layer path is graded against).
fn ref_quant(act: &[f32], rows: usize, cols: usize) -> (Vec<f32>, Vec<f32>) {
    let mut q = vec![0.0f32; rows * cols];
    let mut scale = vec![0.0f32; rows];
    for r in 0..rows {
        let row = &act[r * cols..r * cols + cols];
        let gamma = row.iter().fold(0.0f32, |m, &v| m.max(v.abs()));
        if gamma == 0.0 {
            continue;
        }
        let s = 127.0f32 / gamma;
        for c in 0..cols {
            q[r * cols + c] = (row[c] * s).round_ties_even().clamp(-128.0, 127.0);
        }
        scale[r] = gamma / 127.0;
    }
    (q, scale)
}

/// Reference ternary linear: `y[r,n] = weight_scale · act_scale[r] · Σ_k
/// q_act[r,k] · trit[n,k]`, with `trits` the `[n_out, k_in]` weight.
fn ref_ternary_linear(
    act: &[f32],
    trits: &[Trit],
    n_out: usize,
    k_in: usize,
    weight_scale: f32,
    rows: usize,
) -> Vec<f32> {
    let (q, ascale) = ref_quant(act, rows, k_in);
    let mut out = vec![0.0f32; rows * n_out];
    for r in 0..rows {
        for n in 0..n_out {
            let wrow = &trits[n * k_in..n * k_in + k_in];
            let qrow = &q[r * k_in..r * k_in + k_in];
            let mut acc = 0.0f32;
            for k in 0..k_in {
                match wrow[k].get() {
                    1 => acc += qrow[k],
                    -1 => acc -= qrow[k],
                    _ => {}
                }
            }
            out[r * n_out + n] = acc * weight_scale * ascale[r];
        }
    }
    out
}

/// Largest relative error between `got` and `want` (absolute error where `want`
/// is tiny, to avoid divide-by-zero blow-ups on near-zero entries).
fn max_rel_err(got: &[f32], want: &[f32]) -> f32 {
    assert_eq!(got.len(), want.len(), "length mismatch in error check");
    let mut worst = 0.0f32;
    for (&g, &w) in got.iter().zip(want.iter()) {
        let denom = w.abs().max(1e-3);
        let e = (g - w).abs() / denom;
        if e > worst {
            worst = e;
        }
    }
    worst
}

// --------------------------------------------------------------------------- //
// 1. TernaryLinear forward vs reference.
// --------------------------------------------------------------------------- //

#[test]
fn ternary_linear_matches_reference() {
    let backend = CpuBackend::new();
    let mut rng = Rng::new(0xC0FF_EE12_3456_789A);

    // A non-block-aligned K (300 = 256 + 44) exercises the multi-block TQ2_0 pack
    // with a zero-padded tail; a few output channels and a small batch.
    let n_out = 17;
    let k_in = 300;
    let m = 5;
    let weight_scale = 0.37_f32;

    let trits: Vec<Trit> = (0..n_out * k_in).map(|_| rng.next_trit()).collect();
    let act: Vec<f32> = (0..m * k_in).map(|_| rng.next_f32(3.0)).collect();

    let linear = TernaryLinear::new(&backend, &trits, n_out, k_in, weight_scale)
        .expect("construct ternary linear");
    assert_eq!(linear.n_out, n_out);
    assert_eq!(linear.k_in, k_in);
    assert_eq!(linear.scales, vec![weight_scale; n_out]);

    let mut got = vec![f32::NAN; m * n_out];
    linear
        .forward(&backend, &act, m, &mut got)
        .expect("forward");

    let want = ref_ternary_linear(&act, &trits, n_out, k_in, weight_scale, m);
    let rel = max_rel_err(&got, &want);
    assert!(
        rel <= 1e-4,
        "TernaryLinear forward rel err {rel} exceeds 1e-4"
    );
    assert!(got.iter().all(|v| v.is_finite()), "non-finite output");
}

#[test]
fn ternary_linear_shape_errors() {
    let backend = CpuBackend::new();
    let trits = vec![Trit::POS; 4 * 6];
    let linear = TernaryLinear::new(&backend, &trits, 4, 6, 1.0).expect("construct");

    // Wrong activation length for m = 2 (needs 12, supply 10).
    let act = vec![1.0f32; 10];
    let mut out = vec![0.0f32; 8];
    assert!(linear.forward(&backend, &act, 2, &mut out).is_err());

    // Wrong output length.
    let act = vec![1.0f32; 12];
    let mut out = vec![0.0f32; 7];
    assert!(linear.forward(&backend, &act, 2, &mut out).is_err());

    // Constructor: trits length mismatch.
    assert!(TernaryLinear::new(&backend, &trits, 4, 7, 1.0).is_err());
}

// --------------------------------------------------------------------------- //
// 2. Relu2Mlp forward vs a transformers BitNetMLP-shaped torch reference.
// --------------------------------------------------------------------------- //

/// Build a `TernaryLinear` from an `i8` trit slice (`{-1,0,1}`).
fn linear_from_i8(
    backend: &CpuBackend,
    w: &[i8],
    n_out: usize,
    k_in: usize,
    scale: f32,
) -> TernaryLinear {
    let trits: Vec<Trit> = w.iter().map(|&v| Trit::from_i8(v).unwrap()).collect();
    TernaryLinear::new(backend, &trits, n_out, k_in, scale).expect("construct linear")
}

/// Golden generated by a torch snippet mirroring the Rust W1.58A8 path:
/// for each of gate/up/down, per-token int8 absmax quant (`round_ties_even`,
/// clamp `[-128,127]`), ternary matmul, then `weight_scale · act_scale` fold;
/// the gate output passes through `relu(z)²` and is multiplied elementwise by the
/// up output before `down`. `transformers` `BitNetMLP` is
/// `down(ffn_sub_norm(relu2(gate(x)) * up(x)))` with `act_fn = relu2`; the
/// `ffn_sub_norm` (unit weight) is wired in WF-4, so this reference omits it and
/// matches the `Relu2Mlp` body exactly. `m = 3`, `n_embd = 8`, `n_ff = 16`.
#[test]
fn relu2_mlp_matches_torch() {
    let backend = CpuBackend::new();

    let n_embd = 8usize;
    let n_ff = 16usize;
    let m = 3usize;

    #[rustfmt::skip]
    let gate_w: [i8; 128] = [
        -1, 1, 0, 0, -1, 1, 0, 1, 0, 1, 0, 0, 1, -1, -1, 0,
        1, 0, -1, 0, 0, 1, 0, 1, 1, 0, 1, -1, 0, 0, -1, -1,
        0, 0, 1, 0, 1, 1, 0, -1, 1, -1, 0, -1, -1, 1, 1, 0,
        -1, 1, -1, -1, 0, -1, 0, 0, 0, 1, 1, -1, 1, -1, 1, 0,
        -1, 1, -1, 0, 0, 0, 1, -1, 0, 0, 0, 1, -1, 0, 1, 1,
        -1, 0, 1, -1, 0, 1, 0, -1, 1, 0, 0, 1, -1, -1, 0, 1,
        0, 0, 1, 1, 0, -1, -1, 1, 0, -1, -1, 0, 0, -1, -1, -1,
        -1, -1, 1, 1, 1, 0, -1, -1, 1, 0, 1, 1, 0, -1, -1, 1,
    ];
    #[rustfmt::skip]
    let up_w: [i8; 128] = [
        -1, 1, 0, 1, 1, 0, 0, 0, 1, 0, -1, 0, 1, -1, 1, 1,
        0, 1, 1, 1, 0, 0, 0, 1, -1, 0, -1, 0, -1, 0, -1, 0,
        1, 1, 0, -1, 0, 1, 1, 1, -1, 1, 0, 1, -1, 1, -1, 0,
        1, -1, -1, 0, 1, 1, -1, 0, 0, 0, 1, -1, 0, 1, -1, 1,
        -1, 0, 0, -1, 1, 1, 0, 1, 0, 1, 0, 1, -1, -1, -1, 0,
        -1, 1, -1, 0, -1, 0, 1, -1, 1, 0, 1, 1, 0, 0, -1, -1,
        1, -1, 0, 1, 1, 1, -1, 0, -1, -1, 1, -1, 0, -1, -1, -1,
        0, -1, -1, 1, 1, -1, 0, 0, 1, 0, 1, -1, 0, 1, 0, -1,
    ];
    #[rustfmt::skip]
    let down_w: [i8; 128] = [
        -1, 1, -1, 1, 1, -1, 0, 0, 0, 0, 1, -1, 1, -1, 1, -1,
        -1, -1, -1, -1, 0, -1, -1, 0, 0, 1, 1, 0, -1, -1, 1, 0,
        1, -1, 0, 0, -1, -1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0,
        1, 1, -1, -1, 1, 1, -1, -1, 1, -1, 1, 0, -1, 0, 0, 0,
        -1, 0, 1, -1, 0, 1, -1, 0, 0, -1, -1, -1, -1, 0, -1, -1,
        1, -1, 1, 1, 0, 0, 0, -1, 0, 0, 1, 0, 0, -1, 1, 0,
        -1, 0, -1, -1, 1, 1, 0, 0, -1, 1, 1, -1, 0, -1, 1, -1,
        1, 0, 1, 0, 0, 0, 0, -1, 0, 1, 0, -1, -1, 0, 0, 0,
    ];

    let gate_s = 0.7f32;
    let up_s = 1.3f32;
    let down_s = 0.9f32;

    #[rustfmt::skip]
    let x: [f32; 24] = [
        0.0280848, 1.81962, -1.7041, -0.763918, 1.16651, -0.435736, -0.4094, -0.833583,
        1.37861, 0.981006, 0.6409, -1.12393, -1.6235, 0.216321, 0.592558, -0.923424,
        -0.559595, 1.35074, 0.159319, 0.0902367, -0.492201, -1.81118, -1.88051, -0.95603,
    ];
    #[rustfmt::skip]
    let want: [f32; 24] = [
        19.0772, -29.0962, 1.37246, 13.9991, -13.4501, -11.8032, -3.84289, 4.11738,
        -1.45049, 12.5172, 4.35148, 5.8557, 1.7191, -5.10359, 7.68225, -1.45049,
        -48.2883, 23.9342, 8.18802, -7.55817, 23.3044, 13.4368, -26.8735, 10.7074,
    ];

    let mlp = Relu2Mlp {
        gate: Projection::Ternary(linear_from_i8(&backend, &gate_w, n_ff, n_embd, gate_s)),
        up: Projection::Ternary(linear_from_i8(&backend, &up_w, n_ff, n_embd, up_s)),
        down: Projection::Ternary(linear_from_i8(&backend, &down_w, n_embd, n_ff, down_s)),
        // Empty sub-norm => skipped, matching the no-sub-norm torch reference here.
        ffn_sub_norm: Vec::new(),
        rms_eps: 1e-5,
    };

    let mut got = vec![f32::NAN; m * n_embd];
    mlp.forward(&backend, &x, m, &mut got).expect("mlp forward");

    let rel = max_rel_err(&got, &want);
    assert!(rel <= 2e-3, "Relu2Mlp vs torch rel err {rel} exceeds 2e-3");
    assert!(got.iter().all(|v| v.is_finite()), "non-finite MLP output");
}

// --------------------------------------------------------------------------- //
// 3. TransformerBlock assembly / smoke.
// --------------------------------------------------------------------------- //

fn rand_linear(backend: &CpuBackend, rng: &mut Rng, n_out: usize, k_in: usize) -> Projection {
    let trits: Vec<Trit> = (0..n_out * k_in).map(|_| rng.next_trit()).collect();
    // Small positive scale keeps activations in a sane range across the block.
    Projection::Ternary(
        TernaryLinear::new(backend, &trits, n_out, k_in, 0.05).expect("construct linear"),
    )
}

fn tiny_cfg() -> ModelConfig {
    // n_embd 16, n_head 4, n_head_kv 2, head_dim 4, n_ff 32.
    ModelConfig {
        arch: "bitnet".to_owned(),
        n_layers: 1,
        n_embd: 16,
        n_head: 4,
        n_head_kv: 2,
        head_dim: 4,
        n_ff: 32,
        n_ctx: 64,
        rope_theta: 10000.0,
        rms_eps: 1e-5,
    }
}

fn build_block(backend: &CpuBackend, rng: &mut Rng, cfg: &ModelConfig) -> TransformerBlock {
    let n_embd = cfg.n_embd as usize;
    let n_ff = cfg.n_ff as usize;
    let head_dim = cfg.head_dim() as usize;
    let q_width = cfg.n_head as usize * head_dim;
    let kv_width = cfg.n_head_kv as usize * head_dim;

    TransformerBlock {
        attn_norm: (0..n_embd).map(|_| 1.0 + rng.next_f32(0.1)).collect(),
        q_proj: rand_linear(backend, rng, q_width, n_embd),
        k_proj: rand_linear(backend, rng, kv_width, n_embd),
        v_proj: rand_linear(backend, rng, kv_width, n_embd),
        o_proj: rand_linear(backend, rng, n_embd, q_width),
        // Empty sub-norm => skipped, keeping the WF-3 assembly smoke test unchanged.
        attn_sub_norm: Vec::new(),
        q_bias: Vec::new(),
        k_bias: Vec::new(),
        v_bias: Vec::new(),
        q_norm: Vec::new(),
        k_norm: Vec::new(),
        ffn_norm: (0..n_embd).map(|_| 1.0 + rng.next_f32(0.1)).collect(),
        mlp: Mlp::Relu2(Relu2Mlp {
            gate: rand_linear(backend, rng, n_ff, n_embd),
            up: rand_linear(backend, rng, n_ff, n_embd),
            down: rand_linear(backend, rng, n_embd, n_ff),
            ffn_sub_norm: Vec::new(),
            rms_eps: 1e-5,
        }),
    }
}

#[test]
fn transformer_block_smoke() {
    let backend = CpuBackend::new();
    let mut rng = Rng::new(0x5EED_1234_ABCD_0001);
    let cfg = tiny_cfg();
    let n_embd = cfg.n_embd as usize;

    let block = build_block(&backend, &mut rng, &cfg);

    // Prefill: three tokens at positions 0,1,2.
    let seq = 3usize;
    let x: Vec<f32> = (0..seq * n_embd).map(|_| rng.next_f32(1.0)).collect();
    let positions = [0usize, 1, 2];

    let mut kv = tritium_nn::KvCache::new(
        cfg.n_ctx as usize,
        cfg.n_head_kv as usize,
        cfg.head_dim() as usize,
    );

    let mut out = vec![f32::NAN; seq * n_embd];
    block
        .forward(&backend, &x, &positions, &mut kv, &cfg, &mut out)
        .expect("block forward");

    // Shape + finiteness.
    assert_eq!(out.len(), seq * n_embd);
    assert!(out.iter().all(|v| v.is_finite()), "non-finite block output");

    // The KV cache advanced by `seq` tokens (attention path ran).
    assert_eq!(kv.len, seq);

    // Residual paths are observably present: out = x + attn + mlp. With nonzero
    // weights the block output must differ from the bare input on at least one
    // element, yet stay finite (i.e. the residual add did not no-op the input
    // away nor blow it up).
    let changed = out
        .iter()
        .zip(x.iter())
        .any(|(&o, &xi)| (o - xi).abs() > 1e-6);
    assert!(
        changed,
        "block output identical to input — residual or sublayers inert"
    );

    // Incremental decode of a 4th token at position 3 reads the cached prefix and
    // appends one more row.
    let x2: Vec<f32> = (0..n_embd).map(|_| rng.next_f32(1.0)).collect();
    let mut out2 = vec![f32::NAN; n_embd];
    block
        .forward(&backend, &x2, &[3usize], &mut kv, &cfg, &mut out2)
        .expect("incremental forward");
    assert_eq!(kv.len, seq + 1);
    assert!(
        out2.iter().all(|v| v.is_finite()),
        "non-finite decode output"
    );
}

#[test]
fn transformer_block_shape_errors() {
    let backend = CpuBackend::new();
    let mut rng = Rng::new(0x1111_2222_3333_4444);
    let cfg = tiny_cfg();
    let n_embd = cfg.n_embd as usize;
    let block = build_block(&backend, &mut rng, &cfg);
    let mut kv = tritium_nn::KvCache::new(
        cfg.n_ctx as usize,
        cfg.n_head_kv as usize,
        cfg.head_dim() as usize,
    );

    // positions length disagrees with seq (x is 2 tokens, positions is 1).
    let x = vec![0.1f32; 2 * n_embd];
    let mut out = vec![0.0f32; 2 * n_embd];
    assert!(
        block
            .forward(&backend, &x, &[0usize], &mut kv, &cfg, &mut out)
            .is_err()
    );

    // out length disagrees with x.
    let x = vec![0.1f32; n_embd];
    let mut bad_out = vec![0.0f32; n_embd - 1];
    assert!(
        block
            .forward(&backend, &x, &[0usize], &mut kv, &cfg, &mut bad_out)
            .is_err()
    );
}
