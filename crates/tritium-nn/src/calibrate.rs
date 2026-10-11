//! Activation calibration and the AWQ-style salience fold.
//!
//! **Promoted out of `crates/tritium-nn/tests/common/mod.rs`, where it lived for the whole SALT
//! campaign.** Every published ladder number was measured under this fold at α = 0.75, yet nothing
//! in the shipping tree could call it — `tritium quantize` reads a bare safetensors file and has no
//! way to compute activation statistics. That gap is why the CLI's artifacts are a *different
//! configuration* from the research runs (measured: the fold is worth 2.3× at T=2, 8.5% at T=3, and
//! 1.1% at T=4). This module is the prerequisite for closing it.
//!
//! # Why this is model-aware, not a tensor op
//!
//! [`fold`] takes an [`Arch`] and **returns a modified one**: it scales weight columns by the
//! per-channel salience and divides the *preceding norm* by the same factor, so the product is
//! unchanged while the weights become easier to quantize. That requires knowing which norm feeds
//! which projection — a property of the layer graph, not of any single tensor. Any command offering
//! calibration therefore has to load a model, not a file of tensors.
//!
//! # Scope
//!
//! The Gram/curvature machinery (`Gram`, `GramSet`, `damped_inverse`, `calibrate_gram`) stays in the
//! test tree deliberately: per-group curvature allocation was measured **negative** — 12.4% lower
//! weight SSE and 12% worse perplexity at identical bits — so it is a research artifact, not a
//! shipping path.
//!
//! [`forward_observed`] exposes all actual projection inputs and pre-transform
//! outputs for a caller-owned backward phase. The legacy salience fold still
//! skips `o_proj` and the tied head for its unchanged structural reasons.

use tritium_train::Tape;
use tritium_train::nn::{AttentionQkNorm, attention_heads_traced_with_qk_norm};
use tritium_train::tape::ValueId;

mod staged;
#[cfg(feature = "tokenizer")]
pub use staged::StagedTextCaptureLimits;
pub use staged::{
    RecordedOriginalSalienceCapture, RecordedStagedCapture, SourceCalibrationModel,
    StagedCalibrationLimits, StagedCalibrationModel, StagedCaptureLimits,
    replay_original_salience_capture, replay_staged_capture,
};

use crate::{Mlp, ModelRunner, Projection};
use tritium_quantize::engine::SaliencePair;
pub use tritium_quantize::engine::{SalienceFoldError, divide_rows, scale_cols, smooth_scales};

/// Learned per-head query/key RMSNorm weights, held fp in the calibration graph.
#[derive(Debug, Clone, PartialEq)]
pub struct QkNorm {
    pub query: Vec<f32>,
    pub key: Vec<f32>,
}

/// Dims + fp 1D norms for the tape forward (the parts held fp, not quantized/trained).
#[derive(Debug, Clone)]
pub struct Arch {
    pub attn_norms: Vec<Vec<f32>>,
    pub ffn_norms: Vec<Vec<f32>>,
    /// Exactly one optional pair per layer; weights each have `head_dim` entries.
    pub qk_norms: Vec<Option<QkNorm>>,
    pub out_norm: Vec<f32>,
    pub n_embd: usize,
    pub n_head: usize,
    pub n_head_kv: usize,
    pub head_dim: usize,
    pub ff: usize,
    pub vocab: usize,
    pub eps: f32,
    pub theta: f32,
    pub n_layers: usize,
}

/// Unwrap a `Dense` projection to `(weights, n_out, k_in)`. Panics on compact or backend-owned weights:
/// calibration reads the fp master, so a quantized model is a caller error, not a runtime case.
pub fn dense(p: &Projection) -> (Vec<f32>, usize, usize) {
    match p {
        Projection::Dense(d) => (d.weights.clone(), d.n_out, d.k_in),
        Projection::Salt(_)
        | Projection::Tensor(_)
        | Projection::HostSaltV2(_)
        | Projection::Ternary(_)
        | Projection::Q2(_) => {
            panic!("from_hf builds Dense projections")
        }
        #[cfg(feature = "cuda")]
        Projection::SaltV2(_) => panic!("from_hf builds Dense projections"),
    }
}

/// Extract the tape `Arch`, the flat fp weights (index 0 = tied token_embd, then per layer
/// q,k,v,o,gate,up,down), and their `(n_out, k_in)` shapes from a loaded runner.
pub fn extract(runner: &ModelRunner) -> (Arch, Vec<Vec<f32>>, Vec<(usize, usize)>) {
    let cfg = &runner.config;
    let w = &runner.weights;
    assert!(w.lm_head.is_none(), "assumes tied lm-head");
    let arch = Arch {
        attn_norms: w.layers.iter().map(|b| b.attn_norm.clone()).collect(),
        ffn_norms: w.layers.iter().map(|b| b.ffn_norm.clone()).collect(),
        qk_norms: w
            .layers
            .iter()
            .map(|b| {
                if b.q_norm.is_empty() && b.k_norm.is_empty() {
                    None
                } else {
                    assert_eq!(b.q_norm.len(), cfg.head_dim() as usize);
                    assert_eq!(b.k_norm.len(), cfg.head_dim() as usize);
                    Some(QkNorm {
                        query: b.q_norm.clone(),
                        key: b.k_norm.clone(),
                    })
                }
            })
            .collect(),
        out_norm: w.output_norm.clone(),
        n_embd: cfg.n_embd as usize,
        n_head: cfg.n_head as usize,
        n_head_kv: cfg.n_head_kv as usize,
        head_dim: cfg.head_dim() as usize,
        ff: match &w.layers[0].mlp {
            Mlp::SwiGlu(m) => dense(&m.gate).1,
            Mlp::Relu2(_) => panic!("SwiGLU"),
        },
        vocab: w.vocab,
        eps: cfg.rms_eps,
        theta: cfg.rope_theta,
        n_layers: w.layers.len(),
    };
    let mut fp: Vec<Vec<f32>> = vec![
        w.token_embd
            .as_dense()
            .expect("training reference requires dense token embedding")
            .to_vec(),
    ];
    let mut shapes: Vec<(usize, usize)> = vec![(w.vocab, arch.n_embd)];
    for b in w.layers.iter() {
        let (gate, up, down) = match &b.mlp {
            Mlp::SwiGlu(m) => (&m.gate, &m.up, &m.down),
            Mlp::Relu2(_) => unreachable!(),
        };
        for p in [&b.q_proj, &b.k_proj, &b.v_proj, &b.o_proj, gate, up, down] {
            let (wv, n, k) = dense(p);
            fp.push(wv);
            shapes.push((n, k));
        }
    }
    (arch, fp, shapes)
}

/// HuggingFace tensor names for the weights [`extract`] returns, **in the same order**.
///
/// Anything that writes a quantized artifact has to label each matrix with the name the loader
/// will look it up by, and [`extract`] returns a bare `Vec<Vec<f32>>`. Deriving the names in the
/// consumer would mean a second copy of the naming convention living outside this crate, which is
/// how a rename turns into an artifact that loads but is silently wired wrong. This delegates to
/// the same `NameSchema` the loader itself uses, so there is exactly one spelling of each name.
///
/// Index 0 is the tied token embedding; then per layer `q, k, v, o, gate, up, down`.
/// `extract_names_match_weights` pins the parallel to [`extract`].
pub fn weight_names(a: &Arch) -> Vec<String> {
    let schema = crate::model::NameSchema::Hf;
    let mut names = vec![schema.top("token_embd").to_owned()];
    for li in 0..a.n_layers {
        for slot in ["q", "k", "v", "o", "gate", "up", "down"] {
            names.push(schema.layer(li, slot));
        }
    }
    names
}

/// HuggingFace names of the fp 1-D norms, paired with the [`Arch`] fields holding them.
///
/// These are the tensors [`fold`] modifies, and `ModelWeights::load_salt` reads them from the fp
/// master rather than from the SALT bundle ("only 1D norms come from the fp master"). A converter
/// that writes the bundle but not these norms produces a model that loads and is *wrong* — the
/// weights carry the fold, the norms do not. Returned in `(name, values)` pairs so a writer cannot
/// mis-pair them.
pub fn norm_tensors(a: &Arch) -> Vec<(String, &[f32])> {
    let schema = crate::model::NameSchema::Hf;
    let mut out = Vec::with_capacity(2 * a.n_layers + 1);
    for li in 0..a.n_layers {
        out.push((schema.layer(li, "attn_norm"), a.attn_norms[li].as_slice()));
        out.push((schema.layer(li, "ffn_norm"), a.ffn_norms[li].as_slice()));
        if let Some(norm) = &a.qk_norms[li] {
            out.push((schema.layer(li, "q_norm"), norm.query.as_slice()));
            out.push((schema.layer(li, "k_norm"), norm.key.as_slice()));
        }
    }
    out.push((schema.top("output_norm").to_owned(), a.out_norm.as_slice()));
    out
}

/// The whole-model forward on the tape (validated bit-exact vs `ModelRunner` in
/// `tape_model_conformance`): embed_gather → N GQA blocks → SwiGLU → tied head. `wids[0]` is the
/// tied embed/head; then per layer `q,k,v,o,gate,up,down`. Returns `[tokens.len(), vocab]` logits.
pub fn forward(t: &mut Tape, wids: &[ValueId], a: &Arch, tokens: &[u32]) -> ValueId {
    forward_observed(t, wids, a, tokens, &mut |_, _| {})
}

/// An activation transform applied at a projection input: `(which projection, layer, values, seq, cols)`.
///
/// Values are mutated in place and re-entered on the tape, so the transform can be any precision
/// model at all — int8 per row, int8 per group, ternary planes, or identity.
pub type ActFn<'a> = dyn FnMut(Tap, usize, &mut [f32], usize, usize) + 'a;

/// [`forward`] with activations quantized at every projection input.
///
/// Every perplexity this repo publishes is weight-quantized only: the tape consumes fp32
/// activations. The shipping runtime does not — it int8-quantizes before each projection, which
/// costs +2.76% at T=4 and DOUBLES the excess over fp. Measuring any other activation precision
/// meant a second forward pass nobody had written.
///
/// **Every projection input is covered**, `o_proj`'s included: it used to live inside
/// [`tritium_train::nn::attention`], where nothing could reach it, so results carried an upper-bound caveat with one
/// of seven projections per layer unpenalised. [`tritium_train::nn::attention_heads`] splits the output projection off
/// so this pass can tap and quantize it like any other.
pub fn forward_aq(
    t: &mut Tape,
    wids: &[ValueId],
    a: &Arch,
    tokens: &[u32],
    aq: &mut ActFn<'_>,
) -> ValueId {
    forward_impl(
        t,
        wids,
        a,
        tokens,
        &mut ForwardHooks {
            head: true,
            transform: Some(aq),
            input: &mut |_, _, _, _, _, _| {},
            projection: &mut |_, _| {},
        },
    )
}

/// Per-input-channel second moments for the projections whose scale is foldable.
#[derive(Debug, Clone)]
pub struct Calib {
    pub attn_in: Vec<Vec<f64>>, // [layer][n_embd] — input to q, k, v
    pub ffn_in: Vec<Vec<f64>>,  // [layer][n_embd] — input to gate, up
    pub down_in: Vec<Vec<f64>>, // [layer][ff]     — input to down
    pub rows: usize,
}

impl Calib {
    pub fn new(a: &Arch) -> Self {
        Self {
            attn_in: vec![vec![0.0; a.n_embd]; a.n_layers],
            ffn_in: vec![vec![0.0; a.n_embd]; a.n_layers],
            down_in: vec![vec![0.0; a.ff]; a.n_layers],
            rows: 0,
        }
    }
}

/// Accumulate `Σ x²` per column of a `[seq, cols]` tape value.
pub fn accumulate(t: &Tape, id: ValueId, seq: usize, cols: usize, acc: &mut [f64]) {
    let v = t.value(id);
    for r in 0..seq {
        for (c, a) in acc.iter_mut().enumerate() {
            let x = f64::from(v[r * cols + c]);
            *a += x * x;
        }
    }
}

/// One calibration forward. Mirrors [`forward`] exactly, tapping the three foldable inputs.
///
/// Only the diagonal (`Σ x²` per channel) is accumulated, which is all the salience fold needs. The
/// full Gram `Σ x xᵀ` that sequential error compensation would require stays in the test tree with
/// the rest of the curvature machinery — see the module docs for why.
/// Which projection group a tap is measuring the input to.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Tap {
    /// Input to q, k, v — post-RMSNorm, so already unit-RMS per token.
    AttnIn,
    /// Input to gate, up — post-RMSNorm.
    FfnIn,
    /// Input to down — `silu(gate) * up`, the FFN intermediate. NOT normalised, and the widest
    /// activation in the block, so this is where activation outliers are worst.
    DownIn,
    /// Input to `o_proj` — the concatenated head outputs, `[seq, n_head * head_dim]`.
    ///
    /// The one projection input a block used to hide inside `attention`, which is why every
    /// activation-precision number in this repo carried an "upper bound: 1 of 7 projections is
    /// unpenalised" caveat, and why the salience proxy falls back to a global mean for `o`. Never a
    /// calibration tap for the fold — the fold's other half divides the preceding norm, and `o_proj`
    /// has no norm in front of it, only attention.
    OProjIn,
    /// Input to the tied output head. Never a calibration tap; used by [`forward_aq`] so an
    /// activation-precision sweep does not silently leave the largest projection at fp32.
    Head,
}

/// A calibration tap: `(which projection, layer index, tape, value, seq, cols)`.
///
/// The tape and value are passed rather than a slice so a caller that only wants a reduction never
/// materialises a copy — [`accumulate`] reads straight out of the tape.
pub type TapFn<'a> = dyn FnMut(Tap, usize, &Tape, ValueId, usize, usize) + 'a;

/// Actual consumer of a weight in the supported tied-head GQA/SwiGLU graph.
/// These are runtime observations, not registered calibration evidence.
#[non_exhaustive]
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ProjectionRole {
    /// Token gather; its output has hidden width, not vocabulary width.
    Embedding,
    /// Query projection before RoPE.
    Query,
    /// Key projection before RoPE.
    Key,
    /// Value projection.
    Value,
    /// Concatenated attention heads to the residual stream.
    Output,
    /// SwiGLU gate projection before SiLU.
    Gate,
    /// SwiGLU up projection before multiplication.
    Up,
    /// Gated intermediate to the residual stream.
    Down,
    /// Final normalized hidden to the tied language head.
    Head,
}

/// Actual original projection input/output IDs on the caller-owned tape.
/// IDs are valid only on that tape. Callbacks borrow it only for their invocation;
/// callers may retain this metadata and inspect values/gradients after backward.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct ProjectionTap {
    /// Actual graph consumer, distinguishing tied embedding and head uses.
    pub role: ProjectionRole,
    /// Transformer layer, absent for embedding/head.
    pub layer: Option<usize>,
    /// Index in [`extract`] / [`weight_names`], not a generated display name.
    pub weight_index: usize,
    /// Actual weight value ID; tied consumers share this ID.
    pub weight: ValueId,
    /// Original activation ID. Only token gather has no dense input ID;
    /// its actual token indices are the caller's `tokens` argument.
    pub input: Option<ValueId>,
    /// Actual projection output ID, before RoPE/SiLU or later graph operations.
    pub output: ValueId,
    /// Observed token count.
    pub samples: usize,
    /// Original weight matrix row count (vocabulary for token gather/head).
    pub rows: usize,
    /// Original weight matrix column count.
    pub columns: usize,
}

/// Observation callback at actual projection seams. It produces no gradients,
/// hashes or evidence itself. `Tape::backward` for a declared caller objective
/// supplies gradients of these exact output IDs; activations are not Fisher.
pub type ProjectionTapFn<'a> = dyn FnMut(&Tape, ProjectionTap) + 'a;

/// The real differentiable forward with all original projection observations.
/// Caller ownership permits post-objective backward inspection without retaining
/// a borrowed internal tape or running a second neural forward. Legacy forward,
/// activation transform and fold calibration are adapters over this implementation.
///
/// Scope/shape/ValueId contracts are the same as [`forward`] and [`Arch`]: tied
/// head, full-RoPE GQA, SwiGLU, original dense reference weights. This is not a
/// Qwen hybrid/MTP tap implementation, cache verifier, RSS bound or receipt.
pub fn forward_observed(
    t: &mut Tape,
    wids: &[ValueId],
    a: &Arch,
    tokens: &[u32],
    observe: &mut ProjectionTapFn<'_>,
) -> ValueId {
    forward_impl(
        t,
        wids,
        a,
        tokens,
        &mut ForwardHooks {
            head: true,
            transform: None,
            input: &mut |_, _, _, _, _, _| {},
            projection: observe,
        },
    )
}

struct ForwardHooks<'call, 'callback> {
    // Legacy fold calibration deliberately stops before vocabulary-sized logits.
    // Full/observed/AQ forwards always include the head; no requested tap fallback.
    head: bool,
    transform: Option<&'call mut ActFn<'callback>>,
    input: &'call mut TapFn<'callback>,
    projection: &'call mut ProjectionTapFn<'callback>,
}
impl ForwardHooks<'_, '_> {
    fn input(
        &mut self,
        t: &mut Tape,
        id: ValueId,
        kind: Tap,
        li: usize,
        seq: usize,
        cols: usize,
    ) -> ValueId {
        (self.input)(kind, li, t, id, seq, cols);
        if let Some(transform) = &mut self.transform {
            let mut values = t.value(id).to_vec();
            transform(kind, li, &mut values, seq, cols);
            t.leaf(values) // Preserve forward_aq's existing graph-detaching transform semantics.
        } else {
            id
        }
    }
}

fn forward_impl(
    t: &mut Tape,
    wids: &[ValueId],
    a: &Arch,
    tokens: &[u32],
    hooks: &mut ForwardHooks<'_, '_>,
) -> ValueId {
    let seq = tokens.len();
    assert_eq!(a.qk_norms.len(), a.n_layers);
    for norm in a.qk_norms.iter().flatten() {
        assert_eq!(norm.query.len(), a.head_dim);
        assert_eq!(norm.key.len(), a.head_dim);
        assert!(norm.query.iter().chain(&norm.key).all(|v| v.is_finite()));
        assert!(a.eps.is_finite() && a.eps > 0.0);
    }
    let mut hidden = t.embed_gather(wids[0], tokens, a.vocab, a.n_embd);
    (hooks.projection)(
        t,
        ProjectionTap {
            role: ProjectionRole::Embedding,
            layer: None,
            weight_index: 0,
            weight: wids[0],
            input: None,
            output: hidden,
            samples: seq,
            rows: a.vocab,
            columns: a.n_embd,
        },
    );
    let qd = a.n_head * a.head_dim;
    let kvd = a.n_head_kv * a.head_dim;
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        let an = t.leaf(a.attn_norms[li].clone());
        let xn = t.rmsnorm(hidden, an, seq, a.n_embd, a.eps);
        let xn = hooks.input(t, xn, Tap::AttnIn, li, seq, a.n_embd);
        let qk_norm = a.qk_norms[li].as_ref().map(|norm| AttentionQkNorm {
            query: t.leaf(norm.query.clone()),
            key: t.leaf(norm.key.clone()),
            eps: a.eps,
        });
        let trace = attention_heads_traced_with_qk_norm(
            t,
            xn,
            wids[base],
            wids[base + 1],
            wids[base + 2],
            seq,
            a.n_embd,
            a.n_head,
            a.n_head_kv,
            a.head_dim,
            a.theta,
            qk_norm,
        );
        for (offset, role, output, rows) in [
            (0, ProjectionRole::Query, trace.query, qd),
            (1, ProjectionRole::Key, trace.key, kvd),
            (2, ProjectionRole::Value, trace.value, kvd),
        ] {
            (hooks.projection)(
                t,
                ProjectionTap {
                    role,
                    layer: Some(li),
                    weight_index: base + offset,
                    weight: wids[base + offset],
                    input: Some(xn),
                    output,
                    samples: seq,
                    rows,
                    columns: a.n_embd,
                },
            );
        }
        let heads = hooks.input(t, trace.heads, Tap::OProjIn, li, seq, qd);
        let attn = t.dense_matmul(heads, wids[base + 3], seq, a.n_embd, qd);
        (hooks.projection)(
            t,
            ProjectionTap {
                role: ProjectionRole::Output,
                layer: Some(li),
                weight_index: base + 3,
                weight: wids[base + 3],
                input: Some(heads),
                output: attn,
                samples: seq,
                rows: a.n_embd,
                columns: qd,
            },
        );
        hidden = t.add(hidden, attn);
        let fnw = t.leaf(a.ffn_norms[li].clone());
        let hn = t.rmsnorm(hidden, fnw, seq, a.n_embd, a.eps);
        let hn = hooks.input(t, hn, Tap::FfnIn, li, seq, a.n_embd);
        let g = t.dense_matmul(hn, wids[base + 4], seq, a.ff, a.n_embd);
        let u = t.dense_matmul(hn, wids[base + 5], seq, a.ff, a.n_embd);
        for (offset, role, output) in [(4, ProjectionRole::Gate, g), (5, ProjectionRole::Up, u)] {
            (hooks.projection)(
                t,
                ProjectionTap {
                    role,
                    layer: Some(li),
                    weight_index: base + offset,
                    weight: wids[base + offset],
                    input: Some(hn),
                    output,
                    samples: seq,
                    rows: a.ff,
                    columns: a.n_embd,
                },
            );
        }
        let ga = t.silu(g);
        let gated = t.mul(ga, u);
        let gated = hooks.input(t, gated, Tap::DownIn, li, seq, a.ff);
        let down = t.dense_matmul(gated, wids[base + 6], seq, a.n_embd, a.ff);
        (hooks.projection)(
            t,
            ProjectionTap {
                role: ProjectionRole::Down,
                layer: Some(li),
                weight_index: base + 6,
                weight: wids[base + 6],
                input: Some(gated),
                output: down,
                samples: seq,
                rows: a.n_embd,
                columns: a.ff,
            },
        );
        hidden = t.add(hidden, down);
    }
    if !hooks.head {
        return hidden;
    }
    let onw = t.leaf(a.out_norm.clone());
    let fnorm = t.rmsnorm(hidden, onw, seq, a.n_embd, a.eps);
    let fnorm = hooks.input(t, fnorm, Tap::Head, a.n_layers, seq, a.n_embd);
    let logits = t.dense_matmul(fnorm, wids[0], seq, a.vocab, a.n_embd);
    (hooks.projection)(
        t,
        ProjectionTap {
            role: ProjectionRole::Head,
            layer: None,
            weight_index: 0,
            weight: wids[0],
            input: Some(fnorm),
            output: logits,
            samples: seq,
            rows: a.vocab,
            columns: a.n_embd,
        },
    );
    logits
}

/// Run the calibration forward pass and hand each tap's raw activations to `tap`.
///
/// [`calibrate`] is this with a closure that reduces to second moments. Anything wanting the
/// activations themselves — the A8 headroom measurement, for one — needs them undiscarded, and
/// duplicating this forward pass in a test would be a second implementation to keep in sync.
pub fn calibrate_tapped(weights: &[Vec<f32>], a: &Arch, tokens: &[u32], tap: &mut TapFn<'_>) {
    let mut t = Tape::new();
    let wids: Vec<ValueId> = weights.iter().map(|w| t.leaf(w.clone())).collect();
    forward_impl(
        &mut t,
        &wids,
        a,
        tokens,
        &mut ForwardHooks {
            head: false,
            transform: None,
            input: &mut |kind, li, t, id, seq, cols| {
                if matches!(kind, Tap::AttnIn | Tap::FfnIn | Tap::DownIn) {
                    tap(kind, li, t, id, seq, cols);
                }
            },
            projection: &mut |_, _| {},
        },
    );
}

pub fn calibrate(weights: &[Vec<f32>], a: &Arch, tokens: &[u32], c: &mut Calib) {
    let seq = tokens.len();
    calibrate_tapped(weights, a, tokens, &mut |kind, li, t, id, seq, cols| {
        let acc = match kind {
            Tap::AttnIn => &mut c.attn_in[li],
            Tap::FfnIn => &mut c.ffn_in[li],
            Tap::DownIn => &mut c.down_in[li],
            // Neither the tied head nor `o_proj` has a foldable scale — the fold's other half
            // divides the norm feeding the projection, and these two have no norm in front of
            // them. `calibrate_tapped` never emits either; only `forward_aq` does.
            Tap::Head | Tap::OProjIn => return,
        };
        accumulate(t, id, seq, cols, acc);
    });
    c.rows += seq;
}

/// Apply the salience fold. Returns the rebalanced weights plus the `Arch` whose fp norms absorb
/// the inverse. Cancellation is algebraic, not a bit-identical finite-precision logit claim.
/// Panics on invalid input; use [`try_fold`] for fallible conversion.
///
/// # The two projections this deliberately does not fold
///
/// Five of the seven per-layer projections are folded below. The other two are skipped for
/// **different structural reasons**, and both are worth stating because the sensitivity proxy falls
/// back to a global mean for exactly these — and the embedding is the most sensitive tensor in the
/// model, at roughly 1000× the median.
///
/// **`o_proj` has no norm in front of it.** Its input is the concatenated attention head outputs,
/// and the fold's other half has to land somewhere that cancels it exactly. There is no candidate:
/// scaling `v_proj`'s output rows would work arithmetically for the value path, but the head
/// outputs are `softmax(QKᵀ)·V`, and the softmax sits between `v` and here. Not a missing tap —
/// there is nothing to fold *into*. (`Tap::OProjIn` exists for [`forward_aq`], which quantizes that
/// activation; it is not a calibration tap and `calibrate_tapped` never emits it.)
///
/// **The tied head is blocked by the tie, not by a missing norm.** `out_norm` sits directly in
/// front of it, so the fold would be well-formed in isolation. But the head's weights *are* the
/// embedding table, and scaling its columns rescales every gathered token vector by the same
/// per-channel `s`. That vector enters the residual stream unnormalised, so the scale propagates
/// through the whole network with nothing to cancel it — RMSNorm renormalises globally and cannot
/// undo a per-channel scale. An **untied** head would be foldable; this one is not.
pub fn fold(
    fp: &[Vec<f32>],
    shapes: &[(usize, usize)],
    a: &Arch,
    c: &Calib,
    alpha: f64,
) -> (Vec<Vec<f32>>, Arch) {
    try_fold(fp, shapes, a, c, alpha).expect("invalid model-aware paired salience fold")
}

/// Checked graph adapter to the shared paired salience numerical module.
///
/// Valid legacy results keep their exact ordered f32 operations. The caller's
/// weights/architecture remain unchanged on any error. This currently clones
/// the dense model, as the legacy path did; it is not out-of-core/RSS admission.
/// # Errors
/// Malformed tied-head SwiGLU inventory, statistics, alpha or nonfinite values.
pub fn try_fold(
    fp: &[Vec<f32>],
    shapes: &[(usize, usize)],
    a: &Arch,
    c: &Calib,
    alpha: f64,
) -> Result<(Vec<Vec<f32>>, Arch), SalienceFoldError> {
    if c.rows == 0 {
        return Err(SalienceFoldError::EmptyStatistics);
    }
    if !alpha.is_finite() || !(0.0..=1.0).contains(&alpha) {
        return Err(SalienceFoldError::Alpha);
    }
    let count = a.n_layers.checked_mul(7).and_then(|n| n.checked_add(1));
    let q_rows = a.n_head.checked_mul(a.head_dim);
    let kv_rows = a.n_head_kv.checked_mul(a.head_dim);
    if count != Some(fp.len())
        || shapes.len() != fp.len()
        || a.attn_norms.len() != a.n_layers
        || a.ffn_norms.len() != a.n_layers
        || c.attn_in.len() != a.n_layers
        || c.ffn_in.len() != a.n_layers
        || c.down_in.len() != a.n_layers
        || a.n_embd == 0
        || a.ff == 0
        || a.n_layers == 0
        || a.vocab == 0
        || q_rows.is_none_or(|n| n == 0)
        || kv_rows.is_none_or(|n| n == 0)
        || shapes.first() != Some(&(a.vocab, a.n_embd))
        || a.out_norm.len() != a.n_embd
        || shapes.iter().zip(fp).any(|(&(rows, cols), values)| {
            rows == 0 || cols == 0 || rows.checked_mul(cols) != Some(values.len())
        })
    {
        return Err(SalienceFoldError::Geometry);
    }
    let mut w = fp.to_vec();
    let mut arch = a.clone();
    for li in 0..a.n_layers {
        let base = 1 + 7 * li;
        if c.attn_in[li].len() != a.n_embd
            || c.ffn_in[li].len() != a.n_embd
            || c.down_in[li].len() != a.ff
            || Some(shapes[base].0) != q_rows
            || Some(shapes[base + 1].0) != kv_rows
            || Some(shapes[base + 2].0) != kv_rows
            || (0..3).any(|k| shapes[base + k].1 != a.n_embd)
            || shapes[base + 3].0 != a.n_embd
            || Some(shapes[base + 3].1) != q_rows
            || shapes[base + 4] != (a.ff, a.n_embd)
            || shapes[base + 5] != (a.ff, a.n_embd)
            || shapes[base + 6] != (a.n_embd, a.ff)
        {
            return Err(SalienceFoldError::Geometry);
        }
        let [q, k, v, _o, gate, up, down] = &mut w[base..base + 7] else {
            unreachable!("validated layer inventory")
        };
        {
            let mut projections = [q.as_mut_slice(), k.as_mut_slice(), v.as_mut_slice()];
            SaliencePair {
                projections: &mut projections,
                inverse: &mut arch.attn_norms[li],
                inverse_columns: 1,
            }
            .apply(&c.attn_in[li], c.rows, alpha)?;
        }
        {
            let mut projections = [gate.as_mut_slice(), up.as_mut_slice()];
            SaliencePair {
                projections: &mut projections,
                inverse: &mut arch.ffn_norms[li],
                inverse_columns: 1,
            }
            .apply(&c.ffn_in[li], c.rows, alpha)?;
        }
        let mut projections = [down.as_mut_slice()];
        SaliencePair {
            projections: &mut projections,
            inverse: up,
            inverse_columns: a.n_embd,
        }
        .apply(&c.down_in[li], c.rows, alpha)?;
    }
    Ok((w, arch))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn arch(n_layers: usize) -> Arch {
        Arch {
            attn_norms: vec![vec![1.0; 4]; n_layers],
            ffn_norms: vec![vec![1.0; 4]; n_layers],
            qk_norms: vec![None; n_layers],
            out_norm: vec![1.0; 4],
            n_embd: 4,
            n_head: 2,
            n_head_kv: 1,
            head_dim: 2,
            ff: 8,
            vocab: 16,
            eps: 1e-5,
            theta: 10_000.0,
            n_layers,
        }
    }

    /// `weight_names` has to stay parallel to `extract`, which pushes the tied embedding and then
    /// seven projections per layer. Nothing in the type system enforces that; this does.
    #[test]
    fn weight_names_match_extract_layout() {
        for n_layers in [1usize, 2, 5] {
            assert_eq!(weight_names(&arch(n_layers)).len(), 1 + 7 * n_layers);
        }
    }

    /// A rename in the loader's schema would otherwise show up as an artifact that writes fine and
    /// resolves nothing — pin the exact strings the loader looks up.
    #[test]
    fn weight_names_are_the_hf_spellings() {
        let names = weight_names(&arch(2));
        assert_eq!(names[0], "model.embed_tokens.weight");
        assert_eq!(names[1], "model.layers.0.self_attn.q_proj.weight");
        assert_eq!(names[7], "model.layers.0.mlp.down_proj.weight");
        assert_eq!(names[8], "model.layers.1.self_attn.q_proj.weight");
    }

    /// The norms are the half of the fold that lives outside the SALT bundle. Mis-pairing a name
    /// with the wrong vector produces a model that loads and is wrong, so the pairing is returned
    /// as tuples and asserted here.
    #[test]
    fn norm_tensors_pair_every_norm_with_its_name() {
        let mut a = arch(2);
        a.attn_norms[1] = vec![7.0; 4];
        a.ffn_norms[0] = vec![9.0; 4];
        let norms = norm_tensors(&a);
        assert_eq!(norms.len(), 2 * 2 + 1);
        let lookup = |name: &str| {
            norms
                .iter()
                .find(|(n, _)| n == name)
                .unwrap_or_else(|| panic!("missing {name}"))
                .1
        };
        assert_eq!(lookup("model.layers.1.input_layernorm.weight")[0], 7.0);
        assert_eq!(
            lookup("model.layers.0.post_attention_layernorm.weight")[0],
            9.0
        );
        assert_eq!(lookup("model.norm.weight").len(), 4);
    }
}
