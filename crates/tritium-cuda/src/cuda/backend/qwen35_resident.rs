//! Device-resident Qwen3.5/Qwen3.6 decode executor (SALT decode campaign, Phase 2).
//!
//! The host-orchestrated path runs every projection as its own round trip: scan the
//! activation, upload it, launch, block on the download, scan the result. For the
//! Qwen3.6-27B bundle that is ~500 round trips per token, and after the row-stream
//! GEMV cut the projections to 13 ms, 23 of the 37.6 ms per token were spent outside
//! any kernel. This executor keeps the whole step on the device: activations,
//! DeltaNet state, convolution state and the KV cache never leave it, and a step
//! returns only the four bytes of the next greedy token.
//!
//! **Numerics.** Fast tier throughout: the row-stream GEMV reassociates the K-sum and
//! the decode kernels use CUDA transcendentals. It is gated on relative error and
//! greedy-token identity against the host forward, not on equality, and it is only
//! offered where the caller has opted into the fast tier.
//!
//! **Shape.** Deliberately ADR 0044 D8's `Executor` surface -- `prefill`, `step`,
//! `reset` over backend-owned state -- so the runner consolidation can move it behind
//! `TernaryBackend::resident` once tensors have a backend-neutral handle, instead of
//! rewriting it.

use std::sync::Arc;

use super::salt_v2_runtime::{
    SALT_STREAM_DESCRIPTOR_WORDS, launch_salt_v2_stream_multi_on, launch_salt_v2_stream_on,
    salt_stream_descriptor, salt_v2_stream_dispatch,
};
use super::*;

/// Largest context the executor's attention kernel keeps scores for in shared memory.
///
/// 8192 scores are 32 KiB, inside the default per-block shared budget. Longer contexts
/// need a split-context attention kernel, which is later work.
pub const QWEN35_RESIDENT_MAX_CONTEXT: usize = 8192;

/// One token mixer's weights, borrowed from the host model for the build.
#[derive(Debug)]
#[non_exhaustive]
pub enum Qwen35ResidentMixerSpec<'a> {
    /// A Gated DeltaNet (`linear_attention`) layer.
    DeltaNet {
        /// `[conv_width, hidden]`: query, key and value, concatenated.
        qkv: Arc<SaltV2ResidentTensor>,
        /// `[value_width, hidden]`: the output gate `z`.
        z: Arc<SaltV2ResidentTensor>,
        /// `[value_heads, hidden]`: beta logits.
        b: Arc<SaltV2ResidentTensor>,
        /// `[value_heads, hidden]`: decay logits.
        a: Arc<SaltV2ResidentTensor>,
        /// `[hidden, value_width]`: output projection.
        out: Arc<SaltV2ResidentTensor>,
        /// `[conv_width, conv_kernel]` depthwise taps.
        conv_weight: &'a [f32],
        /// `[value_head_dim]` gated-norm weight (applied directly, not `1 + w`).
        norm_weight: &'a [f32],
        /// `[value_heads]`.
        dt_bias: &'a [f32],
        /// `[value_heads]`.
        a_log: &'a [f32],
    },
    /// A gated causal GQA (`full_attention`) layer.
    Attention {
        /// `[2 * heads * head_dim, hidden]`: query and gate interleaved per head.
        q: Arc<SaltV2ResidentTensor>,
        /// `[kv_heads * head_dim, hidden]`.
        k: Arc<SaltV2ResidentTensor>,
        /// `[kv_heads * head_dim, hidden]`.
        v: Arc<SaltV2ResidentTensor>,
        /// `[hidden, heads * head_dim]`.
        o: Arc<SaltV2ResidentTensor>,
        /// `[head_dim]` zero-centered query head norm.
        q_norm: &'a [f32],
        /// `[head_dim]` zero-centered key head norm.
        k_norm: &'a [f32],
    },
}

/// One decoder layer.
#[derive(Debug)]
pub struct Qwen35ResidentLayerSpec<'a> {
    /// `[hidden]` zero-centered input norm.
    pub input_norm: &'a [f32],
    /// The token mixer.
    pub mixer: Qwen35ResidentMixerSpec<'a>,
    /// `[hidden]` zero-centered post-attention norm.
    pub post_attention_norm: &'a [f32],
    /// `[intermediate, hidden]`.
    pub gate: Arc<SaltV2ResidentTensor>,
    /// `[intermediate, hidden]`.
    pub up: Arc<SaltV2ResidentTensor>,
    /// `[hidden, intermediate]`.
    pub down: Arc<SaltV2ResidentTensor>,
}

/// Everything the executor needs, borrowed from a loaded host model.
#[derive(Debug)]
pub struct Qwen35ResidentSpec<'a> {
    /// `[vocab, hidden]` token table (must carry no rotation).
    pub embedding: Arc<SaltV2ResidentTensor>,
    /// Decoder layers in order.
    pub layers: Vec<Qwen35ResidentLayerSpec<'a>>,
    /// `[hidden]` zero-centered final norm.
    pub final_norm: &'a [f32],
    /// `[vocab, hidden]` untied language head.
    pub lm_head: Arc<SaltV2ResidentTensor>,
    /// RMSNorm epsilon shared by every norm in the model.
    pub rms_norm_eps: f32,
    /// DeltaNet query/key heads.
    pub deltanet_key_heads: usize,
    /// DeltaNet value (recurrent-state) heads.
    pub deltanet_value_heads: usize,
    /// DeltaNet per-head query/key width.
    pub deltanet_key_head_dim: usize,
    /// DeltaNet per-head value width.
    pub deltanet_value_head_dim: usize,
    /// DeltaNet depthwise conv taps.
    pub deltanet_conv_kernel: usize,
    /// Full-attention query heads.
    pub attention_heads: usize,
    /// Full-attention key/value heads.
    pub attention_kv_heads: usize,
    /// Full-attention head width.
    pub attention_head_dim: usize,
    /// Rotated prefix of each head (NeoX half-rotation).
    pub attention_rotary_dim: usize,
    /// RoPE base.
    pub rope_theta: f32,
    /// Context the executor allocates KV for; at most [`QWEN35_RESIDENT_MAX_CONTEXT`].
    pub max_context: usize,
}

struct DeltaNetLayer {
    qkv: Arc<SaltV2ResidentTensor>,
    z: Arc<SaltV2ResidentTensor>,
    b: Arc<SaltV2ResidentTensor>,
    a: Arc<SaltV2ResidentTensor>,
    out: Arc<SaltV2ResidentTensor>,
    conv_weight: CudaSlice<f32>,
    norm_weight: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    a_log: CudaSlice<f32>,
    conv_state: CudaSlice<f32>,
    recurrent_state: CudaSlice<f32>,
}

struct AttentionLayer {
    q: Arc<SaltV2ResidentTensor>,
    k: Arc<SaltV2ResidentTensor>,
    v: Arc<SaltV2ResidentTensor>,
    o: Arc<SaltV2ResidentTensor>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    key_cache: CudaSlice<f32>,
    value_cache: CudaSlice<f32>,
}

enum Mixer {
    DeltaNet(Box<DeltaNetLayer>),
    Attention(Box<AttentionLayer>),
}

/// Several projections of one input, launched as one fused row-stream GEMV.
struct FusedGroup {
    descriptors: CudaSlice<u64>,
    tensor_count: u32,
    total_rows: u32,
}

struct Layer {
    /// The mixer's input projections fused (DeltaNet qkv|z|b|a, attention q|k|v).
    fused_in: Option<FusedGroup>,
    /// The MLP's gate|up fused.
    fused_mlp: Option<FusedGroup>,
    input_norm: CudaSlice<f32>,
    post_attention_norm: CudaSlice<f32>,
    mixer: Mixer,
    gate: Arc<SaltV2ResidentTensor>,
    up: Arc<SaltV2ResidentTensor>,
    down: Arc<SaltV2ResidentTensor>,
}

struct Kernels {
    stream_gemv: CudaFunction,
    stream_gemv_multi: CudaFunction,
    gather: CudaFunction,
    recurrent: CudaFunction,
    add_rmsnorm: CudaFunction,
    attn_prep: CudaFunction,
    attention: CudaFunction,
    sigmoid_mul: CudaFunction,
    swiglu: CudaFunction,
    conv: CudaFunction,
    prep: CudaFunction,
    gated_rmsnorm: CudaFunction,
    argmax_partial: CudaFunction,
    argmax_final: CudaFunction,
}

/// Scratch activations, sized once at build.
struct Scratch {
    ctrl: CudaSlice<i32>,
    token: CudaSlice<u32>,
    residual: CudaSlice<f32>,
    normalized: CudaSlice<f32>,
    branch: CudaSlice<f32>,
    qkv: CudaSlice<f32>,
    z: CudaSlice<f32>,
    b: CudaSlice<f32>,
    a: CudaSlice<f32>,
    convolved: CudaSlice<f32>,
    kk: CudaSlice<f32>,
    qq: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    decay: CudaSlice<f32>,
    core: CudaSlice<f32>,
    gated: CudaSlice<f32>,
    fused_query: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    query: CudaSlice<f32>,
    attention_gate: CudaSlice<f32>,
    attended: CudaSlice<f32>,
    mlp_gate: CudaSlice<f32>,
    mlp_up: CudaSlice<f32>,
    mlp_act: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    argmax_value: CudaSlice<f32>,
    argmax_index: CudaSlice<i32>,
}

/// Argmax stage-one blocks and threads: enough blocks to spread a 248K vocabulary.
const ARGMAX_BLOCKS: u32 = 128;
const ARGMAX_THREADS: u32 = 1024;

/// A Qwen3.5/Qwen3.6 decoder resident on one CUDA device.
///
/// Built by [`CudaBackend::build_qwen35_resident`]. Holds its own KV cache and
/// recurrent state; [`Self::reset`] starts a new sequence.
pub struct Qwen35Resident {
    /// Per-warp shared table bytes for fused launches over `hidden` columns.
    fused_table_bytes: u32,
    stream: Arc<CudaStream>,
    kernels: Kernels,
    embedding: Arc<SaltV2ResidentTensor>,
    layers: Vec<Layer>,
    final_norm: CudaSlice<f32>,
    lm_head: Arc<SaltV2ResidentTensor>,
    inv_freq: CudaSlice<f32>,
    scratch: Scratch,
    hidden: usize,
    vocab: usize,
    eps: f32,
    key_heads: usize,
    value_heads: usize,
    key_head_dim: usize,
    value_head_dim: usize,
    conv_kernel: usize,
    heads: usize,
    kv_heads: usize,
    head_dim: usize,
    rotary_dim: usize,
    max_context: usize,
    position: usize,
}

impl core::fmt::Debug for Qwen35Resident {
    fn fmt(&self, formatter: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        formatter
            .debug_struct("Qwen35Resident")
            .field("layers", &self.layers.len())
            .field("hidden", &self.hidden)
            .field("vocab", &self.vocab)
            .field("position", &self.position)
            .field("max_context", &self.max_context)
            .finish_non_exhaustive()
    }
}

fn invalid(message: impl Into<String>) -> BackendError {
    BackendError::InvalidInput(message.into())
}

fn to_i32(value: usize, name: &str) -> Result<i32, BackendError> {
    i32::try_from(value).map_err(|_| invalid(format!("{name} exceeds the i32 kernel ABI")))
}

fn to_u32(value: usize, name: &str) -> Result<u32, BackendError> {
    u32::try_from(value).map_err(|_| invalid(format!("{name} exceeds the u32 kernel ABI")))
}

impl CudaBackend {
    /// Build a device-resident Qwen3.5/Qwen3.6 decoder from a loaded host model's
    /// resident SALT V2 tensors and fp32 parameters.
    ///
    /// Every projection must be one the row-stream GEMV serves (B3, scale group 128,
    /// 256-aligned width) and belong to this backend's context.
    ///
    /// # Errors
    /// Rejects a tensor the executor cannot run, inconsistent geometry, a context
    /// above [`QWEN35_RESIDENT_MAX_CONTEXT`], or a device allocation failure.
    pub fn build_qwen35_resident(
        &self,
        spec: &Qwen35ResidentSpec<'_>,
    ) -> Result<Qwen35Resident, BackendError> {
        let hidden = spec.embedding.columns;
        let vocab = spec.embedding.rows;
        if spec.max_context == 0 || spec.max_context > QWEN35_RESIDENT_MAX_CONTEXT {
            return Err(invalid(format!(
                "resident Qwen context {} must be in 1..={QWEN35_RESIDENT_MAX_CONTEXT}",
                spec.max_context
            )));
        }
        let key_width = spec.deltanet_key_heads * spec.deltanet_key_head_dim;
        let value_width = spec.deltanet_value_heads * spec.deltanet_value_head_dim;
        let conv_width = 2 * key_width + value_width;
        let query_width = spec.attention_heads * spec.attention_head_dim;
        let kv_width = spec.attention_kv_heads * spec.attention_head_dim;
        if spec.deltanet_key_heads == 0
            || !spec
                .deltanet_value_heads
                .is_multiple_of(spec.deltanet_key_heads)
            || spec.attention_kv_heads == 0
            || !spec.attention_heads.is_multiple_of(spec.attention_kv_heads)
            || !spec.attention_rotary_dim.is_multiple_of(2)
            || spec.attention_rotary_dim > spec.attention_head_dim
            || spec.attention_head_dim > 1024
            || spec.deltanet_value_head_dim > 1024
            || spec.deltanet_key_head_dim > 1024
        {
            return Err(invalid("resident Qwen head geometry is inconsistent"));
        }

        let check = |tensor: &SaltV2ResidentTensor,
                     rows: usize,
                     columns: usize,
                     name: &str|
         -> Result<(), BackendError> {
            self.validate_salt_v2_resident_context(tensor)?;
            if tensor.rows != rows || tensor.columns != columns {
                return Err(invalid(format!(
                    "resident Qwen {name} is {}x{}, expected {rows}x{columns}",
                    tensor.rows, tensor.columns
                )));
            }
            if salt_v2_stream_dispatch(tensor.columns, tensor.scale_group_size, tensor.codec_tag)
                .is_none()
            {
                return Err(invalid(format!(
                    "resident Qwen {name} is not a shape the row-stream GEMV serves"
                )));
            }
            Ok(())
        };
        self.validate_salt_v2_resident_context(&spec.embedding)?;
        check(&spec.lm_head, vocab, hidden, "lm_head")?;
        let intermediate = spec
            .layers
            .first()
            .map(|layer| layer.gate.rows)
            .ok_or_else(|| invalid("resident Qwen needs at least one layer"))?;

        let upload = |values: &[f32], expected: usize, name: &str| {
            if values.len() != expected {
                return Err(invalid(format!(
                    "resident Qwen {name} has {} values, expected {expected}",
                    values.len()
                )));
            }
            self.stream
                .clone_htod(values)
                .map_err(|error| driver_err("upload resident Qwen parameter", &error))
        };
        let zeros = |len: usize, name: &str| {
            self.stream
                .alloc_zeros::<f32>(len)
                .map_err(|error| alloc_or_backend(name, &error, len * core::mem::size_of::<f32>()))
        };

        let mut layers = Vec::with_capacity(spec.layers.len());
        for (index, layer) in spec.layers.iter().enumerate() {
            check(&layer.gate, intermediate, hidden, "gate")?;
            check(&layer.up, intermediate, hidden, "up")?;
            check(&layer.down, hidden, intermediate, "down")?;
            let mixer = match &layer.mixer {
                Qwen35ResidentMixerSpec::DeltaNet {
                    qkv,
                    z,
                    b,
                    a,
                    out,
                    conv_weight,
                    norm_weight,
                    dt_bias,
                    a_log,
                } => {
                    check(qkv, conv_width, hidden, "DeltaNet qkv")?;
                    check(z, value_width, hidden, "DeltaNet z")?;
                    check(b, spec.deltanet_value_heads, hidden, "DeltaNet b")?;
                    check(a, spec.deltanet_value_heads, hidden, "DeltaNet a")?;
                    check(out, hidden, value_width, "DeltaNet out")?;
                    Mixer::DeltaNet(Box::new(DeltaNetLayer {
                        qkv: Arc::clone(qkv),
                        z: Arc::clone(z),
                        b: Arc::clone(b),
                        a: Arc::clone(a),
                        out: Arc::clone(out),
                        conv_weight: upload(
                            conv_weight,
                            conv_width * spec.deltanet_conv_kernel,
                            "conv weight",
                        )?,
                        norm_weight: upload(
                            norm_weight,
                            spec.deltanet_value_head_dim,
                            "gated norm weight",
                        )?,
                        dt_bias: upload(dt_bias, spec.deltanet_value_heads, "dt bias")?,
                        a_log: upload(a_log, spec.deltanet_value_heads, "a log")?,
                        conv_state: zeros(
                            conv_width * spec.deltanet_conv_kernel,
                            "allocate DeltaNet conv state",
                        )?,
                        recurrent_state: zeros(
                            spec.deltanet_value_heads
                                * spec.deltanet_key_head_dim
                                * spec.deltanet_value_head_dim,
                            "allocate DeltaNet recurrent state",
                        )?,
                    }))
                }
                Qwen35ResidentMixerSpec::Attention {
                    q,
                    k,
                    v,
                    o,
                    q_norm,
                    k_norm,
                } => {
                    check(q, 2 * query_width, hidden, "attention q")?;
                    check(k, kv_width, hidden, "attention k")?;
                    check(v, kv_width, hidden, "attention v")?;
                    check(o, hidden, query_width, "attention o")?;
                    Mixer::Attention(Box::new(AttentionLayer {
                        q: Arc::clone(q),
                        k: Arc::clone(k),
                        v: Arc::clone(v),
                        o: Arc::clone(o),
                        q_norm: upload(q_norm, spec.attention_head_dim, "q norm")?,
                        k_norm: upload(k_norm, spec.attention_head_dim, "k norm")?,
                        key_cache: zeros(spec.max_context * kv_width, "allocate key cache")?,
                        value_cache: zeros(spec.max_context * kv_width, "allocate value cache")?,
                    }))
                }
            };
            let _ = index;
            layers.push(Layer {
                fused_in: None,
                fused_mlp: None,
                input_norm: upload(layer.input_norm, hidden, "input norm")?,
                post_attention_norm: upload(layer.post_attention_norm, hidden, "post norm")?,
                mixer,
                gate: Arc::clone(&layer.gate),
                up: Arc::clone(&layer.up),
                down: Arc::clone(&layer.down),
            });
        }

        // RoPE inverse frequencies, built exactly as the host builds them so the only
        // difference left in the rotation is the device `sincosf`.
        let half = spec.attention_rotary_dim / 2;
        let inv_freq: Vec<f32> = (0..half)
            .map(|lane| {
                1.0 / spec
                    .rope_theta
                    .powf((2 * lane) as f32 / spec.attention_rotary_dim as f32)
            })
            .collect();

        let module = &self.qwen35_decode_module;
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|error| driver_err("resolve resident Qwen kernel", &error))
        };
        let kernels = Kernels {
            stream_gemv: self.func_salt_v2_stream.clone(),
            stream_gemv_multi: self.func_salt_v2_stream_multi.clone(),
            gather: self.func_salt_v2_gather.clone(),
            recurrent: self.func_deltanet_step.clone(),
            add_rmsnorm: function("q35_add_rmsnorm")?,
            attn_prep: function("q35_attn_prep")?,
            attention: function("q35_attention")?,
            sigmoid_mul: function("q35_sigmoid_mul")?,
            swiglu: function("q35_swiglu")?,
            conv: function("q35_deltanet_conv")?,
            prep: function("q35_deltanet_prep")?,
            gated_rmsnorm: function("q35_gated_rmsnorm")?,
            argmax_partial: function("q35_argmax_partial")?,
            argmax_final: function("q35_argmax_final")?,
        };

        let scratch = Scratch {
            ctrl: self
                .stream
                .alloc_zeros::<i32>(4)
                .map_err(|error| alloc_or_backend("allocate resident ctrl", &error, 16))?,
            token: self
                .stream
                .alloc_zeros::<u32>(1)
                .map_err(|error| alloc_or_backend("allocate resident token", &error, 4))?,
            residual: zeros(hidden, "allocate residual")?,
            normalized: zeros(hidden, "allocate normalized")?,
            branch: zeros(hidden, "allocate branch")?,
            qkv: zeros(conv_width, "allocate qkv")?,
            z: zeros(value_width, "allocate z")?,
            b: zeros(spec.deltanet_value_heads, "allocate b")?,
            a: zeros(spec.deltanet_value_heads, "allocate a")?,
            convolved: zeros(conv_width, "allocate convolved")?,
            kk: zeros(key_width, "allocate kk")?,
            qq: zeros(key_width, "allocate qq")?,
            beta: zeros(spec.deltanet_value_heads, "allocate beta")?,
            decay: zeros(spec.deltanet_value_heads, "allocate decay")?,
            core: zeros(value_width, "allocate core")?,
            gated: zeros(value_width, "allocate gated")?,
            fused_query: zeros(2 * query_width, "allocate fused query")?,
            key: zeros(kv_width, "allocate key")?,
            value: zeros(kv_width, "allocate value")?,
            query: zeros(query_width, "allocate query")?,
            attention_gate: zeros(query_width, "allocate attention gate")?,
            attended: zeros(query_width, "allocate attended")?,
            mlp_gate: zeros(intermediate, "allocate mlp gate")?,
            mlp_up: zeros(intermediate, "allocate mlp up")?,
            mlp_act: zeros(intermediate, "allocate mlp act")?,
            logits: zeros(vocab, "allocate logits")?,
            argmax_value: zeros(ARGMAX_BLOCKS as usize, "allocate argmax values")?,
            argmax_index: self
                .stream
                .alloc_zeros::<i32>(ARGMAX_BLOCKS as usize)
                .map_err(|error| alloc_or_backend("allocate argmax indices", &error, 512))?,
        };

        // Fused input groups. Each descriptor bakes in a scratch buffer's device
        // address, which is stable for the executor's life because scratch is
        // allocated once and never reallocated.
        let fuse = |members: &[(&SaltV2ResidentTensor, &CudaSlice<f32>)]| {
            let mut words = Vec::with_capacity(members.len() * SALT_STREAM_DESCRIPTOR_WORDS);
            let mut first_row = 0u32;
            for (tensor, output) in members {
                let address = crate::cuda::graph_raw::dptr(*output, &self.stream);
                words.extend(salt_stream_descriptor(
                    tensor,
                    &self.stream,
                    address,
                    first_row,
                )?);
                first_row = first_row
                    .checked_add(to_u32(tensor.rows, "fused rows")?)
                    .ok_or_else(|| invalid("fused row count overflows u32"))?;
            }
            let descriptors = self
                .stream
                .clone_htod(&words)
                .map_err(|error| driver_err("upload fused GEMV descriptors", &error))?;
            Ok::<_, BackendError>(FusedGroup {
                descriptors,
                tensor_count: to_u32(members.len(), "fused tensor count")?,
                total_rows: first_row,
            })
        };
        for layer in &mut layers {
            layer.fused_mlp = Some(fuse(&[
                (&layer.gate, &scratch.mlp_gate),
                (&layer.up, &scratch.mlp_up),
            ])?);
            layer.fused_in = Some(match &layer.mixer {
                Mixer::DeltaNet(mixer) => fuse(&[
                    (&mixer.qkv, &scratch.qkv),
                    (&mixer.z, &scratch.z),
                    (&mixer.b, &scratch.b),
                    (&mixer.a, &scratch.a),
                ])?,
                Mixer::Attention(mixer) => fuse(&[
                    (&mixer.q, &scratch.fused_query),
                    (&mixer.k, &scratch.key),
                    (&mixer.v, &scratch.value),
                ])?,
            });
        }
        let fused_table_bytes = salt_v2_stream_dispatch(hidden, 128, 1)
            .ok_or_else(|| invalid("hidden width is not one the row-stream GEMV serves"))?;

        Ok(Qwen35Resident {
            fused_table_bytes,
            stream: Arc::clone(&self.stream),
            kernels,
            embedding: Arc::clone(&spec.embedding),
            layers,
            final_norm: upload(spec.final_norm, hidden, "final norm")?,
            lm_head: Arc::clone(&spec.lm_head),
            inv_freq: self
                .stream
                .clone_htod(&inv_freq)
                .map_err(|error| driver_err("upload RoPE frequencies", &error))?,
            scratch,
            hidden,
            vocab,
            eps: spec.rms_norm_eps,
            key_heads: spec.deltanet_key_heads,
            value_heads: spec.deltanet_value_heads,
            key_head_dim: spec.deltanet_key_head_dim,
            value_head_dim: spec.deltanet_value_head_dim,
            conv_kernel: spec.deltanet_conv_kernel,
            heads: spec.attention_heads,
            kv_heads: spec.attention_kv_heads,
            head_dim: spec.attention_head_dim,
            rotary_dim: spec.attention_rotary_dim,
            max_context: spec.max_context,
            position: 0,
        })
    }
}

/// Launch a prepared kernel. Free-standing so a caller can hold disjoint mutable
/// borrows of the executor's fields in the builder while calling it.
fn run(
    builder: &mut cudarc::driver::LaunchArgs<'_>,
    cfg: LaunchConfig,
    name: &str,
) -> Result<(), BackendError> {
    // SAFETY: every buffer handed to these kernels was sized from the validated
    // geometry at build, and each kernel bounds-checks its lanes against the sizes
    // it is given.
    #[allow(unsafe_code)]
    unsafe {
        builder
            .launch(cfg)
            .map(|_| ())
            .map_err(|error| driver_err(name, &error))
    }
}

/// Whether fused input projections are enabled (default on; `TRITIUM_QWEN35_FUSED=0`
/// launches each projection separately, for A/B).
fn fused_enabled() -> bool {
    std::env::var("TRITIUM_QWEN35_FUSED").as_deref() != Ok("0")
}

fn elementwise(n: usize) -> LaunchConfig {
    let threads = 256u32;
    LaunchConfig {
        grid_dim: ((n as u32).div_ceil(threads), 1, 1),
        block_dim: (threads, 1, 1),
        shared_mem_bytes: 0,
    }
}

#[allow(clippy::too_many_arguments)]
fn add_rmsnorm(
    stream: &CudaStream,
    kernel: &CudaFunction,
    residual: &mut CudaSlice<f32>,
    branch: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    normalized: &mut CudaSlice<f32>,
    n: i32,
    eps: f32,
    add: bool,
) -> Result<(), BackendError> {
    let add = i32::from(add);
    // Every Qwen3.5 decoder norm is zero-centered, `x * (1 + w)`.
    let one_plus = 1i32;
    let mut builder = stream.launch_builder(kernel);
    builder
        .arg(residual)
        .arg(branch)
        .arg(weight)
        .arg(normalized)
        .arg(&n)
        .arg(&eps)
        .arg(&add)
        .arg(&one_plus);
    run(
        &mut builder,
        LaunchConfig {
            grid_dim: (1, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        },
        "launch q35_add_rmsnorm",
    )
}

impl Qwen35Resident {
    /// Tokens consumed so far in the current sequence.
    #[must_use]
    pub const fn position(&self) -> usize {
        self.position
    }

    /// Start a new sequence: zero every recurrent, convolution and KV buffer.
    ///
    /// # Errors
    /// Returns a driver failure.
    pub fn reset(&mut self) -> Result<(), BackendError> {
        for layer in &mut self.layers {
            match &mut layer.mixer {
                Mixer::DeltaNet(state) => {
                    self.stream
                        .memset_zeros(&mut state.conv_state)
                        .map_err(|error| driver_err("reset conv state", &error))?;
                    self.stream
                        .memset_zeros(&mut state.recurrent_state)
                        .map_err(|error| driver_err("reset recurrent state", &error))?;
                }
                Mixer::Attention(state) => {
                    self.stream
                        .memset_zeros(&mut state.key_cache)
                        .map_err(|error| driver_err("reset key cache", &error))?;
                    self.stream
                        .memset_zeros(&mut state.value_cache)
                        .map_err(|error| driver_err("reset value cache", &error))?;
                }
            }
        }
        self.position = 0;
        Ok(())
    }

    /// Feed `tokens` in order, returning the greedy token after the last one.
    ///
    /// Each token runs as one decode step; batched prefill is later work.
    ///
    /// # Errors
    /// As [`Self::step`]; rejects an empty prompt.
    pub fn prefill(&mut self, tokens: &[u32]) -> Result<u32, BackendError> {
        let (&last, rest) = tokens
            .split_last()
            .ok_or_else(|| invalid("resident Qwen prefill needs at least one token"))?;
        for &token in rest {
            self.forward(token)?;
        }
        self.step(last)
    }

    /// Consume `token` and return the greedy next token. Four bytes cross the bus.
    ///
    /// # Errors
    /// Rejects a token outside the vocabulary or a full context, or returns a
    /// driver failure.
    pub fn step(&mut self, token: u32) -> Result<u32, BackendError> {
        self.forward(token)?;
        self.argmax()?;
        let mut next = [0u32; 1];
        self.stream
            .memcpy_dtoh(&self.scratch.token, &mut next)
            .map_err(|error| driver_err("read resident Qwen token", &error))?;
        Ok(next[0])
    }

    /// Consume `token` and return the full logits row, for sampling other than
    /// greedy and for parity checks. Downloads `vocab` floats.
    ///
    /// # Errors
    /// As [`Self::step`].
    pub fn step_logits(&mut self, token: u32) -> Result<Vec<f32>, BackendError> {
        self.forward(token)?;
        let mut logits = vec![0.0f32; self.vocab];
        self.stream
            .memcpy_dtoh(&self.scratch.logits, &mut logits)
            .map_err(|error| driver_err("read resident Qwen logits", &error))?;
        Ok(logits)
    }

    /// Run one token through every layer and the language head, leaving the
    /// logits in scratch.
    fn forward(&mut self, token: u32) -> Result<(), BackendError> {
        if token as usize >= self.vocab {
            return Err(invalid(format!(
                "token {token} outside the {}-row vocabulary",
                self.vocab
            )));
        }
        if self.position >= self.max_context {
            return Err(invalid(format!(
                "resident Qwen context of {} is full",
                self.max_context
            )));
        }
        let ctrl = [to_i32(self.position, "position")?, 0, 0, 0];
        self.stream
            .memcpy_htod(&ctrl, &mut self.scratch.ctrl)
            .map_err(|error| driver_err("upload resident ctrl", &error))?;
        self.stream
            .memcpy_htod(&[token], &mut self.scratch.token)
            .map_err(|error| driver_err("upload resident token", &error))?;

        self.gather()?;
        let n = to_i32(self.hidden, "hidden")?;
        for index in 0..self.layers.len() {
            // The previous layer's MLP output is folded into the residual here,
            // fused with this layer's input norm; layer 0 starts from the embedding.
            add_rmsnorm(
                &self.stream,
                &self.kernels.add_rmsnorm,
                &mut self.scratch.residual,
                &self.scratch.branch,
                &self.layers[index].input_norm,
                &mut self.scratch.normalized,
                n,
                self.eps,
                index != 0,
            )?;
            if matches!(self.layers[index].mixer, Mixer::DeltaNet(_)) {
                self.deltanet(index)?;
            } else {
                self.attention(index)?;
            }
            add_rmsnorm(
                &self.stream,
                &self.kernels.add_rmsnorm,
                &mut self.scratch.residual,
                &self.scratch.branch,
                &self.layers[index].post_attention_norm,
                &mut self.scratch.normalized,
                n,
                self.eps,
                true,
            )?;
            self.mlp(index)?;
        }
        add_rmsnorm(
            &self.stream,
            &self.kernels.add_rmsnorm,
            &mut self.scratch.residual,
            &self.scratch.branch,
            &self.final_norm,
            &mut self.scratch.normalized,
            n,
            self.eps,
            true,
        )?;
        launch_salt_v2_stream_on(
            &self.stream,
            &self.kernels.stream_gemv,
            &self.lm_head,
            &self.scratch.normalized,
            1,
            &mut self.scratch.logits,
        )?;
        self.position += 1;
        Ok(())
    }

    fn gather(&mut self) -> Result<(), BackendError> {
        let embedding = &self.embedding;
        let selected = 1u32;
        let n = to_u32(embedding.rows, "vocab")?;
        let k = to_u32(embedding.columns, "hidden")?;
        let tile_count = to_u32(embedding.tile_count, "tile count")?;
        let plane_count = to_u32(embedding.plane_count, "plane count")?;
        let payload_bytes = embedding.receipt.payload_bytes();
        let scale_count = embedding.receipt.scale_bytes() / core::mem::size_of::<u16>() as u64;
        let index_metadata = embedding
            .index_metadata
            .as_ref()
            .unwrap_or(&embedding.payload);
        let mut builder = self.stream.launch_builder(&self.kernels.gather);
        builder
            .arg(&embedding.payload)
            .arg(&embedding.scales)
            .arg(index_metadata)
            .arg(&self.scratch.token)
            .arg(&mut self.scratch.residual)
            .arg(&selected)
            .arg(&n)
            .arg(&k)
            .arg(&embedding.codec_tag)
            .arg(&embedding.scale_group_size)
            .arg(&tile_count)
            .arg(&plane_count)
            .arg(&payload_bytes)
            .arg(&scale_count)
            .arg(&embedding.allocation_map_bytes)
            .arg(&embedding.rank_prefix_count)
            .arg(&embedding.terminal_map_value);
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (k.div_ceil(256), 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            "launch resident embedding gather",
        )
    }

    fn deltanet(&mut self, index: usize) -> Result<(), BackendError> {
        let key_width = self.key_heads * self.key_head_dim;
        let group = to_u32(self.value_heads / self.key_heads, "group size")?;
        let key_heads = to_i32(self.key_heads, "key heads")?;
        let value_heads = to_i32(self.value_heads, "value heads")?;
        let key_head_dim = to_i32(self.key_head_dim, "key head dim")?;
        let value_head_dim = to_i32(self.value_head_dim, "value head dim")?;
        let kernel_taps = to_i32(self.conv_kernel, "conv kernel")?;
        let dk = to_u32(self.key_head_dim, "key head dim")?;
        let dv = to_u32(self.value_head_dim, "value head dim")?;
        let value_head_blocks = to_u32(self.value_heads, "value heads")?;
        let prep_blocks = to_u32(self.key_heads + 1, "prep blocks")?;
        let query_scale = 1.0f32 / (self.key_head_dim as f32).sqrt();
        // The host reference's `QK_L2_EPSILON`.
        let l2_epsilon = 1e-6f32;
        let eps = self.eps;
        let hidden_u32 = to_u32(self.hidden, "hidden")?;
        let fused_table_bytes = self.fused_table_bytes;
        let (stream, kernels, scratch) = (&self.stream, &self.kernels, &mut self.scratch);
        let entry = &mut self.layers[index];
        let fused = if fused_enabled() {
            entry.fused_in.as_ref()
        } else {
            None
        };
        let Mixer::DeltaNet(layer) = &mut entry.mixer else {
            return Err(invalid("resident Qwen layer is not a DeltaNet layer"));
        };
        let gemv = &kernels.stream_gemv;
        match fused {
            Some(group) => launch_salt_v2_stream_multi_on(
                stream,
                &kernels.stream_gemv_multi,
                &group.descriptors,
                group.tensor_count,
                group.total_rows,
                hidden_u32,
                fused_table_bytes,
                &scratch.normalized,
            )?,
            None => {
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.qkv,
                    &scratch.normalized,
                    1,
                    &mut scratch.qkv,
                )?;
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.z,
                    &scratch.normalized,
                    1,
                    &mut scratch.z,
                )?;
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.b,
                    &scratch.normalized,
                    1,
                    &mut scratch.b,
                )?;
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.a,
                    &scratch.normalized,
                    1,
                    &mut scratch.a,
                )?;
            }
        }

        let conv_width = layer.qkv.rows;
        let width = to_i32(conv_width, "conv width")?;
        let mut builder = stream.launch_builder(&kernels.conv);
        builder
            .arg(&scratch.qkv)
            .arg(&mut layer.conv_state)
            .arg(&layer.conv_weight)
            .arg(&mut scratch.convolved)
            .arg(&width)
            .arg(&kernel_taps);
        run(
            &mut builder,
            elementwise(conv_width),
            "launch q35_deltanet_conv",
        )?;

        let mut builder = stream.launch_builder(&kernels.prep);
        builder
            .arg(&scratch.convolved)
            .arg(&scratch.b)
            .arg(&scratch.a)
            .arg(&layer.a_log)
            .arg(&layer.dt_bias)
            .arg(&mut scratch.kk)
            .arg(&mut scratch.qq)
            .arg(&mut scratch.beta)
            .arg(&mut scratch.decay)
            .arg(&key_heads)
            .arg(&value_heads)
            .arg(&key_head_dim)
            .arg(&query_scale)
            .arg(&l2_epsilon);
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (prep_blocks, 1, 1),
                block_dim: (dk.max(64), 1, 1),
                shared_mem_bytes: 0,
            },
            "launch q35_deltanet_prep",
        )?;

        let value = scratch.convolved.slice(2 * key_width..);
        let mut builder = stream.launch_builder(&kernels.recurrent);
        builder
            .arg(&mut layer.recurrent_state)
            .arg(&scratch.kk)
            .arg(&scratch.qq)
            .arg(&value)
            .arg(&scratch.beta)
            .arg(&scratch.decay)
            .arg(&mut scratch.core)
            .arg(&dk)
            .arg(&dv)
            .arg(&group);
        // One thread per value lane, each writing only its own column of its own
        // head's state -- updated in place, with no staging copy, because the
        // executor never rolls a step back. The lanes of a head are split across
        // warp-sized blocks: one block per head is only 48 blocks on a 128-SM part,
        // and measured 2.17 ms/token that way.
        let (lane_blocks, lane_threads) = if dv.is_multiple_of(32) {
            (dv / 32, 32)
        } else {
            (1, dv)
        };
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (value_head_blocks, lane_blocks, 1),
                block_dim: (lane_threads, 1, 1),
                shared_mem_bytes: 2 * dk * core::mem::size_of::<f32>() as u32,
            },
            "launch DeltaNet recurrent step",
        )?;

        let mut builder = stream.launch_builder(&kernels.gated_rmsnorm);
        builder
            .arg(&scratch.core)
            .arg(&scratch.z)
            .arg(&layer.norm_weight)
            .arg(&mut scratch.gated)
            .arg(&value_head_dim)
            .arg(&eps);
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (value_head_blocks, 1, 1),
                block_dim: (dv.max(64), 1, 1),
                shared_mem_bytes: 0,
            },
            "launch q35_gated_rmsnorm",
        )?;

        launch_salt_v2_stream_on(
            stream,
            gemv,
            &layer.out,
            &scratch.gated,
            1,
            &mut scratch.branch,
        )
    }

    fn attention(&mut self, index: usize) -> Result<(), BackendError> {
        let heads = to_i32(self.heads, "heads")?;
        let kv_heads = to_i32(self.kv_heads, "kv heads")?;
        let head_dim = to_i32(self.head_dim, "head dim")?;
        let rotary_dim = to_i32(self.rotary_dim, "rotary dim")?;
        let threads = to_u32(self.head_dim, "head dim")?;
        let prep_blocks = to_u32(self.heads + self.kv_heads, "prep blocks")?;
        let head_blocks = to_u32(self.heads, "heads")?;
        let score_bytes = to_u32(
            self.max_context * core::mem::size_of::<f32>(),
            "attention scores",
        )?;
        let query_width = self.heads * self.head_dim;
        let query_width_i32 = to_i32(query_width, "query width")?;
        let scale = 1.0f32 / (self.head_dim as f32).sqrt();
        let eps = self.eps;
        let hidden_u32 = to_u32(self.hidden, "hidden")?;
        let fused_table_bytes = self.fused_table_bytes;
        let (stream, kernels, scratch, inv_freq) = (
            &self.stream,
            &self.kernels,
            &mut self.scratch,
            &self.inv_freq,
        );
        let entry = &mut self.layers[index];
        let fused = if fused_enabled() {
            entry.fused_in.as_ref()
        } else {
            None
        };
        let Mixer::Attention(layer) = &mut entry.mixer else {
            return Err(invalid("resident Qwen layer is not an attention layer"));
        };
        let gemv = &kernels.stream_gemv;
        match fused {
            Some(group) => launch_salt_v2_stream_multi_on(
                stream,
                &kernels.stream_gemv_multi,
                &group.descriptors,
                group.tensor_count,
                group.total_rows,
                hidden_u32,
                fused_table_bytes,
                &scratch.normalized,
            )?,
            None => {
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.q,
                    &scratch.normalized,
                    1,
                    &mut scratch.fused_query,
                )?;
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.k,
                    &scratch.normalized,
                    1,
                    &mut scratch.key,
                )?;
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.v,
                    &scratch.normalized,
                    1,
                    &mut scratch.value,
                )?;
            }
        }

        let mut builder = stream.launch_builder(&kernels.attn_prep);
        builder
            .arg(&scratch.fused_query)
            .arg(&scratch.key)
            .arg(&scratch.value)
            .arg(&layer.q_norm)
            .arg(&layer.k_norm)
            .arg(inv_freq)
            .arg(&mut scratch.query)
            .arg(&mut scratch.attention_gate)
            .arg(&mut layer.key_cache)
            .arg(&mut layer.value_cache)
            .arg(&scratch.ctrl)
            .arg(&heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(&rotary_dim)
            .arg(&eps);
        // One block per query head plus one per KV head, one thread per head lane;
        // the cache row written is `ctrl[0] < max_context`, checked in `forward`.
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (prep_blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: threads * core::mem::size_of::<f32>() as u32,
            },
            "launch q35_attn_prep",
        )?;

        let mut builder = stream.launch_builder(&kernels.attention);
        builder
            .arg(&scratch.query)
            .arg(&layer.key_cache)
            .arg(&layer.value_cache)
            .arg(&mut scratch.attended)
            .arg(&scratch.ctrl)
            .arg(&heads)
            .arg(&kv_heads)
            .arg(&head_dim)
            .arg(&scale);
        // The shared score buffer holds `max_context` floats and the kernel reads
        // `ctrl[0] + 1 <= max_context` positions.
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (head_blocks, 1, 1),
                block_dim: (threads, 1, 1),
                shared_mem_bytes: score_bytes,
            },
            "launch q35_attention",
        )?;

        let mut builder = stream.launch_builder(&kernels.sigmoid_mul);
        builder
            .arg(&mut scratch.attended)
            .arg(&scratch.attention_gate)
            .arg(&query_width_i32);
        run(
            &mut builder,
            elementwise(query_width),
            "launch q35_sigmoid_mul",
        )?;

        launch_salt_v2_stream_on(
            stream,
            gemv,
            &layer.o,
            &scratch.attended,
            1,
            &mut scratch.branch,
        )
    }

    fn mlp(&mut self, index: usize) -> Result<(), BackendError> {
        let hidden_u32 = to_u32(self.hidden, "hidden")?;
        let fused_table_bytes = self.fused_table_bytes;
        let (stream, kernels, scratch) = (&self.stream, &self.kernels, &mut self.scratch);
        let layer = &self.layers[index];
        let fused = if fused_enabled() {
            layer.fused_mlp.as_ref()
        } else {
            None
        };
        let gemv = &kernels.stream_gemv;
        match fused {
            Some(group) => launch_salt_v2_stream_multi_on(
                stream,
                &kernels.stream_gemv_multi,
                &group.descriptors,
                group.tensor_count,
                group.total_rows,
                hidden_u32,
                fused_table_bytes,
                &scratch.normalized,
            )?,
            None => {
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.gate,
                    &scratch.normalized,
                    1,
                    &mut scratch.mlp_gate,
                )?;
                launch_salt_v2_stream_on(
                    stream,
                    gemv,
                    &layer.up,
                    &scratch.normalized,
                    1,
                    &mut scratch.mlp_up,
                )?;
            }
        }
        let n = layer.gate.rows;
        let n_i32 = to_i32(n, "intermediate")?;
        let mut builder = stream.launch_builder(&kernels.swiglu);
        builder
            .arg(&scratch.mlp_gate)
            .arg(&scratch.mlp_up)
            .arg(&mut scratch.mlp_act)
            .arg(&n_i32);
        run(&mut builder, elementwise(n), "launch q35_swiglu")?;
        launch_salt_v2_stream_on(
            stream,
            gemv,
            &layer.down,
            &scratch.mlp_act,
            1,
            &mut scratch.branch,
        )
    }

    fn argmax(&mut self) -> Result<(), BackendError> {
        let n = to_i32(self.vocab, "vocab")?;
        let mut builder = self.stream.launch_builder(&self.kernels.argmax_partial);
        builder
            .arg(&self.scratch.logits)
            .arg(&n)
            .arg(&mut self.scratch.argmax_value)
            .arg(&mut self.scratch.argmax_index);
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (ARGMAX_BLOCKS, 1, 1),
                block_dim: (ARGMAX_THREADS, 1, 1),
                shared_mem_bytes: 0,
            },
            "launch q35_argmax_partial",
        )?;
        let partials = ARGMAX_BLOCKS as i32;
        let mut builder = self.stream.launch_builder(&self.kernels.argmax_final);
        builder
            .arg(&self.scratch.argmax_value)
            .arg(&self.scratch.argmax_index)
            .arg(&partials)
            .arg(&mut self.scratch.token);
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (1, 1, 1),
                block_dim: (32, 1, 1),
                shared_mem_bytes: 0,
            },
            "launch q35_argmax_final",
        )
    }
}
