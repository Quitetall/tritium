//! DFlash block-diffusion drafter on the resident Qwen3.6 executor.
//!
//! The drafter (`z-lab/Qwen3.6-27B-DFlash`: five Qwen3 layers) conditions on the
//! target's residual stream after a few layers, projected once per consumed
//! position into each layer's key/value cache, and drafts a block of tokens in a
//! single non-autoregressive pass from `[anchor, mask, mask, ...]`. It shares the
//! target's token embedding and language head.
//!
//! Its numerics never reach the output: the target keeps a drafted token only if
//! it equals the target's own greedy choice. So the drafter runs on int8 weights
//! (one f32 scale per row), f16 tensor-core products and CUDA transcendentals,
//! which halves its bytes and lets it sit beside the target in VRAM.

use super::*;

/// The drafter's shape, from its `config.json`.
#[derive(Debug, Clone)]
pub struct DFlashConfig {
    /// Hidden width (the target's).
    pub hidden: usize,
    /// Query heads.
    pub heads: usize,
    /// Key/value heads.
    pub kv_heads: usize,
    /// MLP width.
    pub intermediate: usize,
    /// Target decoder layers whose residual stream conditions the drafter.
    pub target_layers: Vec<usize>,
    /// Tokens per drafted block, the anchor included.
    pub block_size: usize,
    /// The mask token filling the block after the anchor.
    pub mask_token: u32,
    /// RoPE base.
    pub rope_theta: f32,
    /// RMSNorm epsilon.
    pub rms_eps: f32,
    /// Window of the sliding (causal) layers.
    pub sliding_window: usize,
    /// Per layer: causal sliding-window attention (`true`) or full attention over
    /// the context and the whole block (`false`).
    pub sliding: Vec<bool>,
    /// Weight precision on the device: 8 (one scale per row) or 4 (one f16 scale
    /// per 64 columns of a row). The drafter's numerics never reach the output.
    pub weight_bits: u8,
}

/// Head width the drafter kernels are written for.
const HEAD_DIM: usize = 128;

/// A drafter weight on the device.
enum W8 {
    /// int8 with one f32 scale per row.
    Int8 {
        weight: CudaSlice<i8>,
        scale: CudaSlice<f32>,
        rows: usize,
        columns: usize,
    },
    /// Offset-8 nibbles with one f16 scale per 64 columns, laid out for
    /// `dflash_w4_rows*` (see `dflash.cu`).
    Int4 {
        weight: CudaSlice<u8>,
        scale: CudaSlice<u16>,
        rows: usize,
        columns: usize,
    },
}

impl W8 {
    const fn shape(&self) -> (usize, usize) {
        match self {
            Self::Int8 { rows, columns, .. } | Self::Int4 { rows, columns, .. } => {
                (*rows, *columns)
            }
        }
    }
}

struct DraftLayer {
    input_norm: CudaSlice<f32>,
    post_norm: CudaSlice<f32>,
    q_norm: CudaSlice<f32>,
    k_norm: CudaSlice<f32>,
    q: W8,
    /// k | v.
    kv: W8,
    o: W8,
    /// gate | up.
    gate_up: W8,
    down: W8,
    /// `[capacity, kv_heads * 128]`.
    key_cache: CudaSlice<f32>,
    value_cache: CudaSlice<f32>,
    sliding: bool,
}

struct Functions {
    split_act: CudaFunction,
    rows16: CudaFunction,
    rows64: CudaFunction,
    w4_rows16: CudaFunction,
    w4_rows64: CudaFunction,
    concat: CudaFunction,
    norm_rope: CudaFunction,
    attention: CudaFunction,
    swiglu: CudaFunction,
}

/// The drafter's weights, caches and scratch.
pub(super) struct Drafter {
    config: DFlashConfig,
    functions: Functions,
    fc: W8,
    hidden_norm: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    layers: Vec<DraftLayer>,
    inv_freq: CudaSlice<f32>,
    /// f16 activation staging, `[64, max k]`.
    act16: CudaSlice<u16>,
    /// `[64, layers * hidden]` concatenated target features.
    concat: CudaSlice<f32>,
    /// `[64, hidden]` projected and normalized context.
    ctx_hidden: CudaSlice<f32>,
    ctx_norm: CudaSlice<f32>,
    /// `[64, 2 * kv width]`.
    kv: CudaSlice<f32>,
    /// `[block, hidden]` block residual, normalized input and branch.
    h: CudaSlice<f32>,
    hn: CudaSlice<f32>,
    branch: CudaSlice<f32>,
    /// `[block, heads * 128]`.
    q: CudaSlice<f32>,
    q_rot: CudaSlice<f32>,
    attn: CudaSlice<f32>,
    /// `[block, 2 * intermediate]` and `[block, intermediate]`.
    gate_up: CudaSlice<f32>,
    act: CudaSlice<f32>,
    /// `[block]` block token ids.
    tokens: CudaSlice<u32>,
    /// Positions whose target features are in the caches.
    pub(super) context: usize,
    capacity: usize,
}

impl std::fmt::Debug for Drafter {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter
            .debug_struct("Drafter")
            .field("config", &self.config)
            .field("context", &self.context)
            .finish_non_exhaustive()
    }
}

fn quantize(
    stream: &Arc<CudaStream>,
    values: &[f32],
    rows: usize,
    columns: usize,
    name: &str,
    bits: u8,
) -> Result<W8, BackendError> {
    if values.len() != rows * columns || !rows.is_multiple_of(16) || !columns.is_multiple_of(64) {
        return Err(invalid(format!(
            "drafter tensor {name} is {} values for [{rows}, {columns}] (rows must be a \
             multiple of 16, columns of 64)",
            values.len()
        )));
    }
    if bits == 4 {
        return quantize4(stream, values, rows, columns);
    }
    let mut weight = vec![0i8; values.len()];
    let mut scale = vec![0f32; rows];
    for (row, (out, values)) in weight
        .chunks_mut(columns)
        .zip(values.chunks(columns))
        .enumerate()
    {
        let peak = values
            .iter()
            .fold(0.0f32, |peak, value| peak.max(value.abs()));
        let step = if peak > 0.0 { peak / 127.0 } else { 1.0 };
        scale[row] = step;
        for (slot, value) in out.iter_mut().zip(values) {
            *slot = (value / step).round().clamp(-127.0, 127.0) as i8;
        }
    }
    Ok(W8::Int8 {
        weight: stream
            .clone_htod(&weight)
            .map_err(|error| driver_err("upload drafter weight", &error))?,
        scale: stream
            .clone_htod(&scale)
            .map_err(|error| driver_err("upload drafter scale", &error))?,
        rows,
        columns,
    })
}

/// 4-bit quantization: per row and 64-column chunk, `scale = max|w| / 7` and
/// nibble `round(w / scale) + 8` clamped to 1..=15, packed so quad thread c's
/// sixteen columns of a chunk are two words with each column pair at
/// `(word >> 4k) & 0x000F000F`.
fn quantize4(
    stream: &Arc<CudaStream>,
    values: &[f32],
    rows: usize,
    columns: usize,
) -> Result<W8, BackendError> {
    let chunks = columns / 64;
    let mut weight = vec![0u8; rows * columns / 2];
    let mut scale = vec![0u16; rows * chunks];
    for row in 0..rows {
        for chunk in 0..chunks {
            let source = &values[row * columns + chunk * 64..row * columns + chunk * 64 + 64];
            let peak = source
                .iter()
                .fold(0.0f32, |peak, value| peak.max(value.abs()));
            let step = half::f16::from_f32(if peak > 0.0 { peak / 7.0 } else { 1.0 });
            scale[row * chunks + chunk] = step.to_bits();
            let step = step.to_f32();
            let nibble = |value: f32| ((value / step).round().clamp(-7.0, 7.0) as i32 + 8) as u32;
            for quad in 0..4 {
                for word in 0..2 {
                    let mut packed = 0u32;
                    for j in 0..8 {
                        let column = 16 * quad + 8 * word + j;
                        let shift = if j % 2 == 0 {
                            4 * (j / 2)
                        } else {
                            16 + 4 * (j / 2)
                        };
                        packed |= nibble(source[column]) << shift;
                    }
                    let at = row * columns / 2 + chunk * 32 + quad * 8 + word * 4;
                    weight[at..at + 4].copy_from_slice(&packed.to_le_bytes());
                }
            }
        }
    }
    Ok(W8::Int4 {
        weight: stream
            .clone_htod(&weight)
            .map_err(|error| driver_err("upload drafter weight", &error))?,
        scale: stream
            .clone_htod(&scale)
            .map_err(|error| driver_err("upload drafter scale", &error))?,
        rows,
        columns,
    })
}

/// Fetch one drafter tensor as f32 with its shape.
pub type DFlashFetch<'a> = dyn FnMut(&str) -> Result<(Vec<f32>, Vec<usize>), BackendError> + 'a;

impl Drafter {
    pub(super) fn build(
        stream: &Arc<CudaStream>,
        module: &Arc<CudaModule>,
        config: DFlashConfig,
        capacity: usize,
        fetch: &mut DFlashFetch<'_>,
    ) -> Result<Self, BackendError> {
        let hidden = config.hidden;
        let kv_width = config.kv_heads * HEAD_DIM;
        let q_width = config.heads * HEAD_DIM;
        let block = config.block_size;
        if config.sliding.is_empty() || block == 0 || block > QWEN35_VERIFY_MAX {
            return Err(invalid(
                "drafter config: no layers, or a block above the verify limit",
            ));
        }
        let bits = config.weight_bits;
        if bits != 4 && bits != 8 {
            return Err(invalid("drafter weight bits must be 4 or 8"));
        }
        let function = |name: &str| {
            module
                .load_function(name)
                .map_err(|error| driver_err("resolve drafter kernel", &error))
        };
        let functions = Functions {
            split_act: function("dflash_split_act")?,
            rows16: function("dflash_w8_rows16")?,
            rows64: function("dflash_w8_rows64")?,
            w4_rows16: function("dflash_w4_rows16")?,
            w4_rows64: function("dflash_w4_rows64")?,
            concat: function("dflash_concat_features")?,
            norm_rope: function("dflash_norm_rope")?,
            attention: function("dflash_attention")?,
            swiglu: function("dflash_swiglu_fused")?,
        };
        let mut matrix = |name: &str, rows: usize, columns: usize| {
            let (values, shape) = fetch(name)?;
            if shape != [rows, columns] {
                return Err(invalid(format!(
                    "drafter tensor {name} has shape {shape:?}"
                )));
            }
            quantize(stream, &values, rows, columns, name, bits)
        };
        let fc = matrix("fc.weight", hidden, config.target_layers.len() * hidden)?;
        let mut vector = |name: &str, len: usize| {
            let (values, shape) = fetch(name)?;
            if shape != [len] {
                return Err(invalid(format!(
                    "drafter tensor {name} has shape {shape:?}"
                )));
            }
            stream
                .clone_htod(&values)
                .map_err(|error| driver_err("upload drafter vector", &error))
        };
        let hidden_norm = vector("hidden_norm.weight", hidden)?;
        let norm = vector("norm.weight", hidden)?;
        let zeros = |len: usize, what: &str| {
            stream
                .alloc_zeros::<f32>(len)
                .map_err(|error| alloc_or_backend(what, &error, len * 4))
        };
        let mut layers = Vec::with_capacity(config.sliding.len());
        for (index, &sliding) in config.sliding.iter().enumerate() {
            let name = |suffix: &str| format!("layers.{index}.{suffix}");
            let fuse =
                |fetch: &mut DFlashFetch<'_>, a: &str, b: &str, rows: usize, columns: usize| {
                    let (mut first, shape) = fetch(a)?;
                    if shape != [rows, columns] {
                        return Err(invalid(format!("drafter tensor {a} has shape {shape:?}")));
                    }
                    let (second, shape) = fetch(b)?;
                    if shape != [rows, columns] {
                        return Err(invalid(format!("drafter tensor {b} has shape {shape:?}")));
                    }
                    first.extend_from_slice(&second);
                    quantize(stream, &first, 2 * rows, columns, a, bits)
                };
            let kv = fuse(
                fetch,
                &name("self_attn.k_proj.weight"),
                &name("self_attn.v_proj.weight"),
                kv_width,
                hidden,
            )?;
            let gate_up = fuse(
                fetch,
                &name("mlp.gate_proj.weight"),
                &name("mlp.up_proj.weight"),
                config.intermediate,
                hidden,
            )?;
            let (values, _) = fetch(&name("self_attn.q_proj.weight"))?;
            let q = quantize(stream, &values, q_width, hidden, "q_proj", bits)?;
            let (values, _) = fetch(&name("self_attn.o_proj.weight"))?;
            let o = quantize(stream, &values, hidden, q_width, "o_proj", bits)?;
            let (values, _) = fetch(&name("mlp.down_proj.weight"))?;
            let down = quantize(
                stream,
                &values,
                hidden,
                config.intermediate,
                "down_proj",
                bits,
            )?;
            let mut vector = |suffix: &str, len: usize| {
                let (values, shape) = fetch(&name(suffix))?;
                if shape != [len] {
                    return Err(invalid(format!(
                        "drafter tensor {suffix} has shape {shape:?}"
                    )));
                }
                stream
                    .clone_htod(&values)
                    .map_err(|error| driver_err("upload drafter vector", &error))
            };
            layers.push(DraftLayer {
                input_norm: vector("input_layernorm.weight", hidden)?,
                post_norm: vector("post_attention_layernorm.weight", hidden)?,
                q_norm: vector("self_attn.q_norm.weight", HEAD_DIM)?,
                k_norm: vector("self_attn.k_norm.weight", HEAD_DIM)?,
                q,
                kv,
                o,
                gate_up,
                down,
                key_cache: zeros(capacity * kv_width, "allocate drafter key cache")?,
                value_cache: zeros(capacity * kv_width, "allocate drafter value cache")?,
                sliding,
            });
        }
        let inv_freq: Vec<f32> = (0..HEAD_DIM / 2)
            .map(|lane| 1.0 / config.rope_theta.powf((2 * lane) as f32 / HEAD_DIM as f32))
            .collect();
        let features = config.target_layers.len() * hidden;
        let max_k = features.max(config.intermediate).max(q_width).max(hidden);
        let rows = PREFILL_CHUNK;
        Ok(Self {
            fc,
            hidden_norm,
            norm,
            layers,
            inv_freq: stream
                .clone_htod(&inv_freq)
                .map_err(|error| driver_err("upload drafter RoPE", &error))?,
            act16: stream.alloc_zeros::<u16>(rows * max_k).map_err(|error| {
                alloc_or_backend("allocate drafter staging", &error, rows * max_k * 2)
            })?,
            concat: zeros(rows * features, "allocate drafter features")?,
            ctx_hidden: zeros(rows * hidden, "allocate drafter context")?,
            ctx_norm: zeros(rows * hidden, "allocate drafter context")?,
            kv: zeros(rows * 2 * kv_width, "allocate drafter kv")?,
            h: zeros(block * hidden, "allocate drafter block")?,
            hn: zeros(block * hidden, "allocate drafter block")?,
            branch: zeros(block * hidden, "allocate drafter block")?,
            q: zeros(block * q_width, "allocate drafter query")?,
            q_rot: zeros(block * q_width, "allocate drafter query")?,
            attn: zeros(block * q_width, "allocate drafter attention")?,
            gate_up: zeros(block * 2 * config.intermediate, "allocate drafter mlp")?,
            act: zeros(block * config.intermediate, "allocate drafter mlp")?,
            tokens: stream
                .alloc_zeros::<u32>(block)
                .map_err(|error| alloc_or_backend("allocate drafter tokens", &error, block * 4))?,
            context: 0,
            capacity,
            functions,
            config,
        })
    }

    /// `out[r * out_stride + row] = input[r] . weight[row]` for `m` rows of `input`.
    fn w8(
        functions: &Functions,
        stream: &CudaStream,
        act16: &mut CudaSlice<u16>,
        weight: &W8,
        input: &CudaSlice<f32>,
        m: usize,
        out: &mut CudaSlice<f32>,
    ) -> Result<(), BackendError> {
        let (rows, columns) = weight.shape();
        let k = to_u32(columns, "drafter columns")?;
        let n = to_u32(rows, "drafter rows")?;
        let m_u32 = to_u32(m, "drafter rows")?;
        let padded = m_u32.div_ceil(8) * 8;
        let mut builder = stream.launch_builder(&functions.split_act);
        builder.arg(input).arg(&mut *act16).arg(&m_u32).arg(&k);
        run(
            &mut builder,
            LaunchConfig {
                grid_dim: (padded, 1, 1),
                block_dim: (256, 1, 1),
                shared_mem_bytes: 0,
            },
            "launch dflash_split_act",
        )?;
        let cfg = LaunchConfig {
            grid_dim: (n / 16, 1, 1),
            block_dim: (128, 1, 1),
            shared_mem_bytes: 0,
        };
        match weight {
            W8::Int8 { weight, scale, .. } => {
                let function = if m <= 16 {
                    &functions.rows16
                } else {
                    &functions.rows64
                };
                let mut builder = stream.launch_builder(function);
                builder
                    .arg(weight)
                    .arg(scale)
                    .arg(&*act16)
                    .arg(out)
                    .arg(&n)
                    .arg(&k)
                    .arg(&m_u32)
                    .arg(&n);
                run(&mut builder, cfg, "launch dflash W8 GEMM")
            }
            W8::Int4 { weight, scale, .. } => {
                let function = if m <= 16 {
                    &functions.w4_rows16
                } else {
                    &functions.w4_rows64
                };
                let mut builder = stream.launch_builder(function);
                builder
                    .arg(weight)
                    .arg(scale)
                    .arg(&*act16)
                    .arg(out)
                    .arg(&n)
                    .arg(&k)
                    .arg(&m_u32)
                    .arg(&n);
                run(&mut builder, cfg, "launch dflash W4 GEMM")
            }
        }
    }

    /// Project `rows` positions' captured target features (`features`,
    /// `[layers, PREFILL_CHUNK, hidden]`) into every layer's key/value cache at
    /// positions `self.context ..`.
    pub(super) fn ingest(
        &mut self,
        stream: &Arc<CudaStream>,
        add_rmsnorm_fn: &CudaFunction,
        features: &CudaSlice<f32>,
        rows: usize,
    ) -> Result<(), BackendError> {
        if rows == 0 {
            return Ok(());
        }
        if rows > PREFILL_CHUNK || self.context + rows > self.capacity {
            return Err(invalid("drafter context is full"));
        }
        let hidden = self.config.hidden;
        let layers_n = self.config.target_layers.len();
        let kv_width = self.config.kv_heads * HEAD_DIM;
        let total = to_u32(layers_n * rows * hidden, "drafter features")?;
        let (layers_u32, rows_u32, hidden_u32, stride) = (
            to_u32(layers_n, "layers")?,
            to_u32(rows, "rows")?,
            to_u32(hidden, "hidden")?,
            to_u32(PREFILL_CHUNK, "chunk")?,
        );
        let mut builder = stream.launch_builder(&self.functions.concat);
        builder
            .arg(features)
            .arg(&mut self.concat)
            .arg(&layers_u32)
            .arg(&rows_u32)
            .arg(&hidden_u32)
            .arg(&stride);
        run(
            &mut builder,
            elementwise(total as usize),
            "launch dflash_concat_features",
        )?;
        Self::w8(
            &self.functions,
            stream,
            &mut self.act16,
            &self.fc,
            &self.concat,
            rows,
            &mut self.ctx_hidden,
        )?;
        plain_rmsnorm(
            stream,
            add_rmsnorm_fn,
            &mut self.ctx_hidden,
            &self.branch,
            &self.hidden_norm,
            &mut self.ctx_norm,
            hidden,
            self.config.rms_eps,
            false,
            rows,
        )?;
        let position0 = to_i32(self.context, "drafter position")?;
        let kv_stride = to_u32(2 * kv_width, "kv width")?;
        let out_stride = to_u32(kv_width, "kv width")?;
        let value_offset = out_stride;
        let zero = 0u32;
        for layer in &mut self.layers {
            Self::w8(
                &self.functions,
                stream,
                &mut self.act16,
                &layer.kv,
                &self.ctx_norm,
                rows,
                &mut self.kv,
            )?;
            let start = self.context * kv_width;
            let mut keys = layer.key_cache.slice_mut(start..start + rows * kv_width);
            let mut values = layer.value_cache.slice_mut(start..start + rows * kv_width);
            let mut builder = stream.launch_builder(&self.functions.norm_rope);
            builder
                .arg(&self.kv)
                .arg(&kv_stride)
                .arg(&zero)
                .arg(&layer.k_norm)
                .arg(&self.inv_freq)
                .arg(&mut keys)
                .arg(&out_stride)
                .arg(&position0)
                .arg(&self.config.rms_eps)
                .arg(&mut values)
                .arg(&value_offset);
            run(
                &mut builder,
                LaunchConfig {
                    grid_dim: (to_u32(self.config.kv_heads, "kv heads")?, rows_u32, 1),
                    block_dim: (HEAD_DIM as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                "launch dflash_norm_rope (context)",
            )?;
        }
        self.context += rows;
        Ok(())
    }

    /// Run the drafter over `[anchor, mask, ...]` at `position` (the anchor's
    /// position, which must be the context length), leaving the final normalized
    /// block hidden states in [`Self::hidden`].
    pub(super) fn draft_hidden(
        &mut self,
        stream: &Arc<CudaStream>,
        add_rmsnorm_fn: &CudaFunction,
        gather_fn: &CudaFunction,
        embedding: &SaltV2ResidentTensor,
        anchor: u32,
        position: usize,
    ) -> Result<(), BackendError> {
        if position != self.context {
            return Err(invalid(format!(
                "drafter context holds {} positions, asked to draft at {position}",
                self.context
            )));
        }
        let block = self.config.block_size;
        if position + block > self.capacity {
            return Err(invalid("drafter context is full"));
        }
        let hidden = self.config.hidden;
        let q_width = self.config.heads * HEAD_DIM;
        let kv_width = self.config.kv_heads * HEAD_DIM;
        let eps = self.config.rms_eps;
        let mut ids = vec![self.config.mask_token; block];
        ids[0] = anchor;
        stream
            .memcpy_htod(&ids, &mut self.tokens)
            .map_err(|error| driver_err("upload drafter block", &error))?;
        let block_u32 = to_u32(block, "block")?;
        gather_rows(
            stream,
            gather_fn,
            embedding,
            &self.tokens,
            &mut self.h,
            block_u32,
        )?;

        let position0 = to_i32(position, "drafter position")?;
        let q_stride = to_u32(q_width, "q width")?;
        let kv_stride = to_u32(2 * kv_width, "kv width")?;
        let kv_out = to_u32(kv_width, "kv width")?;
        let zero = 0u32;
        let null = 0u64;
        let heads = to_i32(self.config.heads, "heads")?;
        let kv_heads = to_i32(self.config.kv_heads, "kv heads")?;
        let block_i32 = to_i32(block, "block")?;
        let scale = 1.0f32 / (HEAD_DIM as f32).sqrt();
        let window = to_i32(self.config.sliding_window, "window")?;
        let span_bytes = to_u32((position + block) * 4, "drafter attention scores")?;
        if span_bytes > 48 * 1024 {
            return Err(invalid(
                "drafter context exceeds the attention kernel's shared memory",
            ));
        }
        let intermediate = to_u32(self.config.intermediate, "intermediate")?;
        for index in 0..self.layers.len() {
            let layer = &mut self.layers[index];
            plain_rmsnorm(
                stream,
                add_rmsnorm_fn,
                &mut self.h,
                &self.branch,
                &layer.input_norm,
                &mut self.hn,
                hidden,
                eps,
                index != 0,
                block,
            )?;
            Self::w8(
                &self.functions,
                stream,
                &mut self.act16,
                &layer.q,
                &self.hn,
                block,
                &mut self.q,
            )?;
            let mut builder = stream.launch_builder(&self.functions.norm_rope);
            builder
                .arg(&self.q)
                .arg(&q_stride)
                .arg(&zero)
                .arg(&layer.q_norm)
                .arg(&self.inv_freq)
                .arg(&mut self.q_rot)
                .arg(&q_stride)
                .arg(&position0)
                .arg(&eps)
                .arg(&null)
                .arg(&zero);
            run(
                &mut builder,
                LaunchConfig {
                    grid_dim: (to_u32(self.config.heads, "heads")?, block_u32, 1),
                    block_dim: (HEAD_DIM as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                "launch dflash_norm_rope (query)",
            )?;
            Self::w8(
                &self.functions,
                stream,
                &mut self.act16,
                &layer.kv,
                &self.hn,
                block,
                &mut self.kv,
            )?;
            let start = position * kv_width;
            let mut keys = layer.key_cache.slice_mut(start..start + block * kv_width);
            let mut values = layer.value_cache.slice_mut(start..start + block * kv_width);
            let mut builder = stream.launch_builder(&self.functions.norm_rope);
            builder
                .arg(&self.kv)
                .arg(&kv_stride)
                .arg(&zero)
                .arg(&layer.k_norm)
                .arg(&self.inv_freq)
                .arg(&mut keys)
                .arg(&kv_out)
                .arg(&position0)
                .arg(&eps)
                .arg(&mut values)
                .arg(&kv_out);
            run(
                &mut builder,
                LaunchConfig {
                    grid_dim: (to_u32(self.config.kv_heads, "kv heads")?, block_u32, 1),
                    block_dim: (HEAD_DIM as u32, 1, 1),
                    shared_mem_bytes: 0,
                },
                "launch dflash_norm_rope (block keys)",
            )?;
            let causal = i32::from(layer.sliding);
            let layer_window = if layer.sliding { window } else { 0 };
            let mut builder = stream.launch_builder(&self.functions.attention);
            builder
                .arg(&self.q_rot)
                .arg(&layer.key_cache)
                .arg(&layer.value_cache)
                .arg(&mut self.attn)
                .arg(&position0)
                .arg(&block_i32)
                .arg(&causal)
                .arg(&layer_window)
                .arg(&heads)
                .arg(&kv_heads)
                .arg(&scale);
            run(
                &mut builder,
                LaunchConfig {
                    grid_dim: (to_u32(self.config.heads, "heads")?, block_u32, 1),
                    block_dim: (128, 1, 1),
                    shared_mem_bytes: span_bytes,
                },
                "launch dflash_attention",
            )?;
            Self::w8(
                &self.functions,
                stream,
                &mut self.act16,
                &layer.o,
                &self.attn,
                block,
                &mut self.branch,
            )?;
            plain_rmsnorm(
                stream,
                add_rmsnorm_fn,
                &mut self.h,
                &self.branch,
                &layer.post_norm,
                &mut self.hn,
                hidden,
                eps,
                true,
                block,
            )?;
            Self::w8(
                &self.functions,
                stream,
                &mut self.act16,
                &layer.gate_up,
                &self.hn,
                block,
                &mut self.gate_up,
            )?;
            let mut builder = stream.launch_builder(&self.functions.swiglu);
            builder
                .arg(&self.gate_up)
                .arg(&mut self.act)
                .arg(&intermediate)
                .arg(&block_u32);
            run(
                &mut builder,
                elementwise(self.config.intermediate * block),
                "launch dflash_swiglu_fused",
            )?;
            Self::w8(
                &self.functions,
                stream,
                &mut self.act16,
                &layer.down,
                &self.act,
                block,
                &mut self.branch,
            )?;
        }
        plain_rmsnorm(
            stream,
            add_rmsnorm_fn,
            &mut self.h,
            &self.branch,
            &self.norm,
            &mut self.hn,
            hidden,
            eps,
            true,
            block,
        )
    }

    /// The last draft's final normalized hidden states, `[block, hidden]`.
    pub(super) const fn hidden(&self) -> &CudaSlice<f32> {
        &self.hn
    }

    pub(super) const fn block_size(&self) -> usize {
        self.config.block_size
    }

    /// Forget the context past `positions` (a reset or a restored snapshot).
    pub(super) fn truncate(&mut self, positions: usize) {
        self.context = self.context.min(positions);
    }
}

/// `out = x / rms(x) * w` per row (Qwen3's plain-weight norm), after
/// `residual += branch` when `add`.
#[allow(clippy::too_many_arguments)]
fn plain_rmsnorm(
    stream: &CudaStream,
    kernel: &CudaFunction,
    residual: &mut CudaSlice<f32>,
    branch: &CudaSlice<f32>,
    weight: &CudaSlice<f32>,
    normalized: &mut CudaSlice<f32>,
    n: usize,
    eps: f32,
    add: bool,
    rows: usize,
) -> Result<(), BackendError> {
    let n = to_i32(n, "hidden")?;
    let add = i32::from(add);
    let one_plus = 0i32;
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
            grid_dim: (to_u32(rows, "rows")?, 1, 1),
            block_dim: (1024, 1, 1),
            shared_mem_bytes: 0,
        },
        "launch drafter rmsnorm",
    )
}
