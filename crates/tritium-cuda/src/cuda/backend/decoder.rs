//! One resident language loop over standard or ordered hybrid mixer data.
use super::semantic::CudaTensor;
use super::*;
use tritium_spec::*;

enum Mixer {
    Attention {
        matrices: [Arc<dyn DeviceBuffer>; 4],
        vectors: Vec<Option<CudaSlice<f32>>>,
        gated: bool,
        cache: Box<attention::Cache>,
    },
    Delta {
        matrices: [Arc<dyn DeviceBuffer>; 5],
        state: Box<delta_mixer::State>,
    },
}
impl Mixer {
    fn len(&self) -> usize {
        match self {
            Self::Attention { cache, .. } => cache.len(),
            Self::Delta { state, .. } => state.len(),
        }
    }
    fn reset(&mut self) {
        match self {
            Self::Attention { cache, .. } => cache.publish(0),
            Self::Delta { state, .. } => state.reset(),
        }
    }
    fn abort(&mut self) {
        if let Self::Delta { state, .. } = self {
            state.abort();
        }
    }
    fn ready(&self, end: usize) -> bool {
        match self {
            Self::Delta { state, .. } => state.staged_len() == Some(end),
            Self::Attention { .. } => true,
        }
    }
    fn commit(&mut self, end: usize) {
        match self {
            Self::Attention { cache, .. } => cache.publish(end),
            Self::Delta { state, .. } => state.commit(),
        }
    }
}
struct Layer {
    input: CudaSlice<f32>,
    post: CudaSlice<f32>,
    mlp: [Arc<dyn DeviceBuffer>; 3],
    activation: DecoderActivation,
    sub: Option<CudaSlice<f32>>,
    sub_eps: f32,
    mixer: Mixer,
    staged: Option<usize>,
}
struct Workspace {
    hidden: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    projection: CudaSlice<f32>,
    query: Option<CudaSlice<f32>>,
    key: Option<CudaSlice<f32>>,
    value: Option<CudaSlice<f32>>,
    mixed: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    a: Option<CudaSlice<f32>>,
    b: Option<CudaSlice<f32>>,
    c: Option<CudaSlice<f32>>,
    d: Option<CudaSlice<f32>>,
    attention_gate: Option<CudaSlice<f32>>,
    convolved: Option<CudaSlice<f32>>,
    rotate: Option<CudaSlice<f32>>,
    rope: Option<CudaSlice<f32>>,
    last: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    ids: CudaSlice<u32>,
}
struct Decoder {
    backend: Arc<CudaBackend>,
    geometry: DecoderGeometry,
    semantics: DecoderSemantics,
    limits: DecoderLimits,
    embedding: Arc<dyn DeviceBuffer>,
    head: Arc<dyn DeviceBuffer>,
    _retained: Vec<Arc<dyn DeviceBuffer>>,
    final_norm: CudaSlice<f32>,
    layers: Vec<Layer>,
    workspace: Workspace,
    frequencies: Vec<f64>,
    rope_host: Vec<f32>,
    cursor: usize,
    payload: usize,
    has_delta: bool,
}
enum ViewMixer<'a> {
    Attention {
        matrices: [&'a Arc<dyn DeviceBuffer>; 4],
        vectors: [&'a [f32]; 6],
        gated: bool,
    },
    Delta(&'a DecoderDeltaNet<'a>),
}
struct View<'a> {
    input: &'a [f32],
    post: &'a [f32],
    mlp: [&'a Arc<dyn DeviceBuffer>; 3],
    activation: DecoderActivation,
    sub: &'a [f32],
    sub_eps: f32,
    mixer: ViewMixer<'a>,
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn mul(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("decoder extent overflow"))
}
fn add(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("decoder payload overflow"))
}
fn zeros<T: Default + Clone>(n: usize) -> Result<Vec<T>, BackendError> {
    let mut out = Vec::new();
    out.try_reserve_exact(n)
        .map_err(|_| BackendError::OutOfMemory {
            requested: n.saturating_mul(core::mem::size_of::<T>()),
        })?;
    out.resize(n, T::default());
    Ok(out)
}
fn required(data: &mut Option<CudaSlice<f32>>) -> Result<&mut CudaSlice<f32>, BackendError> {
    data.as_mut()
        .ok_or_else(|| invalid("admitted decoder workspace missing"))
}
fn vector(values: &[f32], width: usize, optional: bool) -> Result<usize, BackendError> {
    if (values.len() != width && !(optional && values.is_empty()))
        || values.iter().any(|v| !v.is_finite())
    {
        return Err(invalid("decoder vector geometry or values"));
    }
    Ok(values.len())
}

pub(super) fn create(
    backend: Arc<CudaBackend>,
    graph: DecoderGraph<'_>,
    limits: DecoderLimits,
) -> Result<Option<Box<dyn ResidentDecoder>>, BackendError> {
    let Some(budget) = backend.resident_decoder_budget else {
        return Ok(None);
    };
    let g = graph.geometry;
    let s = graph.semantics;
    let attention = AttentionGeometry {
        query_heads: g.query_heads,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
    };
    let (q, kv) = attention.widths()?;
    if g.hidden == 0
        || g.intermediate == 0
        || g.vocabulary == 0
        || g.head_dim > 1024
        || !g.head_dim.is_multiple_of(2)
        || s.rotary_dim == 0
        || !s.rotary_dim.is_multiple_of(2)
        || s.rotary_dim > g.head_dim
        || !g.rms_eps.is_finite()
        || g.rms_eps <= 0.
        || !g.rope_theta.is_finite()
        || g.rope_theta <= 0.
        || limits.max_context == 0
        || limits.max_segment_tokens == 0
        || limits.max_segment_tokens > limits.max_context
        || u32::try_from(limits.max_context).is_err()
        || budget == 0
    {
        return Err(invalid("unsupported decoder geometry or limits"));
    }
    let mut views = Vec::new();
    match &graph.layers {
        DecoderLayers::Standard(layers) => {
            if s != DecoderSemantics::standard(g.head_dim) {
                return Err(invalid("standard numerical semantics changed"));
            }
            views
                .try_reserve_exact(layers.len())
                .map_err(|_| invalid("decoder view allocation"))?;
            for l in *layers {
                views.push(View {
                    input: l.attention_norm,
                    post: l.ffn_norm,
                    mlp: [l.gate, l.up, l.down],
                    activation: l.activation,
                    sub: l.ffn_sub_norm,
                    sub_eps: l.ffn_sub_eps,
                    mixer: ViewMixer::Attention {
                        matrices: [l.query, l.key, l.value, l.attention_output],
                        vectors: [
                            l.attention_sub_norm,
                            l.query_bias,
                            l.key_bias,
                            l.value_bias,
                            l.query_norm,
                            l.key_norm,
                        ],
                        gated: false,
                    },
                });
            }
        }
        DecoderLayers::Hybrid(layers) => {
            if !s.zero_centered_norm || !s.qwen_rope_f32 {
                return Err(invalid("hybrid numerical semantics missing"));
            }
            views
                .try_reserve_exact(layers.len())
                .map_err(|_| invalid("hybrid view allocation"))?;
            for l in *layers {
                views.push(View {
                    input: l.input_norm,
                    post: l.post_norm,
                    mlp: [l.gate, l.up, l.down],
                    activation: DecoderActivation::SwiGlu,
                    sub: &[],
                    sub_eps: g.rms_eps,
                    mixer: match &l.mixer {
                        DecoderMixer::FullAttention(a) => ViewMixer::Attention {
                            matrices: [a.query_gate, a.key, a.value, a.output],
                            vectors: [&[], &[], &[], &[], a.query_norm, a.key_norm],
                            gated: true,
                        },
                        DecoderMixer::DeltaNet(d) => ViewMixer::Delta(d),
                        _ => return Err(BackendError::UnsupportedTensor),
                    },
                });
            }
        }
        _ => return Err(BackendError::UnsupportedTensor),
    }
    if views.is_empty() {
        return Err(invalid("decoder has no layers"));
    }
    let m = limits.max_segment_tokens;
    let mut unique = Vec::<usize>::new();
    unique
        .try_reserve_exact(add(
            add(mul(views.len(), 8)?, 2)?,
            graph.retained_matrices.len(),
        )?)
        .map_err(|_| invalid("decoder inventory allocation"))?;
    let mut matrix_bytes = 0;
    let mut rotate_width = 0;
    let mut check = |handle: &Arc<dyn DeviceBuffer>,
                     rows: usize,
                     cols: usize,
                     batch: usize,
                     executed: bool|
     -> Result<(), BackendError> {
        let tensor = CudaTensor::checked(&backend, handle.as_ref())?;
        if tensor.dimensions() != (rows, cols) {
            return Err(invalid("decoder matrix geometry mismatch"));
        }
        CudaBackend::check_grad_launch_bounds(batch, rows, cols)?;
        let identity = tensor as *const CudaTensor as usize;
        if !unique.contains(&identity) {
            matrix_bytes = add(matrix_bytes, tensor.len_bytes())?;
            unique.push(identity);
        }
        if executed && !matches!(tensor.basis(), Basis::Identity) {
            rotate_width = rotate_width.max(cols);
        }
        Ok(())
    };
    check(graph.embedding, g.vocabulary, g.hidden, 1, false)?;
    check(graph.head, g.vocabulary, g.hidden, 1, true)?;
    for handle in graph.retained_matrices {
        let (rows, cols) = CudaTensor::checked(&backend, handle.as_ref())?.dimensions();
        check(handle, rows, cols, 1, false)?;
    }
    let mut preserved = vector(graph.final_norm, g.hidden, false)?;
    let mut state_bytes = 0;
    let mut has_attention = false;
    let mut has_delta = false;
    let mut aux_a = 0;
    let mut aux_b = 0;
    let mut aux_c = 0;
    let mut aux_d = 0;
    let mut gate_width = 0;
    let mut conv_width = 0;
    let mut mixed_width = 0;
    let mut widest = g.vocabulary.max(g.hidden).max(g.intermediate);
    for l in &views {
        preserved = add(preserved, vector(l.input, g.hidden, false)?)?;
        preserved = add(preserved, vector(l.post, g.hidden, false)?)?;
        preserved = add(preserved, vector(l.sub, g.intermediate, true)?)?;
        if !matches!(
            l.activation,
            DecoderActivation::SwiGlu | DecoderActivation::Relu2
        ) || (l.activation == DecoderActivation::SwiGlu && !l.sub.is_empty())
            || (!l.sub.is_empty() && (!l.sub_eps.is_finite() || l.sub_eps <= 0.))
        {
            return Err(invalid("unsupported MLP semantics"));
        }
        for (handle, (rows, cols)) in l.mlp.iter().zip([
            (g.intermediate, g.hidden),
            (g.intermediate, g.hidden),
            (g.hidden, g.intermediate),
        ]) {
            check(handle, rows, cols, m, true)?;
        }
        match &l.mixer {
            ViewMixer::Attention {
                matrices,
                vectors,
                gated,
            } => {
                has_attention = true;
                mixed_width = mixed_width.max(q);
                widest = widest.max(q).max(kv);
                let query_width = if *gated { mul(2, q)? } else { q };
                widest = widest.max(query_width);
                for (handle, (rows, cols)) in matrices.iter().zip([
                    (query_width, g.hidden),
                    (kv, g.hidden),
                    (kv, g.hidden),
                    (g.hidden, q),
                ]) {
                    check(handle, rows, cols, m, true)?;
                }
                for (i, (values, width)) in vectors
                    .iter()
                    .zip([q, q, kv, kv, g.head_dim, g.head_dim])
                    .enumerate()
                {
                    preserved = add(preserved, vector(values, width, !*gated || i < 4)?)?;
                }
                state_bytes = add(state_bytes, mul(mul(mul(2, limits.max_context)?, kv)?, 4)?)?;
                if *gated {
                    aux_a = aux_a.max(query_width);
                    gate_width = gate_width.max(q);
                }
            }
            ViewMixer::Delta(d) => {
                has_delta = true;
                let (cv, value) = d.geometry.widths()?;
                let heads = d.geometry.recurrence.value_heads;
                if d.geometry.rms_eps.to_bits() != g.rms_eps.to_bits() {
                    return Err(invalid("DeltaNet epsilon differs from graph"));
                }
                state_bytes = add(
                    state_bytes,
                    delta_mixer::State::payload_bytes(d.geometry, d.weights, limits)?,
                )?;
                for (handle, (rows, cols)) in
                    [d.qkv, d.z, d.beta, d.decay, d.output].into_iter().zip([
                        (cv, g.hidden),
                        (value, g.hidden),
                        (heads, g.hidden),
                        (heads, g.hidden),
                        (g.hidden, value),
                    ])
                {
                    check(handle, rows, cols, m, true)?;
                }
                aux_a = aux_a.max(cv);
                aux_b = aux_b.max(value);
                aux_c = aux_c.max(heads);
                aux_d = aux_d.max(heads);
                conv_width = conv_width.max(cv);
                mixed_width = mixed_width.max(value);
                widest = widest.max(cv).max(value).max(heads);
            }
        }
    }
    let q_width = if has_attention { q } else { 0 };
    let kv_width = if has_attention { kv } else { 0 };
    let rope_width = if has_attention { s.rotary_dim } else { 0 };
    let widths = [
        g.hidden,
        g.hidden,
        g.hidden,
        q_width,
        kv_width,
        kv_width,
        mixed_width,
        g.intermediate,
        g.intermediate,
        aux_a,
        aux_b,
        aux_c,
        aux_d,
        gate_width,
        conv_width,
        rotate_width,
        rope_width,
    ];
    let mut workspace = 0;
    for width in widths {
        let n = mul(m, width)?;
        u32::try_from(n).map_err(|_| invalid("decoder workspace exceeds u32"))?;
        workspace = add(workspace, n)?;
    }
    if has_attention {
        u32::try_from(mul(m, g.query_heads)?).map_err(|_| invalid("attention grid exceeds u32"))?;
    }
    workspace = add(workspace, add(add(g.hidden, g.vocabulary)?, widest)?)?;
    let payload = add(
        add(matrix_bytes, state_bytes)?,
        add(mul(add(preserved, workspace)?, 4)?, mul(m, 4)?)?,
    )?;
    if payload > budget {
        return Err(BackendError::TensorPayloadBudgetExceeded {
            requested: payload as u64,
            limit: budget as u64,
        });
    }
    // All source/ownership/geometry/vector/state/workspace admission passed before device allocation.
    let alloc = |n| {
        backend
            .stream
            .alloc_zeros::<f32>(n)
            .map_err(|e| alloc_or_backend("common decoder workspace", &e, n * 4))
    };
    let optional = |width| {
        if width == 0 {
            Ok(None)
        } else {
            alloc(m * width).map(Some)
        }
    };
    let upload = |v: &[f32]| {
        backend.stream.clone_htod(v).map_err(|e| {
            alloc_or_backend("decoder preserved vector", &e, core::mem::size_of_val(v))
        })
    };
    let mut ones = zeros::<f32>(widest)?;
    ones.fill(1.);
    let w = Workspace {
        hidden: alloc(m * g.hidden)?,
        normed: alloc(m * g.hidden)?,
        projection: alloc(m * g.hidden)?,
        query: optional(q_width)?,
        key: optional(kv_width)?,
        value: optional(kv_width)?,
        mixed: alloc(m * mixed_width)?,
        gate: alloc(m * g.intermediate)?,
        up: alloc(m * g.intermediate)?,
        a: optional(aux_a)?,
        b: optional(aux_b)?,
        c: optional(aux_c)?,
        d: optional(aux_d)?,
        attention_gate: optional(gate_width)?,
        convolved: optional(conv_width)?,
        rotate: optional(rotate_width)?,
        rope: optional(rope_width)?,
        last: alloc(g.hidden)?,
        logits: alloc(g.vocabulary)?,
        ones: upload(&ones)?,
        ids: backend
            .stream
            .alloc_zeros::<u32>(m)
            .map_err(|e| alloc_or_backend("decoder token IDs", &e, m * 4))?,
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(views.len())
        .map_err(|_| invalid("decoder layer allocation"))?;
    for l in views {
        let mixer = match l.mixer {
            ViewMixer::Attention {
                matrices,
                vectors,
                gated,
            } => {
                let mut owned = Vec::new();
                owned
                    .try_reserve_exact(6)
                    .map_err(|_| invalid("attention vector allocation"))?;
                for values in vectors {
                    owned.push(if values.is_empty() {
                        None
                    } else {
                        Some(upload(values)?)
                    });
                }
                Mixer::Attention {
                    matrices: matrices.map(Arc::clone),
                    vectors: owned,
                    gated,
                    cache: Box::new(attention::Cache::new(
                        &backend,
                        attention,
                        AttentionLimits {
                            max_context: limits.max_context,
                            max_segment_tokens: m,
                            max_device_payload_bytes: usize::MAX,
                        },
                    )?),
                }
            }
            ViewMixer::Delta(d) => Mixer::Delta {
                matrices: [d.qkv, d.z, d.beta, d.decay, d.output].map(Arc::clone),
                state: Box::new(delta_mixer::State::new(
                    &backend, d.geometry, d.weights, limits,
                )?),
            },
        };
        layers.push(Layer {
            input: upload(l.input)?,
            post: upload(l.post)?,
            mlp: l.mlp.map(Arc::clone),
            activation: l.activation,
            sub: if l.sub.is_empty() {
                None
            } else {
                Some(upload(l.sub)?)
            },
            sub_eps: l.sub_eps,
            mixer,
            staged: None,
        });
    }
    let mut retained = Vec::new();
    retained
        .try_reserve_exact(graph.retained_matrices.len())
        .map_err(|_| invalid("retained matrix inventory"))?;
    retained.extend(graph.retained_matrices.iter().map(|h| Arc::clone(h)));
    let mut frequencies = zeros::<f64>(rope_width / 2)?;
    for (j, f) in frequencies.iter_mut().enumerate() {
        *f = if s.qwen_rope_f32 {
            f64::from(1. / g.rope_theta.powf((2 * j) as f32 / s.rotary_dim as f32))
        } else {
            f64::from(g.rope_theta).powf(-2. * j as f64 * (1. / g.head_dim as f64))
        };
    }
    Ok(Some(Box::new(Decoder {
        final_norm: upload(graph.final_norm)?,
        rope_host: zeros(m * rope_width)?,
        embedding: Arc::clone(graph.embedding),
        head: Arc::clone(graph.head),
        _retained: retained,
        backend,
        geometry: g,
        semantics: s,
        limits,
        layers,
        workspace: w,
        frequencies,
        cursor: 0,
        payload,
        has_delta,
    })))
}

fn norm(
    b: &CudaBackend,
    x: &mut CudaSlice<f32>,
    weight: Option<&CudaSlice<f32>>,
    rows: usize,
    cols: usize,
    eps: f32,
    zero: bool,
) -> Result<(), BackendError> {
    let Some(weight) = weight else {
        return Ok(());
    };
    let cols = cols as u32;
    let zero = u32::from(zero);
    let mut launch = b.stream.launch_builder(&b.func_decoder_norm);
    launch.arg(x).arg(weight).arg(&cols).arg(&eps).arg(&zero);
    #[allow(unsafe_code)]
    // SAFETY: admitted row/vector extents bound one canonical reduction block per row.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| driver_err("common decoder norm", &e))?;
    Ok(())
}
fn bias(
    b: &CudaBackend,
    x: &mut CudaSlice<f32>,
    weight: Option<&CudaSlice<f32>>,
    rows: usize,
    cols: usize,
) -> Result<(), BackendError> {
    let Some(weight) = weight else {
        return Ok(());
    };
    let count = (rows * cols) as u32;
    let cols = cols as u32;
    let mut launch = b.stream.launch_builder(&b.func_decoder_bias);
    launch.arg(x).arg(weight).arg(&cols).arg(&count);
    #[allow(unsafe_code)]
    // SAFETY: admitted vector width and active rows bound every channel index.
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("common decoder bias", &e))?;
    Ok(())
}
fn rope(
    b: &CudaBackend,
    x: &mut CudaSlice<f32>,
    table: &CudaSlice<f32>,
    n: usize,
    heads: usize,
    dim: usize,
    rotary: usize,
) -> Result<(), BackendError> {
    let pairs = (n * heads * (rotary / 2)) as u32;
    let heads = heads as u32;
    let dim = dim as u32;
    let rotary = rotary as u32;
    let mut launch = b.stream.launch_builder(&b.func_decoder_rope);
    launch
        .arg(x)
        .arg(table)
        .arg(&heads)
        .arg(&dim)
        .arg(&rotary)
        .arg(&pairs);
    #[allow(unsafe_code)]
    // SAFETY: even admitted rotary prefix and bounded active rows prove every disjoint pair.
    unsafe { launch.launch(LaunchConfig::for_num_elems(pairs)) }
        .map_err(|e| driver_err("common decoder RoPE", &e))?;
    Ok(())
}
fn residual(
    b: &CudaBackend,
    x: &mut CudaSlice<f32>,
    delta: &CudaSlice<f32>,
    count: usize,
) -> Result<(), BackendError> {
    let count = count as u32;
    let mut launch = b.stream.launch_builder(&b.func_decoder_add);
    launch.arg(x).arg(delta).arg(&count);
    #[allow(unsafe_code)]
    // SAFETY: residual/projection allocations share the admitted hidden-row extent.
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("common decoder residual", &e))?;
    Ok(())
}
fn project(
    b: &CudaBackend,
    handle: &Arc<dyn DeviceBuffer>,
    input: &CudaSlice<f32>,
    rotate: &mut Option<CudaSlice<f32>>,
    ones: &CudaSlice<f32>,
    out: &mut CudaSlice<f32>,
    n: usize,
) -> Result<(), BackendError> {
    CudaTensor::checked(b, handle.as_ref())?.project_device(b, input, rotate.as_mut(), ones, out, n)
}
impl Decoder {
    fn run(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        if cancelled() {
            return Ok(None);
        }
        let end = add(self.cursor, tokens.len())?;
        if tokens.is_empty()
            || tokens.len() > self.limits.max_segment_tokens
            || end > self.limits.max_context
            || positions.len() != tokens.len()
            || !positions.iter().copied().eq(self.cursor..end)
            || tokens
                .iter()
                .any(|&t| t as usize >= self.geometry.vocabulary)
            || self.layers.iter().any(|l| l.mixer.len() != self.cursor)
            || (self.has_delta && self.cursor != 0 && tokens.len() != 1)
        {
            return Err(invalid("common decoder invalid segment or cache cursor"));
        }
        let g = self.geometry;
        let s = self.semantics;
        let n = tokens.len();
        let q = g.query_heads * g.head_dim;
        let kv = g.kv_heads * g.head_dim;
        let hc = n * g.hidden;
        let b = &self.backend;
        let w = &mut self.workspace;
        let mut result = zeros::<f32>(g.vocabulary)?;
        for (row, &position) in positions.iter().enumerate() {
            for (j, &frequency) in self.frequencies.iter().enumerate() {
                let (sine, cosine) = if s.qwen_rope_f32 {
                    let (s, c) = ((position as f32) * (frequency as f32)).sin_cos();
                    (f64::from(s), f64::from(c))
                } else {
                    (position as f64 * frequency).sin_cos()
                };
                self.rope_host[row * s.rotary_dim + j] = cosine as f32;
                self.rope_host[row * s.rotary_dim + j + s.rotary_dim / 2] = sine as f32;
            }
        }
        b.stream
            .memcpy_htod(tokens, &mut w.ids.slice_mut(..n))
            .map_err(|e| driver_err("common decoder tokens", &e))?;
        if let Some(table) = &mut w.rope {
            b.stream
                .memcpy_htod(
                    &self.rope_host[..n * s.rotary_dim],
                    &mut table.slice_mut(..n * s.rotary_dim),
                )
                .map_err(|e| driver_err("common decoder position metadata", &e))?;
        }
        CudaTensor::checked(b, self.embedding.as_ref())?.gather_device(
            b,
            &w.ids,
            &mut w.hidden,
            n,
        )?;
        for l in &mut self.layers {
            if cancelled() {
                return Ok(None);
            }
            b.stream
                .memcpy_dtod(&w.hidden.slice(..hc), &mut w.normed.slice_mut(..hc))
                .map_err(|e| driver_err("decoder input norm staging", &e))?;
            norm(
                b,
                &mut w.normed,
                Some(&l.input),
                n,
                g.hidden,
                g.rms_eps,
                s.zero_centered_norm,
            )?;
            match &mut l.mixer {
                Mixer::Attention {
                    matrices,
                    vectors,
                    gated,
                    cache,
                } => {
                    if *gated {
                        project(
                            b,
                            &matrices[0],
                            &w.normed,
                            &mut w.rotate,
                            &w.ones,
                            required(&mut w.a)?,
                            n,
                        )?;
                        let dim = g.head_dim as u32;
                        let count = (n * q) as u32;
                        let mut launch = b.stream.launch_builder(&b.func_qwen_query_gate);
                        launch
                            .arg(
                                w.a.as_ref()
                                    .ok_or_else(|| invalid("fused query workspace"))?,
                            )
                            .arg(required(&mut w.query)?)
                            .arg(required(&mut w.attention_gate)?)
                            .arg(&dim)
                            .arg(&count);
                        #[allow(unsafe_code)]
                        // SAFETY: checked 2Q per-head fused geometry and Q output extents bound each split.
                        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                            .map_err(|e| driver_err("Qwen query gate split", &e))?;
                    } else {
                        project(
                            b,
                            &matrices[0],
                            &w.normed,
                            &mut w.rotate,
                            &w.ones,
                            required(&mut w.query)?,
                            n,
                        )?;
                    }
                    project(
                        b,
                        &matrices[1],
                        &w.normed,
                        &mut w.rotate,
                        &w.ones,
                        required(&mut w.key)?,
                        n,
                    )?;
                    project(
                        b,
                        &matrices[2],
                        &w.normed,
                        &mut w.rotate,
                        &w.ones,
                        required(&mut w.value)?,
                        n,
                    )?;
                    bias(b, required(&mut w.query)?, vectors[1].as_ref(), n, q)?;
                    bias(b, required(&mut w.key)?, vectors[2].as_ref(), n, kv)?;
                    bias(b, required(&mut w.value)?, vectors[3].as_ref(), n, kv)?;
                    norm(
                        b,
                        required(&mut w.query)?,
                        vectors[4].as_ref(),
                        n * g.query_heads,
                        g.head_dim,
                        g.rms_eps,
                        s.zero_centered_norm,
                    )?;
                    norm(
                        b,
                        required(&mut w.key)?,
                        vectors[5].as_ref(),
                        n * g.kv_heads,
                        g.head_dim,
                        g.rms_eps,
                        s.zero_centered_norm,
                    )?;
                    let table = w.rope.as_ref().ok_or_else(|| invalid("RoPE workspace"))?;
                    rope(
                        b,
                        required(&mut w.query)?,
                        table,
                        n,
                        g.query_heads,
                        g.head_dim,
                        s.rotary_dim,
                    )?;
                    rope(
                        b,
                        required(&mut w.key)?,
                        table,
                        n,
                        g.kv_heads,
                        g.head_dim,
                        s.rotary_dim,
                    )?;
                    cache.stage_device(
                        w.query.as_ref().ok_or_else(|| invalid("query workspace"))?,
                        w.key.as_ref().ok_or_else(|| invalid("key workspace"))?,
                        w.value.as_ref().ok_or_else(|| invalid("value workspace"))?,
                        &mut w.mixed,
                        n,
                        1. / (g.head_dim as f32).sqrt(),
                    )?;
                    if *gated {
                        let count = (n * q) as u32;
                        let mut launch = b.stream.launch_builder(&b.func_qwen_attention_gate);
                        launch
                            .arg(&mut w.mixed)
                            .arg(
                                w.attention_gate
                                    .as_ref()
                                    .ok_or_else(|| invalid("gate workspace"))?,
                            )
                            .arg(&count);
                        #[allow(unsafe_code)]
                        // SAFETY: admitted Q active rows bound attention and per-head gate arrays.
                        unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                            .map_err(|e| driver_err("Qwen attention gate", &e))?;
                    }
                    norm(
                        b,
                        &mut w.mixed,
                        vectors[0].as_ref(),
                        n,
                        q,
                        g.rms_eps,
                        s.zero_centered_norm,
                    )?;
                    project(
                        b,
                        &matrices[3],
                        &w.mixed,
                        &mut w.rotate,
                        &w.ones,
                        &mut w.projection,
                        n,
                    )?;
                }
                Mixer::Delta {
                    matrices, state, ..
                } => {
                    project(
                        b,
                        &matrices[0],
                        &w.normed,
                        &mut w.rotate,
                        &w.ones,
                        required(&mut w.a)?,
                        n,
                    )?;
                    project(
                        b,
                        &matrices[1],
                        &w.normed,
                        &mut w.rotate,
                        &w.ones,
                        required(&mut w.b)?,
                        n,
                    )?;
                    project(
                        b,
                        &matrices[2],
                        &w.normed,
                        &mut w.rotate,
                        &w.ones,
                        required(&mut w.c)?,
                        n,
                    )?;
                    project(
                        b,
                        &matrices[3],
                        &w.normed,
                        &mut w.rotate,
                        &w.ones,
                        required(&mut w.d)?,
                        n,
                    )?;
                    state.stage_device(
                        [
                            w.a.as_ref().ok_or_else(|| invalid("QKV workspace"))?,
                            w.b.as_ref().ok_or_else(|| invalid("Z workspace"))?,
                            w.c.as_ref().ok_or_else(|| invalid("B workspace"))?,
                            w.d.as_ref().ok_or_else(|| invalid("A workspace"))?,
                        ],
                        required(&mut w.convolved)?,
                        &mut w.mixed,
                        n,
                    )?;
                    project(
                        b,
                        &matrices[4],
                        &w.mixed,
                        &mut w.rotate,
                        &w.ones,
                        &mut w.projection,
                        n,
                    )?;
                }
            }
            residual(b, &mut w.hidden, &w.projection, hc)?;
            b.stream
                .memcpy_dtod(&w.hidden.slice(..hc), &mut w.normed.slice_mut(..hc))
                .map_err(|e| driver_err("decoder post norm staging", &e))?;
            norm(
                b,
                &mut w.normed,
                Some(&l.post),
                n,
                g.hidden,
                g.rms_eps,
                s.zero_centered_norm,
            )?;
            project(
                b,
                &l.mlp[0],
                &w.normed,
                &mut w.rotate,
                &w.ones,
                &mut w.gate,
                n,
            )?;
            project(
                b,
                &l.mlp[1],
                &w.normed,
                &mut w.rotate,
                &w.ones,
                &mut w.up,
                n,
            )?;
            let count = (n * g.intermediate) as u32;
            let function = match l.activation {
                DecoderActivation::SwiGlu => &b.func_semantic_swiglu,
                DecoderActivation::Relu2 => &b.func_decoder_relu2,
                _ => return Err(BackendError::UnsupportedTensor),
            };
            let mut launch = b.stream.launch_builder(function);
            launch.arg(&mut w.gate).arg(&w.up).arg(&count);
            #[allow(unsafe_code)]
            // SAFETY: gate/up arrays carry the admitted active MLP row extent.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| driver_err("common gated MLP", &e))?;
            drop(launch);
            norm(
                b,
                &mut w.gate,
                l.sub.as_ref(),
                n,
                g.intermediate,
                l.sub_eps,
                false,
            )?;
            project(
                b,
                &l.mlp[2],
                &w.gate,
                &mut w.rotate,
                &w.ones,
                &mut w.projection,
                n,
            )?;
            residual(b, &mut w.hidden, &w.projection, hc)?;
            l.staged = Some(end);
        }
        if cancelled() {
            return Ok(None);
        }
        b.stream
            .memcpy_dtod(&w.hidden.slice((n - 1) * g.hidden..hc), &mut w.last)
            .map_err(|e| driver_err("common decoder final row", &e))?;
        norm(
            b,
            &mut w.last,
            Some(&self.final_norm),
            1,
            g.hidden,
            g.rms_eps,
            s.zero_centered_norm,
        )?;
        project(
            b,
            &self.head,
            &w.last,
            &mut w.rotate,
            &w.ones,
            &mut w.logits,
            1,
        )?;
        b.stream
            .memcpy_dtoh(&w.logits, &mut result)
            .map_err(|e| driver_err("common decoder final logits", &e))?;
        if result.iter().any(|v| !v.is_finite()) {
            return Err(invalid("resident decoder non-finite final logits"));
        }
        if cancelled() {
            return Ok(None);
        }
        if self
            .layers
            .iter()
            .any(|l| l.staged != Some(end) || !l.mixer.ready(end))
        {
            return Err(invalid("common graph staging extent mismatch"));
        }
        for l in &mut self.layers {
            l.mixer.commit(end);
            l.staged = None;
        }
        self.cursor = end;
        Ok(Some(result))
    }
}
impl ResidentDecoder for Decoder {
    fn limits(&self) -> DecoderLimits {
        self.limits
    }
    fn payload_bytes(&self) -> usize {
        self.payload
    }
    fn len(&self) -> usize {
        self.cursor
    }
    fn reset(&mut self) {
        self.cursor = 0;
        for l in &mut self.layers {
            l.mixer.reset();
            l.staged = None;
        }
    }
    fn forward(
        &mut self,
        tokens: &[u32],
        positions: &[usize],
        cancelled: &dyn Fn() -> bool,
    ) -> Result<Option<Vec<f32>>, BackendError> {
        let result = self.run(tokens, positions, cancelled);
        if !matches!(&result, Ok(Some(_))) {
            for l in &mut self.layers {
                l.mixer.abort();
                l.staged = None;
            }
        }
        result
    }
}
