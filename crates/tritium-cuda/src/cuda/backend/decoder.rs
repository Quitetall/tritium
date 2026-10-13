//! Whole standard decoder execution over admitted shared native tensor handles.
use super::semantic::CudaTensor;
use super::*;
use tritium_spec::{
    AttentionGeometry, AttentionLimits, DecoderActivation, DecoderGeometry, DecoderGraph,
    DecoderLayer, DecoderLimits, ResidentDecoder,
};

struct Layer {
    matrices: [Arc<dyn DeviceBuffer>; 7],
    vectors: Vec<Option<CudaSlice<f32>>>,
    activation: DecoderActivation,
    sub_eps: f32,
    cache: attention::Cache,
}
struct Workspace {
    hidden: CudaSlice<f32>,
    normed: CudaSlice<f32>,
    query: CudaSlice<f32>,
    key: CudaSlice<f32>,
    value: CudaSlice<f32>,
    attention: CudaSlice<f32>,
    projection: CudaSlice<f32>,
    gate: CudaSlice<f32>,
    up: CudaSlice<f32>,
    rotated_hidden: CudaSlice<f32>,
    rotated_query: CudaSlice<f32>,
    rotated_ffn: CudaSlice<f32>,
    rope: CudaSlice<f32>,
    last: CudaSlice<f32>,
    logits: CudaSlice<f32>,
    ones: CudaSlice<f32>,
    ids: CudaSlice<u32>,
}
struct Decoder {
    backend: Arc<CudaBackend>,
    geometry: DecoderGeometry,
    limits: DecoderLimits,
    embedding: Arc<dyn DeviceBuffer>,
    head: Arc<dyn DeviceBuffer>,
    final_norm: CudaSlice<f32>,
    layers: Vec<Layer>,
    workspace: Workspace,
    frequencies: Vec<f64>,
    rope_host: Vec<f32>,
    cursor: usize,
    payload: usize,
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
fn mul(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_mul(b)
        .ok_or_else(|| invalid("resident decoder extent overflow"))
}
fn add(a: usize, b: usize) -> Result<usize, BackendError> {
    a.checked_add(b)
        .ok_or_else(|| invalid("resident decoder payload overflow"))
}
fn host_zeroed<T: Clone + Default>(len: usize) -> Result<Vec<T>, BackendError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(len)
        .map_err(|_| BackendError::OutOfMemory {
            requested: len.saturating_mul(core::mem::size_of::<T>()),
        })?;
    values.resize(len, T::default());
    Ok(values)
}
fn matrices<'a>(layer: &'a DecoderLayer<'a>) -> [&'a Arc<dyn DeviceBuffer>; 7] {
    [
        layer.query,
        layer.key,
        layer.value,
        layer.attention_output,
        layer.gate,
        layer.up,
        layer.down,
    ]
}
fn vectors<'a>(layer: &'a DecoderLayer<'a>) -> [&'a [f32]; 9] {
    [
        layer.attention_norm,
        layer.ffn_norm,
        layer.attention_sub_norm,
        layer.query_bias,
        layer.key_bias,
        layer.value_bias,
        layer.query_norm,
        layer.key_norm,
        layer.ffn_sub_norm,
    ]
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
    let a = AttentionGeometry {
        query_heads: g.query_heads,
        kv_heads: g.kv_heads,
        head_dim: g.head_dim,
    };
    let (q, kv) = a.widths()?;
    if g.hidden == 0
        || g.intermediate == 0
        || g.vocabulary == 0
        || graph.layers.is_empty()
        || !g.head_dim.is_multiple_of(2)
        || g.head_dim > 1024
        || !g.rms_eps.is_finite()
        || g.rms_eps <= 0.
        || !g.rope_theta.is_finite()
        || g.rope_theta <= 0.
        || limits.max_context == 0
        || limits.max_segment_tokens == 0
        || limits.max_segment_tokens > limits.max_context
        || budget == 0
        || u32::try_from(limits.max_context).is_err()
    {
        return Err(invalid("unsupported resident decoder geometry or limits"));
    }
    let m = limits.max_segment_tokens;
    for width in [g.hidden, q, kv, g.intermediate, g.head_dim] {
        u32::try_from(mul(m, width)?)
            .map_err(|_| invalid("resident decoder workspace exceeds u32"))?;
    }
    u32::try_from(mul(m, g.query_heads)?)
        .map_err(|_| invalid("resident attention grid exceeds u32"))?;
    let mut unique: Vec<Arc<dyn DeviceBuffer>> = Vec::new();
    unique
        .try_reserve_exact(add(mul(graph.layers.len(), 7)?, 2)?)
        .map_err(|_| invalid("resident weight inventory allocation"))?;
    let mut weight_bytes = 0;
    let mut check_matrix =
        |handle: &Arc<dyn DeviceBuffer>, rows, cols, batch| -> Result<(), BackendError> {
            if CudaTensor::checked(&backend, handle.as_ref())?.dimensions() != (rows, cols) {
                return Err(invalid("resident decoder matrix geometry mismatch"));
            }
            CudaBackend::check_grad_launch_bounds(batch, rows, cols)?;
            if !unique.iter().any(|prior| Arc::ptr_eq(prior, handle)) {
                weight_bytes = add(weight_bytes, handle.len_bytes())?;
                unique.push(Arc::clone(handle));
            }
            Ok(())
        };
    check_matrix(graph.embedding, g.vocabulary, g.hidden, 1)?;
    check_matrix(graph.head, g.vocabulary, g.hidden, 1)?;
    let mut vector_elements = g.hidden;
    if graph.final_norm.len() != g.hidden || graph.final_norm.iter().any(|v| !v.is_finite()) {
        return Err(invalid("resident final norm geometry or values"));
    }
    for layer in graph.layers {
        for (handle, (rows, cols)) in matrices(layer).into_iter().zip([
            (q, g.hidden),
            (kv, g.hidden),
            (kv, g.hidden),
            (g.hidden, q),
            (g.intermediate, g.hidden),
            (g.intermediate, g.hidden),
            (g.hidden, g.intermediate),
        ]) {
            check_matrix(handle, rows, cols, m)?;
        }
        if !matches!(
            layer.activation,
            DecoderActivation::SwiGlu | DecoderActivation::Relu2
        ) || (layer.activation == DecoderActivation::SwiGlu && !layer.ffn_sub_norm.is_empty())
            || (!layer.ffn_sub_norm.is_empty()
                && (!layer.ffn_sub_eps.is_finite() || layer.ffn_sub_eps <= 0.))
        {
            return Err(invalid("unsupported resident MLP settings"));
        }
        for (i, (values, width)) in vectors(layer)
            .into_iter()
            .zip([
                g.hidden,
                g.hidden,
                q,
                q,
                kv,
                kv,
                g.head_dim,
                g.head_dim,
                g.intermediate,
            ])
            .enumerate()
        {
            if (i < 2 || !values.is_empty()) && values.len() != width
                || values.iter().any(|v| !v.is_finite())
            {
                return Err(invalid("resident layer vector geometry or values"));
            }
            vector_elements = add(vector_elements, values.len())?;
        }
    }
    // Fixed maximum-segment workspace plus all full-capacity KV arrays.
    let widest = [g.hidden, q, kv, g.intermediate, g.vocabulary]
        .into_iter()
        .max()
        .unwrap_or(0);
    let row_workspace = add(
        add(mul(4, g.hidden)?, mul(3, q)?)?,
        add(add(mul(2, kv)?, mul(3, g.intermediate)?)?, g.head_dim)?,
    )?;
    let workspace_elements = add(
        mul(m, row_workspace)?,
        add(add(g.hidden, g.vocabulary)?, widest)?,
    )?;
    let cache_elements = mul(mul(mul(2, limits.max_context)?, kv)?, graph.layers.len())?;
    let payload = add(
        weight_bytes,
        add(
            mul(
                add(add(vector_elements, workspace_elements)?, cache_elements)?,
                4,
            )?,
            mul(m, 4)?,
        )?,
    )?;
    if payload > budget {
        return Err(BackendError::TensorPayloadBudgetExceeded {
            requested: payload as u64,
            limit: budget as u64,
        });
    }
    // No device allocation occurs until the entire graph/ownership/payload plan passes.
    let alloc = |len| {
        backend
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| alloc_or_backend("resident decoder workspace", &e, len * 4))
    };
    let upload = |values: &[f32]| {
        backend.stream.clone_htod(values).map_err(|e| {
            alloc_or_backend(
                "resident decoder vector",
                &e,
                core::mem::size_of_val(values),
            )
        })
    };
    let mut ones = host_zeroed::<f32>(widest)?;
    ones.fill(1.);
    let ones = upload(&ones)?;
    let workspace = Workspace {
        hidden: alloc(m * g.hidden)?,
        normed: alloc(m * g.hidden)?,
        query: alloc(m * q)?,
        key: alloc(m * kv)?,
        value: alloc(m * kv)?,
        attention: alloc(m * q)?,
        projection: alloc(m * g.hidden)?,
        gate: alloc(m * g.intermediate)?,
        up: alloc(m * g.intermediate)?,
        rotated_hidden: alloc(m * g.hidden)?,
        rotated_query: alloc(m * q)?,
        rotated_ffn: alloc(m * g.intermediate)?,
        rope: alloc(m * g.head_dim)?,
        last: alloc(g.hidden)?,
        logits: alloc(g.vocabulary)?,
        ones,
        ids: backend
            .stream
            .alloc_zeros::<u32>(m)
            .map_err(|e| alloc_or_backend("resident token IDs", &e, m * 4))?,
    };
    let mut layers = Vec::new();
    layers
        .try_reserve_exact(graph.layers.len())
        .map_err(|_| invalid("resident layer allocation"))?;
    let attention_limits = AttentionLimits {
        max_context: limits.max_context,
        max_segment_tokens: m,
        max_device_payload_bytes: usize::MAX,
    };
    for layer in graph.layers {
        let mut norms = Vec::new();
        norms
            .try_reserve_exact(9)
            .map_err(|_| invalid("resident vector allocation"))?;
        for values in vectors(layer) {
            norms.push(if values.is_empty() {
                None
            } else {
                Some(upload(values)?)
            });
        }
        layers.push(Layer {
            matrices: matrices(layer).map(Arc::clone),
            vectors: norms,
            activation: layer.activation,
            sub_eps: layer.ffn_sub_eps,
            cache: attention::Cache::new(&backend, a, attention_limits)?,
        });
    }
    let mut frequencies = host_zeroed::<f64>(g.head_dim / 2)?;
    let inv = 1. / g.head_dim as f64;
    for (j, frequency) in frequencies.iter_mut().enumerate() {
        *frequency = f64::from(g.rope_theta).powf(-2. * j as f64 * inv);
    }
    let decoder = Decoder {
        final_norm: upload(graph.final_norm)?,
        rope_host: host_zeroed(m * g.head_dim)?,
        backend,
        geometry: g,
        limits,
        embedding: Arc::clone(graph.embedding),
        head: Arc::clone(graph.head),
        layers,
        workspace,
        frequencies,
        cursor: 0,
        payload,
    };
    Ok(Some(Box::new(decoder)))
}

fn norm(
    backend: &CudaBackend,
    data: &mut CudaSlice<f32>,
    weights: Option<&CudaSlice<f32>>,
    rows: usize,
    cols: usize,
    eps: f32,
) -> Result<(), BackendError> {
    let Some(weights) = weights else {
        return Ok(());
    };
    let cols = cols as u32;
    let mut launch = backend.stream.launch_builder(&backend.func_decoder_norm);
    launch.arg(data).arg(weights).arg(&cols).arg(&eps);
    #[allow(unsafe_code)]
    // SAFETY: construction validates vector width; active row count fits allocated data.
    unsafe {
        launch.launch(LaunchConfig {
            grid_dim: (rows as u32, 1, 1),
            block_dim: (256, 1, 1),
            shared_mem_bytes: 0,
        })
    }
    .map_err(|e| driver_err("resident decoder RMSNorm", &e))?;
    Ok(())
}
fn bias(
    backend: &CudaBackend,
    data: &mut CudaSlice<f32>,
    weights: Option<&CudaSlice<f32>>,
    rows: usize,
    cols: usize,
) -> Result<(), BackendError> {
    let Some(weights) = weights else {
        return Ok(());
    };
    let count = (rows * cols) as u32;
    let cols = cols as u32;
    let mut launch = backend.stream.launch_builder(&backend.func_decoder_bias);
    launch.arg(data).arg(weights).arg(&cols).arg(&count);
    #[allow(unsafe_code)]
    // SAFETY: admitted active geometry and immutable vector width bound each lane.
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("resident decoder bias", &e))?;
    Ok(())
}
fn rope(
    backend: &CudaBackend,
    data: &mut CudaSlice<f32>,
    table: &CudaSlice<f32>,
    rows: usize,
    heads: usize,
    dim: usize,
) -> Result<(), BackendError> {
    let pairs = (rows * heads * (dim / 2)) as u32;
    let heads = heads as u32;
    let dim = dim as u32;
    let mut launch = backend.stream.launch_builder(&backend.func_decoder_rope);
    launch
        .arg(data)
        .arg(table)
        .arg(&heads)
        .arg(&dim)
        .arg(&pairs);
    #[allow(unsafe_code)]
    // SAFETY: even admitted head width, active row counts and table length bound all pairs.
    unsafe { launch.launch(LaunchConfig::for_num_elems(pairs)) }
        .map_err(|e| driver_err("resident decoder RoPE", &e))?;
    Ok(())
}
fn residual(
    backend: &CudaBackend,
    hidden: &mut CudaSlice<f32>,
    delta: &CudaSlice<f32>,
    count: usize,
) -> Result<(), BackendError> {
    let count = count as u32;
    let mut launch = backend.stream.launch_builder(&backend.func_decoder_add);
    launch.arg(hidden).arg(delta).arg(&count);
    #[allow(unsafe_code)]
    // SAFETY: residual and projection allocations share admitted [max_segment, hidden].
    unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
        .map_err(|e| driver_err("resident decoder residual", &e))?;
    Ok(())
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
        for layer in &mut self.layers {
            layer.cache.publish(0);
        }
    }
    fn forward(
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
                .any(|&token| token as usize >= self.geometry.vocabulary)
            || self
                .layers
                .iter()
                .any(|layer| layer.cache.len() != self.cursor)
        {
            return Err(invalid("resident decoder invalid segment or cache cursor"));
        }
        let g = self.geometry;
        let n = tokens.len();
        let q = g.query_heads * g.head_dim;
        let kv = g.kv_heads * g.head_dim;
        let h_count = n * g.hidden;
        let b = &self.backend;
        let w = &mut self.workspace;
        let mut result = host_zeroed::<f32>(g.vocabulary)?;
        for (row, &position) in positions.iter().enumerate() {
            for (j, &frequency) in self.frequencies.iter().enumerate() {
                let (s, c) = (position as f64 * frequency).sin_cos();
                self.rope_host[row * g.head_dim + j] = c as f32;
                self.rope_host[row * g.head_dim + j + g.head_dim / 2] = s as f32;
            }
        }
        b.stream
            .memcpy_htod(tokens, &mut w.ids.slice_mut(..n))
            .map_err(|e| driver_err("resident tokens", &e))?;
        b.stream
            .memcpy_htod(
                &self.rope_host[..n * g.head_dim],
                &mut w.rope.slice_mut(..n * g.head_dim),
            )
            .map_err(|e| driver_err("resident position metadata", &e))?;
        CudaTensor::checked(b, self.embedding.as_ref())?.gather_device(
            b,
            &w.ids,
            &mut w.hidden,
            n,
        )?;
        for layer in &mut self.layers {
            if cancelled() {
                return Ok(None);
            }
            b.stream
                .memcpy_dtod(
                    &w.hidden.slice(..h_count),
                    &mut w.normed.slice_mut(..h_count),
                )
                .map_err(|e| driver_err("resident input norm staging", &e))?;
            norm(
                b,
                &mut w.normed,
                layer.vectors[0].as_ref(),
                n,
                g.hidden,
                g.rms_eps,
            )?;
            for (index, output) in [(0, &mut w.query), (1, &mut w.key), (2, &mut w.value)] {
                CudaTensor::checked(b, layer.matrices[index].as_ref())?.project_device(
                    b,
                    &w.normed,
                    &mut w.rotated_hidden,
                    &w.ones,
                    output,
                    n,
                )?;
            }
            bias(b, &mut w.query, layer.vectors[3].as_ref(), n, q)?;
            bias(b, &mut w.key, layer.vectors[4].as_ref(), n, kv)?;
            bias(b, &mut w.value, layer.vectors[5].as_ref(), n, kv)?;
            norm(
                b,
                &mut w.query,
                layer.vectors[6].as_ref(),
                n * g.query_heads,
                g.head_dim,
                g.rms_eps,
            )?;
            norm(
                b,
                &mut w.key,
                layer.vectors[7].as_ref(),
                n * g.kv_heads,
                g.head_dim,
                g.rms_eps,
            )?;
            rope(b, &mut w.query, &w.rope, n, g.query_heads, g.head_dim)?;
            rope(b, &mut w.key, &w.rope, n, g.kv_heads, g.head_dim)?;
            layer.cache.stage_device(
                &w.query,
                &w.key,
                &w.value,
                &mut w.attention,
                n,
                1. / (g.head_dim as f32).sqrt(),
            )?;
            norm(
                b,
                &mut w.attention,
                layer.vectors[2].as_ref(),
                n,
                q,
                g.rms_eps,
            )?;
            CudaTensor::checked(b, layer.matrices[3].as_ref())?.project_device(
                b,
                &w.attention,
                &mut w.rotated_query,
                &w.ones,
                &mut w.projection,
                n,
            )?;
            residual(b, &mut w.hidden, &w.projection, h_count)?;
            b.stream
                .memcpy_dtod(
                    &w.hidden.slice(..h_count),
                    &mut w.normed.slice_mut(..h_count),
                )
                .map_err(|e| driver_err("resident MLP norm staging", &e))?;
            norm(
                b,
                &mut w.normed,
                layer.vectors[1].as_ref(),
                n,
                g.hidden,
                g.rms_eps,
            )?;
            for (index, output) in [(4, &mut w.gate), (5, &mut w.up)] {
                CudaTensor::checked(b, layer.matrices[index].as_ref())?.project_device(
                    b,
                    &w.normed,
                    &mut w.rotated_hidden,
                    &w.ones,
                    output,
                    n,
                )?;
            }
            let count = (n * g.intermediate) as u32;
            let function = match layer.activation {
                DecoderActivation::SwiGlu => &b.func_semantic_swiglu,
                DecoderActivation::Relu2 => &b.func_decoder_relu2,
                _ => return Err(invalid("unsupported MLP activation")),
            };
            let mut launch = b.stream.launch_builder(function);
            launch.arg(&mut w.gate).arg(&w.up).arg(&count);
            #[allow(unsafe_code)]
            // SAFETY: fixed admitted gate/up arrays bound the active element count.
            unsafe { launch.launch(LaunchConfig::for_num_elems(count)) }
                .map_err(|e| driver_err("resident gated activation", &e))?;
            drop(launch);
            norm(
                b,
                &mut w.gate,
                layer.vectors[8].as_ref(),
                n,
                g.intermediate,
                layer.sub_eps,
            )?;
            CudaTensor::checked(b, layer.matrices[6].as_ref())?.project_device(
                b,
                &w.gate,
                &mut w.rotated_ffn,
                &w.ones,
                &mut w.projection,
                n,
            )?;
            residual(b, &mut w.hidden, &w.projection, h_count)?;
        }
        if cancelled() {
            return Ok(None);
        }
        b.stream
            .memcpy_dtod(&w.hidden.slice((n - 1) * g.hidden..h_count), &mut w.last)
            .map_err(|e| driver_err("resident final row", &e))?;
        norm(
            b,
            &mut w.last,
            Some(&self.final_norm),
            1,
            g.hidden,
            g.rms_eps,
        )?;
        CudaTensor::checked(b, self.head.as_ref())?.project_device(
            b,
            &w.last,
            &mut w.rotated_hidden,
            &w.ones,
            &mut w.logits,
            1,
        )?;
        b.stream
            .memcpy_dtoh(&w.logits, &mut result)
            .map_err(|e| driver_err("resident final logits", &e))?;
        if result.iter().any(|v| !v.is_finite()) {
            return Err(invalid("resident decoder non-finite final logits"));
        }
        if cancelled() {
            return Ok(None);
        }
        // All layers have staged this extent and final readback surfaced device
        // work errors. Publication cannot fail and never copies committed KV.
        for layer in &mut self.layers {
            layer.cache.publish(end);
        }
        self.cursor = end;
        Ok(Some(result))
    }
}
