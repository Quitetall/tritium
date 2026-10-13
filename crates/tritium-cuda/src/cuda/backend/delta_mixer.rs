//! Complete device-f32 DeltaNet mixer, separate from exact legacy recurrence.
use super::*;
use tritium_spec::{
    DecoderLimits, DeltaMixerGeometry, DeltaMixerSegment, DeltaMixerWeights, ResidentDeltaMixer,
};

#[derive(Debug)]
pub(super) struct State {
    stream: Arc<CudaStream>,
    convolution_function: CudaFunction,
    mixer_function: CudaFunction,
    geometry: DeltaMixerGeometry,
    limits: DecoderLimits,
    convolution: CudaSlice<f32>,
    norm: CudaSlice<f32>,
    dt_bias: CudaSlice<f32>,
    a_log: CudaSlice<f32>,
    conv_current: CudaSlice<f32>,
    conv_staging: CudaSlice<f32>,
    recurrent_current: CudaSlice<f32>,
    recurrent_staging: CudaSlice<f32>,
    cursor: usize,
    pending: Option<usize>,
}
#[derive(Debug)]
struct Mixer {
    state: State,
    convolved: CudaSlice<f32>,
    payload: usize,
    qkv: CudaSlice<f32>,
    z: CudaSlice<f32>,
    beta: CudaSlice<f32>,
    decay: CudaSlice<f32>,
    output: CudaSlice<f32>,
    host_output: Vec<f32>,
}
fn invalid(message: &str) -> BackendError {
    BackendError::InvalidInput(message.into())
}
impl State {
    pub(super) fn len(&self) -> usize {
        self.cursor
    }
    pub(super) fn staged_len(&self) -> Option<usize> {
        self.pending
    }
    pub(super) fn abort(&mut self) {
        self.pending = None;
    }
    pub(super) fn reset(&mut self) {
        self.cursor = 0;
        self.pending = None;
    }
    pub(super) fn payload_bytes(
        geometry: DeltaMixerGeometry,
        weights: DeltaMixerWeights<'_>,
        limits: DecoderLimits,
    ) -> Result<usize, BackendError> {
        let full = geometry.payload_bytes(weights, limits, usize::MAX)?;
        let g = geometry.recurrence;
        let (conv, value) = geometry.widths()?;
        if g.key_head_dim > 1024
            || g.value_head_dim > 1024
            || [
                g.value_heads,
                g.group_size,
                geometry.conv_kernel,
                limits.max_context,
                limits.max_segment_tokens * conv,
                limits.max_segment_tokens * value,
                limits.max_segment_tokens * g.value_heads,
            ]
            .into_iter()
            .any(|n| u32::try_from(n).is_err())
        {
            return Err(BackendError::UnsupportedTensor);
        }
        Ok(full - 2 * limits.max_segment_tokens * (conv + value + g.value_heads) * 4)
    }
    pub(super) fn new(
        backend: &CudaBackend,
        geometry: DeltaMixerGeometry,
        weights: DeltaMixerWeights<'_>,
        limits: DecoderLimits,
    ) -> Result<Self, BackendError> {
        Self::payload_bytes(geometry, weights, limits)?;
        let conv = geometry.widths()?.0 * geometry.conv_kernel;
        let recurrent = geometry.recurrence.elements()?;
        let alloc = |len| {
            backend
                .stream
                .alloc_zeros::<f32>(len)
                .map_err(|e| alloc_or_backend("DeltaNet owned state", &e, len * 4))
        };
        let upload = |values: &[f32]| {
            backend.stream.clone_htod(values).map_err(|e| {
                alloc_or_backend(
                    "DeltaNet preserved coefficients",
                    &e,
                    core::mem::size_of_val(values),
                )
            })
        };
        Ok(Self {
            stream: Arc::clone(&backend.stream),
            convolution_function: backend.func_delta_convolution.clone(),
            mixer_function: backend.func_delta_mixer.clone(),
            geometry,
            limits,
            convolution: upload(weights.convolution)?,
            norm: upload(weights.norm)?,
            dt_bias: upload(weights.dt_bias)?,
            a_log: upload(weights.a_log)?,
            conv_current: alloc(conv)?,
            conv_staging: alloc(conv)?,
            recurrent_current: alloc(recurrent)?,
            recurrent_staging: alloc(recurrent)?,
            cursor: 0,
            pending: None,
        })
    }
    fn extent(&self, tokens: usize) -> Result<usize, BackendError> {
        let end = self
            .cursor
            .checked_add(tokens)
            .ok_or_else(|| invalid("DeltaNet mixer cursor overflow"))?;
        if tokens == 0
            || tokens > self.limits.max_segment_tokens
            || end > self.limits.max_context
            || (self.cursor != 0 && tokens != 1)
        {
            return Err(invalid(
                "DeltaNet mixer segment exceeds limits or cached one-token contract",
            ));
        }
        Ok(end)
    }
    pub(super) fn stage_device(
        &mut self,
        inputs: [&CudaSlice<f32>; 4],
        convolved: &mut CudaSlice<f32>,
        output: &mut CudaSlice<f32>,
        tokens: usize,
    ) -> Result<(), BackendError> {
        let [qkv, z, beta, decay] = inputs;
        self.pending = None;
        let end = self.extent(tokens)?;
        let g = self.geometry.recurrence;
        let (width, value_width) = self.geometry.widths()?;
        for (data, expected) in [
            (qkv, tokens * width),
            (z, tokens * value_width),
            (beta, tokens * g.value_heads),
            (decay, tokens * g.value_heads),
            (&*output, tokens * value_width),
            (&*convolved, tokens * width),
        ] {
            if !Arc::ptr_eq(data.stream(), &self.stream) {
                return Err(invalid("DeltaNet mixer operand belongs to another stream"));
            }
            if data.len() < expected {
                return Err(BackendError::ShapeMismatch {
                    expected,
                    got: data.len(),
                });
            }
        }
        if self.cursor == 0 {
            self.stream
                .memset_zeros(&mut self.conv_staging)
                .map_err(|e| driver_err("initialize DeltaNet convolution", &e))?;
            self.stream
                .memset_zeros(&mut self.recurrent_staging)
                .map_err(|e| driver_err("initialize DeltaNet recurrence", &e))?;
        } else {
            self.stream
                .memcpy_dtod(&self.conv_current, &mut self.conv_staging)
                .map_err(|e| driver_err("stage DeltaNet convolution", &e))?;
            self.stream
                .memcpy_dtod(&self.recurrent_current, &mut self.recurrent_staging)
                .map_err(|e| driver_err("stage DeltaNet recurrence", &e))?;
        }
        let width = width as u32;
        let kernel = self.geometry.conv_kernel as u32;
        let tokens = tokens as u32;
        let mut convolution = self.stream.launch_builder(&self.convolution_function);
        convolution
            .arg(qkv)
            .arg(&self.convolution)
            .arg(&mut self.conv_staging)
            .arg(&mut *convolved)
            .arg(&width)
            .arg(&kernel)
            .arg(&tokens);
        #[allow(unsafe_code)]
        // SAFETY: one lane owns a channel; construction bounds all history,
        // active-row, coefficient and kernel-ABI extents before allocation.
        unsafe { convolution.launch(LaunchConfig::for_num_elems(width)) }
            .map_err(|e| driver_err("device DeltaNet convolution", &e))?;
        drop(convolution);
        let heads = g.value_heads as u32;
        let dk = g.key_head_dim as u32;
        let dv = g.value_head_dim as u32;
        let group = g.group_size as u32;
        let scale = 1. / (g.key_head_dim as f32).sqrt();
        let eps = self.geometry.rms_eps;
        let mut mixer = self.stream.launch_builder(&self.mixer_function);
        mixer
            .arg(&*convolved)
            .arg(z)
            .arg(beta)
            .arg(decay)
            .arg(&self.a_log)
            .arg(&self.dt_bias)
            .arg(&self.norm)
            .arg(&mut self.recurrent_staging)
            .arg(output)
            .arg(&heads)
            .arg(&dk)
            .arg(&dv)
            .arg(&group)
            .arg(&tokens)
            .arg(&scale)
            .arg(&eps);
        #[allow(unsafe_code)]
        // SAFETY: each block owns a value head; lanes own state columns and
        // active output. All lanes participate in every per-token barrier.
        unsafe {
            mixer.launch(LaunchConfig {
                grid_dim: (heads, 1, 1),
                block_dim: (dv.next_power_of_two().max(32), 1, 1),
                shared_mem_bytes: (2 * dk + dv + 3) * 4,
            })
        }
        .map_err(|e| driver_err("device DeltaNet gated mixer", &e))?;
        self.pending = Some(end);
        Ok(())
    }
    pub(super) fn commit(&mut self) {
        if let Some(end) = self.pending.take() {
            core::mem::swap(&mut self.conv_current, &mut self.conv_staging);
            core::mem::swap(&mut self.recurrent_current, &mut self.recurrent_staging);
            self.cursor = end;
        }
    }
}
pub(super) fn create(
    backend: &CudaBackend,
    geometry: DeltaMixerGeometry,
    weights: DeltaMixerWeights<'_>,
    limits: DecoderLimits,
    budget: usize,
) -> Result<Option<Box<dyn ResidentDeltaMixer>>, BackendError> {
    let payload = geometry.payload_bytes(weights, limits, budget)?;
    let g = geometry.recurrence;
    let (conv_width, value_width) = geometry.widths()?;
    if g.key_head_dim > 1024
        || g.value_head_dim > 1024
        || [
            g.value_heads,
            g.group_size,
            geometry.conv_kernel,
            limits.max_context,
            limits.max_segment_tokens * conv_width,
            limits.max_segment_tokens * value_width,
            limits.max_segment_tokens * g.value_heads,
        ]
        .into_iter()
        .any(|n| u32::try_from(n).is_err())
    {
        return Ok(None);
    }
    let alloc = |len| {
        backend
            .stream
            .alloc_zeros::<f32>(len)
            .map_err(|e| alloc_or_backend("DeltaNet mixer payload", &e, len * 4))
    };
    let m = limits.max_segment_tokens;
    let mut host_output = Vec::new();
    host_output
        .try_reserve_exact(m * value_width)
        .map_err(|_| BackendError::OutOfMemory {
            requested: m * value_width * 4,
        })?;
    host_output.resize(m * value_width, 0.);
    Ok(Some(Box::new(Mixer {
        state: State::new(backend, geometry, weights, limits)?,
        convolved: alloc(m * conv_width)?,
        payload,
        qkv: alloc(m * conv_width)?,
        z: alloc(m * value_width)?,
        beta: alloc(m * g.value_heads)?,
        decay: alloc(m * g.value_heads)?,
        output: alloc(m * value_width)?,
        host_output,
    })))
}
impl ResidentDeltaMixer for Mixer {
    fn geometry(&self) -> DeltaMixerGeometry {
        self.state.geometry
    }
    fn limits(&self) -> DecoderLimits {
        self.state.limits
    }
    fn payload_bytes(&self) -> usize {
        self.payload
    }
    fn len(&self) -> usize {
        self.state.cursor
    }
    fn staged_len(&self) -> Option<usize> {
        self.state.pending
    }
    fn abort(&mut self) {
        self.state.pending = None;
    }
    fn reset(&mut self) {
        self.state.cursor = 0;
        self.state.pending = None;
    }
    fn commit(&mut self) {
        self.state.commit();
    }
    fn stage(&mut self, segment: DeltaMixerSegment<'_>) -> Result<(), BackendError> {
        self.state.pending = None;
        self.state.extent(segment.tokens)?;
        let (conv, value) = self.state.geometry.widths()?;
        let heads = self.state.geometry.recurrence.value_heads;
        let n = segment.tokens;
        for (data, expected) in [
            (segment.qkv, n * conv),
            (segment.z, n * value),
            (segment.beta_logits, n * heads),
            (segment.decay_logits, n * heads),
            (&*segment.output, n * value),
        ] {
            if data.len() != expected {
                return Err(BackendError::ShapeMismatch {
                    expected,
                    got: data.len(),
                });
            }
        }
        if segment
            .qkv
            .iter()
            .chain(segment.z)
            .chain(segment.beta_logits)
            .chain(segment.decay_logits)
            .any(|v| !v.is_finite())
        {
            return Err(invalid("DeltaNet mixer projected operands must be finite"));
        }
        let stream = &self.state.stream;
        stream
            .memcpy_htod(segment.qkv, &mut self.qkv.slice_mut(..n * conv))
            .map_err(|e| driver_err("upload DeltaNet QKV", &e))?;
        stream
            .memcpy_htod(segment.z, &mut self.z.slice_mut(..n * value))
            .map_err(|e| driver_err("upload DeltaNet Z", &e))?;
        stream
            .memcpy_htod(segment.beta_logits, &mut self.beta.slice_mut(..n * heads))
            .map_err(|e| driver_err("upload DeltaNet B", &e))?;
        stream
            .memcpy_htod(segment.decay_logits, &mut self.decay.slice_mut(..n * heads))
            .map_err(|e| driver_err("upload DeltaNet A", &e))?;
        let result = (|| {
            self.state.stage_device(
                [&self.qkv, &self.z, &self.beta, &self.decay],
                &mut self.convolved,
                &mut self.output,
                n,
            )?;
            self.state
                .stream
                .memcpy_dtoh(
                    &self.output.slice(..n * value),
                    &mut self.host_output[..n * value],
                )
                .map_err(|e| driver_err("read DeltaNet mixer output", &e))?;
            if self.host_output[..n * value].iter().any(|v| !v.is_finite()) {
                return Err(invalid("DeltaNet mixer produced non-finite output"));
            }
            Ok(())
        })();
        if let Err(error) = result {
            self.state.pending = None;
            return Err(error);
        }
        segment
            .output
            .copy_from_slice(&self.host_output[..n * value]);
        Ok(())
    }
    fn read_committed(
        &self,
        convolution: &mut [f32],
        recurrent: &mut [f32],
    ) -> Result<(), BackendError> {
        for (got, expected) in [
            (convolution.len(), self.state.conv_current.len()),
            (recurrent.len(), self.state.recurrent_current.len()),
        ] {
            if got != expected {
                return Err(BackendError::ShapeMismatch { expected, got });
            }
        }
        if self.state.cursor == 0 {
            convolution.fill(0.);
            recurrent.fill(0.);
            return Ok(());
        }
        self.state
            .stream
            .memcpy_dtoh(&self.state.conv_current, convolution)
            .map_err(|e| driver_err("read committed DeltaNet convolution", &e))?;
        self.state
            .stream
            .memcpy_dtoh(&self.state.recurrent_current, recurrent)
            .map_err(|e| driver_err("read committed DeltaNet recurrence", &e))
    }
}
