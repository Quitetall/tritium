//! Private captured-callback execution; no complete owner VJP tensor.
use super::error;
use crate::{
    model_level::NativePackedMatrix,
    torch_native::{checked_product, cpu_f32, tensor_shape},
};
use dlpark::legacy::Dlpack;
use pyo3::PyResult;
use std::{io, sync::Arc};
use tritium_core::AdditiveAdjointSpan;
use tritium_cpu::CpuBackend;
use tritium_spec::DeviceBuffer;

pub(crate) enum Input<'a> {
    Matmul {
        x: &'a [f32],
        dy: &'a [f32],
        batch: usize,
    },
    Embed {
        ids: &'a [i64],
        dy: &'a [f32],
    },
}

impl<'a> Input<'a> {
    pub(crate) fn matmul(
        matrix: &NativePackedMatrix,
        x: &'a Dlpack,
        dy: &'a Dlpack,
    ) -> PyResult<Self> {
        let xs = tensor_shape(x, "activation")?;
        let ys = tensor_shape(dy, "cotangent")?;
        let (rows, cols, _) = matrix.dimensions();
        if xs.last() != Some(&cols)
            || ys.last() != Some(&rows)
            || xs[..xs.len() - 1] != ys[..ys.len() - 1]
        {
            return Err(error("native scale adjoint shape mismatch"));
        }
        Ok(Self::Matmul {
            x: cpu_f32(x, "activation")?,
            dy: cpu_f32(dy, "cotangent")?,
            batch: checked_product(&xs[..xs.len() - 1], "batch")?,
        })
    }

    pub(crate) fn embed(
        matrix: &NativePackedMatrix,
        ids: &'a Dlpack,
        dy: &'a Dlpack,
    ) -> PyResult<Self> {
        let shape = tensor_shape(ids, "indices")?;
        let grad = tensor_shape(dy, "cotangent")?;
        let (rows, cols, _) = matrix.dimensions();
        if grad.last() != Some(&cols) || grad[..grad.len() - 1] != shape {
            return Err(error("native embedding scale adjoint shape mismatch"));
        }
        let ids = ids.cpu_data_slice::<i64>().map_err(error)?;
        if ids.iter().any(|&i| i < 0 || i as u64 >= rows as u64) {
            return Err(error("native embedding index out of range"));
        }
        Ok(Self::Embed {
            ids,
            dy: cpu_f32(dy, "cotangent")?,
        })
    }
}

pub(crate) struct Program<'a> {
    input: Input<'a>,
    handle: Arc<dyn DeviceBuffer>,
    row: Vec<f32>,
    ids: Vec<usize>,
    scales: usize,
    count: usize,
    pub maximum_values: usize,
}

impl<'a> Program<'a> {
    pub(crate) const fn count(&self) -> usize {
        self.count
    }
    pub(crate) fn new(
        input: Input<'a>,
        handle: Arc<dyn DeviceBuffer>,
        dimensions: (usize, usize, usize),
    ) -> PyResult<Self> {
        let (_, cols, maximum) = dimensions;
        let cpu = CpuBackend::new();
        let scales = cpu.additive_stored_scale_count(&*handle).map_err(error)?;
        let weights = cpu.additive_present_weight_count(&*handle).map_err(error)?;
        let count = scales
            .checked_add(weights)
            .ok_or_else(|| error("native gradient count overflow"))?;
        let index_count = match &input {
            Input::Embed { ids, .. } => ids.len(),
            _ => 0,
        };
        let transferred = match &input {
            Input::Matmul { x, dy, .. } => x.len().checked_add(dy.len()),
            Input::Embed { dy, .. } => Some(dy.len()),
        }
        .ok_or_else(|| error("native span transfer overflow"))?;
        let overhead = transferred
            .checked_add(cols)
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| n.checked_add(index_count.checked_mul(size_of::<usize>())?))
            .ok_or_else(|| error("native span scratch overflow"))?;
        let maximum_values = maximum
            .checked_sub(overhead)
            .map(|n| n / 8)
            .ok_or_else(|| error("native scale adjoint output/scratch exceeds byte limit"))?;
        if count != 0 && maximum_values == 0 {
            return Err(error(
                "native scale adjoint output/scratch exceeds byte limit",
            ));
        }
        let mut row = Vec::new();
        row.try_reserve_exact(cols).map_err(error)?;
        row.resize(cols, 0.0);
        let mut indices = Vec::new();
        indices.try_reserve_exact(index_count).map_err(error)?;
        if let Input::Embed { ids, .. } = &input {
            indices.extend(ids.iter().map(|&n| n as usize));
        }
        Ok(Self {
            input,
            handle,
            row,
            ids: indices,
            scales,
            count,
            maximum_values,
        })
    }

    pub(crate) fn fill(&mut self, start: usize, out: &mut [f64]) -> io::Result<()> {
        if start.checked_add(out.len()).is_none_or(|n| n > self.count) {
            return Err(io::Error::other("capture span differs from current owner"));
        }
        let s = self.scales.saturating_sub(start).min(out.len());
        let (scales, weights) = out.split_at_mut(s);
        let cpu = CpuBackend::new();
        let map = |e: tritium_spec::BackendError| io::Error::other(e.to_string());
        match &self.input {
            Input::Matmul { x, dy, batch } => {
                if s != 0 || self.count == 0 {
                    cpu.accumulate_additive_matmul_scale_vjp_span(
                        &*self.handle,
                        x,
                        dy,
                        *batch,
                        &mut self.row,
                        AdditiveAdjointSpan::new(start, scales),
                    )
                    .map_err(map)?;
                }
                if !weights.is_empty() || self.count == 0 {
                    cpu.accumulate_additive_matmul_weight_vjp_span(
                        &*self.handle,
                        x,
                        dy,
                        *batch,
                        &mut self.row,
                        AdditiveAdjointSpan::new((start + s).saturating_sub(self.scales), weights),
                    )
                    .map_err(map)?;
                }
            }
            Input::Embed { dy, .. } => {
                if s != 0 || self.count == 0 {
                    cpu.accumulate_additive_embed_scale_vjp_span(
                        &*self.handle,
                        &self.ids,
                        dy,
                        &mut self.row,
                        AdditiveAdjointSpan::new(start, scales),
                    )
                    .map_err(map)?;
                }
                if !weights.is_empty() || self.count == 0 {
                    cpu.accumulate_additive_embed_weight_vjp_span(
                        &*self.handle,
                        &self.ids,
                        dy,
                        &mut self.row,
                        AdditiveAdjointSpan::new((start + s).saturating_sub(self.scales), weights),
                    )
                    .map_err(map)?;
                }
            }
        }
        Ok(())
    }
}
