//! Explicit device-f32 region; no host intermediate activation transfers.
use super::*;

/// Execution identity, not the stored wire/fitting numerical identity.
pub const QTEA_DEVICE_SWIGLU_NUMERICAL_ID: &str = "qtea.cuda-device-swiglu.f32.v1";

/// Logical active workspace footprint, possibly reused capacity, NOT physical/model qualification.
/// Excludes other weights/calls, caller buffers and allocator/driver/module/events.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub struct QteaSwiGluAllocationLedger {
    /// Unique encoded bytes among the three selected handles.
    pub encoded_resident_bytes: u64,
    /// One input activation upload and resident buffer.
    pub input_bytes: u64,
    /// Gate plus up device buffers; gate is reused for the activated product.
    pub hidden_bytes: u64,
    /// Final device output and host output element bytes.
    pub output_bytes: u64,
    /// Four-byte status buffer, zero for empty calls.
    pub status_bytes: u64,
    /// All per-call device requests, including status.
    pub device_staging_bytes: u64,
    /// Actual host output Vec capacity plus four-byte status value.
    pub host_staging_capacity_bytes: usize,
    /// Selected unique encoded weights plus per-call device requests.
    pub requested_device_peak_bytes: u64,
    /// Always zero: no auxiliary weight shadow.
    pub dense_weight_bytes: u64,
    /// This call reused already admitted capacity, not prior activation values.
    pub workspace_reused: bool,
    /// This call's active workspace remains cached; empty calls select no workspace.
    pub workspace_retained: bool,
}

fn plan(
    resident: u64,
    batch: usize,
    width: usize,
    hidden: usize,
    limits: QteaCudaLimits,
) -> Result<QteaSwiGluAllocationLedger, BackendError> {
    if limits.max_device_staging_bytes == 0 || limits.max_host_staging_bytes == 0 {
        return Err(invalid("QTEA SwiGLU staging limits must be positive"));
    }
    let input = batch
        .checked_mul(width)
        .ok_or_else(|| invalid("QTEA SwiGLU input overflow"))?;
    let mid = batch
        .checked_mul(hidden)
        .ok_or_else(|| invalid("QTEA SwiGLU hidden overflow"))?;
    u32::try_from(input).map_err(|_| invalid("QTEA SwiGLU output exceeds u32 kernel ABI"))?;
    u32::try_from(mid).map_err(|_| invalid("QTEA SwiGLU hidden exceeds u32 kernel ABI"))?;
    let input_bytes = u64::try_from(byte_count(input)?)
        .map_err(|_| invalid("QTEA SwiGLU input bytes exceed u64"))?;
    let hidden_bytes = u64::try_from(byte_count(mid)?)
        .map_err(|_| invalid("QTEA SwiGLU hidden bytes exceed u64"))?
        .checked_mul(2)
        .ok_or_else(|| invalid("QTEA SwiGLU hidden bytes overflow"))?;
    let status = if batch == 0 { 0 } else { 4 };
    let device = input_bytes
        .checked_mul(2)
        .and_then(|n| n.checked_add(hidden_bytes))
        .and_then(|n| n.checked_add(status))
        .ok_or_else(|| invalid("QTEA SwiGLU staging overflow"))?;
    let host = byte_count(input)?
        .checked_add(status as usize)
        .ok_or_else(|| invalid("QTEA SwiGLU host staging overflow"))?;
    if device > limits.max_device_staging_bytes || host > limits.max_host_staging_bytes {
        return Err(invalid("QTEA SwiGLU staging exceeds caller limit"));
    }
    Ok(QteaSwiGluAllocationLedger {
        encoded_resident_bytes: resident,
        input_bytes,
        hidden_bytes,
        output_bytes: input_bytes,
        status_bytes: status,
        device_staging_bytes: device,
        host_staging_capacity_bytes: host,
        requested_device_peak_bytes: resident
            .checked_add(device)
            .ok_or_else(|| invalid("QTEA SwiGLU peak overflow"))?,
        dense_weight_bytes: 0,
        workspace_reused: false,
        workspace_retained: false,
    })
}

impl CudaBackend {
    /// Gate/up/device-SiLU/down with one input upload and output/status readbacks.
    /// Separate ordered projections are unchanged; device expf need not bit-match
    /// host exp. This explicit operation does not require/alter backend defaults.
    /// Private staging makes caller output unchanged on every refusal.
    /// # Errors
    /// Foreign context, geometry/ABI, nonfinite input/intermediate/output, positive
    /// per-call caps, allocation/copy/kernel failure. No host or dense fallback.
    #[allow(clippy::too_many_arguments)] // Three explicit weight roles in one bounded region.
    pub fn qtea_swiglu_device_into(
        &self,
        gate: &QteaResidentTensor,
        up: &QteaResidentTensor,
        down: &QteaResidentTensor,
        input: &[f32],
        batch: usize,
        output: &mut [f32],
        limits: QteaCudaLimits,
    ) -> Result<QteaSwiGluAllocationLedger, BackendError> {
        for t in [gate, up, down] {
            self.check_qtea_context(t)?;
        }
        let (f, k) = gate.shape();
        if up.shape() != (f, k)
            || down.shape() != (k, f)
            || batch.checked_mul(k) != Some(input.len())
            || input.len() != output.len()
        {
            return Err(invalid("QTEA SwiGLU geometry mismatch"));
        }
        if input.iter().any(|x| !x.is_finite()) {
            return Err(invalid("QTEA SwiGLU input nonfinite"));
        }
        let mut resident = 0u64;
        let weights = [gate, up, down];
        for (i, t) in weights.iter().enumerate() {
            if !weights[..i]
                .iter()
                .any(|earlier| std::ptr::eq(*earlier, *t))
            {
                resident = resident
                    .checked_add(
                        u64::try_from(t.encoded_bytes())
                            .map_err(|_| invalid("QTEA SwiGLU resident exceeds u64"))?,
                    )
                    .ok_or_else(|| invalid("QTEA SwiGLU resident sum overflow"))?;
            }
        }
        let mut ledger = plan(resident, batch, k, f, limits)?;
        if batch == 0 {
            return Ok(ledger);
        }
        self.with_swiglu_workspace(
            SwiGluWorkspaceKind::PackedQtea,
            batch,
            k,
            f,
            Some(limits),
            |w, reused, retained| {
                ledger.workspace_reused = reused;
                ledger.workspace_retained = retained;
                ledger.host_staging_capacity_bytes = byte_count(w.host_output.capacity())?
                    .checked_add(4)
                    .filter(|n| *n <= limits.max_host_staging_bytes)
                    .ok_or_else(|| invalid("QTEA SwiGLU host capacity exceeds limit"))?;
                let hidden = batch * f; // Checked by plan before allocation.
                self.stream
                    .memset_zeros(&mut w.invalid)
                    .map_err(|e| driver_err("reset QTEA SwiGLU status", &e))?;
                self.stream
                    .memcpy_htod(input, &mut w.input)
                    .map_err(|e| driver_err("upload QTEA SwiGLU input", &e))?;
                let hidden_count =
                    u32::try_from(hidden).map_err(|_| invalid("QTEA SwiGLU hidden ABI"))?;
                let out_count =
                    u32::try_from(output.len()).map_err(|_| invalid("QTEA SwiGLU output ABI"))?;
                self.launch_qtea_matmul(gate, &w.input, &mut w.gate, hidden_count)?;
                self.launch_qtea_matmul(up, &w.input, &mut w.up, hidden_count)?;
                let mut launch = self.stream.launch_builder(&gate.kernels.swiglu);
                launch
                    .arg(&mut w.gate)
                    .arg(&w.up)
                    .arg(&mut w.invalid)
                    .arg(&hidden_count);
                // SAFETY: exact same-context disjoint buffers, checked u32 count and
                // private initialized status. Every thread writes one gate element;
                // status uses atomicOr, and all launches share the ordered backend stream.
                #[allow(unsafe_code)]
                unsafe { launch.launch(LaunchConfig::for_num_elems(hidden_count)) }
                    .map_err(|e| driver_err("launch QTEA device SwiGLU", &e))?;
                drop(launch);
                self.launch_qtea_matmul(down, &w.gate, &mut w.output, out_count)?;
                let mut flag = [0u32];
                self.stream
                    .memcpy_dtoh(&w.output, &mut w.host_output)
                    .map_err(|e| driver_err("read QTEA SwiGLU output", &e))?;
                self.stream
                    .memcpy_dtoh(&w.invalid, &mut flag)
                    .map_err(|e| driver_err("read QTEA SwiGLU status", &e))?;
                self.stream
                    .synchronize()
                    .map_err(|e| driver_err("finish QTEA SwiGLU", &e))?;
                if flag[0] != 0 || w.host_output.iter().any(|x| !x.is_finite()) {
                    return Err(invalid("QTEA device SwiGLU nonfinite intermediate/output"));
                }
                output.copy_from_slice(&w.host_output);
                Ok(ledger)
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn exact_logical_staging_includes_two_hidden_buffers_and_status() {
        let caps = QteaCudaLimits {
            max_device_staging_bytes: 84,
            max_host_staging_bytes: 20,
        };
        let p = plan(100, 1, 4, 6, caps).unwrap();
        assert_eq!(
            (
                p.input_bytes,
                p.hidden_bytes,
                p.output_bytes,
                p.status_bytes
            ),
            (16, 48, 16, 4)
        );
        assert_eq!(p.requested_device_peak_bytes, 184);
        assert!(
            plan(
                100,
                1,
                4,
                6,
                QteaCudaLimits {
                    max_device_staging_bytes: 83,
                    ..caps
                }
            )
            .is_err()
        );
        assert!(
            plan(
                100,
                1,
                4,
                6,
                QteaCudaLimits {
                    max_host_staging_bytes: 19,
                    ..caps
                }
            )
            .is_err()
        );
        assert!(plan(100, usize::MAX, 4, 6, caps).is_err());
        assert!(plan(100, 1, usize::MAX, 6, caps).is_err());
        assert_eq!(
            plan(
                100,
                0,
                4,
                6,
                QteaCudaLimits {
                    max_device_staging_bytes: 1,
                    max_host_staging_bytes: 1
                }
            )
            .unwrap()
            .device_staging_bytes,
            0
        );
        assert!(
            plan(
                100,
                0,
                4,
                6,
                QteaCudaLimits {
                    max_device_staging_bytes: 0,
                    max_host_staging_bytes: 1
                }
            )
            .is_err()
        );
    }
}
