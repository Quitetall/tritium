//! Bounded scalar WebAssembly adapter for the portable training manifest.

use tritium_spec::{
    BackendError, TrainBackendError, TrainBackendV1, TrainBufferDataMutV1, TrainBufferDataRefV1,
    TrainCapabilitiesV1, TrainLimitsV1, TrainOutputV1, TrainReceiptV1, TrainRequestV1,
};
use tritium_train::CpuTrainBackendV1;

const BACKEND_ID: &str = "wasm.portable.v1";
const MAX_BUFFER_BYTES: u64 = 8 * 1024 * 1024;
const MAX_CALLER_BYTES: u64 = 64 * 1024 * 1024;

const LIMITS: TrainLimitsV1 = TrainLimitsV1 {
    max_rank: 4,
    max_elements: MAX_BUFFER_BYTES,
    max_bytes: MAX_BUFFER_BYTES,
};

/// Complete scalar portable-training backend for one WebAssembly linear-memory
/// instance.
///
/// The numeric implementation is the same deterministic semantic reference
/// compiled into the guest. It does not call a host CPU fallback. Every caller
/// buffer is capped at 8 MiB and their combined payload at 64 MiB before the
/// reference executor can mutate output. The manifest's own Conv/Attention and
/// SALT reader scratch limits bound temporary storage to less than another
/// 128 MiB, keeping a prepared execution below a conservative 192 MiB linear
/// memory envelope.
#[derive(Clone, Debug)]
pub struct WasmTrainBackendV1 {
    reference: CpuTrainBackendV1,
    physical_device: String,
}

impl WasmTrainBackendV1 {
    /// Bind receipts to the actual engine/browser identity supplied by the host
    /// controller.
    ///
    /// # Errors
    /// Returns [`BackendError::InvalidInput`] for an empty identity. Release
    /// admission separately rejects structural or unversioned identities.
    pub fn new(physical_device: impl Into<String>) -> Result<Self, BackendError> {
        let physical_device = physical_device.into();
        if physical_device.trim().is_empty() {
            return Err(BackendError::InvalidInput(
                "WASM physical device identity must not be empty".to_owned(),
            ));
        }
        Ok(Self {
            reference: CpuTrainBackendV1::new(),
            physical_device,
        })
    }

    /// Combined caller-owned payload ceiling checked before dispatch.
    #[must_use]
    pub const fn max_caller_bytes() -> u64 {
        MAX_CALLER_BYTES
    }
}

impl TrainBackendV1 for WasmTrainBackendV1 {
    fn capabilities(&self) -> TrainCapabilitiesV1 {
        let mut capabilities = self.reference.capabilities();
        capabilities.backend_id = BACKEND_ID.to_owned();
        capabilities.limits = LIMITS;
        capabilities.device_resident = true;
        capabilities
    }

    fn execute(
        &self,
        request: TrainRequestV1<'_>,
        output: &mut TrainOutputV1<'_>,
    ) -> Result<TrainReceiptV1, TrainBackendError> {
        request.validate_with_limits(output, LIMITS)?;
        let caller_bytes = caller_bytes(&request, output)?;
        if caller_bytes > MAX_CALLER_BYTES {
            return Err(TrainBackendError::Backend {
                code: "wasm_arena_limit".to_owned(),
                message: format!(
                    "caller payload requires {caller_bytes} bytes, bounded WASM arena permits {MAX_CALLER_BYTES}"
                ),
            });
        }

        let mut receipt = self.reference.execute(request, output)?;
        receipt.backend_id = BACKEND_ID.to_owned();
        receipt.backend_build = backend_build_identity();
        receipt.physical_device = Some(self.physical_device.clone());
        receipt.limits = LIMITS;
        receipt.host_transfers = 0;
        receipt.device_resident = true;
        Ok(receipt)
    }
}

fn backend_build_identity() -> String {
    format!(
        "{}@{}+{}",
        env!("CARGO_PKG_NAME"),
        env!("CARGO_PKG_VERSION"),
        env!("TRITIUM_SOURCE_ID")
    )
}

fn caller_bytes(
    request: &TrainRequestV1<'_>,
    output: &TrainOutputV1<'_>,
) -> Result<u64, TrainBackendError> {
    request
        .inputs
        .iter()
        .map(|buffer| match buffer.data {
            TrainBufferDataRefV1::F32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataRefV1::U32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataRefV1::Bytes(data) => Some(data.len() as u64),
        })
        .chain(output.buffers.iter().map(|buffer| match &buffer.data {
            TrainBufferDataMutV1::F32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataMutV1::U32(data) => (data.len() as u64).checked_mul(4),
            TrainBufferDataMutV1::Bytes(data) => Some(data.len() as u64),
        }))
        .try_fold(0_u64, |total, bytes| {
            total
                .checked_add(bytes.ok_or_else(arena_overflow)?)
                .ok_or_else(arena_overflow)
        })
}

fn arena_overflow() -> TrainBackendError {
    TrainBackendError::Backend {
        code: "wasm_arena_overflow".to_owned(),
        message: "caller payload byte count overflowed u64".to_owned(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_empty_physical_identity() {
        assert!(WasmTrainBackendV1::new("  ").is_err());
    }

    #[test]
    fn exposes_bounded_complete_capabilities() {
        let backend = WasmTrainBackendV1::new("wasm32:test").expect("valid identity");
        let capabilities = backend.capabilities();
        assert_eq!(capabilities.backend_id, BACKEND_ID);
        assert_eq!(capabilities.supported_operations.len(), 36);
        assert_eq!(capabilities.limits, LIMITS);
        assert!(capabilities.device_resident);
        assert_eq!(WasmTrainBackendV1::max_caller_bytes(), 64 * 1024 * 1024);
    }
}
