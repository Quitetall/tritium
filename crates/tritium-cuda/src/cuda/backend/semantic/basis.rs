//! Exact shared-core basis ordering on device storage, without a host fallback.

use super::*;
use cudarc::driver::{LaunchConfig, PushKernelArg};

/// Input/output-axis basis on an already uploaded row-major activation buffer.
pub(super) fn apply(
    backend: &CudaBackend,
    values: &mut CudaSlice<f32>,
    cols: usize,
    basis: Basis,
    inverse: bool,
) -> Result<(), BackendError> {
    let (block, signed, seed, domain) = match basis {
        Basis::Identity => return Ok(()),
        Basis::Hadamard { block } => (block, false, 0_u64, 0_u64),
        Basis::SignedRht {
            block,
            seed,
            domain,
        } => (block, true, seed, domain),
        _ => return Err(BackendError::UnsupportedTensor),
    };
    if cols == 0
        || !block.is_power_of_two()
        || !cols.is_multiple_of(block as usize)
        || !values.len().is_multiple_of(cols)
    {
        return Err(invalid("CUDA basis buffer geometry mismatch"));
    }
    let count = u32::try_from(values.len())
        .map_err(|_| invalid("CUDA basis buffer exceeds u32 index ABI"))?;
    let cols = u32::try_from(cols).map_err(|_| invalid("CUDA basis columns exceed u32"))?;
    if count == 0 {
        return Ok(());
    }
    let cfg = LaunchConfig::for_num_elems(count);
    if signed && !inverse {
        let mut launch = backend
            .stream
            .launch_builder(&backend.func_semantic_basis_sign);
        launch
            .arg(&mut *values)
            .arg(&count)
            .arg(&cols)
            .arg(&seed)
            .arg(&domain);
        #[allow(unsafe_code)]
        // SAFETY: checked row geometry bounds the one-writer coordinate kernel.
        unsafe { launch.launch(cfg) }.map_err(|e| driver_err("CUDA basis signs", &e))?;
    }
    let pairs = count / 2;
    let mut width = 1_u32;
    while width < block {
        let mut launch = backend
            .stream
            .launch_builder(&backend.func_semantic_basis_butterfly);
        launch
            .arg(&mut *values)
            .arg(&pairs)
            .arg(&cols)
            .arg(&block)
            .arg(&width);
        #[allow(unsafe_code)]
        // SAFETY: each stage partitions admitted blocks into disjoint pairs.
        // The owning stream orders stages; no kernel reads a concurrent pair.
        unsafe { launch.launch(LaunchConfig::for_num_elems(pairs)) }
            .map_err(|e| driver_err("CUDA basis butterfly", &e))?;
        width *= 2;
    }
    // This retains the original shipping sqrt-reciprocal bits; core explicitly
    // freezes their equivalence for every power-of-two block on this f32 domain.
    let normalization = 1.0 / (block as f32).sqrt();
    let inverse_sign = u32::from(inverse && signed);
    let mut launch = backend
        .stream
        .launch_builder(&backend.func_semantic_basis_finish);
    launch
        .arg(&mut *values)
        .arg(&count)
        .arg(&cols)
        .arg(&normalization)
        .arg(&inverse_sign)
        .arg(&seed)
        .arg(&domain);
    #[allow(unsafe_code)]
    // SAFETY: each thread normalizes/signs one checked device coordinate.
    unsafe { launch.launch(cfg) }.map_err(|e| driver_err("CUDA basis normalization", &e))?;
    Ok(())
}
