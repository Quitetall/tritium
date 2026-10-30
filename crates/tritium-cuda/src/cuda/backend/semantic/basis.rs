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
    if matches!(basis, Basis::Identity) {
        return Ok(());
    }
    let active = values.len();
    apply_prefix(backend, values, active, cols, basis, inverse)
}

/// Transform only active whole rows of a larger reserved device workspace.
pub(super) fn apply_prefix(
    backend: &CudaBackend,
    values: &mut CudaSlice<f32>,
    active: usize,
    cols: usize,
    basis: Basis,
    inverse: bool,
) -> Result<(), BackendError> {
    if active > values.len() || cols == 0 || !active.is_multiple_of(cols) {
        return Err(invalid("CUDA active basis prefix geometry mismatch"));
    }
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
    if cols == 0 || !block.is_power_of_two() || !cols.is_multiple_of(block as usize) {
        return Err(invalid("CUDA basis buffer geometry mismatch"));
    }
    let count =
        u32::try_from(active).map_err(|_| invalid("CUDA basis buffer exceeds u32 index ABI"))?;
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn active_prefix_preserves_inactive_reservation_bit_for_bit() {
        let backend = match CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) if std::env::var("TRITIUM_REQUIRE_CUDA").as_deref() == Ok("1") => {
                panic!("required CUDA: {error}")
            }
            Err(error) => {
                eprintln!("SKIP active basis prefix: {error}");
                return;
            }
        };
        // Shared decoder scratch need not end on this projection's row width.
        let original: Vec<f32> = (0..256 * 128 + 7)
            .map(|i| ((i * 7 % 31) as f32 - 15.) / 64.)
            .collect();
        for basis in [
            Basis::Identity,
            Basis::Hadamard { block: 128 },
            Basis::SignedRht {
                block: 128,
                seed: 19,
                domain: 27,
            },
        ] {
            for inverse in [false, true] {
                let mut expected = original.clone();
                if inverse {
                    apply_inverse_basis(&mut expected[..128], basis).unwrap();
                } else {
                    apply_basis(&mut expected[..128], basis).unwrap();
                }
                let mut data = backend.stream.clone_htod(&original).unwrap();
                apply_prefix(&backend, &mut data, 128, 128, basis, inverse).unwrap();
                let got = backend.stream.clone_dtoh(&data).unwrap();
                assert_eq!(got, expected);
                for active in [1, original.len() + 128] {
                    assert!(
                        apply_prefix(&backend, &mut data, active, 128, basis, inverse).is_err()
                    );
                    assert_eq!(backend.stream.clone_dtoh(&data).unwrap(), got);
                }
            }
        }
    }
}
