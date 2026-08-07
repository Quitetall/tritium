//! Exact scalar CPU reference for SALT V2 additive ternary matrices.

use core::fmt;

use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SALT_V2_ALLOCATION_TILE_SIZE, SaltV2Package, SaltV2Tensor, SaltV2Transform,
};

/// The scalar operation order used by the SALT V2 CPU reference.
///
/// For each row, physical group128 segments are visited in increasing linear
/// coefficient order. Within a group, planes are visited from zero upward and
/// columns are visited from low to high. Each plane first reduces hard-trit
/// add/sub/skip activation contributions, then applies its f16 scale exactly
/// once before adding to the row output. No tree reduction or parallel
/// reassociation is permitted.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2ReductionOrder {
    /// Rows, then group128 segments, planes, and columns, all in ascending order.
    RowMajorGroupThenPlaneThenColumn,
}

/// Exact work and heap-allocation accounting for one SALT V2 reference matvec.
///
/// The caller-owned `*_into` path performs no heap allocation. In particular,
/// hard trits are streamed directly into per-plane/group accumulators, so
/// `dense_weight_bytes()` is always zero. The convenience
/// allocating path owns only the returned output buffer; its size is reported
/// separately by `output_bytes()` and is not temporary scratch.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SaltV2MatVecReceipt {
    codec: SaltV2Codec,
    rows: usize,
    columns: usize,
    weight_coefficients_visited: u64,
    plane_coefficients_visited: u64,
    output_bytes: u64,
    temporary_heap_bytes: u64,
    dense_weight_bytes: u64,
    reduction_order: SaltV2ReductionOrder,
}

impl SaltV2MatVecReceipt {
    /// Physical codec of the validated semantic package.
    #[must_use]
    pub fn codec(self) -> SaltV2Codec {
        self.codec
    }

    /// Number of output rows visited.
    #[must_use]
    pub fn rows(self) -> usize {
        self.rows
    }

    /// Number of activation columns reduced per output row.
    #[must_use]
    pub fn columns(self) -> usize {
        self.columns
    }

    /// Number of logical matrix coefficients reconstructed exactly once.
    #[must_use]
    pub fn weight_coefficients_visited(self) -> u64 {
        self.weight_coefficients_visited
    }

    /// Number of present plane coefficients read across all reconstructed weights.
    #[must_use]
    pub fn plane_coefficients_visited(self) -> u64 {
        self.plane_coefficients_visited
    }

    /// Bytes occupied by the caller-owned or returned output slice.
    #[must_use]
    pub fn output_bytes(self) -> u64 {
        self.output_bytes
    }

    /// Temporary heap bytes allocated by the execution path.
    ///
    /// This is zero for both APIs. The allocating convenience API reports its
    /// returned `Vec<f32>` separately through [`Self::output_bytes`].
    #[must_use]
    pub fn temporary_heap_bytes(self) -> u64 {
        self.temporary_heap_bytes
    }

    /// Bytes allocated for a dense reconstructed weight matrix.
    ///
    /// This is always zero: no dense weight reconstruction occurs.
    #[must_use]
    pub fn dense_weight_bytes(self) -> u64 {
        self.dense_weight_bytes
    }

    /// Explicit scalar reduction order used for this result.
    #[must_use]
    pub fn reduction_order(self) -> SaltV2ReductionOrder {
        self.reduction_order
    }
}

/// Output and accounting from the allocating SALT V2 reference matvec.
#[derive(Clone, Debug, PartialEq)]
pub struct SaltV2MatVec {
    /// Deterministic row-major output vector.
    pub output: Vec<f32>,
    /// Exact work and temporary-memory accounting.
    pub receipt: SaltV2MatVecReceipt,
}

/// Errors from checked SALT V2 scalar reconstruction and matvec execution.
#[derive(Clone, Debug, PartialEq, Eq)]
#[non_exhaustive]
pub enum SaltV2CpuError {
    /// A tensor index was outside the package tensor table.
    TensorIndexOutOfBounds {
        /// Requested tensor index.
        index: usize,
        /// Number of tensors in the package.
        tensors: usize,
    },
    /// The tensor requires an activation transform this CPU reference does not implement.
    UnsupportedTransform {
        /// Exact transform identity rejected by the runtime.
        transform: SaltV2Transform,
    },
    /// Matrix execution requires exactly two semantic dimensions.
    ExpectedMatrix {
        /// Supplied tensor rank.
        rank: usize,
    },
    /// A semantic dimension did not fit the host `usize`.
    DimensionTooLarge {
        /// Zero-based dimension axis.
        axis: usize,
        /// Semantic `u64` dimension.
        value: u64,
    },
    /// Multiplying matrix dimensions overflowed the host `usize`.
    DimensionProductOverflow,
    /// The shape product disagreed with the validated tensor coefficient count.
    SemanticLengthMismatch {
        /// Coefficients derived from the two matrix dimensions.
        shape_coefficients: usize,
        /// Coefficients declared by the semantic tensor.
        tensor_coefficients: usize,
    },
    /// The activation vector length did not match the matrix column count.
    InputLengthMismatch {
        /// Matrix column count.
        expected: usize,
        /// Supplied activation count.
        got: usize,
    },
    /// The output slice length did not match the matrix row count.
    OutputLengthMismatch {
        /// Matrix row count.
        expected: usize,
        /// Supplied output count.
        got: usize,
    },
    /// A requested linear coefficient was outside the tensor.
    CoefficientIndexOutOfBounds {
        /// Requested linear row-major coefficient index.
        index: usize,
        /// Number of coefficients in the tensor.
        coefficients: usize,
    },
    /// A validated tensor's private tile/plane layout was internally inconsistent.
    SemanticLayoutMismatch {
        /// Coefficient whose tile, trit, or scale could not be located.
        coefficient: usize,
    },
    /// An activation was NaN or infinite.
    NonFiniteActivation {
        /// Activation index.
        index: usize,
        /// Raw `f32` bits.
        bits: u32,
    },
    /// Scalar multiplication or accumulation produced NaN or infinity.
    NonFiniteOutput {
        /// Output row being reduced.
        row: usize,
        /// Column where an add/sub overflowed, or the terminal column of a group whose scale or
        /// output accumulation became non-finite.
        column: usize,
    },
    /// Receipt or output-size arithmetic overflowed.
    AccountingOverflow {
        /// Accounting field that could not be represented.
        field: &'static str,
    },
    /// The allocating convenience API could not reserve its output vector.
    AllocationFailed {
        /// Number of requested `f32` output elements.
        requested_elements: usize,
    },
}

impl fmt::Display for SaltV2CpuError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::TensorIndexOutOfBounds { index, tensors } => {
                write!(
                    f,
                    "SALT V2 tensor index {index} is outside {tensors} tensors"
                )
            }
            Self::UnsupportedTransform { transform } => {
                write!(f, "SALT V2 CPU does not support transform {transform:?}")
            }
            Self::ExpectedMatrix { rank } => {
                write!(f, "SALT V2 matvec requires rank 2, got rank {rank}")
            }
            Self::DimensionTooLarge { axis, value } => {
                write!(
                    f,
                    "SALT V2 dimension {axis} value {value} exceeds host usize"
                )
            }
            Self::DimensionProductOverflow => {
                write!(f, "SALT V2 matrix dimension product overflows host usize")
            }
            Self::SemanticLengthMismatch {
                shape_coefficients,
                tensor_coefficients,
            } => write!(
                f,
                "SALT V2 shape has {shape_coefficients} coefficients but tensor has {tensor_coefficients}"
            ),
            Self::InputLengthMismatch { expected, got } => {
                write!(
                    f,
                    "SALT V2 matvec expected {expected} activations, got {got}"
                )
            }
            Self::OutputLengthMismatch { expected, got } => {
                write!(f, "SALT V2 matvec expected {expected} outputs, got {got}")
            }
            Self::CoefficientIndexOutOfBounds {
                index,
                coefficients,
            } => write!(
                f,
                "SALT V2 coefficient index {index} is outside {coefficients} coefficients"
            ),
            Self::SemanticLayoutMismatch { coefficient } => write!(
                f,
                "SALT V2 semantic layout cannot locate coefficient {coefficient}"
            ),
            Self::NonFiniteActivation { index, bits } => {
                write!(f, "SALT V2 activation {index} is non-finite ({bits:#010x})")
            }
            Self::NonFiniteOutput { row, column } => write!(
                f,
                "SALT V2 output row {row} became non-finite at column {column}"
            ),
            Self::AccountingOverflow { field } => {
                write!(f, "SALT V2 {field} accounting overflow")
            }
            Self::AllocationFailed { requested_elements } => write!(
                f,
                "could not reserve {requested_elements} SALT V2 output elements"
            ),
        }
    }
}

impl std::error::Error for SaltV2CpuError {}

/// Reconstruct one row-major SALT V2 coefficient without materializing a tensor.
///
/// Present additive planes are reduced in ascending plane order. Each hard trit
/// is multiplied by its zero-point-free f16 tensor-declared group scale in `f32`.
///
/// # Errors
/// Returns [`SaltV2CpuError::CoefficientIndexOutOfBounds`] for an invalid index.
/// Returns [`SaltV2CpuError::UnsupportedTransform`] rather than exposing a
/// coefficient whose activation-domain transform has not been applied.
/// The defensive [`SaltV2CpuError::SemanticLayoutMismatch`] is returned if a
/// value created inside `tritium-format` ever violates its validated layout.
pub fn salt_v2_coefficient(
    tensor: &SaltV2Tensor,
    linear_index: usize,
) -> Result<f32, SaltV2CpuError> {
    reject_unsupported_transform(tensor)?;
    if linear_index >= tensor.logical_coefficients() {
        return Err(SaltV2CpuError::CoefficientIndexOutOfBounds {
            index: linear_index,
            coefficients: tensor.logical_coefficients(),
        });
    }
    reconstruct_coefficient(tensor, linear_index)
}

/// Execute deterministic SALT V2 matrix-vector multiplication into caller memory.
///
/// The selected package tensor must be a row-major `[rows, columns]` matrix.
/// This path performs no heap allocation and never constructs dense weights.
/// Hard-trit activation contributions cancel within each plane/group before
/// that group's f16 scale is applied once.
/// Every activation is validated before the first output write. If a later row
/// overflows, previously completed output rows may already have been written,
/// but a non-finite value is never stored in `output`.
///
/// # Errors
/// Returns a typed error for a missing tensor, unsupported activation transform,
/// invalid matrix dimensions or lengths, arithmetic overflow, non-finite
/// activation, non-finite result, or an impossible inconsistency in the
/// validated semantic tensor layout.
pub fn salt_v2_matvec_into(
    package: &SaltV2Package,
    tensor_index: usize,
    activation: &[f32],
    output: &mut [f32],
) -> Result<SaltV2MatVecReceipt, SaltV2CpuError> {
    let tensor =
        package
            .tensors()
            .get(tensor_index)
            .ok_or(SaltV2CpuError::TensorIndexOutOfBounds {
                index: tensor_index,
                tensors: package.tensors().len(),
            })?;
    reject_unsupported_transform(tensor)?;
    let (rows, columns) = matrix_shape(tensor.dims(), tensor.logical_coefficients())?;
    validate_activation(activation, columns)?;
    if output.len() != rows {
        return Err(SaltV2CpuError::OutputLengthMismatch {
            expected: rows,
            got: output.len(),
        });
    }

    let receipt = work_receipt(package.codec(), tensor, rows, columns)?;
    let scale_group_size = tensor.scale_group_size();
    for (row, output_value) in output.iter_mut().enumerate() {
        let row_base = row
            .checked_mul(columns)
            .ok_or(SaltV2CpuError::DimensionProductOverflow)?;
        let row_end = row_base
            .checked_add(columns)
            .ok_or(SaltV2CpuError::DimensionProductOverflow)?;
        let mut accumulator = 0.0_f32;
        let mut coefficient = row_base;
        while coefficient < row_end {
            let tile_index = coefficient / SALT_V2_ALLOCATION_TILE_SIZE;
            let tile_base = tile_index
                .checked_mul(SALT_V2_ALLOCATION_TILE_SIZE)
                .ok_or(SaltV2CpuError::DimensionProductOverflow)?;
            let local_start = coefficient - tile_base;
            let tile = tensor
                .tiles()
                .get(tile_index)
                .ok_or(SaltV2CpuError::SemanticLayoutMismatch { coefficient })?;
            let group_index = local_start / scale_group_size;
            let group_end = group_index
                .checked_add(1)
                .and_then(|value| value.checked_mul(scale_group_size))
                .ok_or(SaltV2CpuError::DimensionProductOverflow)?
                .min(tile.logical_len());
            let segment_len = (group_end - local_start).min(row_end - coefficient);
            let segment_end = coefficient
                .checked_add(segment_len)
                .ok_or(SaltV2CpuError::DimensionProductOverflow)?;
            let terminal_column = segment_end - row_base - 1;

            for plane in tile.planes() {
                let scale = plane
                    .scales()
                    .get(group_index)
                    .ok_or(SaltV2CpuError::SemanticLayoutMismatch { coefficient })?;
                let mut group_accumulator = 0.0_f32;
                for current in coefficient..segment_end {
                    let column = current - row_base;
                    let local = current - tile_base;
                    let trit =
                        plane
                            .trits()
                            .get(local)
                            .ok_or(SaltV2CpuError::SemanticLayoutMismatch {
                                coefficient: current,
                            })?;
                    match trit.get() {
                        -1 => group_accumulator -= activation[column],
                        0 => {}
                        1 => group_accumulator += activation[column],
                        _ => {
                            return Err(SaltV2CpuError::SemanticLayoutMismatch {
                                coefficient: current,
                            });
                        }
                    }
                    if !group_accumulator.is_finite() {
                        return Err(SaltV2CpuError::NonFiniteOutput { row, column });
                    }
                }
                let contribution = group_accumulator * scale.to_f32();
                if !contribution.is_finite() {
                    return Err(SaltV2CpuError::NonFiniteOutput {
                        row,
                        column: terminal_column,
                    });
                }
                accumulator += contribution;
                if !accumulator.is_finite() {
                    return Err(SaltV2CpuError::NonFiniteOutput {
                        row,
                        column: terminal_column,
                    });
                }
            }
            coefficient = segment_end;
        }
        *output_value = accumulator;
    }
    Ok(receipt)
}

/// Execute deterministic SALT V2 matrix-vector multiplication and return a vector.
///
/// This convenience API allocates only the returned `rows`-element output. The
/// receipt distinguishes those output bytes from zero temporary heap bytes and
/// zero dense-weight bytes.
///
/// # Errors
/// Returns the errors documented by [`salt_v2_matvec_into`], plus
/// [`SaltV2CpuError::AllocationFailed`] if the output cannot be reserved.
pub fn salt_v2_matvec(
    package: &SaltV2Package,
    tensor_index: usize,
    activation: &[f32],
) -> Result<SaltV2MatVec, SaltV2CpuError> {
    let tensor =
        package
            .tensors()
            .get(tensor_index)
            .ok_or(SaltV2CpuError::TensorIndexOutOfBounds {
                index: tensor_index,
                tensors: package.tensors().len(),
            })?;
    reject_unsupported_transform(tensor)?;
    let (rows, columns) = matrix_shape(tensor.dims(), tensor.logical_coefficients())?;
    validate_activation(activation, columns)?;
    let mut output = Vec::new();
    output
        .try_reserve_exact(rows)
        .map_err(|_| SaltV2CpuError::AllocationFailed {
            requested_elements: rows,
        })?;
    output.resize(rows, 0.0);
    let receipt = salt_v2_matvec_into(package, tensor_index, activation, &mut output)?;
    Ok(SaltV2MatVec { output, receipt })
}

fn validate_activation(activation: &[f32], expected_columns: usize) -> Result<(), SaltV2CpuError> {
    if activation.len() != expected_columns {
        return Err(SaltV2CpuError::InputLengthMismatch {
            expected: expected_columns,
            got: activation.len(),
        });
    }
    for (index, value) in activation.iter().copied().enumerate() {
        if !value.is_finite() {
            return Err(SaltV2CpuError::NonFiniteActivation {
                index,
                bits: value.to_bits(),
            });
        }
    }
    Ok(())
}

fn matrix_shape(
    dimensions: &[u64],
    tensor_coefficients: usize,
) -> Result<(usize, usize), SaltV2CpuError> {
    if dimensions.len() != 2 {
        return Err(SaltV2CpuError::ExpectedMatrix {
            rank: dimensions.len(),
        });
    }
    let rows = usize::try_from(dimensions[0]).map_err(|_| SaltV2CpuError::DimensionTooLarge {
        axis: 0,
        value: dimensions[0],
    })?;
    let columns =
        usize::try_from(dimensions[1]).map_err(|_| SaltV2CpuError::DimensionTooLarge {
            axis: 1,
            value: dimensions[1],
        })?;
    let shape_coefficients = rows
        .checked_mul(columns)
        .ok_or(SaltV2CpuError::DimensionProductOverflow)?;
    if shape_coefficients != tensor_coefficients {
        return Err(SaltV2CpuError::SemanticLengthMismatch {
            shape_coefficients,
            tensor_coefficients,
        });
    }
    Ok((rows, columns))
}

fn reject_unsupported_transform(tensor: &SaltV2Tensor) -> Result<(), SaltV2CpuError> {
    match tensor.transform() {
        SaltV2Transform::None => Ok(()),
        transform => Err(SaltV2CpuError::UnsupportedTransform { transform }),
    }
}

fn reconstruct_coefficient(
    tensor: &SaltV2Tensor,
    linear_index: usize,
) -> Result<f32, SaltV2CpuError> {
    let tile_index = linear_index / SALT_V2_ALLOCATION_TILE_SIZE;
    let local_index = linear_index % SALT_V2_ALLOCATION_TILE_SIZE;
    let tile = tensor
        .tiles()
        .get(tile_index)
        .ok_or(SaltV2CpuError::SemanticLayoutMismatch {
            coefficient: linear_index,
        })?;
    let mut weight = 0.0_f32;
    for plane in tile.planes() {
        let trit =
            plane
                .trits()
                .get(local_index)
                .ok_or(SaltV2CpuError::SemanticLayoutMismatch {
                    coefficient: linear_index,
                })?;
        let scale = plane
            .scales()
            .get(local_index / tensor.scale_group_size())
            .ok_or(SaltV2CpuError::SemanticLayoutMismatch {
                coefficient: linear_index,
            })?;
        weight += trit.to_f32() * scale.to_f32();
    }
    Ok(weight)
}

fn work_receipt(
    codec: SaltV2Codec,
    tensor: &SaltV2Tensor,
    rows: usize,
    columns: usize,
) -> Result<SaltV2MatVecReceipt, SaltV2CpuError> {
    let weight_coefficients_visited =
        u64::try_from(tensor.logical_coefficients()).map_err(|_| {
            SaltV2CpuError::AccountingOverflow {
                field: "weight coefficient",
            }
        })?;
    let mut plane_coefficients_visited = 0_u64;
    for tile in tensor.tiles() {
        let tile_coefficients =
            u64::try_from(tile.logical_len()).map_err(|_| SaltV2CpuError::AccountingOverflow {
                field: "plane coefficient",
            })?;
        let plane_count =
            u64::try_from(tile.planes().len()).map_err(|_| SaltV2CpuError::AccountingOverflow {
                field: "plane coefficient",
            })?;
        plane_coefficients_visited = plane_coefficients_visited
            .checked_add(tile_coefficients.checked_mul(plane_count).ok_or(
                SaltV2CpuError::AccountingOverflow {
                    field: "plane coefficient",
                },
            )?)
            .ok_or(SaltV2CpuError::AccountingOverflow {
                field: "plane coefficient",
            })?;
    }
    let output_bytes = u64::try_from(rows)
        .map_err(|_| SaltV2CpuError::AccountingOverflow {
            field: "output byte",
        })?
        .checked_mul(u64::try_from(core::mem::size_of::<f32>()).map_err(|_| {
            SaltV2CpuError::AccountingOverflow {
                field: "output byte",
            }
        })?)
        .ok_or(SaltV2CpuError::AccountingOverflow {
            field: "output byte",
        })?;
    Ok(SaltV2MatVecReceipt {
        codec,
        rows,
        columns,
        weight_coefficients_visited,
        plane_coefficients_visited,
        output_bytes,
        temporary_heap_bytes: 0,
        dense_weight_bytes: 0,
        reduction_order: SaltV2ReductionOrder::RowMajorGroupThenPlaneThenColumn,
    })
}

#[cfg(test)]
mod tests {
    #[allow(unsafe_code)]
    mod allocation_probe {
        use std::alloc::{GlobalAlloc, Layout, System};
        use std::cell::Cell;

        thread_local! {
            static TRACKING: Cell<bool> = const { Cell::new(false) };
            static ALLOCATIONS: Cell<usize> = const { Cell::new(0) };
        }

        pub(super) struct TrackingSystem;

        // SAFETY: every allocation operation delegates to `System` with the
        // exact pointer and layout supplied by the caller.
        unsafe impl GlobalAlloc for TrackingSystem {
            unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
                record_allocation();
                // SAFETY: This allocator delegates the unchanged layout to the
                // process-wide system allocator.
                unsafe { System.alloc(layout) }
            }

            unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
                record_allocation();
                // SAFETY: This allocator delegates the unchanged layout to the
                // process-wide system allocator.
                unsafe { System.alloc_zeroed(layout) }
            }

            unsafe fn dealloc(&self, pointer: *mut u8, layout: Layout) {
                // SAFETY: `pointer` and `layout` came from this allocator's
                // direct delegation to `System`.
                unsafe { System.dealloc(pointer, layout) }
            }

            unsafe fn realloc(&self, pointer: *mut u8, layout: Layout, new_size: usize) -> *mut u8 {
                record_allocation();
                // SAFETY: `pointer` and `layout` came from `System`; the new
                // size is forwarded unchanged.
                unsafe { System.realloc(pointer, layout, new_size) }
            }
        }

        fn record_allocation() {
            let tracking = TRACKING.try_with(Cell::get).unwrap_or(false);
            if tracking {
                let _ = ALLOCATIONS.try_with(|allocations| {
                    allocations.set(allocations.get().saturating_add(1));
                });
            }
        }

        pub(super) fn count_during<T>(operation: impl FnOnce() -> T) -> (T, usize) {
            struct TrackingGuard;

            impl Drop for TrackingGuard {
                fn drop(&mut self) {
                    let _ = TRACKING.try_with(|tracking| tracking.set(false));
                }
            }

            TRACKING.with(|tracking| {
                assert!(!tracking.replace(true), "allocation probe cannot be nested");
            });
            ALLOCATIONS.with(|allocations| allocations.set(0));
            let guard = TrackingGuard;
            let value = operation();
            let count = ALLOCATIONS.with(Cell::get);
            drop(guard);
            (value, count)
        }
    }

    #[global_allocator]
    static TEST_ALLOCATOR: allocation_probe::TrackingSystem = allocation_probe::TrackingSystem;

    use half::f16;
    use tritium_format::salt_v2::SaltV2Codec;
    use tritium_format::salt_v2_package::{
        SALT_V2_ALLOCATION_TILE_SIZE, SALT_V2_SCALE_GROUP_SIZE, SaltV2Package, SaltV2Plane,
        SaltV2Tensor, SaltV2Tile, SaltV2Transform, read_salt_v2_package, write_salt_v2_package,
    };

    use super::*;

    fn plane(len: usize, plane_index: usize, tile_index: usize) -> SaltV2Plane {
        let trits = (0..len)
            .map(|index| {
                if index % 4 == 0 {
                    0
                } else if (index + plane_index + tile_index).is_multiple_of(2) {
                    1
                } else {
                    -1
                }
            })
            .collect();
        let scales = (0..len.div_ceil(SALT_V2_SCALE_GROUP_SIZE))
            .map(|group| {
                f16::from_f32(
                    0.25 + plane_index as f32 * 0.125
                        + tile_index as f32 * 0.03125
                        + group as f32 * 0.015625,
                )
            })
            .collect();
        SaltV2Plane::new(trits, scales).expect("valid reference plane")
    }

    fn tile(len: usize, plane_count: usize, tile_index: usize) -> SaltV2Tile {
        SaltV2Tile::new(
            (0..plane_count)
                .map(|plane_index| plane(len, plane_index, tile_index))
                .collect(),
        )
        .expect("valid reference tile")
    }

    fn matrix_package(codec: SaltV2Codec) -> SaltV2Package {
        let tensor = SaltV2Tensor::new(
            "projection",
            vec![3, 173],
            vec![tile(256, 1, 0), tile(256, 3, 1), tile(7, 2, 2)],
        )
        .expect("valid ragged matrix");
        SaltV2Package::new(codec, vec![tensor]).expect("valid package")
    }

    fn g64_matrix_package(codec: SaltV2Codec) -> SaltV2Package {
        let tiles = (0..9)
            .map(|tile_index| {
                let trits = (0..256)
                    .map(|index| match (index + tile_index) % 4 {
                        0 => 0,
                        1 | 2 => 1,
                        _ => -1,
                    })
                    .collect();
                let scales = (0..4)
                    .map(|group| f16::from_f32(0.125 + (tile_index * 4 + group) as f32 / 64.0))
                    .collect();
                SaltV2Tile::new(vec![
                    SaltV2Plane::new_with_scale_group_size(trits, scales, 64).unwrap(),
                ])
                .unwrap()
            })
            .collect();
        let tensor = SaltV2Tensor::new_with_layout(
            "g64-projection",
            vec![4, 576],
            SaltV2Transform::None,
            64,
            tiles,
        )
        .unwrap();
        SaltV2Package::new(codec, vec![tensor]).unwrap()
    }

    fn independent_dense_reconstruction(tensor: &SaltV2Tensor) -> Vec<f32> {
        (0..tensor.logical_coefficients())
            .map(|coefficient| {
                let tile = &tensor.tiles()[coefficient / SALT_V2_ALLOCATION_TILE_SIZE];
                let local = coefficient % SALT_V2_ALLOCATION_TILE_SIZE;
                tile.planes().iter().fold(0.0_f32, |weight, plane| {
                    let scale = plane.scales()[local / SALT_V2_SCALE_GROUP_SIZE].to_f32();
                    weight + plane.trits()[local].to_f32() * scale
                })
            })
            .collect()
    }

    fn independent_dense_matvec(
        dense_weights: &[f32],
        rows: usize,
        columns: usize,
        x: &[f32],
    ) -> Vec<f32> {
        assert_eq!(dense_weights.len(), rows * columns);
        dense_weights
            .chunks_exact(columns)
            .map(|dense_row| {
                dense_row
                    .iter()
                    .copied()
                    .zip(x.iter().copied())
                    .fold(0.0_f32, |sum, (weight, activation)| {
                        sum + weight * activation
                    })
            })
            .collect()
    }

    #[test]
    fn every_codec_matches_independent_dense_math_for_ragged_planes_and_rows() {
        let activation = (0..173)
            .map(|index| (index as f32 - 81.0) / 29.0)
            .collect::<Vec<_>>();

        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let package = matrix_package(codec);
            let encoded = write_salt_v2_package(&package).expect("encode physical package");
            let decoded = read_salt_v2_package(&encoded.bytes).expect("decode physical package");
            let dense_weights = independent_dense_reconstruction(&decoded.package.tensors()[0]);
            let expected = independent_dense_matvec(&dense_weights, 3, 173, &activation);
            let actual = salt_v2_matvec(&decoded.package, 0, &activation)
                .expect("reference matvec over decoded hard trits");
            let repeated = salt_v2_matvec(&decoded.package, 0, &activation)
                .expect("repeat deterministic reference matvec");

            for (row, (actual, expected)) in actual.output.iter().zip(&expected).enumerate() {
                let tolerance = 2.0e-5_f32 * expected.abs().max(1.0);
                assert!(
                    (actual - expected).abs() <= tolerance,
                    "codec {codec:?} row {row}: group-plane {actual} versus dense {expected}"
                );
            }
            assert_eq!(actual, repeated, "codec {codec:?} must be bit-stable");
            assert_eq!(actual.receipt.codec(), codec);
            assert_eq!(actual.receipt.rows(), 3);
            assert_eq!(actual.receipt.columns(), 173);
            assert_eq!(actual.receipt.weight_coefficients_visited(), 519);
            assert_eq!(actual.receipt.plane_coefficients_visited(), 1_038);
            assert_eq!(actual.receipt.output_bytes(), 12);
            assert_eq!(actual.receipt.temporary_heap_bytes(), 0);
            assert_eq!(actual.receipt.dense_weight_bytes(), 0);
            assert_eq!(
                actual.receipt.reduction_order(),
                SaltV2ReductionOrder::RowMajorGroupThenPlaneThenColumn
            );
        }
    }

    #[test]
    fn g64_runtime_matches_independent_dense_matvec() {
        let activation = (0..576)
            .map(|index| (index as f32 - 283.0) / 71.0)
            .collect::<Vec<_>>();
        for codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
            let package = g64_matrix_package(codec);
            let encoded = write_salt_v2_package(&package).unwrap();
            let decoded = read_salt_v2_package(&encoded.bytes).unwrap();
            let tensor = &decoded.package.tensors()[0];
            let dense = (0..tensor.logical_coefficients())
                .map(|coefficient| {
                    let tile = &tensor.tiles()[coefficient / SALT_V2_ALLOCATION_TILE_SIZE];
                    let local = coefficient % SALT_V2_ALLOCATION_TILE_SIZE;
                    tile.planes()[0].trits()[local].to_f32()
                        * tile.planes()[0].scales()[local / 64].to_f32()
                })
                .collect::<Vec<_>>();
            let expected = independent_dense_matvec(&dense, 4, 576, &activation);
            let actual = salt_v2_matvec(&decoded.package, 0, &activation).unwrap();
            for (got, expected) in actual.output.iter().zip(expected) {
                let tolerance = 2.0e-5 * expected.abs().max(1.0);
                assert!((got - expected).abs() <= tolerance);
            }
        }
    }

    #[test]
    fn caller_owned_output_path_allocates_no_heap_scratch() {
        let package = matrix_package(SaltV2Codec::B3);
        let activation = vec![0.5; 173];
        let mut output = [0.0; 3];

        let receipt = salt_v2_matvec_into(&package, 0, &activation, &mut output)
            .expect("reference matvec into");

        assert!(output.iter().all(|value| value.is_finite()));
        assert_eq!(receipt.temporary_heap_bytes(), 0);
        assert_eq!(receipt.dense_weight_bytes(), 0);
        assert_eq!(receipt.output_bytes(), 12);
    }

    #[test]
    fn coefficient_reconstruction_handles_tile_and_group_boundaries() {
        let package = matrix_package(SaltV2Codec::D2);
        let tensor = &package.tensors()[0];
        let dense = independent_dense_reconstruction(tensor);

        for index in [0, 127, 128, 255, 256, 511, 512, 518] {
            assert_eq!(salt_v2_coefficient(tensor, index), Ok(dense[index]));
        }
        assert_eq!(
            salt_v2_coefficient(tensor, 519),
            Err(SaltV2CpuError::CoefficientIndexOutOfBounds {
                index: 519,
                coefficients: 519,
            })
        );
    }

    #[test]
    fn rejects_bad_rank_lengths_and_tensor_index() {
        let vector =
            SaltV2Tensor::new("vector", vec![3], vec![tile(3, 1, 0)]).expect("valid vector tensor");
        let vector_package =
            SaltV2Package::new(SaltV2Codec::D2, vec![vector]).expect("valid package");
        let mut one_output = [0.0];
        assert_eq!(
            salt_v2_matvec_into(&vector_package, 0, &[1.0; 3], &mut one_output),
            Err(SaltV2CpuError::ExpectedMatrix { rank: 1 })
        );

        let package = matrix_package(SaltV2Codec::D2);
        let mut output = [0.0; 3];
        assert_eq!(
            salt_v2_matvec_into(&package, 1, &[0.0; 173], &mut output),
            Err(SaltV2CpuError::TensorIndexOutOfBounds {
                index: 1,
                tensors: 1,
            })
        );
        assert_eq!(
            salt_v2_matvec_into(&package, 0, &[0.0; 172], &mut output),
            Err(SaltV2CpuError::InputLengthMismatch {
                expected: 173,
                got: 172,
            })
        );
        assert_eq!(
            salt_v2_matvec_into(&package, 0, &[0.0; 173], &mut output[..2]),
            Err(SaltV2CpuError::OutputLengthMismatch {
                expected: 3,
                got: 2,
            })
        );
    }

    #[test]
    fn checked_shape_logic_rejects_overflow_and_semantic_mismatch() {
        assert_eq!(
            matrix_shape(&[usize::MAX as u64, 2], 0),
            Err(SaltV2CpuError::DimensionProductOverflow)
        );
        assert_eq!(
            matrix_shape(&[2, 3], 5),
            Err(SaltV2CpuError::SemanticLengthMismatch {
                shape_coefficients: 6,
                tensor_coefficients: 5,
            })
        );
    }

    #[test]
    fn rejects_nonfinite_activation_before_writing_output() {
        let package = matrix_package(SaltV2Codec::D2);
        let mut activation = vec![0.0; 173];
        activation[71] = f32::NAN;
        let mut output = [7.0; 3];

        assert_eq!(
            salt_v2_matvec_into(&package, 0, &activation, &mut output),
            Err(SaltV2CpuError::NonFiniteActivation {
                index: 71,
                bits: f32::NAN.to_bits(),
            })
        );
        assert_eq!(output, [7.0; 3]);
    }

    #[test]
    fn allocating_path_preflights_activation_before_output_allocation() {
        let package = matrix_package(SaltV2Codec::D2);
        let short_activation = vec![0.0; 172];
        let mut nonfinite_activation = vec![0.0; 173];
        nonfinite_activation[71] = f32::NAN;

        let (short_result, short_allocations) =
            allocation_probe::count_during(|| salt_v2_matvec(&package, 0, &short_activation));
        let (nonfinite_result, nonfinite_allocations) =
            allocation_probe::count_during(|| salt_v2_matvec(&package, 0, &nonfinite_activation));

        assert_eq!(
            short_result,
            Err(SaltV2CpuError::InputLengthMismatch {
                expected: 173,
                got: 172,
            })
        );
        assert_eq!(
            nonfinite_result,
            Err(SaltV2CpuError::NonFiniteActivation {
                index: 71,
                bits: f32::NAN.to_bits(),
            })
        );
        assert_eq!(short_allocations, 0);
        assert_eq!(nonfinite_allocations, 0);
    }

    #[test]
    fn rejects_nonfinite_output_without_storing_it() {
        let plane = SaltV2Plane::new(vec![1], vec![f16::MAX]).expect("valid max scale");
        let tile = SaltV2Tile::new(vec![plane]).expect("valid tile");
        let tensor = SaltV2Tensor::new("overflow", vec![1, 1], vec![tile]).expect("valid matrix");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
        let mut output = [19.0];

        assert_eq!(
            salt_v2_matvec_into(&package, 0, &[f32::MAX], &mut output),
            Err(SaltV2CpuError::NonFiniteOutput { row: 0, column: 0 })
        );
        assert_eq!(output, [19.0]);
    }

    #[test]
    fn group_plane_cancellation_happens_before_f16_scale_application() {
        let plane = SaltV2Plane::new(vec![1, -1], vec![f16::MAX]).expect("valid cancelling plane");
        let tile = SaltV2Tile::new(vec![plane]).expect("valid tile");
        let tensor =
            SaltV2Tensor::new("cancellation", vec![1, 2], vec![tile]).expect("valid matrix");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");

        let actual = salt_v2_matvec(&package, 0, &[f32::MAX, f32::MAX])
            .expect("opposite trits cancel before the maximum f16 scale is applied");

        assert_eq!(actual.output, [0.0]);
    }

    #[test]
    fn signed_rht_fails_closed_before_writing_output() {
        let transform = tritium_format::salt_v2_package::SaltV2Transform::SignedRht {
            seed: 7,
            domain: 11,
        };
        let tensor = SaltV2Tensor::new_with_transform(
            "rotated",
            vec![1, 2],
            transform,
            vec![
                SaltV2Tile::new(vec![
                    SaltV2Plane::new(vec![1, -1], vec![f16::ONE]).expect("valid plane"),
                ])
                .expect("valid tile"),
            ],
        )
        .expect("valid transformed matrix");
        let package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).expect("valid package");
        let mut output = [23.0];

        assert_eq!(
            salt_v2_matvec_into(&package, 0, &[1.0, 2.0], &mut output),
            Err(SaltV2CpuError::UnsupportedTransform { transform })
        );
        assert_eq!(output, [23.0]);
        assert_eq!(
            salt_v2_coefficient(&package.tensors()[0], 0),
            Err(SaltV2CpuError::UnsupportedTransform { transform })
        );
    }

    #[test]
    fn fixture_really_covers_short_tail_and_ragged_p() {
        let package = matrix_package(SaltV2Codec::D2);
        let tensor = &package.tensors()[0];
        assert_eq!(tensor.tiles().len(), 3);
        assert_eq!(tensor.tiles()[2].logical_len(), 7);
        assert_eq!(
            tensor
                .tiles()
                .iter()
                .map(|tile| tile.planes().len())
                .collect::<Vec<_>>(),
            [1, 3, 2]
        );
        assert_eq!(SALT_V2_ALLOCATION_TILE_SIZE, 256);
    }
}
