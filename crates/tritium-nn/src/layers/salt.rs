//! Packed additive SALT projection.
//!
//! [`SaltLinear`] retains dense and sparse additive planes and reconstructs one
//! 256-weight block at a time during the A8 contraction. It matches
//! [`DenseLinear`](super::DenseLinear) bit-for-bit without retaining an `N × K`
//! fp32 matrix.

use tritium_format::{PackedSaltRow, SaltRow};
use tritium_train::ops::ste::{fast_hadamard, group_is_rotatable};

use crate::error::NnError;
use crate::layers::packed_salt::PackedSaltMatrix;
use crate::ops::{
    quantize_activation_int8, quantize_activation_int8_grouped, quantize_activation_ternary,
};

/// A bias-free additive ternary projection backed by packed SALT rows.
#[derive(Clone, Debug)]
pub struct SaltLinear {
    matrix: PackedSaltMatrix,
    /// Hadamard group width the weights were fitted under, from the bundle header.
    ///
    /// `Some(g)` means the stored weights are `W·H`, so this projection MUST rotate each `g`-wide
    /// slice of the activation before quantizing: `H·H = I`, hence `W·x = (W·H)·(H·x)`. Skipping it
    /// computes `W·H·x` — wrong, but not detectably wrong, which is why the bundle version gates it.
    rotation_group: Option<usize>,
    /// How the activation is quantized before the projection. See [`ActivationPrecision`].
    activation: ActivationPrecision,
}

/// How a SALT projection quantizes its activation.
///
/// Every variant costs the same memory — activations are transient — so this is an accuracy and
/// arithmetic choice, not a size one.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
#[non_exhaustive]
pub enum ActivationPrecision {
    /// One int8 absmax per token, over the whole row. BitNet's reference quantizer and what ships.
    #[default]
    PerToken,
    /// One int8 absmax per `g` inputs. Recovers 64% of the A8 tax on an UNROTATED artifact and
    /// almost nothing on a rotated one — the Hadamard already whitens the outliers it exploits.
    PerGroup(usize),
    /// Additive ternary planes on the geometric ladder, `planes·log2(3)` bits per value: the weight
    /// side's representation applied to the activation, so a kernel could contract trits against
    /// trits with no multiplies. `planes = 5` is 7.92 bits, under int8, at 243 of its 256 levels.
    ///
    /// No such kernel exists here yet — `PackedSaltMatrix::project_rows` reconstructs to f32 and
    /// multiplies — so today this buys the accuracy and not the arithmetic.
    Ternary {
        /// Ternary planes per value, `1..=9`.
        planes: usize,
        /// Inputs sharing one ladder anchor.
        group: usize,
    },
}

impl SaltLinear {
    pub(crate) fn from_packed_matrix(
        matrix: PackedSaltMatrix,
        rotation_group: Option<usize>,
    ) -> Self {
        Self {
            matrix,
            rotation_group,
            activation: ActivationPrecision::PerToken,
        }
    }

    /// Set how this projection quantizes its activation. See [`ActivationPrecision`].
    pub fn set_activation_precision(&mut self, precision: ActivationPrecision) {
        self.activation = precision;
    }

    /// Switch to per-group int8, or back to the shipping per-token absmax.
    pub fn set_activation_group(&mut self, group: Option<usize>) {
        self.activation = match group {
            Some(g) => ActivationPrecision::PerGroup(g),
            None => ActivationPrecision::PerToken,
        };
    }

    /// The activation precision in force.
    #[must_use]
    pub const fn activation_precision(&self) -> ActivationPrecision {
        self.activation
    }

    /// Build a projection from one packed SALT row per output channel.
    ///
    /// Validation consumes fixed one-block scratch; no dense row or matrix is materialized.
    ///
    /// # Errors
    /// [`NnError::Shape`] if dimensions or packed row geometry disagree, or
    /// [`NnError::Backend`] if a TQ2_0 plane is malformed.
    pub fn new(rows: Vec<SaltRow>, n_out: usize, k_in: usize) -> Result<Self, NnError> {
        if n_out == 0 || rows.len() != n_out {
            return Err(NnError::Shape {
                expected: n_out.max(1),
                got: rows.len(),
            });
        }
        if k_in == 0 || u32::try_from(k_in).is_err() {
            return Err(NnError::Shape {
                expected: if k_in == 0 { 1 } else { u32::MAX as usize },
                got: k_in,
            });
        }
        if let Some(row) = rows.iter().find(|row| row.k != k_in) {
            return Err(NnError::Shape {
                expected: k_in,
                got: row.k,
            });
        }
        let rows = rows
            .into_iter()
            .map(PackedSaltRow::try_from)
            .collect::<Result<Vec<_>, _>>()
            .map_err(|error| NnError::Backend(error.to_string()))?;
        Self::from_packed_rows(rows, n_out, k_in)
    }

    /// Build directly from validated dense/sparse SALT rows without expanding residuals.
    ///
    /// # Errors
    /// [`NnError::Shape`] if matrix geometry disagrees with the rows, or
    /// [`NnError::Backend`] if a packed plane contains a non-finite scale.
    pub fn from_packed_rows(
        rows: Vec<PackedSaltRow>,
        n_out: usize,
        k_in: usize,
    ) -> Result<Self, NnError> {
        Ok(Self {
            matrix: PackedSaltMatrix::new(rows, n_out, k_in)?,
            // Raw rows carry no bundle header, so there is nothing to say they were fitted in a
            // rotated basis. Callers that know otherwise build through the bundle path.
            rotation_group: None,
            activation: ActivationPrecision::PerToken,
        })
    }

    /// Packed plane payload bytes retained by this projection.
    #[must_use]
    pub fn packed_bytes(&self) -> usize {
        self.matrix.packed_bytes()
    }

    /// Total retained arena and row/plane metadata bytes, excluding the struct itself.
    /// Cloned projections share arenas, so summing this value across clones double-counts them.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        self.matrix.resident_bytes()
    }

    /// Number of residual planes retained in sparse form.
    #[must_use]
    pub const fn sparse_plane_count(&self) -> usize {
        self.matrix.sparse_plane_count()
    }

    /// Output feature count (`N`).
    #[must_use]
    pub const fn n_out(&self) -> usize {
        self.matrix.n_out()
    }

    /// Input feature count (`K`).
    #[must_use]
    pub const fn k_in(&self) -> usize {
        self.matrix.k_in()
    }

    /// A8 forward matching `salt_rows_to_dense → DenseLinear::new` bit-for-bit.
    ///
    /// Packed rows remain resident. Each worker uses fixed 256-element reconstruction
    /// scratch, preserving plane-order weight reconstruction and global `K` dot order.
    ///
    /// # Errors
    /// [`NnError::Shape`] on operand/output mismatch, or [`NnError::Backend`] if
    /// activation scratch cannot be allocated or a retained packed block cannot
    /// be decoded.
    pub fn forward(&self, act: &[f32], m: usize, out: &mut [f32]) -> Result<(), NnError> {
        let act_len = m.checked_mul(self.k_in()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: act.len(),
        })?;
        if act.len() != act_len {
            return Err(NnError::Shape {
                expected: act_len,
                got: act.len(),
            });
        }
        let out_len = m.checked_mul(self.n_out()).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: out.len(),
        })?;
        if out.len() != out_len {
            return Err(NnError::Shape {
                expected: out_len,
                got: out.len(),
            });
        }

        // Rotate BEFORE quantizing. The fit is in the rotated basis, so the int8 grid the
        // activation lands on must be the rotated one too — and the Hadamard whitens, which should
        // make the activation easier to quantize, not harder.
        let rotated_scratch;
        let act = match self.rotation_group {
            None => act,
            Some(group) => {
                let mut buf = act.to_vec();
                for row in buf.chunks_mut(self.k_in()) {
                    for slice in row.chunks_mut(group) {
                        // `k_in` need not divide by `group` — SmolLM2's `n_embd` is 576 against the
                        // shipping `g256`, leaving a 64-wide tail the fitter does not rotate.
                        if group_is_rotatable(slice.len()) {
                            fast_hadamard(slice);
                        }
                    }
                }
                rotated_scratch = buf;
                &rotated_scratch
            }
        };

        let mut q_act = zeroed_scratch(act_len, "SALT quantized activations")?;
        let mut act_scale = zeroed_scratch(m, "SALT activation scales")?;
        match self.activation {
            ActivationPrecision::PerToken => {
                quantize_activation_int8(act, m, self.k_in(), &mut q_act, &mut act_scale)?;
            }
            // These write dequantized values and a scale of 1, so the fold below is unchanged.
            ActivationPrecision::PerGroup(group) => quantize_activation_int8_grouped(
                act,
                m,
                self.k_in(),
                group,
                &mut q_act,
                &mut act_scale,
            )?,
            ActivationPrecision::Ternary { planes, group } => quantize_activation_ternary(
                act,
                m,
                self.k_in(),
                planes,
                group,
                &mut q_act,
                &mut act_scale,
            )?,
        }

        self.matrix.project_rows(&q_act, m, out)?;
        for (row, scale) in out.chunks_mut(self.n_out()).zip(act_scale) {
            for value in row {
                *value *= scale;
            }
        }
        Ok(())
    }
}

fn zeroed_scratch(len: usize, name: &str) -> Result<Vec<f32>, NnError> {
    let mut values = Vec::new();
    values.try_reserve_exact(len).map_err(|error| {
        NnError::Backend(format!(
            "allocate {name} scratch for {len} f32 values: {error}"
        ))
    })?;
    values.resize(len, 0.0);
    Ok(values)
}
