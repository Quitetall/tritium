//! Grouped ternary 2-D convolution forward and VJP for portable training.
//!
//! Layout is NCHW. Forward lowers each `(batch, group)` to a canonical
//! `im2col` matrix and reuses ternary [`matmul`](super::matmul). Backward uses
//! matching matmul VJP plus ordered `col2im` scatter, keeping overlap sums
//! deterministic across portable backends.

use core::fmt;

use super::matmul;

/// Maximum spatial output rows materialized by portable Conv2d im2col.
pub const CONV2D_PATCH_TILE_ROWS: usize = 32;

/// Invalid Conv2d geometry or caller-owned buffer contract.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum Conv2dError {
    /// Geometry contains zero, incompatible grouped channels, or no output.
    InvalidGeometry(&'static str),
    /// Checked geometry arithmetic exceeded host `usize`.
    ArithmeticOverflow,
    /// One supplied flat buffer has wrong element count.
    BufferLength {
        /// Stable buffer label.
        buffer: &'static str,
        /// Exact count required by geometry.
        expected: usize,
        /// Count supplied by caller.
        got: usize,
    },
}

impl fmt::Display for Conv2dError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidGeometry(field) => write!(formatter, "invalid Conv2d geometry: {field}"),
            Self::ArithmeticOverflow => formatter.write_str("Conv2d geometry arithmetic overflow"),
            Self::BufferLength {
                buffer,
                expected,
                got,
            } => write!(
                formatter,
                "Conv2d {buffer} needs {expected} elements, received {got}"
            ),
        }
    }
}

impl std::error::Error for Conv2dError {}

#[derive(Clone, Copy)]
struct Geometry {
    height_out: usize,
    width_out: usize,
    c_in_per_group: usize,
    c_out_per_group: usize,
    patch_rows: usize,
    patch_columns: usize,
    input_elements: usize,
    weight_elements: usize,
    output_elements: usize,
    scratch_elements: usize,
}

/// Explicit asymmetric geometry for NCHW 2-D convolution.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Conv2dCfg {
    /// Batch size.
    pub batch: usize,
    /// Input channels, divisible by `groups`.
    pub c_in: usize,
    /// Output channels, divisible by `groups`.
    pub c_out: usize,
    /// Input height.
    pub input_h: usize,
    /// Input width.
    pub input_w: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Vertical stride.
    pub stride_h: usize,
    /// Horizontal stride.
    pub stride_w: usize,
    /// Vertical dilation.
    pub dilation_h: usize,
    /// Horizontal dilation.
    pub dilation_w: usize,
    /// Top zero padding.
    pub pad_top: usize,
    /// Bottom zero padding.
    pub pad_bottom: usize,
    /// Left zero padding.
    pub pad_left: usize,
    /// Right zero padding.
    pub pad_right: usize,
    /// Convolution groups.
    pub groups: usize,
}

impl Conv2dCfg {
    /// Output `(height, width)`, or `(0, 0)` for malformed/degenerate geometry.
    #[must_use]
    pub fn output_hw(self) -> (usize, usize) {
        self.geometry()
            .map(|geometry| (geometry.height_out, geometry.width_out))
            .unwrap_or((0, 0))
    }

    /// Input channels owned by one group.
    #[must_use]
    pub fn c_in_per_group(self) -> usize {
        if self.groups != 0 && self.c_in.is_multiple_of(self.groups) {
            self.c_in / self.groups
        } else {
            0
        }
    }

    /// Output channels owned by one group.
    #[must_use]
    pub fn c_out_per_group(self) -> usize {
        if self.groups != 0 && self.c_out.is_multiple_of(self.groups) {
            self.c_out / self.groups
        } else {
            0
        }
    }

    /// Flattened kernel coefficients for one output channel.
    #[must_use]
    pub fn kernel_elements_per_output(self) -> usize {
        product(&[self.c_in_per_group(), self.kernel_h, self.kernel_w]).unwrap_or(0)
    }

    /// Expected input-buffer elements.
    #[must_use]
    pub fn input_elements(self) -> usize {
        product(&[self.batch, self.c_in, self.input_h, self.input_w]).unwrap_or(0)
    }

    /// Expected weight-buffer elements.
    #[must_use]
    pub fn weight_elements(self) -> usize {
        product(&[self.c_out, self.kernel_elements_per_output()]).unwrap_or(0)
    }

    /// Expected output-buffer elements.
    #[must_use]
    pub fn output_elements(self) -> usize {
        self.geometry()
            .map(|geometry| geometry.output_elements)
            .unwrap_or(0)
    }

    /// Maximum temporary `f32` elements used by one forward patch/output tile.
    ///
    /// This excludes caller-owned output and VJP result buffers.
    #[must_use]
    pub fn max_scratch_elements(self) -> usize {
        self.geometry()
            .map(|geometry| geometry.scratch_elements)
            .unwrap_or(0)
    }

    /// Whether geometry and supplied buffers form one supported convolution.
    #[must_use]
    pub fn buffers_fit(
        self,
        input_len: usize,
        weight_len: usize,
        scale_len: usize,
        output_len: usize,
    ) -> bool {
        self.validate_buffers(input_len, weight_len, scale_len, output_len)
            .is_ok()
    }

    /// Validate geometry and exact flat-buffer lengths without allocation.
    ///
    /// # Errors
    /// Returns [`Conv2dError`] for malformed geometry, arithmetic overflow, or
    /// any mismatched buffer length.
    pub fn validate_buffers(
        self,
        input_len: usize,
        weight_len: usize,
        scale_len: usize,
        output_len: usize,
    ) -> Result<(), Conv2dError> {
        let geometry = self.geometry()?;
        for (buffer, expected, got) in [
            ("input", geometry.input_elements, input_len),
            ("weight", geometry.weight_elements, weight_len),
            ("scale", self.c_out, scale_len),
            ("output", geometry.output_elements, output_len),
        ] {
            if expected != got {
                return Err(Conv2dError::BufferLength {
                    buffer,
                    expected,
                    got,
                });
            }
        }
        Ok(())
    }

    fn geometry(self) -> Result<Geometry, Conv2dError> {
        if self.batch == 0
            || self.c_in == 0
            || self.c_out == 0
            || self.input_h == 0
            || self.input_w == 0
            || self.kernel_h == 0
            || self.kernel_w == 0
            || self.stride_h == 0
            || self.stride_w == 0
            || self.dilation_h == 0
            || self.dilation_w == 0
            || self.groups == 0
        {
            return Err(Conv2dError::InvalidGeometry("zero dimension"));
        }
        if !self.c_in.is_multiple_of(self.groups) || !self.c_out.is_multiple_of(self.groups) {
            return Err(Conv2dError::InvalidGeometry("grouped channels"));
        }
        let height_out = checked_output_axis(
            self.input_h,
            self.kernel_h,
            self.stride_h,
            self.dilation_h,
            self.pad_top,
            self.pad_bottom,
        )?;
        let width_out = checked_output_axis(
            self.input_w,
            self.kernel_w,
            self.stride_w,
            self.dilation_w,
            self.pad_left,
            self.pad_right,
        )?;
        if height_out == 0 || width_out == 0 {
            return Err(Conv2dError::InvalidGeometry("kernel exceeds padded input"));
        }
        let c_in_per_group = self.c_in / self.groups;
        let c_out_per_group = self.c_out / self.groups;
        let patch_rows = product(&[height_out, width_out])?;
        let patch_columns = product(&[c_in_per_group, self.kernel_h, self.kernel_w])?;
        let input_elements = product(&[self.batch, self.c_in, self.input_h, self.input_w])?;
        let weight_elements = product(&[self.c_out, patch_columns])?;
        let output_elements = product(&[self.batch, self.c_out, patch_rows])?;
        let tile_rows = patch_rows.min(CONV2D_PATCH_TILE_ROWS);
        let scratch_elements = product(&[tile_rows, patch_columns])?
            .checked_add(product(&[tile_rows, c_out_per_group])?)
            .ok_or(Conv2dError::ArithmeticOverflow)?;
        Ok(Geometry {
            height_out,
            width_out,
            c_in_per_group,
            c_out_per_group,
            patch_rows,
            patch_columns,
            input_elements,
            weight_elements,
            output_elements,
            scratch_elements,
        })
    }
}

fn checked_output_axis(
    input: usize,
    kernel: usize,
    stride: usize,
    dilation: usize,
    pad_before: usize,
    pad_after: usize,
) -> Result<usize, Conv2dError> {
    if kernel == 0 || stride == 0 || dilation == 0 {
        return Err(Conv2dError::InvalidGeometry("zero axis field"));
    }
    let Some(effective) = dilation
        .checked_mul(kernel - 1)
        .and_then(|value| value.checked_add(1))
    else {
        return Err(Conv2dError::ArithmeticOverflow);
    };
    let Some(padded) = input
        .checked_add(pad_before)
        .and_then(|value| value.checked_add(pad_after))
    else {
        return Err(Conv2dError::ArithmeticOverflow);
    };
    if padded < effective {
        Ok(0)
    } else {
        (padded - effective)
            .checked_div(stride)
            .and_then(|value| value.checked_add(1))
            .ok_or(Conv2dError::ArithmeticOverflow)
    }
}

fn product(values: &[usize]) -> Result<usize, Conv2dError> {
    values
        .iter()
        .try_fold(1usize, |total, value| total.checked_mul(*value))
        .ok_or(Conv2dError::ArithmeticOverflow)
}

fn input_index(
    cfg: Conv2dCfg,
    batch: usize,
    channel: usize,
    oh: usize,
    ow: usize,
    kh: usize,
    kw: usize,
) -> Option<usize> {
    let padded_h = oh
        .checked_mul(cfg.stride_h)?
        .checked_add(kh.checked_mul(cfg.dilation_h)?)?;
    let padded_w = ow
        .checked_mul(cfg.stride_w)?
        .checked_add(kw.checked_mul(cfg.dilation_w)?)?;
    let ih = padded_h.checked_sub(cfg.pad_top)?;
    let iw = padded_w.checked_sub(cfg.pad_left)?;
    if ih >= cfg.input_h || iw >= cfg.input_w {
        return None;
    }
    (((batch.checked_mul(cfg.c_in)?.checked_add(channel)?)
        .checked_mul(cfg.input_h)?
        .checked_add(ih)?)
    .checked_mul(cfg.input_w)?)
    .checked_add(iw)
}

fn im2col(
    input: &[f32],
    cfg: Conv2dCfg,
    geometry: Geometry,
    batch: usize,
    group: usize,
    row_start: usize,
    row_count: usize,
) -> Vec<f32> {
    let mut columns = vec![0.0; row_count * geometry.patch_columns];
    for local_row in 0..row_count {
        let row = row_start + local_row;
        let oh = row / geometry.width_out;
        let ow = row % geometry.width_out;
        for ci_local in 0..geometry.c_in_per_group {
            let channel = group * geometry.c_in_per_group + ci_local;
            for kh in 0..cfg.kernel_h {
                for kw in 0..cfg.kernel_w {
                    if let Some(index) = input_index(cfg, batch, channel, oh, ow, kh, kw) {
                        let column = (ci_local * cfg.kernel_h + kh) * cfg.kernel_w + kw;
                        columns[local_row * geometry.patch_columns + column] = input[index];
                    }
                }
            }
        }
    }
    columns
}

/// Fallible scale-folded grouped NCHW Conv2d forward.
///
/// # Errors
/// Returns [`Conv2dError`] before allocation for malformed geometry, overflow,
/// or mismatched input, weight, or scale buffers.
pub fn try_forward(
    input: &[f32],
    weight: &[f32],
    scale: &[f32],
    cfg: &Conv2dCfg,
) -> Result<Vec<f32>, Conv2dError> {
    let cfg = *cfg;
    let geometry = cfg.geometry()?;
    cfg.validate_buffers(
        input.len(),
        weight.len(),
        scale.len(),
        geometry.output_elements,
    )?;
    let mut output = vec![0.0; geometry.output_elements];
    for batch in 0..cfg.batch {
        for group in 0..cfg.groups {
            let weight_start = group * geometry.c_out_per_group * geometry.patch_columns;
            let group_weight = &weight
                [weight_start..weight_start + geometry.c_out_per_group * geometry.patch_columns];
            let scale_start = group * geometry.c_out_per_group;
            let group_scale = &scale[scale_start..scale_start + geometry.c_out_per_group];
            for row_start in (0..geometry.patch_rows).step_by(CONV2D_PATCH_TILE_ROWS) {
                let row_count = (geometry.patch_rows - row_start).min(CONV2D_PATCH_TILE_ROWS);
                let columns = im2col(input, cfg, geometry, batch, group, row_start, row_count);
                let group_output = matmul::forward(
                    &columns,
                    group_weight,
                    group_scale,
                    row_count,
                    geometry.c_out_per_group,
                    geometry.patch_columns,
                );
                for co_local in 0..geometry.c_out_per_group {
                    let channel = group * geometry.c_out_per_group + co_local;
                    for local_row in 0..row_count {
                        let row = row_start + local_row;
                        let oh = row / geometry.width_out;
                        let ow = row % geometry.width_out;
                        let output_index = ((batch * cfg.c_out + channel) * geometry.height_out
                            + oh)
                            * geometry.width_out
                            + ow;
                        output[output_index] =
                            group_output[local_row * geometry.c_out_per_group + co_local];
                    }
                }
            }
        }
    }
    Ok(output)
}

/// Scale-folded grouped NCHW Conv2d forward.
///
/// # Panics
/// Panics when geometry or flat-buffer lengths violate [`try_forward`].
#[must_use]
pub fn forward(input: &[f32], weight: &[f32], scale: &[f32], cfg: &Conv2dCfg) -> Vec<f32> {
    try_forward(input, weight, scale, cfg).expect("valid Conv2d forward contract")
}

/// Fallible VJP returning input, weight, and scale gradients.
///
/// # Errors
/// Returns [`Conv2dError`] before allocation for malformed geometry, overflow,
/// or mismatched input, weight, scale, or output-gradient buffers.
pub fn try_vjp(
    input: &[f32],
    weight: &[f32],
    scale: &[f32],
    cfg: &Conv2dCfg,
    grad_output: &[f32],
) -> Result<Vec<Vec<f32>>, Conv2dError> {
    let cfg = *cfg;
    let geometry = cfg.geometry()?;
    cfg.validate_buffers(input.len(), weight.len(), scale.len(), grad_output.len())?;
    let mut grad_input = vec![0.0; input.len()];
    let mut grad_weight = vec![0.0; weight.len()];
    let mut grad_scale = vec![0.0; scale.len()];
    for batch in 0..cfg.batch {
        for group in 0..cfg.groups {
            let weight_start = group * geometry.c_out_per_group * geometry.patch_columns;
            let group_weight = &weight
                [weight_start..weight_start + geometry.c_out_per_group * geometry.patch_columns];
            let scale_start = group * geometry.c_out_per_group;
            let group_scale = &scale[scale_start..scale_start + geometry.c_out_per_group];
            for row_start in (0..geometry.patch_rows).step_by(CONV2D_PATCH_TILE_ROWS) {
                let row_count = (geometry.patch_rows - row_start).min(CONV2D_PATCH_TILE_ROWS);
                let columns = im2col(input, cfg, geometry, batch, group, row_start, row_count);
                let mut group_grad_output = vec![0.0; row_count * geometry.c_out_per_group];
                for co_local in 0..geometry.c_out_per_group {
                    let channel = group * geometry.c_out_per_group + co_local;
                    for local_row in 0..row_count {
                        let row = row_start + local_row;
                        let oh = row / geometry.width_out;
                        let ow = row % geometry.width_out;
                        let output_index = ((batch * cfg.c_out + channel) * geometry.height_out
                            + oh)
                            * geometry.width_out
                            + ow;
                        group_grad_output[local_row * geometry.c_out_per_group + co_local] =
                            grad_output[output_index];
                    }
                }
                let gradients = matmul::vjp(
                    &columns,
                    group_weight,
                    group_scale,
                    row_count,
                    geometry.c_out_per_group,
                    geometry.patch_columns,
                    &group_grad_output,
                );
                for co_local in 0..geometry.c_out_per_group {
                    let channel = group * geometry.c_out_per_group + co_local;
                    grad_scale[channel] += gradients[2][co_local];
                    for column in 0..geometry.patch_columns {
                        grad_weight[channel * geometry.patch_columns + column] +=
                            gradients[1][co_local * geometry.patch_columns + column];
                    }
                }
                for local_row in 0..row_count {
                    let row = row_start + local_row;
                    let oh = row / geometry.width_out;
                    let ow = row % geometry.width_out;
                    for ci_local in 0..geometry.c_in_per_group {
                        let channel = group * geometry.c_in_per_group + ci_local;
                        for kh in 0..cfg.kernel_h {
                            for kw in 0..cfg.kernel_w {
                                if let Some(index) =
                                    input_index(cfg, batch, channel, oh, ow, kh, kw)
                                {
                                    let column = (ci_local * cfg.kernel_h + kh) * cfg.kernel_w + kw;
                                    grad_input[index] +=
                                        gradients[0][local_row * geometry.patch_columns + column];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
    Ok(vec![grad_input, grad_weight, grad_scale])
}

/// VJP returning input, weight, and per-output-channel scale gradients.
///
/// # Panics
/// Panics when geometry or flat-buffer lengths violate [`try_vjp`].
#[must_use]
pub fn vjp(
    input: &[f32],
    weight: &[f32],
    scale: &[f32],
    cfg: &Conv2dCfg,
    grad_output: &[f32],
) -> Vec<Vec<f32>> {
    try_vjp(input, weight, scale, cfg, grad_output).expect("valid Conv2d VJP contract")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn malformed_or_overflowing_geometry_has_zero_output() {
        let malformed = Conv2dCfg {
            batch: 1,
            c_in: 1,
            c_out: 1,
            input_h: 2,
            input_w: 2,
            kernel_h: 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            pad_top: 0,
            pad_bottom: 0,
            pad_left: 0,
            pad_right: 0,
            groups: 0,
        };
        assert_eq!(malformed.c_in_per_group(), 0);
        assert_eq!(malformed.output_elements(), 0);
        assert!(!malformed.buffers_fit(0, 0, 0, 0));

        let overflow = Conv2dCfg {
            batch: usize::MAX,
            groups: 1,
            ..malformed
        };
        assert_eq!(overflow.output_hw(), (0, 0));
        assert_eq!(overflow.input_elements(), 0);
        assert!(matches!(
            overflow.validate_buffers(0, 0, 0, 0),
            Err(Conv2dError::ArithmeticOverflow)
        ));

        let scratch_overflow = Conv2dCfg {
            batch: 1,
            c_in: 1,
            c_out: 1,
            input_h: 1,
            input_w: 2,
            kernel_h: usize::MAX / 2 + 1,
            kernel_w: 1,
            stride_h: 1,
            stride_w: 1,
            dilation_h: 1,
            dilation_w: 1,
            pad_top: 0,
            pad_bottom: usize::MAX / 2,
            pad_left: 0,
            pad_right: 0,
            groups: 1,
        };
        assert_eq!(scratch_overflow.max_scratch_elements(), 0);
        assert!(matches!(
            scratch_overflow.validate_buffers(2, usize::MAX / 2 + 1, 1, 2),
            Err(Conv2dError::ArithmeticOverflow)
        ));

        let zero_height = Conv2dCfg {
            kernel_h: 0,
            groups: 1,
            ..malformed
        };
        assert_eq!(zero_height.output_hw(), (0, 0));
    }
}
