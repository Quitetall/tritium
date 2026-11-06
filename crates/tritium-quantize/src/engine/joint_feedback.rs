//! Joint fitter adapter over the shared validated Schur feedback controller.

use super::direct::{DirectFitError, JointOutput, reserved_vec};
use crate::{
    ColumnGroup, DensePsdMetric, FeedbackMetric, FeedbackProblem, FeedbackRunError, JointFitMetric,
    fit_with_feedback_precision,
};
use tritium_core::FeedbackDecay;
use tritium_format::AdditiveTensor;
use tritium_schema::{AdditiveLayout, JointFeedbackFitOptions};

pub(super) fn fit(
    layout: AdditiveLayout,
    source: &[f32],
    gram: &[f64],
    options: JointFeedbackFitOptions,
) -> Result<AdditiveTensor, DirectFitError> {
    let rows = layout.rows as usize;
    let cols = layout.cols as usize;
    let group = usize::from(layout.group);
    let mut output = JointOutput::new(layout, options.fitter)?;
    // Reuse the shipping damped full-Gram inverse, not a second factorization implementation.
    let inverse = super::activation_aware::damped_inverse(gram, cols, options.damp)
        .ok_or(DirectFitError::JointFactorization)?;
    let groups = (0..cols)
        .step_by(group)
        .map(|start| ColumnGroup {
            start,
            end: (start + group).min(cols),
        })
        .collect::<Vec<_>>();
    let weights = source.iter().map(|&v| f64::from(v)).collect::<Vec<_>>();
    fit_with_feedback_precision(
        FeedbackProblem {
            rows,
            columns: cols,
            weights: &weights,
            groups: &groups,
            metric: FeedbackMetric::InverseHessian(&inverse),
        },
        FeedbackDecay::new(options.decay, options.decay_ramp)
            .map_err(|_| DirectFitError::InvalidOptions("feedback decay"))?,
        options.precision,
        |request| {
            let start = request.column_start;
            let end = start + request.columns;
            let width = request.columns;
            let mut local = reserved_vec(width * width)?;
            for row in start..end {
                local.extend_from_slice(&gram[row * cols + start..row * cols + end]);
            }
            let metric = DensePsdMetric::new(width, &local).map_err(DirectFitError::Joint)?;
            let mut reconstruction = reserved_vec(rows * width)?;
            let mut input = reserved_vec(width)?;
            for row in 0..rows {
                input.clear();
                for (index, &v) in request.working_weights[row * width..(row + 1) * width]
                    .iter()
                    .enumerate()
                {
                    let narrowed = v as f32;
                    if !narrowed.is_finite() {
                        return Err(DirectFitError::FitterNarrowing {
                            row,
                            column: start + index,
                        });
                    }
                    input.push(narrowed);
                }
                let fitted = output.fit_row(
                    row,
                    request.group_index,
                    &input,
                    JointFitMetric::Dense(&metric),
                )?;
                reconstruction.extend(fitted.into_iter().map(f64::from));
            }
            Ok(reconstruction)
        },
    )
    .map_err(|error| match error {
        FeedbackRunError::Feedback(error) => DirectFitError::Feedback(error),
        FeedbackRunError::Fitter(error) => error,
    })?;
    output.finish()
}
