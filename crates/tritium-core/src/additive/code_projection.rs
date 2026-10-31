//! Shared discrete search; no optimizer, coordinate transform, or allocation.
use super::{AdditiveView, Trit};
use tritium_schema::{PlaneCodec, ScaleAnchor};

const PLANES: usize = 3;
const WIDTH: usize = 4;

/// A rejected code-unit request. Candidate output is unchanged on every error.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CodeProjectionError {
    /// Codec or actual plane capacity is not implemented by this solver.
    UnsupportedProfile,
    /// Unit index, proposal length, or candidate coefficient length is wrong.
    Shape,
    /// Proposal, original decoded value, or squared-change budget is nonfinite.
    NonFinite,
    /// A squared-change budget is negative (including negative zero).
    InvalidBudget,
    /// A selected S34 plane does not have exactly three nonzero coefficients.
    InvalidStructure,
}

/// Observations for one discrete projection, not a model-quality receipt.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CodeProjectionStats {
    /// Whether any present coefficient was changed in candidate output.
    pub changed: bool,
    /// A strictly improving candidate existed but exceeded the movement budget.
    pub trust_limited: bool,
    /// Squared error of the original decoded unit against the proposal.
    pub surrogate_before: f64,
    /// Squared error after accepted projection (before error on trust refusal).
    pub surrogate_after: f64,
    /// Accepted squared decoded movement; zero on refusal or no change.
    pub squared_change: f64,
}

/// Project one native stored-basis code unit into caller-owned candidate trits.
///
/// D2/B3 units are individual row-major logical coefficients; S34 units are
/// row-major four-column blocks. `target` has one or four finite stored-coordinate
/// values respectively. No basis transform is applied. `out` has the complete
/// compact/uniform trit length and is otherwise untouched: initialize it from the
/// parent before constructing a whole candidate. Only an accepted changed unit's
/// present offsets are written. Errors and trust refusal leave it unchanged.
///
/// Dense search is exhaustive across up to three present planes. S34 uses one
/// conditional exact 32-pattern sweep per present plane, not a global optimum.
/// Strict improvement retains original codes on ties. Scales, their declared
/// precision, missing planes and all metadata remain unchanged. Sparse structure
/// is validated for this selected unit only, not for the rest of the payload.
/// Optional budget is an absolute finite nonnegative squared decoded movement;
/// selection and whole-tensor relative trust accounting belong to the caller.
/// Uses fixed stack buffers only; does not retain a dense master or Adam state.
pub fn project_additive_code_unit(
    view: AdditiveView<'_>,
    unit: usize,
    target: &[f32],
    max_squared_change: Option<f64>,
    out: &mut [Trit],
) -> Result<CodeProjectionStats, CodeProjectionError> {
    let width = match view.layout().codec {
        PlaneCodec::D2 | PlaneCodec::B3 => 1,
        PlaneCodec::S34 => WIDTH,
        _ => return Err(CodeProjectionError::UnsupportedProfile),
    };
    if view.plane_count() > PLANES {
        return Err(CodeProjectionError::UnsupportedProfile);
    }
    let rows = usize::try_from(view.layout().rows).map_err(|_| CodeProjectionError::Shape)?;
    let cols = usize::try_from(view.layout().cols).map_err(|_| CodeProjectionError::Shape)?;
    if cols % width != 0 || (width == WIDTH && usize::from(view.layout().group) % WIDTH != 0) {
        return Err(CodeProjectionError::Shape);
    }
    let units_per_row = cols / width;
    let units = rows
        .checked_mul(units_per_row)
        .ok_or(CodeProjectionError::Shape)?;
    if unit >= units || target.len() != width || out.len() != view.trits().len() {
        return Err(CodeProjectionError::Shape);
    }
    if target.iter().any(|value| !value.is_finite()) {
        return Err(CodeProjectionError::NonFinite);
    }
    if let Some(budget) = max_squared_change {
        if !budget.is_finite() {
            return Err(CodeProjectionError::NonFinite);
        }
        if budget.is_sign_negative() {
            return Err(CodeProjectionError::InvalidBudget);
        }
    }
    let row = unit / units_per_row;
    let col = (unit % units_per_row) * width;
    let mut offsets = [0; PLANES];
    let mut scales = [0.0; PLANES];
    let mut codes = [[Trit::ZERO; WIDTH]; PLANES];
    let mut planes = 0;
    for plane in 0..view.plane_count() {
        let Some(tile) = view
            .row_tiles(plane, row)
            .find(|tile| tile.col + tile.len > col)
        else {
            continue;
        };
        if tile.col > col {
            continue;
        }
        if col + width > tile.col + tile.len {
            return Err(CodeProjectionError::Shape);
        }
        let local = col - tile.col;
        let stored = tile.scale
            + if view.layout().law.anchor == ScaleAnchor::Group {
                local / usize::from(view.layout().group)
            } else {
                0
            };
        offsets[planes] = tile.trit + local;
        scales[planes] = view
            .stored_scale(plane, stored)
            .map_err(|_| CodeProjectionError::NonFinite)?;
        codes[planes][..width]
            .copy_from_slice(&view.trits()[offsets[planes]..offsets[planes] + width]);
        if width == WIDTH && codes[planes].iter().filter(|code| code.is_zero()).count() != 1 {
            return Err(CodeProjectionError::InvalidStructure);
        }
        planes += 1;
    }
    let original = decode(&codes, &scales, planes, width);
    if original[..width].iter().any(|value| !value.is_finite()) {
        return Err(CodeProjectionError::NonFinite);
    }
    let before = loss(&original, target);
    let initial = codes;
    if width == 1 {
        let mut best_loss = before;
        for encoded in 0..3usize.pow(planes as u32) {
            let mut cursor = encoded;
            let mut trial = codes;
            for plane in 0..planes {
                trial[plane][0] = if scales[plane] == 0.0 {
                    Trit::ZERO
                } else {
                    [Trit::NEG, Trit::ZERO, Trit::POS][cursor % 3]
                };
                cursor /= 3;
            }
            let value = decode(&trial, &scales, planes, width);
            let candidate_loss = loss(&value, target);
            if candidate_loss < best_loss {
                best_loss = candidate_loss;
                codes = trial;
            }
        }
    } else {
        for plane in 0..planes {
            let mut best = codes[plane];
            let mut best_loss = loss(&decode(&codes, &scales, planes, width), target);
            for zero in 0..WIDTH {
                for signs in 0..8 {
                    codes[plane] = pattern(zero, signs);
                    let candidate_loss = loss(&decode(&codes, &scales, planes, width), target);
                    if candidate_loss < best_loss {
                        best_loss = candidate_loss;
                        best = codes[plane];
                    }
                }
            }
            codes[plane] = best;
        }
    }
    let candidate = decode(&codes, &scales, planes, width);
    let squared_change = loss(&candidate, &original[..width]);
    let changed = codes != initial;
    let trust_limited = changed && max_squared_change.is_some_and(|limit| squared_change > limit);
    if changed && !trust_limited {
        for plane in 0..planes {
            out[offsets[plane]..offsets[plane] + width].copy_from_slice(&codes[plane][..width]);
        }
    }
    Ok(CodeProjectionStats {
        changed: changed && !trust_limited,
        trust_limited,
        surrogate_before: before,
        surrogate_after: if trust_limited {
            before
        } else {
            loss(&candidate, target)
        },
        squared_change: if trust_limited { 0.0 } else { squared_change },
    })
}

fn decode(
    codes: &[[Trit; WIDTH]; PLANES],
    scales: &[f32; PLANES],
    planes: usize,
    width: usize,
) -> [f32; WIDTH] {
    let mut values = [0.0; WIDTH];
    for plane in 0..planes {
        for col in 0..width {
            values[col] += codes[plane][col].to_f32() * scales[plane];
        }
    }
    values
}
fn loss(values: &[f32; WIDTH], target: &[f32]) -> f64 {
    // An overflowing trial must never be selected, including a NaN caused by
    // cancellation after a previous infinite f32 partial sum.
    if values[..target.len()]
        .iter()
        .any(|value| !value.is_finite())
    {
        return f64::INFINITY;
    }
    values
        .iter()
        .zip(target)
        .map(|(&a, &b)| {
            let difference = f64::from(a) - f64::from(b);
            difference * difference
        })
        .sum()
}
fn pattern(zero: usize, signs: usize) -> [Trit; WIDTH] {
    let mut codes = [Trit::ZERO; WIDTH];
    let mut bit = 0;
    for (col, code) in codes.iter_mut().enumerate() {
        if col != zero {
            *code = if signs & (1 << bit) == 0 {
                Trit::NEG
            } else {
                Trit::POS
            };
            bit += 1;
        }
    }
    codes
}
