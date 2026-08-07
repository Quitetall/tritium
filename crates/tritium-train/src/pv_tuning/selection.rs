use core::cmp::Ordering;

use super::PvTernaryStructure;

pub(super) fn selected_units(
    structure: PvTernaryStructure,
    decoded: &[f32],
    proposal: &[f32],
    fraction: f32,
) -> Vec<usize> {
    let width = unit_width(structure);
    let unit_count = decoded.len() / width;
    let keep = ((unit_count as f64) * f64::from(fraction)).ceil() as usize;
    let mut ranked: Vec<(usize, f64)> = (0..unit_count)
        .map(|unit| {
            let start = unit * width;
            let magnitude = (start..start + width)
                .map(|index| (f64::from(proposal[index]) - f64::from(decoded[index])).powi(2))
                .sum();
            (unit, magnitude)
        })
        .collect();
    ranked.sort_by(|left, right| {
        right
            .1
            .partial_cmp(&left.1)
            .unwrap_or(Ordering::Equal)
            .then_with(|| left.0.cmp(&right.0))
    });
    ranked.truncate(keep.max(1).min(unit_count));
    ranked.into_iter().map(|(unit, _)| unit).collect()
}

pub(super) fn unit_surrogate(
    structure: PvTernaryStructure,
    units: &[usize],
    decoded: &[f32],
    proposal: &[f32],
) -> f64 {
    let width = unit_width(structure);
    units
        .iter()
        .flat_map(|unit| {
            let start = unit * width;
            start..start + width
        })
        .map(|index| squared_error(decoded[index], proposal[index]))
        .sum()
}

pub(super) fn squared_error(value: f32, target: f32) -> f64 {
    (f64::from(value) - f64::from(target)).powi(2)
}

const fn unit_width(structure: PvTernaryStructure) -> usize {
    match structure {
        PvTernaryStructure::Dense => 1,
        PvTernaryStructure::S34 => 4,
    }
}
