use tritium_format::{PackedTrainingSaltSnapshot, TrainingSaltPlane};
use tritium_train::PvTernaryWeight;

use super::DevicePvRecoveryError;

pub(super) fn pack_snapshot(
    weight: &PvTernaryWeight,
) -> Result<PackedTrainingSaltSnapshot, DevicePvRecoveryError> {
    let planes = weight
        .planes()
        .iter()
        .map(|plane| TrainingSaltPlane::new(plane.trits(), plane.scales()))
        .collect::<Vec<_>>();
    Ok(PackedTrainingSaltSnapshot::pack(
        weight.rows(),
        weight.cols(),
        weight.group_size(),
        weight.structure(),
        &planes,
    )?)
}
