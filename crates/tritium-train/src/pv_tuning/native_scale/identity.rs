//! Exact-width layout binding. JCS numbers must not round u64 rotation seeds.
use super::PvTuningError;
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScalePrecision,
    Transport,
};

pub(super) fn hash_layout(
    h: &mut blake3::Hasher,
    layout: AdditiveLayout,
) -> Result<(), PvTuningError> {
    h.update(&layout.rows.to_le_bytes());
    h.update(&layout.cols.to_le_bytes());
    h.update(&layout.tile.to_le_bytes());
    h.update(&layout.group.to_le_bytes());
    h.update(&[layout.max_planes]);
    h.update(&[match layout.allocation {
        PlaneAllocation::Uniform => 0,
        PlaneAllocation::PerTile => 1,
        _ => return Err(unsupported()),
    }]);
    h.update(&[match layout.codec {
        PlaneCodec::D2 => 0,
        PlaneCodec::B3 => 1,
        PlaneCodec::S34 => 2,
        _ => return Err(unsupported()),
    }]);
    h.update(&[match layout.law.anchor {
        ScaleAnchor::Group => 0,
        ScaleAnchor::Row => 1,
        ScaleAnchor::Tensor => 2,
        ScaleAnchor::Channel => 3,
        _ => return Err(unsupported()),
    }]);
    match layout.law.relation {
        PlaneRelation::Free => {
            h.update(&[0]);
        }
        PlaneRelation::Tied { num, den } => {
            h.update(&[1]);
            h.update(&num.to_le_bytes());
            h.update(&den.to_le_bytes());
        }
        _ => return Err(unsupported()),
    }
    h.update(&[match layout.law.precision {
        ScalePrecision::F16 => 0,
        ScalePrecision::Bf16 => 1,
        ScalePrecision::F32 => 2,
        _ => return Err(unsupported()),
    }]);
    match layout.basis {
        Basis::Identity => {
            h.update(&[0]);
        }
        Basis::Hadamard { block } => {
            h.update(&[1]);
            h.update(&block.to_le_bytes());
        }
        Basis::SignedRht {
            block,
            seed,
            domain,
        } => {
            h.update(&[2]);
            h.update(&block.to_le_bytes());
            h.update(&seed.to_le_bytes());
            h.update(&domain.to_le_bytes());
        }
        _ => return Err(unsupported()),
    }
    match layout.transport {
        Transport::Raw => {
            h.update(&[0]);
        }
        Transport::JointHuffman { block } => {
            h.update(&[1]);
            h.update(&block.to_le_bytes());
        }
        _ => return Err(unsupported()),
    }
    Ok(())
}
fn unsupported() -> PvTuningError {
    PvTuningError::invalid_weight("unsupported native checkpoint layout field")
}
