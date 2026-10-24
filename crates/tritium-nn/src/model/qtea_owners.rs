//! Shared packed ownership and logical payload accounting after graph admission.
use super::qtea::{HostQteaModelLimits, add_weight, failure, invalid};
use crate::{HostQteaLinear, NnError};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek},
    sync::Arc,
};
use tritium_format::qtea::model::{QteaModelReader, QteaModelTensor};

pub(super) struct Owners {
    pub(super) matrices: BTreeMap<String, Arc<HostQteaLinear>>,
    pub(super) preserved: BTreeMap<String, (Vec<u64>, Vec<f32>)>,
}
pub(super) fn plan<R: Read + Seek>(
    reader: &QteaModelReader<R>,
    limit: usize,
) -> Result<(), NnError> {
    let mut total = 0;
    let mut unique = BTreeSet::new();
    for (name, tensor) in &reader.manifest().tensors {
        let bytes = match tensor {
            QteaModelTensor::Qtea { blob, .. } if unique.insert(*blob.as_bytes()) => {
                usize::try_from(reader.tensor_bytes(name).map_err(failure)?).map_err(failure)?
            }
            QteaModelTensor::Qtea { .. } => 0,
            QteaModelTensor::Vector { length, .. } => usize::try_from(*length)
                .map_err(failure)?
                .checked_mul(4)
                .ok_or_else(|| invalid("QTEA vector size overflow"))?,
            QteaModelTensor::Preserved { shape, .. } => {
                shape.iter().try_fold(4_usize, |n, d| {
                    n.checked_mul(usize::try_from(*d).map_err(failure)?)
                        .ok_or_else(|| invalid("QTEA preserved payload overflow"))
                })?
            }
        };
        add_weight(&mut total, bytes, limit)?;
    }
    Ok(())
}
pub(super) fn read<R: Read + Seek>(
    reader: &mut QteaModelReader<R>,
    limits: HostQteaModelLimits,
) -> Result<Owners, NnError> {
    let names: Vec<_> = reader.manifest().tensors.keys().cloned().collect();
    let mut owners = Owners {
        matrices: BTreeMap::new(),
        preserved: BTreeMap::new(),
    };
    let mut cache = BTreeMap::<[u8; 32], Arc<HostQteaLinear>>::new();
    let mut retained = 0;
    for name in names {
        match reader.manifest().tensors[&name].clone() {
            QteaModelTensor::Qtea {
                blob,
                rows,
                columns,
            } => {
                let key = *blob.as_bytes();
                let matrix = if let Some(matrix) = cache.get(&key) {
                    if matrix.rows() as u64 != rows || matrix.columns() as u64 != columns {
                        return Err(invalid("shared QTEA blob has incompatible owner geometry"));
                    }
                    Arc::clone(matrix)
                } else {
                    let tensor = reader.qtea(&name).map_err(failure)?;
                    add_weight(
                        &mut retained,
                        tensor.buffer_capacity_bytes(),
                        limits.max_weight_bytes,
                    )?;
                    let matrix = Arc::new(HostQteaLinear::from_tensor(
                        tensor,
                        limits.max_staging_bytes,
                    ));
                    cache.insert(key, Arc::clone(&matrix));
                    matrix
                };
                owners.matrices.insert(name, matrix);
            }
            tensor => {
                let (shape, values) = match tensor {
                    QteaModelTensor::Vector { length, .. } => {
                        (vec![length], reader.vector(&name).map_err(failure)?)
                    }
                    QteaModelTensor::Preserved { shape, .. } => {
                        (shape, reader.preserved(&name).map_err(failure)?)
                    }
                    QteaModelTensor::Qtea { .. } => unreachable!(),
                };
                add_weight(
                    &mut retained,
                    values
                        .capacity()
                        .checked_mul(4)
                        .ok_or_else(|| invalid("QTEA preserved capacity overflow"))?,
                    limits.max_weight_bytes,
                )?;
                owners.preserved.insert(name, (shape, values));
            }
        }
    }
    Ok(owners)
}
