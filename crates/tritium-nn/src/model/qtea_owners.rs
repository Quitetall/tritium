//! Shared packed ownership and logical payload accounting after graph admission.
use super::qtea::{add_weight, failure, invalid};
use super::qtea_execution::Execution;
use crate::{HostQteaLinear, NnError};
use std::{
    collections::{BTreeMap, BTreeSet},
    io::{Read, Seek},
    sync::Arc,
};
use tritium_format::qtea::model::{QteaModelReader, QteaModelTensor};

#[derive(Clone)]
pub(super) enum Matrix {
    Host(Arc<HostQteaLinear>),
    #[cfg(feature = "cuda")]
    Cuda(Arc<crate::CudaQteaLinear>),
}
impl Matrix {
    pub(super) fn shape(&self) -> (usize, usize) {
        match self {
            Self::Host(m) => (m.rows(), m.columns()),
            #[cfg(feature = "cuda")]
            Self::Cuda(m) => (m.rows(), m.columns()),
        }
    }
    pub(super) fn projection(self) -> crate::Projection {
        match self {
            Self::Host(m) => crate::Projection::HostQtea(m),
            #[cfg(feature = "cuda")]
            Self::Cuda(m) => crate::Projection::CudaQtea(m),
        }
    }
    pub(super) fn embedding(self) -> crate::TokenEmbedding {
        match self {
            Self::Host(m) => crate::TokenEmbedding::from_host_qtea(m),
            #[cfg(feature = "cuda")]
            Self::Cuda(m) => crate::TokenEmbedding::from_cuda_qtea(m),
        }
    }
}
pub(super) struct Owners {
    pub(super) matrices: BTreeMap<String, Matrix>,
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
pub(super) fn plan_target<R: Read + Seek>(
    reader: &QteaModelReader<R>,
    target: Execution,
) -> Result<(), NnError> {
    #[cfg(feature = "cuda")]
    if let Execution::Cuda(limits) = target {
        if limits.max_device_weight_bytes == 0
            || limits.max_device_staging_bytes == 0
            || limits.host.max_weight_bytes == 0
            || limits.host.max_staging_bytes == 0
        {
            return Err(invalid("QTEA CUDA weight/staging caps must be positive"));
        }
        let (mut device, mut preserved, mut largest) = (0u64, 0usize, 0usize);
        let mut unique = BTreeSet::new();
        for (name, tensor) in &reader.manifest().tensors {
            match tensor {
                QteaModelTensor::Qtea { blob, .. } if unique.insert(*blob.as_bytes()) => {
                    let bytes = reader.tensor_bytes(name).map_err(failure)?;
                    device = device
                        .checked_add(bytes)
                        .filter(|v| *v <= limits.max_device_weight_bytes)
                        .ok_or_else(|| invalid("QTEA aggregate CUDA weight cap exceeded"))?;
                    largest = largest.max(usize::try_from(bytes).map_err(failure)?);
                }
                QteaModelTensor::Qtea { .. } => {}
                other => {
                    let elements = match other {
                        QteaModelTensor::Vector { length, .. } => {
                            usize::try_from(*length).map_err(failure)?
                        }
                        QteaModelTensor::Preserved { shape, .. } => {
                            shape.iter().try_fold(1usize, |n, d| {
                                n.checked_mul(usize::try_from(*d).map_err(failure)?)
                                    .ok_or_else(|| invalid("QTEA preserved shape overflow"))
                            })?
                        }
                        _ => unreachable!(),
                    };
                    add_weight(
                        &mut preserved,
                        elements
                            .checked_mul(4)
                            .ok_or_else(|| invalid("QTEA preserved bytes overflow"))?,
                        limits.host.max_weight_bytes,
                    )?;
                }
            }
        }
        add_weight(&mut preserved, largest, limits.host.max_weight_bytes)?;
        return Ok(());
    }
    plan(reader, target.host().max_weight_bytes)
}

pub(super) fn read_target<R: Read + Seek>(
    reader: &mut QteaModelReader<R>,
    target: Execution,
    #[allow(unused_variables)] backend: Option<&dyn tritium_spec::TernaryBackend>,
) -> Result<Owners, NnError> {
    let limits = target.host();
    let names: Vec<_> = reader.manifest().tensors.keys().cloned().collect();
    let mut owners = Owners {
        matrices: BTreeMap::new(),
        preserved: BTreeMap::new(),
    };
    let mut cache = BTreeMap::<[u8; 32], Matrix>::new();
    let mut retained = 0;
    #[cfg(feature = "cuda")]
    let mut device_used = 0u64;
    for name in names {
        match reader.manifest().tensors[&name].clone() {
            QteaModelTensor::Qtea {
                blob,
                rows,
                columns,
            } => {
                let key = *blob.as_bytes();
                let matrix = if let Some(matrix) = cache.get(&key) {
                    if (matrix.shape().0 as u64, matrix.shape().1 as u64) != (rows, columns) {
                        return Err(invalid("shared QTEA blob has incompatible owner geometry"));
                    }
                    matrix.clone()
                } else {
                    let tensor = reader.qtea(&name).map_err(failure)?;
                    let matrix = match target {
                        Execution::Host(_) => {
                            add_weight(
                                &mut retained,
                                tensor.buffer_capacity_bytes(),
                                limits.max_weight_bytes,
                            )?;
                            Matrix::Host(Arc::new(HostQteaLinear::from_tensor(
                                tensor,
                                limits.max_staging_bytes,
                            )))
                        }
                        #[cfg(feature = "cuda")]
                        Execution::Cuda(cuda) => {
                            let backend = backend
                                .and_then(|b| {
                                    b.as_concrete()
                                        .and_then(|b| b.downcast_ref::<tritium_cuda::CudaBackend>())
                                })
                                .ok_or_else(|| {
                                    invalid("QTEA CUDA loading requires a concrete CUDA backend")
                                })?;
                            let mut peak = retained;
                            add_weight(
                                &mut peak,
                                tensor.buffer_capacity_bytes(),
                                limits.max_weight_bytes,
                            )?;
                            let encoded =
                                u64::try_from(tensor.as_bytes().len()).map_err(failure)?;
                            let remaining = cuda
                                .max_device_weight_bytes
                                .checked_sub(device_used)
                                .ok_or_else(|| {
                                invalid("QTEA CUDA weight accounting underflow")
                            })?;
                            device_used = device_used
                                .checked_add(encoded)
                                .filter(|n| *n <= cuda.max_device_weight_bytes)
                                .ok_or_else(|| {
                                    invalid("QTEA aggregate CUDA weight cap exceeded")
                                })?;
                            Matrix::Cuda(Arc::new(crate::CudaQteaLinear::upload(
                                backend,
                                tensor.view(),
                                remaining,
                                tritium_cuda::QteaCudaLimits {
                                    max_device_staging_bytes: cuda.max_device_staging_bytes,
                                    max_host_staging_bytes: limits.max_staging_bytes,
                                },
                            )?))
                            // tensor is dropped here; never retain a complete host matrix set.
                        }
                    };
                    cache.insert(key, matrix.clone());
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
