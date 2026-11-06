//! Public mixer transaction tests, not numerical or physical-device evidence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, atomic::AtomicU8};
use tritium_core::GemmShape;
use tritium_cpu::CpuBackend;
use tritium_spec::{BackendError, DeviceBuffer, DeviceCaps, MpGemm, TernaryBackend};
use tritium_spec::{DeltaNetGeometry, DeltaNetStep, ResidentDeltaNet};

pub(crate) const ALLOC: u8 = 1;
pub(crate) const STAGE: u8 = 2;
pub(crate) const STEP: u8 = 3;
pub(crate) const WRITE: u8 = 4;
pub(crate) const WRONG_GEOMETRY: u8 = 5;
pub(crate) const OUTPUT: u8 = 6;
pub(crate) const READ: u8 = 7;

#[derive(Debug, Default)]
pub(crate) struct Record {
    pub(crate) creations: usize,
    pub(crate) stages: usize,
    pub(crate) reads: usize,
    pub(crate) writes: Vec<Vec<f32>>,
    pub(crate) committed: Vec<f32>,
    pub(crate) steps: usize,
    pub(crate) drops: usize,
}

pub(crate) struct Adapter {
    pub(crate) cpu: CpuBackend,
    pub(crate) record: Arc<Mutex<Record>>,
    pub(crate) fault: Arc<AtomicU8>,
    pub(crate) supported: AtomicBool,
}

impl Adapter {
    pub(crate) fn new() -> Self {
        Self {
            cpu: CpuBackend::new(),
            record: Arc::default(),
            fault: Arc::default(),
            supported: AtomicBool::new(true),
        }
    }
}

fn fault(flag: &AtomicU8, point: u8) -> Result<(), BackendError> {
    if flag.load(Ordering::SeqCst) == point {
        Err(BackendError::Backend(format!(
            "injected resident failure {point}"
        )))
    } else {
        Ok(())
    }
}

impl TernaryBackend for Adapter {
    fn device_id(&self) -> &str {
        "resident-test"
    }
    fn capabilities(&self) -> DeviceCaps {
        self.cpu.capabilities()
    }
    fn as_concrete(&self) -> Option<&dyn std::any::Any> {
        Some(self)
    }
    fn upload_weights(
        &self,
        bytes: &[u8],
        shape: GemmShape,
        format: tritium_spec::TernaryFormat,
    ) -> Result<Box<dyn DeviceBuffer>, BackendError> {
        self.cpu.upload_weights(bytes, shape, format)
    }
    fn mpgemm(&self, params: MpGemm<'_>) -> Result<(), BackendError> {
        fault(&self.fault, OUTPUT)?;
        self.cpu.mpgemm(params)
    }
    fn resident_deltanet(
        &self,
        mut geometry: DeltaNetGeometry,
    ) -> Result<Option<Box<dyn ResidentDeltaNet>>, BackendError> {
        let elements = geometry.elements()?;
        if !self.supported.load(Ordering::SeqCst) {
            return Ok(None);
        }
        fault(&self.fault, ALLOC)?;
        if self.fault.load(Ordering::SeqCst) == WRONG_GEOMETRY {
            geometry.group_size += 1;
        }
        self.record.lock().unwrap().creations += 1;
        Ok(Some(Box::new(State {
            geometry,
            current: vec![0.0; elements],
            staged: vec![0.0; elements],
            record: Arc::clone(&self.record),
            fault: Arc::clone(&self.fault),
        })))
    }
}

#[derive(Debug)]
struct State {
    geometry: DeltaNetGeometry,
    current: Vec<f32>,
    staged: Vec<f32>,
    pub(crate) record: Arc<Mutex<Record>>,
    pub(crate) fault: Arc<AtomicU8>,
}

impl Drop for State {
    fn drop(&mut self) {
        self.record.lock().unwrap().drops += 1;
    }
}

impl ResidentDeltaNet for State {
    fn geometry(&self) -> DeltaNetGeometry {
        self.geometry
    }
    fn validate_owner(&self, backend: &dyn TernaryBackend) -> Result<(), BackendError> {
        if backend
            .as_concrete()
            .and_then(|any| any.downcast_ref::<Adapter>())
            .is_some_and(|adapter| Arc::ptr_eq(&adapter.record, &self.record))
        {
            Ok(())
        } else {
            Err(BackendError::InvalidInput(
                "resident test owner mismatch".into(),
            ))
        }
    }
    fn stage(&mut self) -> Result<(), BackendError> {
        self.staged.copy_from_slice(&self.current);
        self.record.lock().unwrap().stages += 1;
        fault(&self.fault, STAGE)
    }
    fn step(&mut self, params: DeltaNetStep<'_>) -> Result<(), BackendError> {
        // Deliberately NOT a copied DeltaNet numerical implementation: count
        // staged tokens to expose ownership/rollback through observable output.
        let g = self.geometry;
        for (got, expected) in [
            (
                params.key.len(),
                g.value_heads / g.group_size * g.key_head_dim,
            ),
            (
                params.query.len(),
                g.value_heads / g.group_size * g.key_head_dim,
            ),
            (params.value.len(), g.value_heads * g.value_head_dim),
            (params.beta.len(), g.value_heads),
            (params.decay.len(), g.value_heads),
            (params.output.len(), g.value_heads * g.value_head_dim),
        ] {
            assert_eq!(got, expected);
        }
        self.staged.iter_mut().for_each(|n| *n += 1.0);
        self.record.lock().unwrap().steps += 1;
        params.output.fill(self.staged[0]);
        fault(&self.fault, STEP)
    }
    fn read_staged(&self, out: &mut [f32]) -> Result<(), BackendError> {
        self.record.lock().unwrap().reads += 1;
        fault(&self.fault, READ)?;
        out.copy_from_slice(&self.staged);
        Ok(())
    }
    fn write_staged(&mut self, values: &[f32]) -> Result<(), BackendError> {
        self.record.lock().unwrap().writes.push(values.to_vec());
        self.staged.copy_from_slice(values);
        fault(&self.fault, WRITE)
    }
    fn commit(&mut self) {
        std::mem::swap(&mut self.current, &mut self.staged);
        self.record
            .lock()
            .unwrap()
            .committed
            .clone_from(&self.current);
    }
}
