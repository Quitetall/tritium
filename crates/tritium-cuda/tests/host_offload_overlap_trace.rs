//! Profiling-only driver for ADR 0027's host-offload overlap gate.
//!
//! This deliberately uses only public training APIs. Three large weight leaves
//! feed a shared loss, then either `HostOffloadTrainer::step` consumes their
//! resident gradient collection or production-style backward streams each leaf
//! as it finalizes. Run both paths under Nsight Systems. The test validates
//! pipeline geometry; the traces provide cross-stream overlap evidence.

#![cfg(feature = "cuda")]

use tritium_cuda::CudaBackend;
use tritium_cuda::train::{
    DeviceTape, DeviceTensor, GradientLeafBinding, HostOffloadTrainParam, HostOffloadTrainer,
};
use tritium_train::AdamW;

const LEAVES: usize = 3;
const DEFAULT_ROWS: usize = 4096;
const DEFAULT_COLS: usize = 2048;
const DEFAULT_STEPS: u64 = 3;

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum TracePath {
    Scheduler,
    ProductionStream,
}

fn trace_path() -> TracePath {
    match std::env::var("TRITIUM_OVERLAP_PATH").as_deref() {
        Ok("scheduler") | Err(std::env::VarError::NotPresent) => TracePath::Scheduler,
        Ok("production-stream") => TracePath::ProductionStream,
        Ok(value) => {
            panic!("TRITIUM_OVERLAP_PATH must be scheduler or production-stream, got {value:?}")
        }
        Err(error) => panic!("TRITIUM_OVERLAP_PATH is not valid Unicode: {error}"),
    }
}

fn env_usize(name: &str, default: usize) -> usize {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .ok()
            .filter(|&parsed| parsed > 0)
            .unwrap_or_else(|| panic!("{name} must be a positive usize, got {value:?}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name} is not valid Unicode: {error}"),
    }
}

fn env_u64(name: &str, default: u64) -> u64 {
    match std::env::var(name) {
        Ok(value) => value
            .parse()
            .ok()
            .filter(|&parsed| parsed > 0)
            .unwrap_or_else(|| panic!("{name} must be a positive u64, got {value:?}")),
        Err(std::env::VarError::NotPresent) => default,
        Err(error) => panic!("{name} is not valid Unicode: {error}"),
    }
}

fn seeded_values(mut state: u64, len: usize) -> Vec<f32> {
    (0..len)
        .map(|_| {
            state ^= state >> 12;
            state ^= state << 25;
            state ^= state >> 27;
            let bits = state.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 40;
            (bits as f32 / (1u32 << 24) as f32 - 0.5) * 0.04
        })
        .collect()
}

#[test]
#[ignore = "profiling-only: run under Nsight Systems to evaluate cross-stream overlap"]
fn host_offload_overlap_trace_driver() {
    let path = trace_path();
    let rows = env_usize("TRITIUM_OVERLAP_ROWS", DEFAULT_ROWS);
    let cols = env_usize("TRITIUM_OVERLAP_COLS", DEFAULT_COLS);
    let steps = env_u64("TRITIUM_OVERLAP_STEPS", DEFAULT_STEPS);
    let elements = rows.checked_mul(cols).expect("trace leaf shape overflow");

    let backend = CudaBackend::new(0).expect("open CUDA device 0");
    let input = seeded_values(0x0027_CAFE, cols);
    let target = DeviceTensor::upload(&backend, &vec![1.0 / rows as f32; rows])
        .expect("upload trace target");
    let optimizer = AdamW::new(1e-3);
    let params = (0..LEAVES)
        .map(|leaf| HostOffloadTrainParam {
            master: seeded_values(0x0027_1000 + leaf as u64, elements),
            rows,
            cols,
            salt_planes: 1,
            optimizer,
        })
        .collect();
    let mut trainer =
        HostOffloadTrainer::new_owned(&backend, params).expect("create host-offload trainer");

    for step in 1..=steps {
        let masters: Vec<Vec<f32>> = (0..LEAVES)
            .map(|index| {
                trainer
                    .master(index)
                    .expect("borrow host-offload master")
                    .to_vec()
            })
            .collect();
        let mut tape = DeviceTape::new(&backend, rows).expect("create device tape");
        let x = tape.leaf(&input).expect("upload trace input");
        let weight_ids: Vec<usize> = masters
            .iter()
            .map(|master| tape.leaf(master).expect("upload trace weight"))
            .collect();
        let mut logits = tape
            .matmul(x, weight_ids[0], 1, rows, cols)
            .expect("first trace matmul");
        for &weight in &weight_ids[1..] {
            let branch = tape
                .matmul(x, weight, 1, rows, cols)
                .expect("trace branch matmul");
            logits = tape.add(logits, branch).expect("sum trace branches");
        }
        match path {
            TracePath::Scheduler => {
                let gradients = tape
                    .xent_backward_device(logits, &target, 1, rows, &weight_ids)
                    .expect("retain trace gradients on device");
                assert_eq!(gradients.len(), LEAVES);
                trainer
                    .step(gradients, step)
                    .expect("run host-offload scheduler step");
            }
            TracePath::ProductionStream => {
                let bindings: Vec<_> = weight_ids
                    .iter()
                    .enumerate()
                    .map(|(parameter_index, &leaf_id)| GradientLeafBinding {
                        leaf_id,
                        parameter_index,
                    })
                    .collect();
                let report = tape
                    .xent_backward_into(logits, &target, 1, rows, &bindings, &mut trainer, step)
                    .expect("stream finalized gradients into host-offload AdamW");
                assert_eq!(report.emissions.len(), LEAVES);
            }
        }

        assert_eq!(trainer.completed_step(), step);
    }

    let stats = trainer.stats();
    assert_eq!(stats.peak_in_flight_parameters, 2);
    assert_eq!(stats.largest_parameter_elements, elements);
    assert_eq!(stats.host_optimizer_elements, LEAVES * elements * 3);
    assert_eq!(stats.resident_input_gradient_elements, LEAVES * elements);
    eprintln!(
        "ADR0027_OVERLAP_TRACE path={path:?} leaves={LEAVES} leaf_elements={elements} steps={steps} \
         peak_in_flight={} device_staging_elements={} pinned_staging_elements={}",
        stats.peak_in_flight_parameters,
        stats.peak_optimizer_device_elements,
        stats.pinned_optimizer_host_elements,
    );
}
