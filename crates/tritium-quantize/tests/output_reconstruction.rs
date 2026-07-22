//! Public-seam tests for streamed SALT V2 block/sliding output reconstruction.

use tritium_format::ModelId;
use tritium_quantize::{
    OutputObjectiveWeights, OutputReconstructionAccumulator, OutputReconstructionError,
    OutputReconstructionSchedule, OutputReconstructionScope, OutputReconstructionSpec,
    select_output_reconstruction,
};

const CANDIDATE_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction candidate v1";
const RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction receipt v1";

fn rehash_single_candidate_receipt(bytes: &mut [u8]) {
    const HEADER_BYTES: usize = 112;
    const CANDIDATE_BYTES: usize = 224;
    const CANDIDATE_PAYLOAD_BYTES: usize = CANDIDATE_BYTES - 32;
    let mut candidate = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT);
    candidate.update(&bytes[HEADER_BYTES..HEADER_BYTES + CANDIDATE_PAYLOAD_BYTES]);
    bytes[HEADER_BYTES + CANDIDATE_PAYLOAD_BYTES..HEADER_BYTES + CANDIDATE_BYTES]
        .copy_from_slice(candidate.finalize().as_bytes());

    let mut receipt = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    receipt.update(&bytes[12..44]);
    receipt.update(&bytes[44..76]);
    receipt.update(&1u64.to_le_bytes());
    receipt.update(&bytes[HEADER_BYTES + CANDIDATE_PAYLOAD_BYTES..HEADER_BYTES + CANDIDATE_BYTES]);
    receipt.update(&bytes[76..108]);
    bytes[HEADER_BYTES + CANDIDATE_BYTES..].copy_from_slice(receipt.finalize().as_bytes());
}

fn spec(schedule: OutputReconstructionSchedule, restarts: usize) -> OutputReconstructionSpec {
    OutputReconstructionSpec::new(
        ModelId::from_digest([1; 32]),
        [2; 32],
        [3; 32],
        [4; 32],
        schedule,
        OutputObjectiveWeights::new(1.0, 0.0, 1.0, 1.0).expect("valid weights"),
        1,
        restarts,
    )
    .expect("valid reconstruction spec")
}

fn exact_candidate(
    spec: &OutputReconstructionSpec,
    candidate_id: [u8; 32],
    seed: u64,
    final_student: &[f32],
) -> tritium_quantize::OutputCandidateReceipt {
    let mut candidate =
        OutputReconstructionAccumulator::new(spec, candidate_id, seed).expect("valid candidate");
    for scope in spec.scopes() {
        match scope {
            OutputReconstructionScope::Block { start, end } => {
                let teacher = [*start as f32, *end as f32];
                candidate
                    .observe(*scope, 0, 1, 2, &[true], &teacher, &teacher)
                    .expect("block observation");
            }
            OutputReconstructionScope::FinalLogits => candidate
                .observe(*scope, 0, 1, 2, &[true], &[0.0, 0.0], final_student)
                .expect("logit observation"),
        }
    }
    candidate.finish().expect("complete candidate")
}

#[test]
fn block_and_teacher_logit_objectives_select_best_restart() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 2 }, 2);
    let exact = exact_candidate(&spec, [9; 32], 22, &[0.0, 0.0]);
    let shifted = exact_candidate(&spec, [8; 32], 11, &[2.0, -2.0]);

    assert_eq!(exact.block_output_mse(), 0.0);
    assert_eq!(exact.teacher_kl(), 0.0);
    assert!((exact.teacher_cross_entropy() - std::f64::consts::LN_2).abs() < 1e-12);
    assert!(shifted.teacher_kl() > 1.0);

    let selected = select_output_reconstruction(&spec, vec![shifted, exact.clone()])
        .expect("select complete restarts");
    assert_eq!(selected.selected_candidate_id(), &[9; 32]);
    assert_eq!(selected.selected(), &exact);
    assert_eq!(selected.candidates().len(), 2);

    let bytes = selected.canonical_bytes().expect("canonical receipt");
    let reopened =
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &bytes)
            .expect("strict receipt reopen");
    assert_eq!(reopened, selected);

    let reversed =
        select_output_reconstruction(&spec, selected.candidates().iter().cloned().rev().collect())
            .expect("evaluation order independent");
    assert_eq!(reversed.canonical_bytes().expect("canonical"), bytes);

    let mut corrupt = bytes;
    corrupt[40] ^= 1;
    assert!(matches!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &corrupt),
        Err(OutputReconstructionError::MalformedReceipt(_))
    ));
}

#[test]
fn sliding_schedule_covers_tail_and_is_order_bound() {
    let spec = spec(
        OutputReconstructionSchedule::SlidingWindows {
            block_count: 5,
            window_size: 3,
            stride: 2,
        },
        1,
    );
    assert_eq!(
        spec.scopes(),
        &[
            OutputReconstructionScope::Block { start: 0, end: 3 },
            OutputReconstructionScope::Block { start: 2, end: 5 },
            OutputReconstructionScope::FinalLogits,
        ]
    );

    let mut candidate = OutputReconstructionAccumulator::new(&spec, [5; 32], 7).expect("candidate");
    let error = candidate
        .observe(
            OutputReconstructionScope::Block { start: 2, end: 5 },
            0,
            1,
            1,
            &[true],
            &[1.0],
            &[1.0],
        )
        .expect_err("scope order must be canonical");
    assert!(matches!(
        error,
        OutputReconstructionError::ScopeOrder { .. }
    ));

    assert!(matches!(
        OutputReconstructionSpec::new(
            ModelId::from_digest([1; 32]),
            [2; 32],
            [3; 32],
            [4; 32],
            OutputReconstructionSchedule::Blocks {
                block_count: u32::MAX,
            },
            OutputObjectiveWeights::new(1.0, 0.0, 0.0, 1.0).expect("weights"),
            1,
            1,
        ),
        Err(OutputReconstructionError::CountOverflow)
    ));
}

#[test]
fn selection_rejects_teacher_drift_and_incomplete_restart_sets() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 2);
    let first = exact_candidate(&spec, [1; 32], 1, &[0.0, 0.0]);
    let mut drifted =
        OutputReconstructionAccumulator::new(&spec, [2; 32], 2).expect("second candidate");
    drifted
        .observe(
            OutputReconstructionScope::Block { start: 0, end: 1 },
            0,
            1,
            2,
            &[true],
            &[10.0, 1.0],
            &[10.0, 1.0],
        )
        .expect("drifted block");
    drifted
        .observe(
            OutputReconstructionScope::FinalLogits,
            0,
            1,
            2,
            &[true],
            &[0.0, 0.0],
            &[0.0, 0.0],
        )
        .expect("final logits");
    let drifted = drifted.finish().expect("complete drifted candidate");

    assert!(matches!(
        select_output_reconstruction(&spec, vec![first.clone()]),
        Err(OutputReconstructionError::RestartCount {
            expected: 2,
            got: 1
        })
    ));
    assert!(matches!(
        select_output_reconstruction(&spec, vec![first, drifted]),
        Err(OutputReconstructionError::TeacherEvidenceMismatch)
    ));
}

#[test]
fn observations_reject_nonfinite_values_and_unselected_final_tokens() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let mut candidate = OutputReconstructionAccumulator::new(&spec, [7; 32], 1).expect("candidate");
    assert!(matches!(
        candidate.observe(
            OutputReconstructionScope::Block { start: 0, end: 1 },
            0,
            1,
            1,
            &[true],
            &[f32::NAN],
            &[0.0],
        ),
        Err(OutputReconstructionError::NonFiniteOutput { .. })
    ));

    let mut candidate = OutputReconstructionAccumulator::new(&spec, [7; 32], 1).expect("candidate");
    candidate
        .observe(
            OutputReconstructionScope::Block { start: 0, end: 1 },
            0,
            1,
            1,
            &[true],
            &[0.0],
            &[0.0],
        )
        .expect("block");
    assert!(matches!(
        candidate.observe(
            OutputReconstructionScope::FinalLogits,
            0,
            1,
            2,
            &[false],
            &[0.0, 0.0],
            &[0.0, 0.0],
        ),
        Err(OutputReconstructionError::EmptyTokenSelection)
    ));
}

#[test]
fn strict_reopen_rejects_rehashed_but_unreachable_candidate_metrics() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let candidate = exact_candidate(&spec, [6; 32], 1, &[0.0, 0.0]);
    let receipt = select_output_reconstruction(&spec, vec![candidate]).expect("receipt");

    let mut wrong_objective = receipt.canonical_bytes().expect("canonical");
    wrong_objective[296..304].copy_from_slice(&123.0f64.to_bits().to_le_bytes());
    rehash_single_candidate_receipt(&mut wrong_objective);
    assert!(matches!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(
            &spec,
            &wrong_objective
        ),
        Err(OutputReconstructionError::MalformedReceipt(
            "candidate objective"
        ))
    ));

    let mut wrong_observations = receipt.canonical_bytes().expect("canonical");
    wrong_observations[248..256].copy_from_slice(&99u64.to_le_bytes());
    rehash_single_candidate_receipt(&mut wrong_observations);
    assert!(matches!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(
            &spec,
            &wrong_observations
        ),
        Err(OutputReconstructionError::MalformedReceipt(
            "candidate observations"
        ))
    ));
}

#[test]
fn strict_reopen_rejects_unrepresentable_count_before_candidate_allocation() {
    const MAX_CANDIDATES: usize = (4 * 1024 * 1024 - 144) / 224;
    let spec = spec(
        OutputReconstructionSchedule::Blocks { block_count: 1 },
        MAX_CANDIDATES + 1,
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TSV2OUT\0");
    bytes.extend_from_slice(&1u16.to_le_bytes());
    bytes.extend_from_slice(&0u16.to_le_bytes());
    bytes.extend_from_slice(spec.spec_id());
    bytes.extend_from_slice(&[7; 32]);
    bytes.extend_from_slice(&[8; 32]);
    bytes.extend_from_slice(
        &u32::try_from(spec.restarts())
            .expect("bounded test count")
            .to_le_bytes(),
    );

    assert_eq!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &bytes),
        Err(OutputReconstructionError::ReceiptTooLarge)
    );
}
