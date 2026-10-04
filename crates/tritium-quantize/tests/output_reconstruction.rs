//! Public-seam tests for streamed SALT V2 block/sliding output reconstruction.

use half::f16;
use tritium_core::Trit;
use tritium_format::{
    ModelId,
    salt_v2::SaltV2Codec,
    salt_v2_package::{SaltV2Package, SaltV2Plane, SaltV2ScaleUpdate, SaltV2Tensor, SaltV2Tile},
};
use tritium_quantize::{
    ActivationCache, ActivationCacheBuilder, ActivationCacheSpec, ActivationChunk, ActivationDType,
    ActivationDigest, ActivationWindow, FixedTritScaleRefitAccumulator, OutputObjectiveWeights,
    OutputReconstructionAccumulator, OutputReconstructionActivationLayer,
    OutputReconstructionActivationSet, OutputReconstructionActivationSource,
    OutputReconstructionError, OutputReconstructionSchedule, OutputReconstructionScope,
    OutputReconstructionSpec, RuntimeFinalLogitsAccumulator, fit_fixed_trit_tile_scale_refit,
    fit_fixed_trit_tile_scale_update, output_reconstruction_activation_digest,
    select_output_reconstruction,
};

const CANDIDATE_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction candidate v1";
const RECEIPT_HASH_CONTEXT: &str = "tritium salt v2 output reconstruction receipt v1";

fn rehash_single_candidate_receipt(bytes: &mut [u8]) {
    const HEADER_BYTES: usize = 112;
    const V2_CANDIDATE_BYTES: usize = 272;
    const SCOPE_BYTES: usize = 57;
    let version = u16::from_le_bytes(bytes[8..10].try_into().unwrap());
    let candidate_bytes = if version == 3 {
        let scope_count = u32::from_le_bytes(
            bytes[HEADER_BYTES + 240..HEADER_BYTES + 244]
                .try_into()
                .unwrap(),
        ) as usize;
        V2_CANDIDATE_BYTES + 4 + scope_count * SCOPE_BYTES
    } else {
        V2_CANDIDATE_BYTES
    };
    let candidate_bytes_for_receipt = &bytes[HEADER_BYTES..HEADER_BYTES + candidate_bytes];
    let mut candidate = blake3::Hasher::new_derive_key(if version == 3 {
        "tritium salt v2 output reconstruction candidate v2"
    } else {
        CANDIDATE_HASH_CONTEXT
    });
    if version == 3 {
        for range in [
            0..32,
            32..64,
            64..72,
            72..104,
            104..136,
            136..168,
            168..176,
            176..184,
            184..192,
            192..200,
            200..208,
            208..240,
        ] {
            candidate.update(&candidate_bytes_for_receipt[range]);
        }
        let scope_count =
            u32::from_le_bytes(candidate_bytes_for_receipt[240..244].try_into().unwrap()) as usize;
        candidate.update(&(scope_count as u64).to_le_bytes());
        for ordinal in 0..scope_count {
            let start = 244 + ordinal * SCOPE_BYTES;
            let record = &candidate_bytes_for_receipt[start..start + SCOPE_BYTES];
            candidate.update(&record[..9]);
            candidate.update(&record[9..17]);
            candidate.update(&record[17..25]);
            candidate.update(&record[25..]);
        }
    } else {
        candidate.update(&candidate_bytes_for_receipt[..candidate_bytes - 32]);
    }
    bytes[HEADER_BYTES + candidate_bytes - 32..HEADER_BYTES + candidate_bytes]
        .copy_from_slice(candidate.finalize().as_bytes());

    let mut receipt = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    receipt.update(&bytes[12..44]);
    receipt.update(&bytes[44..76]);
    receipt.update(&1u64.to_le_bytes());
    receipt.update(&bytes[HEADER_BYTES + candidate_bytes - 32..HEADER_BYTES + candidate_bytes]);
    receipt.update(&bytes[76..108]);
    bytes[HEADER_BYTES + candidate_bytes..].copy_from_slice(receipt.finalize().as_bytes());
}

fn legacy_v1_from_v2(bytes: &[u8], candidate_count: usize) -> Vec<u8> {
    const HEADER_BYTES: usize = 112;
    const V2_CANDIDATE_BYTES: usize = 272;
    let v3 = u16::from_le_bytes(bytes[8..10].try_into().unwrap()) == 3;
    let v3_scope_count = if v3 {
        u32::from_le_bytes(
            bytes[HEADER_BYTES + 240..HEADER_BYTES + 244]
                .try_into()
                .unwrap(),
        ) as usize
    } else {
        0
    };
    let candidate_bytes = V2_CANDIDATE_BYTES + if v3 { 4 + v3_scope_count * 57 } else { 0 };
    let mut legacy = bytes[..HEADER_BYTES].to_vec();
    legacy[8..10].copy_from_slice(&1_u16.to_le_bytes());
    let mut candidate_receipts = Vec::new();
    for ordinal in 0..candidate_count {
        let start = HEADER_BYTES + ordinal * candidate_bytes;
        let legacy_start = legacy.len();
        legacy.extend_from_slice(&bytes[start..start + 136]);
        legacy.extend_from_slice(&bytes[start + 184..start + 240]);
        let mut candidate = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT);
        candidate.update(&legacy[legacy_start..]);
        let receipt = *candidate.finalize().as_bytes();
        legacy.extend_from_slice(&receipt);
        candidate_receipts.push(receipt);
    }
    let mut receipt = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    receipt.update(&legacy[12..44]);
    receipt.update(&legacy[44..76]);
    receipt.update(&u64::try_from(candidate_count).unwrap().to_le_bytes());
    for candidate_receipt in candidate_receipts {
        receipt.update(&candidate_receipt);
    }
    receipt.update(&legacy[76..108]);
    legacy.extend_from_slice(receipt.finalize().as_bytes());
    legacy
}

fn v2_from_v3(bytes: &[u8], candidate_count: usize) -> Vec<u8> {
    const HEADER_BYTES: usize = 112;
    const OLD_CANDIDATE_PAYLOAD_BYTES: usize = 240;
    const V2_CANDIDATE_BYTES: usize = 272;
    const SCOPE_BYTES: usize = 57;
    let scope_count = u32::from_le_bytes(
        bytes[HEADER_BYTES + 240..HEADER_BYTES + 244]
            .try_into()
            .unwrap(),
    ) as usize;
    let v3_candidate_bytes = V2_CANDIDATE_BYTES + 4 + scope_count * SCOPE_BYTES;
    let mut v2 = bytes[..HEADER_BYTES].to_vec();
    v2[8..10].copy_from_slice(&2_u16.to_le_bytes());
    let mut candidate_receipts = Vec::new();
    for ordinal in 0..candidate_count {
        let start = HEADER_BYTES + ordinal * v3_candidate_bytes;
        let payload = &bytes[start..start + OLD_CANDIDATE_PAYLOAD_BYTES];
        v2.extend_from_slice(payload);
        let mut candidate = blake3::Hasher::new_derive_key(CANDIDATE_HASH_CONTEXT);
        candidate.update(payload);
        let id = *candidate.finalize().as_bytes();
        v2.extend_from_slice(&id);
        candidate_receipts.push(id);
    }
    let mut receipt = blake3::Hasher::new_derive_key(RECEIPT_HASH_CONTEXT);
    receipt.update(&v2[12..44]);
    receipt.update(&v2[44..76]);
    receipt.update(&(candidate_count as u64).to_le_bytes());
    for candidate_receipt in candidate_receipts {
        receipt.update(&candidate_receipt);
    }
    receipt.update(&v2[76..108]);
    v2.extend_from_slice(receipt.finalize().as_bytes());
    v2
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

#[test]
fn fixed_trit_scale_refit_finds_nonnegative_scales_without_retaining_batches() {
    // Orthogonal fixed-trit group outputs have the exact independent solution [2, 3].
    let mut fit = FixedTritScaleRefitAccumulator::new(2, 8).expect("valid refit");
    fit.observe(&[1.0, 0.0], 2.0).expect("first output row");
    fit.observe(&[0.0, 1.0], 3.0).expect("second output row");
    let result = fit.finish().expect("complete fit");

    assert_eq!(result.observations(), 2);
    assert_eq!(result.scales(), &[2.0, 3.0]);
    assert_eq!(result.squared_error(), 0.0);

    let plane = SaltV2Plane::new(vec![1; 256], vec![f16::ONE, f16::ONE]).unwrap();
    let tensor = SaltV2Tensor::new(
        "weight",
        vec![256],
        vec![SaltV2Tile::new(vec![plane]).unwrap()],
    )
    .unwrap();
    let mut package = SaltV2Package::new(SaltV2Codec::D2, vec![tensor]).unwrap();
    let update = SaltV2ScaleUpdate::new(0, 0, 0, result.to_f16_scales().unwrap()).unwrap();
    package.apply_scale_updates(&[update]).unwrap();
    assert_eq!(
        package.tensors()[0].tiles()[0].planes()[0].scales(),
        &[f16::from_f32(2.0), f16::from_f32(3.0)]
    );
}

#[test]
fn fixed_trit_scale_refit_streams_activation_rows_through_the_stored_trits() {
    let mut fit = FixedTritScaleRefitAccumulator::new(2, 8).expect("valid refit");
    let trits = [
        Trit::from_i8(1).unwrap(),
        Trit::from_i8(0).unwrap(),
        Trit::from_i8(0).unwrap(),
        Trit::from_i8(1).unwrap(),
    ];
    fit.observe_fixed_trit_projection(&trits, &[1.0, 0.0, 0.0, 0.0], 2, 2.0)
        .expect("first activation row");
    fit.observe_fixed_trit_projection(&trits, &[0.0, 0.0, 0.0, 1.0], 2, 3.0)
        .expect("second activation row");

    assert_eq!(fit.finish().unwrap().scales(), &[2.0, 3.0]);
}

#[test]
fn fixed_trit_scale_refit_never_uses_negative_scales() {
    let mut fit = FixedTritScaleRefitAccumulator::new(1, 4).expect("valid refit");
    fit.observe(&[1.0], -2.0).expect("valid output row");
    let result = fit.finish().expect("complete fit");

    assert_eq!(result.scales(), &[0.0]);
    assert_eq!(result.squared_error(), 4.0);
    assert_eq!(result.to_f16_scales().unwrap(), vec![f16::ZERO]);
    assert_eq!(
        result.to_positive_f16_scales().unwrap(),
        vec![f16::from_bits(1)]
    );
}

#[test]
fn fixed_trit_scale_refit_rejects_f16_overflow() {
    let mut fit = FixedTritScaleRefitAccumulator::new(1, 2).expect("valid refit");
    fit.observe(&[1.0], 100_000.0).expect("valid output row");
    let result = fit.finish().expect("finite f64 fit");

    assert_eq!(
        result.to_f16_scales(),
        Err(OutputReconstructionError::ScaleNotRepresentable)
    );
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
    assert_eq!(bytes.len(), 112 + 2 * (272 + 4 + 3 * 57) + 32);
    assert_eq!(&bytes[8..10], &3_u16.to_le_bytes());
    assert_eq!(selected.selected().scope_evidence().len(), 3);
    assert_eq!(
        blake3::hash(&bytes).to_hex().to_string(),
        "2563bc3c5182fffa793d1d9780d5f0c432bae6c013fcb99eed15ecdca208f418"
    );
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
fn candidate_scope_commitments_are_included_in_the_selected_receipt_identity() {
    let spec = spec(
        OutputReconstructionSchedule::SlidingWindows {
            block_count: 4,
            window_size: 2,
            stride: 1,
        },
        1,
    );
    let mut candidate =
        OutputReconstructionAccumulator::new(&spec, [7; 32], 13).expect("valid candidate");
    for scope in spec.scopes() {
        match scope {
            OutputReconstructionScope::Block { start, end } => {
                let values = [*start as f32, *end as f32];
                candidate
                    .observe(*scope, 0, 1, 2, &[true], &values, &values)
                    .expect("block observation");
            }
            OutputReconstructionScope::FinalLogits => candidate
                .observe(*scope, 0, 1, 2, &[true], &[0.0, 0.0], &[1.0, -1.0])
                .expect("logit observation"),
        }
    }

    let candidate = candidate.finish().expect("complete candidate");
    let scopes = candidate.scope_evidence();
    assert_eq!(scopes.len(), spec.scopes().len());
    for (evidence, scope) in scopes.iter().zip(spec.scopes()) {
        assert_eq!(
            evidence.scope(),
            match scope {
                OutputReconstructionScope::Block { start, end } => {
                    tritium_format::RuntimeOutputScope::Block {
                        start: *start,
                        end: *end,
                    }
                }
                OutputReconstructionScope::FinalLogits => {
                    tritium_format::RuntimeOutputScope::FinalLogits
                }
            }
        );
        assert_eq!(
            evidence.observation_count(),
            u64::from(spec.batches_per_scope())
        );
        assert!(evidence.value_count() > 0);
        assert_ne!(evidence.digest(), &[0; 32]);
    }
    assert_ne!(candidate.student_output_digest(), &[0; 32]);
}

#[test]
fn candidate_scope_commitments_preserve_batch_order_and_counts_across_each_scope() {
    let spec = OutputReconstructionSpec::new(
        ModelId::from_digest([1; 32]),
        [2; 32],
        [3; 32],
        [4; 32],
        OutputReconstructionSchedule::Blocks { block_count: 1 },
        OutputObjectiveWeights::new(1.0, 0.0, 1.0, 1.0).expect("valid weights"),
        2,
        1,
    )
    .expect("valid two-batch spec");
    let mut candidate =
        OutputReconstructionAccumulator::new(&spec, [6; 32], 12).expect("valid candidate");
    for scope in spec.scopes() {
        for batch_index in 0..2 {
            let batch = batch_index as f32;
            let teacher = [batch + 1.0, batch + 2.0];
            candidate
                .observe(*scope, batch_index, 1, 2, &[true], &teacher, &teacher)
                .expect("ordered observation");
        }
    }
    let candidate = candidate.finish().expect("complete candidate");
    let scopes = candidate.scope_evidence();
    assert_eq!(scopes.len(), 2);
    for evidence in scopes {
        assert_eq!(evidence.observation_count(), 2);
        assert_eq!(evidence.value_count(), 4);
    }
}

#[test]
fn selected_scope_commitments_reopen_with_the_exact_v3_candidate_receipt() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 2 }, 1);
    let candidate = exact_candidate(&spec, [12; 32], 21, &[0.5, -0.5]);
    let selected = select_output_reconstruction(&spec, vec![candidate]).expect("selected output");
    let bytes = selected.canonical_bytes().expect("canonical receipt");
    assert_eq!(&bytes[8..10], &3_u16.to_le_bytes());

    let reopened =
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &bytes)
            .expect("strict v3 receipt reopen");
    assert_eq!(reopened, selected);
    assert_eq!(
        reopened.selected().scope_evidence().len(),
        spec.scopes().len()
    );
    for (evidence, scope) in reopened
        .selected()
        .scope_evidence()
        .iter()
        .zip(spec.scopes())
    {
        assert_eq!(
            evidence.scope(),
            match scope {
                OutputReconstructionScope::Block { start, end } => {
                    tritium_format::RuntimeOutputScope::Block {
                        start: *start,
                        end: *end,
                    }
                }
                OutputReconstructionScope::FinalLogits => {
                    tritium_format::RuntimeOutputScope::FinalLogits
                }
            }
        );
        assert_eq!(evidence.candidate_id(), &[12; 32]);
        assert_eq!(evidence.initialization_seed(), 21);
    }

    let mut corrupt = bytes;
    corrupt[112 + 240 + 4 + 24] ^= 0x40;
    assert!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &corrupt)
            .is_err()
    );

    let legacy_v2 = v2_from_v3(&selected.canonical_bytes().expect("v3 receipt"), 1);
    let reopened_v2 =
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &legacy_v2)
            .expect("strict v2 compatibility reopen");
    assert!(reopened_v2.selected().scope_evidence().is_empty());
    assert_eq!(
        reopened_v2.canonical_bytes().expect("v2 re-encode"),
        legacy_v2
    );
}

#[test]
fn restart_selection_rejects_mixed_v2_and_v3_candidate_evidence() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 2);
    let selected = select_output_reconstruction(
        &spec,
        vec![
            exact_candidate(&spec, [3; 32], 31, &[0.0, 0.0]),
            exact_candidate(&spec, [4; 32], 41, &[0.0, 0.0]),
        ],
    )
    .expect("select v3 candidates");
    let v2_bytes = v2_from_v3(&selected.canonical_bytes().expect("v3 bytes"), 2);
    let v2 = tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &v2_bytes)
        .expect("reopen v2");
    let mut mixed = selected.candidates().to_vec();
    mixed[0] = v2.candidates()[0].clone();

    assert!(matches!(
        select_output_reconstruction(&spec, mixed),
        Err(OutputReconstructionError::CandidateSpecMismatch)
    ));
}

#[test]
fn selected_runtime_final_logits_use_the_shared_execution_digest_domain() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let candidate = exact_candidate(&spec, [5; 32], 7, &[3.0, -2.0]);
    let selected = select_output_reconstruction(&spec, vec![candidate]).expect("select candidate");
    let mut runtime = RuntimeFinalLogitsAccumulator::new();
    runtime
        .observe(&[3.0, -2.0])
        .expect("observe runtime logits");
    let runtime = runtime.finish().expect("finish runtime evidence");

    assert_eq!(
        selected.selected().runtime_final_logits_digest(),
        runtime.digest()
    );
    assert_eq!(selected.selected().runtime_batch_count(), 1);
    assert_eq!(selected.selected().runtime_logit_count(), 2);
}

#[test]
fn legacy_v1_remains_strictly_inspectable_but_bridge_ineligible() {
    use std::fmt::Write as _;

    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let candidate = exact_candidate(&spec, [6; 32], 1, &[0.0, 0.0]);
    let receipt = select_output_reconstruction(&spec, vec![candidate]).expect("receipt");
    let legacy = legacy_v1_from_v2(&receipt.canonical_bytes().expect("v2 receipt"), 1);
    let mut hex = String::new();
    for byte in &legacy {
        write!(&mut hex, "{byte:02x}").unwrap();
    }
    assert_eq!(
        hex,
        concat!(
            "545356324f55540001000000b729814d3f098f4a5cf471d8f7c5a5385fde515df732ec98f9397e18da27feb0",
            "cd2a8bf91c50d83d49e568653db8d1ef07afd4f96e8c9439cf7b240d6c94b901060606060606060606060606",
            "060606060606060606060606060606060606060601000000b729814d3f098f4a5cf471d8f7c5a5385fde515df",
            "732ec98f9397e18da27feb0060606060606060606060606060606060606060606060606060606060606060601",
            "00000000000000cd2a8bf91c50d83d49e568653db8d1ef07afd4f96e8c9439cf7b240d6c94b90154b5e5a36",
            "fb952e1b18fa2a5fd3ac34d13767d6e8aca269344fe040c660365060200000000000000020000000000000001",
            "000000000000000000000000000000ef39fafe422ee63f00000000000000000000000000000000a5d60afc8",
            "e6a35a9d662024952a44af57caaad3d56203529f97d2049431dae20295810666556aa78fe6a58af1001da3ef",
            "edbb9b73385e2022d9e0c344b222904"
        )
    );

    let inspected =
        tritium_quantize::OutputReconstructionReceipt::validate_legacy_v1_canonical_bytes(
            &spec, &legacy,
        )
        .expect("strict legacy validation");
    assert_eq!(inspected.candidate_count(), 1);
    assert_eq!(inspected.selected_candidate_id(), &[6; 32]);
    assert_eq!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(&spec, &legacy),
        Err(OutputReconstructionError::LegacyReceiptMissingRuntimeEvidence)
    );
    let mut corrupt = legacy;
    *corrupt.last_mut().expect("legacy receipt bytes") ^= 1;
    assert!(matches!(
        tritium_quantize::OutputReconstructionReceipt::validate_legacy_v1_canonical_bytes(
            &spec, &corrupt
        ),
        Err(OutputReconstructionError::MalformedReceipt(_))
    ));
}

#[test]
fn legacy_v1_rejects_reordered_candidate_blocks() {
    const HEADER_BYTES: usize = 112;
    const LEGACY_CANDIDATE_BYTES: usize = 224;
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 2);
    let first = exact_candidate(&spec, [6; 32], 1, &[0.0, 0.0]);
    let second = exact_candidate(&spec, [7; 32], 2, &[1.0, -1.0]);
    let receipt = select_output_reconstruction(&spec, vec![second, first]).expect("receipt");
    let mut legacy = legacy_v1_from_v2(&receipt.canonical_bytes().expect("v2 receipt"), 2);
    let first = legacy[HEADER_BYTES..HEADER_BYTES + LEGACY_CANDIDATE_BYTES].to_vec();
    let second = legacy
        [HEADER_BYTES + LEGACY_CANDIDATE_BYTES..HEADER_BYTES + 2 * LEGACY_CANDIDATE_BYTES]
        .to_vec();
    legacy[HEADER_BYTES..HEADER_BYTES + LEGACY_CANDIDATE_BYTES].copy_from_slice(&second);
    legacy[HEADER_BYTES + LEGACY_CANDIDATE_BYTES..HEADER_BYTES + 2 * LEGACY_CANDIDATE_BYTES]
        .copy_from_slice(&first);

    assert!(matches!(
        tritium_quantize::OutputReconstructionReceipt::validate_legacy_v1_canonical_bytes(
            &spec, &legacy
        ),
        Err(OutputReconstructionError::MalformedReceipt(
            "legacy candidate order"
        ))
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
fn scale_update_candidate_identity_binds_spec_seed_targets_and_f16_values() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let updates = vec![
        SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::from_f32(0.5)]).expect("first update"),
        SaltV2ScaleUpdate::new(0, 0, 1, vec![f16::from_f32(0.75)]).expect("second update"),
    ];
    let identity = spec
        .candidate_id_for_scale_updates(&[31; 32], 23, &updates)
        .expect("canonical updates identify a candidate");

    assert_ne!(identity, [0; 32]);
    assert_eq!(
        identity,
        spec.candidate_id_for_scale_updates(&[31; 32], 23, &updates)
            .expect("identity is deterministic")
    );
    assert_ne!(
        identity,
        spec.candidate_id_for_scale_updates(&[31; 32], 24, &updates)
            .unwrap()
    );
    assert_ne!(
        identity,
        spec.candidate_id_for_scale_updates(&[32; 32], 23, &updates)
            .unwrap()
    );
    let other_spec = OutputReconstructionSpec::new(
        ModelId::from_digest([1; 32]),
        [2; 32],
        [3; 32],
        [5; 32],
        OutputReconstructionSchedule::Blocks { block_count: 1 },
        OutputObjectiveWeights::new(1.0, 0.0, 1.0, 1.0).unwrap(),
        1,
        1,
    )
    .unwrap();
    assert_ne!(
        identity,
        other_spec
            .candidate_id_for_scale_updates(&[31; 32], 23, &updates)
            .unwrap()
    );

    let changed_scale = vec![
        SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::from_f32(0.5)]).unwrap(),
        SaltV2ScaleUpdate::new(0, 0, 1, vec![f16::from_f32(0.875)]).unwrap(),
    ];
    assert_ne!(
        identity,
        spec.candidate_id_for_scale_updates(&[31; 32], 23, &changed_scale)
            .unwrap()
    );

    let reordered_targets = vec![
        SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::from_f32(0.5)]).unwrap(),
        SaltV2ScaleUpdate::new(0, 1, 0, vec![f16::from_f32(0.75)]).unwrap(),
    ];
    assert_ne!(
        identity,
        spec.candidate_id_for_scale_updates(&[31; 32], 23, &reordered_targets)
            .unwrap()
    );
    assert!(matches!(
        spec.candidate_id_for_scale_updates(&[31; 32], 23, &[]),
        Err(OutputReconstructionError::EmptyScaleUpdateSet)
    ));
    assert!(matches!(
        spec.candidate_id_for_scale_updates(&[0; 32], 23, &updates),
        Err(OutputReconstructionError::MissingPackageIdentity)
    ));
    assert!(matches!(
        spec.candidate_id_for_scale_updates(
            &[31; 32],
            23,
            &[updates[1].clone(), updates[0].clone()]
        ),
        Err(OutputReconstructionError::NonCanonicalScaleUpdateOrder)
    ));
    assert!(matches!(
        spec.candidate_id_for_scale_updates(
            &[31; 32],
            23,
            &[updates[0].clone(), updates[0].clone()]
        ),
        Err(OutputReconstructionError::NonCanonicalScaleUpdateOrder)
    ));
    let invalid_scale = [SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::ZERO]).unwrap()];
    assert!(matches!(
        spec.candidate_id_for_scale_updates(&[31; 32], 23, &invalid_scale),
        Err(OutputReconstructionError::InvalidScaleUpdate)
    ));
    let nonfinite_scale = [SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::NAN]).unwrap()];
    assert!(matches!(
        spec.candidate_id_for_scale_updates(&[31; 32], 23, &nonfinite_scale),
        Err(OutputReconstructionError::InvalidScaleUpdate)
    ));

    let candidate = exact_candidate(&spec, identity, 23, &[0.0, 0.0]);
    assert_eq!(candidate.candidate_id(), &identity);
}

#[test]
fn bound_scale_candidate_carries_updates_into_output_evidence() {
    let output_spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let updates = [SaltV2ScaleUpdate::new(0, 0, 0, vec![f16::ONE]).unwrap()];
    let candidate = output_spec
        .scale_update_candidate(&[41; 32], 7, &updates)
        .expect("updates form a content-bound candidate");

    assert_eq!(candidate.updates(), &updates);
    assert_eq!(candidate.parent_package_digest(), &[41; 32]);
    let mut output = OutputReconstructionAccumulator::for_scale_candidate(&output_spec, &candidate)
        .expect("candidate belongs to this output spec");
    output
        .observe(
            OutputReconstructionScope::Block { start: 0, end: 1 },
            0,
            1,
            2,
            &[true],
            &[1.0, 0.0],
            &[1.0, 0.0],
        )
        .unwrap();
    output
        .observe(
            OutputReconstructionScope::FinalLogits,
            0,
            1,
            2,
            &[true],
            &[0.0, 1.0],
            &[0.0, 1.0],
        )
        .unwrap();
    let receipt = output.finish().expect("complete output evidence");
    assert_eq!(receipt.candidate_id(), candidate.candidate_id());

    let other_spec = spec(OutputReconstructionSchedule::Blocks { block_count: 2 }, 1);
    assert!(matches!(
        OutputReconstructionAccumulator::for_scale_candidate(&other_spec, &candidate),
        Err(OutputReconstructionError::CandidateSpecMismatch)
    ));
}

fn output_activation_cache(
    layer_index: u32,
    values: Vec<f32>,
    token_mask: Vec<bool>,
    sequence_ends: Vec<u64>,
) -> ActivationCache {
    let activation_spec = ActivationCacheSpec::new(
        layer_index,
        format!("model.layers.{layer_index}.residual.input"),
        5,
        2,
        ActivationDType::Float16,
        ActivationDigest::from_bytes([9; 32]),
        2,
    )
    .expect("valid per-layer activation spec");
    let mut builder = ActivationCacheBuilder::new(activation_spec.clone());
    builder
        .ingest(
            ActivationChunk::new(&activation_spec, 0, 5, values, token_mask, sequence_ends)
                .expect("valid layer activation chunk"),
        )
        .expect("ingest layer activation chunk");
    builder.finalize().expect("complete layer activation cache")
}

struct TestActivationSource(Vec<ActivationCache>);

impl OutputReconstructionActivationSource for TestActivationSource {
    fn layer_count(&self) -> usize {
        self.0.len()
    }

    fn layer_metadata(&self, layer_index: u32) -> Option<OutputReconstructionActivationLayer> {
        self.0.as_slice().layer_metadata(layer_index)
    }

    fn read_layer_window(
        &self,
        layer_index: u32,
        token_start: u64,
        token_count: u64,
        max_decoded_bytes: u64,
    ) -> Result<ActivationWindow, OutputReconstructionError> {
        self.0.as_slice().read_layer_window(
            layer_index,
            token_start,
            token_count,
            max_decoded_bytes,
        )
    }
}

#[test]
fn scheduled_output_window_reads_aligned_layer_caches_with_one_total_memory_budget() {
    let source = TestActivationSource(vec![
        output_activation_cache(
            0,
            vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0],
            vec![true, false, true, true, false],
            vec![2, 5],
        ),
        output_activation_cache(
            1,
            vec![
                10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0, 50.0, -50.0,
            ],
            vec![true, false, true, true, false],
            vec![2, 5],
        ),
    ]);
    let activation_digest = output_reconstruction_activation_digest(&source)
        .expect("ordered layer caches have a stable set identity");
    let spec = OutputReconstructionSpec::new(
        ModelId::from_digest([1; 32]),
        activation_digest,
        [9; 32],
        [8; 32],
        OutputReconstructionSchedule::SlidingWindows {
            block_count: 2,
            window_size: 2,
            stride: 1,
        },
        OutputObjectiveWeights::new(1.0, 0.0, 1.0, 1.0).unwrap(),
        1,
        1,
    )
    .expect("valid sliding-window reconstruction spec");
    let activations = OutputReconstructionActivationSet::new(&spec, &source)
        .expect("activation set matches output spec");
    let window = activations
        .read_window(
            OutputReconstructionScope::Block { start: 0, end: 2 },
            1,
            3,
            102,
        )
        .expect("read one bounded window across both layers");

    assert_eq!(window.first_block(), 0);
    assert_eq!(window.block_count(), 2);
    assert_eq!(window.decoded_bytes(), 102);
    assert_eq!(
        window.layer(0).unwrap().values(),
        &[2.0, -2.0, 3.0, -3.0, 4.0, -4.0]
    );
    assert_eq!(
        window.layer(1).unwrap().values(),
        &[20.0, -20.0, 30.0, -30.0, 40.0, -40.0]
    );
    assert_eq!(window.layer(0).unwrap().token_mask(), &[false, true, true]);
    assert_eq!(window.layer(0).unwrap().sequence_ends(), &[2]);
    assert!(matches!(
        activations.read_window(
            OutputReconstructionScope::Block { start: 0, end: 2 },
            1,
            3,
            101,
        ),
        Err(OutputReconstructionError::ActivationCache(_))
    ));
}

#[test]
fn fixed_trit_tile_refit_recovers_shared_scales_from_residual_outputs() {
    let activation_spec = ActivationCacheSpec::new(
        0,
        "model.layers.0.attn.q_proj.input",
        3,
        64,
        ActivationDType::Float16,
        ActivationDigest::from_bytes([9; 32]),
        3,
    )
    .unwrap();
    let mut builder = ActivationCacheBuilder::new(activation_spec.clone());
    builder
        .ingest(
            ActivationChunk::new(
                &activation_spec,
                0,
                3,
                {
                    let mut values = vec![0.0; 3 * 64];
                    values[0] = 1.0;
                    values[1] = 2.0;
                    values[64] = 2.0;
                    values[65] = 1.0;
                    values[128] = 1.0;
                    values[129] = -1.0;
                    values
                },
                vec![true; 3],
                vec![3],
            )
            .unwrap(),
        )
        .unwrap();
    let cache = builder.finalize().unwrap();
    let activations = cache.read_window(0, 3, 4096).unwrap();
    let mut trits = vec![Trit::ZERO; 128];
    trits[0] = Trit::from_i8(1).unwrap();
    trits[1] = Trit::from_i8(-1).unwrap();
    trits[64] = Trit::from_i8(1).unwrap();
    trits[65] = Trit::from_i8(1).unwrap();

    // First output row uses 2*(x0-x1); second uses 0.5*(x0+x1).
    let residual_outputs = [-2.0, 1.5, 2.0, 1.5, 4.0, 0.0];
    let fitted =
        fit_fixed_trit_tile_scale_refit(&activations, &residual_outputs, 2, 0, &trits, 64, 16)
            .unwrap();

    assert_eq!(fitted.scales(), &[2.0, 0.5]);
    assert_eq!(fitted.observations(), 6);
    assert!(fitted.squared_error() < 1e-12);

    let candidate = fit_fixed_trit_tile_scale_update(
        &activations,
        &residual_outputs,
        2,
        7,
        0,
        1,
        &trits,
        64,
        16,
    )
    .unwrap();
    assert_eq!(candidate.update().tensor_index(), 7);
    assert_eq!(candidate.update().tile_index(), 0);
    assert_eq!(candidate.update().plane_index(), 1);
    assert_eq!(
        candidate
            .update()
            .scales()
            .iter()
            .map(|scale| scale.to_f32())
            .collect::<Vec<_>>(),
        vec![2.0, 0.5]
    );
    assert!(candidate.squared_error() < 1e-12);
    assert!(matches!(
        fit_fixed_trit_tile_scale_refit(
            &activations,
            &residual_outputs,
            2,
            0,
            &trits[..127],
            64,
            16,
        ),
        Err(OutputReconstructionError::InvalidGeometry)
    ));
}

#[test]
fn scheduled_output_window_rejects_layer_cache_token_alignment_drift() {
    let caches = [
        output_activation_cache(
            0,
            vec![1.0, -1.0, 2.0, -2.0, 3.0, -3.0, 4.0, -4.0, 5.0, -5.0],
            vec![true, false, true, true, false],
            vec![2, 5],
        ),
        output_activation_cache(
            1,
            vec![
                10.0, -10.0, 20.0, -20.0, 30.0, -30.0, 40.0, -40.0, 50.0, -50.0,
            ],
            vec![true, true, false, true, false],
            vec![1, 5],
        ),
    ];
    let spec = OutputReconstructionSpec::new(
        ModelId::from_digest([1; 32]),
        output_reconstruction_activation_digest(caches.as_slice()).unwrap(),
        [9; 32],
        [8; 32],
        OutputReconstructionSchedule::SlidingWindows {
            block_count: 2,
            window_size: 2,
            stride: 1,
        },
        OutputObjectiveWeights::new(1.0, 0.0, 1.0, 1.0).unwrap(),
        1,
        1,
    )
    .unwrap();
    let activations = OutputReconstructionActivationSet::new(&spec, caches.as_slice()).unwrap();

    assert!(matches!(
        activations.read_window(
            OutputReconstructionScope::Block { start: 0, end: 2 },
            0,
            5,
            400,
        ),
        Err(OutputReconstructionError::ActivationWindowMismatch)
    ));
    assert!(matches!(
        activations.read_window(OutputReconstructionScope::FinalLogits, 0, 1, 100),
        Err(OutputReconstructionError::InvalidActivationWindowScope)
    ));
}

#[test]
fn strict_reopen_rejects_rehashed_but_unreachable_candidate_metrics() {
    let spec = spec(OutputReconstructionSchedule::Blocks { block_count: 1 }, 1);
    let candidate = exact_candidate(&spec, [6; 32], 1, &[0.0, 0.0]);
    let receipt = select_output_reconstruction(&spec, vec![candidate]).expect("receipt");

    let mut wrong_objective = receipt.canonical_bytes().expect("canonical");
    wrong_objective[344..352].copy_from_slice(&123.0f64.to_bits().to_le_bytes());
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
    wrong_observations[296..304].copy_from_slice(&99u64.to_le_bytes());
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

    let mut wrong_runtime_count = receipt.canonical_bytes().expect("canonical");
    wrong_runtime_count[280..288].copy_from_slice(&99u64.to_le_bytes());
    rehash_single_candidate_receipt(&mut wrong_runtime_count);
    assert!(matches!(
        tritium_quantize::OutputReconstructionReceipt::from_canonical_bytes(
            &spec,
            &wrong_runtime_count
        ),
        Err(OutputReconstructionError::MalformedReceipt("candidate"))
    ));
}

#[test]
fn strict_reopen_rejects_unrepresentable_count_before_candidate_allocation() {
    const MAX_CANDIDATES: usize = (4 * 1024 * 1024 - 144) / 272;
    let spec = spec(
        OutputReconstructionSchedule::Blocks { block_count: 1 },
        MAX_CANDIDATES + 1,
    );
    let mut bytes = Vec::new();
    bytes.extend_from_slice(b"TSV2OUT\0");
    bytes.extend_from_slice(&2u16.to_le_bytes());
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
