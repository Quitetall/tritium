//! Manual real-artifact gate for standard Q2_0 ↔ TQ2_0 token identity.
//!
//! Run with:
//! `TRITIUM_Q2_0_MODEL=/path/q2.gguf TRITIUM_TQ2_0_MODEL=/path/tq2.gguf \
//!  cargo test -p tritium-nn --test q2_0_interop -- --ignored --nocapture`

use std::collections::BTreeMap;
use std::path::PathBuf;

use half::f16;
use serde::{Deserialize, Serialize};
use serde_json::json;
use tritium_core::Trit;
use tritium_format::{
    GGML_TYPE_Q2_0, GGML_TYPE_TQ2_0, Q2_0_BLOCK_BYTES, TQ2_0_BLOCK_BYTES, TensorOut, pack_q2_0_row,
    pack_tq2_0_row, write_gguf,
};
use tritium_nn::ModelRunner;

// Link CPU registration consumed by ModelRunner::load_cpu.
use tritium_cpu as _;

const ACCEPTANCE_PATH: &str = concat!(
    env!("CARGO_MANIFEST_DIR"),
    "/../../tools/reference/bitnet_accept.json"
);
const QUALIFICATION_PROMPT_COUNT: usize = 3;
const QUALIFICATION_HORIZON: usize = 16;

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
enum EvidenceProfile {
    Smoke,
    Qualification,
}

const fn qualification_profile(prompt_count: usize, horizon: usize) -> EvidenceProfile {
    if prompt_count == QUALIFICATION_PROMPT_COUNT && horizon == QUALIFICATION_HORIZON {
        EvidenceProfile::Qualification
    } else {
        EvidenceProfile::Smoke
    }
}

#[derive(Deserialize)]
struct AcceptanceReference {
    token_ids: Vec<u32>,
    eval_ids: Vec<u32>,
    eos_token_id: u32,
}

fn artifact_path(name: &str) -> PathBuf {
    std::env::var_os(name)
        .map(PathBuf::from)
        .unwrap_or_else(|| panic!("set {name} to a real GGUF artifact"))
}

fn acceptance_prompts(reference: &AcceptanceReference) -> Vec<Vec<u32>> {
    let width = reference.token_ids.len().max(4);
    let mut prompts = vec![reference.token_ids.clone()];
    for start in [
        0,
        reference.eval_ids.len() / 3,
        reference.eval_ids.len() * 2 / 3,
    ] {
        let end = start.saturating_add(width).min(reference.eval_ids.len());
        if end > start {
            let prompt = reference.eval_ids[start..end].to_vec();
            if !prompts.contains(&prompt) {
                prompts.push(prompt);
            }
        }
    }
    prompts
}

fn assert_sibling_equivalence(
    q2_file: &tritium_format::GgufFile,
    q2_bytes: &[u8],
    tq2_file: &tritium_format::GgufFile,
    tq2_bytes: &[u8],
) {
    assert_eq!(q2_file.version, tq2_file.version, "GGUF versions differ");
    assert_eq!(q2_file.metadata, tq2_file.metadata, "GGUF metadata differs");
    assert_eq!(q2_file.tensors.len(), tq2_file.tensors.len());
    for (index, (q2, tq2)) in q2_file.tensors.iter().zip(&tq2_file.tensors).enumerate() {
        assert_eq!(q2.name, tq2.name, "tensor {index} name differs");
        assert_eq!(q2.dims, tq2.dims, "tensor {} shape differs", q2.name);
        if q2.ggml_type == tritium_format::GGML_TYPE_Q2_0 {
            assert_eq!(
                tq2.ggml_type,
                tritium_format::GGML_TYPE_TQ2_0,
                "{} is not a Q2_0/TQ2_0 sibling pair",
                q2.name
            );
            assert_eq!(q2.dims.len(), 2, "{} is not a matrix", q2.name);
            let k = usize::try_from(q2.dims[0]).expect("Q2_0 K fits usize");
            let rows = usize::try_from(q2.dims[1]).expect("Q2_0 N fits usize");
            assert!(
                k.is_multiple_of(tritium_format::QK_K),
                "{} K={k} is not Q2_0/TQ2_0 compatible",
                q2.name
            );
            let q2_groups = k / tritium_format::Q2_0_GROUP_SIZE;
            let tq2_groups = k / tritium_format::QK_K;
            let q2_row_bytes = q2_groups
                .checked_mul(tritium_format::Q2_0_BLOCK_BYTES)
                .expect("Q2_0 row length does not overflow");
            let tq2_row_bytes = tq2_groups
                .checked_mul(tritium_format::TQ2_0_BLOCK_BYTES)
                .expect("TQ2_0 row length does not overflow");
            let q2_start = q2_file
                .tensor_data_offset
                .checked_add(q2.offset)
                .and_then(|offset| usize::try_from(offset).ok())
                .expect("Q2_0 payload offset fits usize");
            let tq2_start = tq2_file
                .tensor_data_offset
                .checked_add(tq2.offset)
                .and_then(|offset| usize::try_from(offset).ok())
                .expect("TQ2_0 payload offset fits usize");
            let q2_end = rows
                .checked_mul(q2_row_bytes)
                .and_then(|length| q2_start.checked_add(length))
                .expect("Q2_0 payload end does not overflow");
            let tq2_end = rows
                .checked_mul(tq2_row_bytes)
                .and_then(|length| tq2_start.checked_add(length))
                .expect("TQ2_0 payload end does not overflow");
            let q2_payload = q2_bytes
                .get(q2_start..q2_end)
                .unwrap_or_else(|| panic!("{} Q2_0 payload out of bounds", q2.name));
            let tq2_payload = tq2_bytes
                .get(tq2_start..tq2_end)
                .unwrap_or_else(|| panic!("{} TQ2_0 payload out of bounds", q2.name));
            let mut q2_trits = vec![tritium_core::Trit::ZERO; k];
            let mut tq2_trits = vec![tritium_core::Trit::ZERO; k];
            let mut q2_scales = vec![half::f16::ZERO; q2_groups];
            let mut tq2_scales = vec![half::f16::ZERO; tq2_groups];
            for row in 0..rows {
                tritium_format::unpack_q2_0_row(
                    &q2_payload[row * q2_row_bytes..(row + 1) * q2_row_bytes],
                    &mut q2_trits,
                    &mut q2_scales,
                )
                .unwrap_or_else(|error| panic!("{} Q2_0 row {row}: {error}", q2.name));
                tritium_format::unpack_tq2_0_row(
                    &tq2_payload[row * tq2_row_bytes..(row + 1) * tq2_row_bytes],
                    &mut tq2_trits,
                    &mut tq2_scales,
                )
                .unwrap_or_else(|error| panic!("{} TQ2_0 row {row}: {error}", q2.name));
                for (group256, tq2_scale) in tq2_scales.iter().enumerate() {
                    assert!(
                        tq2_scale.is_finite(),
                        "{} TQ2_0 row {row} group {group256} scale is non-finite",
                        q2.name
                    );
                    let mut expected_scale = None;
                    for within in 0..4 {
                        let group64 = group256 * 4 + within;
                        let q2_scale = q2_scales[group64];
                        assert!(
                            q2_scale.is_finite(),
                            "{} Q2_0 row {row} group {group64} scale is non-finite",
                            q2.name
                        );
                        let start = group64 * tritium_format::Q2_0_GROUP_SIZE;
                        let end = start + tritium_format::Q2_0_GROUP_SIZE;
                        if q2_scale == half::f16::ZERO {
                            q2_trits[start..end].fill(tritium_core::Trit::ZERO);
                        } else if let Some(previous) = expected_scale {
                            assert_eq!(
                                q2_scale.to_bits(),
                                previous,
                                "{} Q2_0 row {row} has incompatible G64 scales inside G256 group {group256}",
                                q2.name
                            );
                        } else {
                            expected_scale = Some(q2_scale.to_bits());
                        }
                    }
                    let expected_scale = expected_scale.unwrap_or(half::f16::ZERO.to_bits());
                    assert_eq!(
                        tq2_scale.to_bits(),
                        expected_scale,
                        "{} row {row} G256 group {group256} scale differs",
                        q2.name
                    );
                    if *tq2_scale == half::f16::ZERO {
                        let start = group256 * tritium_format::QK_K;
                        tq2_trits[start..start + tritium_format::QK_K]
                            .fill(tritium_core::Trit::ZERO);
                    }
                }
                assert_eq!(
                    q2_trits, tq2_trits,
                    "{} row {row} decoded ternary values differ",
                    q2.name
                );
            }
            continue;
        }
        assert_eq!(q2.ggml_type, tq2.ggml_type, "{} type differs", q2.name);
        assert_eq!(q2.n_bytes, tq2.n_bytes, "{} payload size differs", q2.name);
        if q2.n_bytes == 0 {
            continue;
        }
        let q2_start = q2_file
            .tensor_data_offset
            .checked_add(q2.offset)
            .and_then(|offset| usize::try_from(offset).ok())
            .expect("Q2 sibling payload offset fits usize");
        let tq2_start = tq2_file
            .tensor_data_offset
            .checked_add(tq2.offset)
            .and_then(|offset| usize::try_from(offset).ok())
            .expect("TQ2 sibling payload offset fits usize");
        let len = usize::try_from(q2.n_bytes).expect("sibling payload length fits usize");
        let q2_end = q2_start
            .checked_add(len)
            .expect("Q2 sibling payload end does not overflow");
        let tq2_end = tq2_start
            .checked_add(len)
            .expect("TQ2 sibling payload end does not overflow");
        assert_eq!(
            q2_bytes
                .get(q2_start..q2_end)
                .unwrap_or_else(|| panic!("{} Q2 sibling payload out of bounds", q2.name)),
            tq2_bytes
                .get(tq2_start..tq2_end)
                .unwrap_or_else(|| panic!("{} TQ2 sibling payload out of bounds", q2.name)),
            "{} non-ternary payload differs",
            q2.name
        );
    }
}

#[test]
#[ignore = "real artifacts: set TRITIUM_Q2_0_MODEL and TRITIUM_TQ2_0_MODEL"]
fn q2_0_and_tq2_0_are_token_identical_on_acceptance_prompts() {
    let q2_path = artifact_path("TRITIUM_Q2_0_MODEL");
    let tq2_path = artifact_path("TRITIUM_TQ2_0_MODEL");
    let q2_bytes = std::fs::read(&q2_path).expect("read Q2_0 artifact");
    let tq2_bytes = std::fs::read(&tq2_path).expect("read TQ2_0 artifact");
    let q2_file = tritium_format::read_gguf(&q2_bytes).expect("parse Q2_0 artifact");
    let tq2_file = tritium_format::read_gguf(&tq2_bytes).expect("parse TQ2_0 artifact");
    assert_sibling_equivalence(&q2_file, &q2_bytes, &tq2_file, &tq2_bytes);
    let q2_count = q2_file
        .tensors
        .iter()
        .filter(|tensor| tensor.ggml_type == tritium_format::GGML_TYPE_Q2_0)
        .count();
    let tq2_count = tq2_file
        .tensors
        .iter()
        .filter(|tensor| tensor.ggml_type == tritium_format::GGML_TYPE_TQ2_0)
        .count();
    assert!(
        q2_count > 0,
        "Q2_0 artifact contains no standard Q2_0 tensors"
    );
    assert_eq!(q2_count, tq2_count, "sibling ternary tensor counts differ");

    let acceptance_bytes = std::fs::read(ACCEPTANCE_PATH).expect("read acceptance prompts");
    let reference: AcceptanceReference =
        serde_json::from_slice(&acceptance_bytes).expect("parse acceptance prompts");
    let horizon = std::env::var("TRITIUM_Q2_INTEROP_HORIZON")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid positive horizon"))
        .unwrap_or(QUALIFICATION_HORIZON);
    assert!(horizon > 0, "interop horizon must be positive");
    let prompt_limit = std::env::var("TRITIUM_Q2_INTEROP_PROMPT_LIMIT")
        .ok()
        .map(|value| value.parse::<usize>().expect("valid positive prompt limit"))
        .unwrap_or(usize::MAX);
    assert!(prompt_limit > 0, "prompt limit must be positive");
    let acceptance_prompts = acceptance_prompts(&reference);
    assert_eq!(
        acceptance_prompts.len(),
        QUALIFICATION_PROMPT_COUNT,
        "frozen qualification prompt construction changed"
    );
    let prompts: Vec<Vec<u32>> = acceptance_prompts.into_iter().take(prompt_limit).collect();
    let profile = qualification_profile(prompts.len(), horizon);

    let mut q2 = ModelRunner::load_cpu(&q2_bytes).expect("load Q2_0 model");
    let mut tq2 = ModelRunner::load_cpu(&tq2_bytes).expect("load TQ2_0 model");
    let mut outputs = Vec::with_capacity(prompts.len());
    for (index, prompt) in prompts.iter().enumerate() {
        let q2_tokens = q2
            .generate(prompt, horizon, reference.eos_token_id)
            .unwrap_or_else(|error| panic!("Q2_0 prompt {index}: {error}"));
        let tq2_tokens = tq2
            .generate(prompt, horizon, reference.eos_token_id)
            .unwrap_or_else(|error| panic!("TQ2_0 prompt {index}: {error}"));
        assert_eq!(
            q2_tokens, tq2_tokens,
            "Q2_0/TQ2_0 token divergence on acceptance prompt {index}"
        );
        outputs.push(q2_tokens);
    }

    println!(
        "{}",
        json!({
            "schema": "tritium.q2-interop-token-identity.v2",
            "profile": profile,
            "qualified": profile == EvidenceProfile::Qualification,
            "qualification_contract": {
                "prompt_count": QUALIFICATION_PROMPT_COUNT,
                "horizon": QUALIFICATION_HORIZON,
            },
            "q2_path": q2_path,
            "q2_blake3": blake3::hash(&q2_bytes).to_hex().to_string(),
            "tq2_path": tq2_path,
            "tq2_blake3": blake3::hash(&tq2_bytes).to_hex().to_string(),
            "acceptance_blake3": blake3::hash(&acceptance_bytes).to_hex().to_string(),
            "q2_tensor_count": q2_count,
            "tq2_tensor_count": tq2_count,
            "prompt_count": prompts.len(),
            "horizon": horizon,
            "prompts": prompts,
            "outputs": outputs,
        })
    );
}

fn sibling_artifacts(
    q2_trits: &[Trit],
    q2_scales: [f16; 4],
    tq2_trits: &[Trit],
    tq2_scale: f16,
) -> (
    tritium_format::GgufFile,
    Vec<u8>,
    tritium_format::GgufFile,
    Vec<u8>,
) {
    let mut q2_payload = vec![0u8; 4 * Q2_0_BLOCK_BYTES];
    pack_q2_0_row(q2_trits, &q2_scales, &mut q2_payload).expect("pack Q2_0 sibling");
    let mut tq2_payload = vec![0u8; TQ2_0_BLOCK_BYTES];
    pack_tq2_0_row(tq2_trits, &[tq2_scale], &mut tq2_payload).expect("pack TQ2_0 sibling");
    let metadata = BTreeMap::new();
    let q2_bytes = write_gguf(
        3,
        &metadata,
        &[TensorOut {
            name: "blk.0.w".to_owned(),
            dims: vec![256, 1],
            ggml_type: GGML_TYPE_Q2_0,
            data: &q2_payload,
        }],
    )
    .expect("write Q2_0 sibling");
    let tq2_bytes = write_gguf(
        3,
        &metadata,
        &[TensorOut {
            name: "blk.0.w".to_owned(),
            dims: vec![256, 1],
            ggml_type: GGML_TYPE_TQ2_0,
            data: &tq2_payload,
        }],
    )
    .expect("write TQ2_0 sibling");
    let q2_file = tritium_format::read_gguf(&q2_bytes).expect("parse Q2_0 sibling");
    let tq2_file = tritium_format::read_gguf(&tq2_bytes).expect("parse TQ2_0 sibling");
    (q2_file, q2_bytes, tq2_file, tq2_bytes)
}

#[test]
fn sibling_gate_rejects_different_ternary_payloads() {
    let q2_trits = vec![Trit::POS; 256];
    let tq2_trits = vec![Trit::NEG; 256];
    let (q2_file, q2_bytes, tq2_file, tq2_bytes) = sibling_artifacts(
        &q2_trits,
        [f16::from_f32(0.125); 4],
        &tq2_trits,
        f16::from_f32(0.125),
    );

    let rejected = std::panic::catch_unwind(|| {
        assert_sibling_equivalence(&q2_file, &q2_bytes, &tq2_file, &tq2_bytes);
    });
    assert!(
        rejected.is_err(),
        "same-schema artifacts with different ternary payloads passed sibling gate"
    );
}

#[test]
fn sibling_gate_compares_zero_scale_groups_by_weight_semantics() {
    let q2_trits = vec![Trit::POS; 256];
    let mut tq2_trits = q2_trits.clone();
    tq2_trits[..64].fill(Trit::ZERO);
    let scale = f16::from_f32(0.125);
    let (q2_file, q2_bytes, tq2_file, tq2_bytes) = sibling_artifacts(
        &q2_trits,
        [f16::ZERO, scale, scale, scale],
        &tq2_trits,
        scale,
    );

    assert_sibling_equivalence(&q2_file, &q2_bytes, &tq2_file, &tq2_bytes);
}

#[test]
fn qualification_profile_distinguishes_truncated_smoke() {
    assert_eq!(qualification_profile(3, 16), EvidenceProfile::Qualification);
    assert_eq!(qualification_profile(1, 16), EvidenceProfile::Smoke);
    assert_eq!(qualification_profile(3, 1), EvidenceProfile::Smoke);
}

#[test]
fn qualification_prompt_set_is_frozen_and_complete() {
    let bytes = std::fs::read(ACCEPTANCE_PATH).expect("read acceptance fixture");
    let reference: AcceptanceReference =
        serde_json::from_slice(&bytes).expect("parse acceptance fixture");
    assert_eq!(
        acceptance_prompts(&reference),
        vec![
            vec![128000, 791, 6864, 315, 9822, 374],
            vec![19558, 13, 578, 3363, 374, 264],
            vec![323, 36105, 11, 323, 374, 2162],
        ]
    );
}
