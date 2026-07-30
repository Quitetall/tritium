//! Shared-forward capture identity golden (ADR 0035 WS-A2).
//!
//! One forward pass over a calibration batch may feed the evidence builders
//! of many tensors. These tests pin the property that makes that sound: the
//! canonical dyadic reduction keys evidence identity on global sample
//! ordinals alone, deliberately excluding batch boundaries and orchestration,
//! so shared-forward records must be byte-identical to per-tensor records for
//! the same frozen inputs.

use tritium_quantize::{
    CurvatureSourceId, SaltV2Curvature, SaltV2KroneckerEvidenceBuilder, SaltV2KroneckerEvidenceSpec,
};

const GROUP: usize = 128;

fn spec(
    kind: SaltV2Curvature,
    tensor_index: u64,
    name: &str,
    rows: usize,
    columns: usize,
) -> SaltV2KroneckerEvidenceSpec {
    SaltV2KroneckerEvidenceSpec::new(
        kind,
        CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap(),
        tensor_index,
        name,
        rows,
        columns,
        0.125,
    )
    .unwrap()
}

// Constant within each G128 group per sample: group Grams stay numerically
// inside the strict PSD admission tolerance while differing across samples,
// groups, and streams.
fn grouped_rows(samples: usize, columns: usize, seed: f32) -> Vec<f32> {
    let mut values = Vec::with_capacity(samples * columns);
    for sample in 0..samples {
        for column in 0..columns {
            let group = column / GROUP;
            values.push(seed + sample as f32 + group as f32 * 0.5);
        }
    }
    values
}

fn ramp(count: usize, scale: f32) -> Vec<f32> {
    (0..count).map(|index| index as f32 * scale + 0.5).collect()
}

#[test]
fn shared_forward_feeding_reproduces_per_tensor_records_exactly() {
    // Three tensors: two share input stream 0, one reads wider stream 1. The
    // shared-forward orchestration feeds all three from one pass per batch;
    // the per-tensor path replays the full calibration once per tensor.
    let tensors = [
        (
            spec(SaltV2Curvature::GuidedFisher, 0, "a.weight", 3, GROUP),
            0_usize,
        ),
        (
            spec(SaltV2Curvature::GuidedFisher, 1, "b.weight", 2, GROUP),
            0_usize,
        ),
        (
            spec(SaltV2Curvature::GuidedFisher, 2, "c.weight", 4, 2 * GROUP),
            1_usize,
        ),
    ];
    let batches = [(2_usize, 0.25_f32), (3_usize, -0.5_f32)];
    let weights = [1.0_f64, 2.0, 0.5, 1.0, 3.0];
    let mask = [true, false, true, true, true];

    let mut shared = tensors
        .iter()
        .map(|(spec, stream)| {
            (
                SaltV2KroneckerEvidenceBuilder::new(spec.clone()).unwrap(),
                *stream,
            )
        })
        .collect::<Vec<_>>();
    let mut offset = 0;
    for (samples, scale) in batches {
        let streams = [
            grouped_rows(samples, GROUP, scale),
            grouped_rows(samples, 2 * GROUP, -scale),
        ];
        for (builder, stream) in &mut shared {
            let factors = ramp(samples * builder.spec().rows(), scale * 2.0);
            builder
                .accumulate_batch(
                    &streams[*stream],
                    Some(&factors),
                    samples,
                    Some(&weights[offset..offset + samples]),
                    Some(&mask[offset..offset + samples]),
                )
                .unwrap();
        }
        offset += samples;
    }

    for ((spec, stream), (shared_builder, _)) in tensors.iter().zip(&shared) {
        let mut standalone = SaltV2KroneckerEvidenceBuilder::new(spec.clone()).unwrap();
        let mut offset = 0;
        for (samples, scale) in batches {
            let streams = [
                grouped_rows(samples, GROUP, scale),
                grouped_rows(samples, 2 * GROUP, -scale),
            ];
            let factors = ramp(samples * spec.rows(), scale * 2.0);
            standalone
                .accumulate_batch(
                    &streams[*stream],
                    Some(&factors),
                    samples,
                    Some(&weights[offset..offset + samples]),
                    Some(&mask[offset..offset + samples]),
                )
                .unwrap();
            offset += samples;
        }
        let shared_evidence = shared_builder.finish().unwrap();
        let standalone_evidence = standalone.finish().unwrap();
        assert_eq!(
            shared_evidence.record_digest(),
            standalone_evidence.record_digest()
        );
        assert_eq!(
            shared_evidence.canonical_bytes().unwrap(),
            standalone_evidence.canonical_bytes().unwrap()
        );
    }
}

#[test]
fn shared_forward_rebatching_does_not_change_evidence_identity() {
    // A shared-forward driver may batch the calibration stream differently
    // from a per-tensor replay. Identity is keyed on the global sample
    // ordinal of every leaf, so one 5-sample batch and a 2+3 split must
    // produce the same canonical record, for both input-Hessian and
    // factor-bearing curvature.
    for kind in [SaltV2Curvature::InputHessian, SaltV2Curvature::GuidedFisher] {
        let contract = spec(kind, 9, "layer.weight", 2, GROUP);
        let activations = grouped_rows(5, GROUP, 1.5);
        let factors = ramp(5 * 2, 0.75);
        let weights = [1.0_f64, 2.0, 0.5, 1.0, 3.0];
        let mask = [true, true, false, true, true];
        let dense = !matches!(kind, SaltV2Curvature::InputHessian);

        let mut one_shot = SaltV2KroneckerEvidenceBuilder::new(contract.clone()).unwrap();
        one_shot
            .accumulate_batch(
                &activations,
                dense.then_some(&factors[..]),
                5,
                Some(&weights),
                Some(&mask),
            )
            .unwrap();

        let mut split = SaltV2KroneckerEvidenceBuilder::new(contract).unwrap();
        split
            .accumulate_batch(
                &activations[..2 * GROUP],
                dense.then_some(&factors[..4]),
                2,
                Some(&weights[..2]),
                Some(&mask[..2]),
            )
            .unwrap();
        split
            .accumulate_batch(
                &activations[2 * GROUP..],
                dense.then_some(&factors[4..]),
                3,
                Some(&weights[2..]),
                Some(&mask[2..]),
            )
            .unwrap();

        assert_eq!(
            one_shot.finish().unwrap().canonical_bytes().unwrap(),
            split.finish().unwrap().canonical_bytes().unwrap()
        );
    }
}
