//! Canonical incremental format equivalence; not device/model qualification.
use tritium_core::{PerTilePlan, Trit, UniformPlan};
use tritium_format::{
    AdditivePayload, AdditiveStreamDecoder, AdditiveStreamValidator, AdditiveTensor,
    AdditiveTensorError,
};
use tritium_schema::{
    ADMITTED_LAWS, AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneCounts, Transport,
};

fn stream<'a>(
    layout: AdditiveLayout,
    counts: &'a PlaneCounts,
    length: usize,
) -> Result<AdditiveStreamValidator<'a>, AdditiveTensorError> {
    match counts {
        PlaneCounts::Uniform(n) => AdditiveStreamValidator::new_uniform(layout, *n, length),
        PlaneCounts::PerTile(c) => AdditiveStreamValidator::new_per_tile(layout, c, length),
        _ => unreachable!(),
    }
}
fn borrowed(layout: AdditiveLayout, counts: &PlaneCounts, bytes: &[u8]) -> bool {
    match counts {
        PlaneCounts::Uniform(n) => AdditivePayload::new_uniform(layout, *n, bytes).is_ok(),
        PlaneCounts::PerTile(c) => AdditivePayload::new_per_tile(layout, c, bytes).is_ok(),
        _ => unreachable!(),
    }
}
#[test]
fn arbitrary_chunks_match_shared_canonical_validator_all_admitted_profiles() {
    let mut cases = 0;
    for law in ADMITTED_LAWS {
        for planes in 1..=law.max_planes {
            for codec in [PlaneCodec::D2, PlaneCodec::B3, PlaneCodec::S34] {
                for group in [32, 128] {
                    for mode in 0..3 {
                        let layout = AdditiveLayout {
                            rows: 3,
                            cols: 260,
                            tile: 256,
                            group,
                            max_planes: planes,
                            allocation: if mode == 0 {
                                PlaneAllocation::Uniform
                            } else {
                                PlaneAllocation::PerTile
                            },
                            codec,
                            law: law.law,
                            basis: Basis::Identity,
                            transport: Transport::Raw,
                        };
                        let counts = if mode == 0 {
                            PlaneCounts::Uniform(planes)
                        } else {
                            PlaneCounts::PerTile(if mode == 2 {
                                vec![0; 6]
                            } else {
                                [0, 1, planes, 2, 2, 0].map(|n| n.min(planes)).to_vec()
                            })
                        };
                        let (n, s) = match &counts {
                            PlaneCounts::Uniform(p) => {
                                let g = UniformPlan::new(layout, *p).unwrap();
                                (g.trit_count(), g.scale_count())
                            }
                            PlaneCounts::PerTile(c) => {
                                let g = PerTilePlan::new(layout, c).unwrap();
                                (g.trit_count(), g.scale_count())
                            }
                            _ => unreachable!(),
                        };
                        let values = (0..n)
                            .map(|i| [Trit::ZERO, Trit::NEG, Trit::POS, Trit::NEG][i % 4])
                            .collect();
                        let tensor = match &counts {
                            PlaneCounts::Uniform(p) => {
                                AdditiveTensor::new(layout, *p, values, vec![0.5; s]).unwrap()
                            }
                            PlaneCounts::PerTile(c) => AdditiveTensor::new_per_tile(
                                layout,
                                c.clone(),
                                values,
                                vec![0.5; s],
                            )
                            .unwrap(),
                            _ => unreachable!(),
                        };
                        let bytes = tensor.encode_payload().unwrap();
                        let required =
                            AdditiveStreamDecoder::working_bytes(layout, &counts, bytes.len())
                                .unwrap();
                        assert!(
                            matches!(AdditiveStreamDecoder::new(layout,&counts,bytes.len(),required-1),
                            Err(AdditiveTensorError::DecodeWorkingLimit {required:r,limit}) if r==required && limit==required-1)
                        );
                        assert!(borrowed(layout, &counts, &bytes));
                        for chunk in [1, 2, 3, 7, 159, 160, 161, 65536] {
                            let mut validator = stream(layout, &counts, bytes.len()).unwrap();
                            for part in bytes.chunks(chunk) {
                                validator.push(part).unwrap();
                            }
                            assert_eq!(validator.finish().unwrap(), if s == 0 { 0. } else { 0.5 });
                            let mut decoder =
                                AdditiveStreamDecoder::new(layout, &counts, bytes.len(), required)
                                    .unwrap();
                            for part in bytes.chunks(chunk) {
                                decoder.push(part).unwrap();
                            }
                            assert_eq!(decoder.finish().unwrap(), tensor);
                        }
                        for at in [0, bytes.len() / 2, bytes.len().saturating_sub(1)] {
                            if bytes.is_empty() {
                                continue;
                            }
                            let mut changed = bytes.clone();
                            changed[at] = 0xff;
                            let expected = borrowed(layout, &counts, &changed);
                            let mut validator = stream(layout, &counts, changed.len()).unwrap();
                            let actual = changed
                                .chunks(1)
                                .try_for_each(|c| validator.push(c))
                                .and_then(|()| validator.finish())
                                .is_ok();
                            assert_eq!(actual, expected, "codec={codec:?} mode={mode} at={at}");
                            let mut decoder =
                                AdditiveStreamDecoder::new(layout, &counts, bytes.len(), required)
                                    .unwrap();
                            let decoded = changed
                                .chunks(1)
                                .try_for_each(|part| decoder.push(part))
                                .and_then(|()| decoder.finish());
                            assert_eq!(decoded.is_ok(), expected);
                        }
                        if !bytes.is_empty() {
                            let mut validator = stream(layout, &counts, bytes.len()).unwrap();
                            validator.push(&bytes[..bytes.len() - 1]).unwrap();
                            assert!(validator.finish().is_err());
                        }
                        let mut validator = stream(layout, &counts, bytes.len()).unwrap();
                        assert!(validator.push(&vec![0; bytes.len() + 1]).is_err());
                        assert!(matches!(
                            validator.push(&[]),
                            Err(AdditiveTensorError::StreamPoisoned)
                        ));
                        assert!(matches!(
                            validator.finish(),
                            Err(AdditiveTensorError::StreamPoisoned)
                        ));
                        let mut decoder =
                            AdditiveStreamDecoder::new(layout, &counts, bytes.len(), required)
                                .unwrap();
                        assert!(decoder.push(&vec![0; bytes.len() + 1]).is_err());
                        assert!(matches!(
                            decoder.push(&[]),
                            Err(AdditiveTensorError::StreamPoisoned)
                        ));
                        assert!(matches!(
                            decoder.finish(),
                            Err(AdditiveTensorError::StreamPoisoned)
                        ));
                        cases += 1;
                    }
                }
            }
        }
    }
    assert_eq!(cases, 180);
}
