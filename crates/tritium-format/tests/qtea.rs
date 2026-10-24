use std::io::{Cursor, Read, Write};
use tritium_core::Trit;
use tritium_format::qtea::*;

fn limits() -> QteaLimits {
    QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: 4096,
    }
}
fn meta(rows: usize, columns: usize, ratio: f64) -> QteaMetadata {
    QteaMetadata {
        rows,
        columns,
        recipe: QteaRecipe {
            group_size: 2,
            salient_ratio: ratio,
            delta_coefficient: 0.96,
            fit_iterations: 1,
            rescale_iterations: 2,
            damping: 0.007,
            decay: 1.,
            relative_pivot_floor: 1e-7,
        },
        bindings: QteaBindings {
            weights: [1; 32],
            gram: [2; 32],
            source_model: [3; 32],
            activation_cache: [4; 32],
            token_stream: [5; 32],
        },
    }
}
fn tensor(rows: usize, columns: usize, ratio: f64) -> QteaTensor {
    let m = meta(rows, columns, ratio);
    let trits = (0..rows * columns)
        .map(|i| [Trit::NEG, Trit::ZERO, Trit::POS][i % 3])
        .collect::<Vec<_>>();
    let alpha = (0..rows).map(|i| i as f32 + 1.).collect::<Vec<_>>();
    let beta = (0..rows)
        .map(|i| [0.25, -0.5, 0., -0.][i % 4])
        .collect::<Vec<_>>();
    let multipliers = vec![1.; columns];
    let order = (0..columns).rev().collect::<Vec<_>>();
    let count = if ratio == 0. {
        0
    } else {
        ((columns / 2) as f64 * ratio).round_ties_even().max(1.) as usize * 2
    };
    let mut selected = order[..count].to_vec();
    selected.sort_unstable();
    let indices = selected
        .iter()
        .map(|c| vec![(c % 4) as u8; rows / 4])
        .collect::<Vec<_>>();
    let codes = selected
        .iter()
        .map(|c| vec![if c % 2 == 0 { 0x38 } else { 0xb8 }; rows / 4])
        .collect::<Vec<_>>();
    QteaTensor::from_parts(
        m,
        &trits,
        &multipliers,
        &order,
        std::iter::repeat_n(
            QteaGroupRef {
                alpha: &alpha,
                beta: &beta,
            },
            columns / 2,
        ),
        selected
            .iter()
            .enumerate()
            .map(|(i, &column)| QteaResidualRef {
                column,
                scale_multiplier: 2.,
                indices: &indices[i],
                codes: &codes[i],
            }),
        limits(),
    )
    .unwrap()
}
fn reseal(bytes: &mut [u8]) {
    let end = bytes.len() - 32;
    let mut h = blake3::Hasher::new_derive_key("tritium qtea tensor wire v1");
    h.update(&bytes[..end]);
    bytes[end..].copy_from_slice(h.finalize().as_bytes());
}

#[test]
fn fixed_wire_offsets_radix_digits_f32_bits_and_exact_ledger() {
    let t = tensor(4, 2, 0.);
    let bytes = t.as_bytes();
    let v = t.view();
    let ledger = v.byte_ledger();
    assert_eq!(bytes.len(), 466);
    assert_eq!(&bytes[..8], b"TQTEA1\0\0");
    assert_eq!(&bytes[8..16], &[1, 0, 0, 0, 128, 1, 0, 0]);
    assert_eq!(&bytes[16..24], &466_u64.to_le_bytes());
    assert_eq!(&bytes[24..32], &4_u64.to_le_bytes());
    assert_eq!(&bytes[32..40], &2_u64.to_le_bytes());
    assert_eq!(&bytes[384..386], &[102, 119]);
    assert_eq!(
        u32::from_le_bytes(bytes[414..418].try_into().unwrap()),
        (-0.0_f32).to_bits()
    );
    assert_eq!(
        v.group_parameters(0, 3).unwrap().1.to_bits(),
        (-0.0_f32).to_bits()
    );
    assert_eq!(ledger.header_bytes, 384);
    assert_eq!(ledger.trit_bytes, 2);
    assert_eq!(ledger.group_parameter_bytes, 32);
    assert_eq!(ledger.column_multiplier_bytes, 8);
    assert_eq!(ledger.order_bytes, 8);
    assert_eq!(ledger.residual_code_bytes, 0);
    assert_eq!(
        ledger.total_bytes,
        ledger.header_bytes
            + ledger.trit_bytes
            + ledger.group_parameter_bytes
            + ledger.column_multiplier_bytes
            + ledger.order_bytes
            + ledger.residual_metadata_bytes
            + ledger.residual_position_bytes
            + ledger.residual_code_bytes
            + ledger.checksum_bytes
    );
    assert!(t.buffer_capacity_bytes() >= bytes.len());
    assert!(QteaTensor::handle_size_bytes() > 0);
    let hex = include_str!("fixtures/qtea_v1.hex")
        .split_whitespace()
        .collect::<String>();
    let independent = (0..hex.len())
        .step_by(2)
        .map(|i| u8::from_str_radix(&hex[i..i + 2], 16).unwrap())
        .collect::<Vec<_>>();
    assert_eq!(
        bytes, independent,
        "frozen independent V1 bytes must remain readable/writable"
    );
    assert!(QteaTensorView::from_bytes(&independent, limits()).is_ok());
}

#[test]
fn borrowed_owned_stream_roundtrips_and_all_trit_position_tails() {
    for rows in [4, 8, 16, 20] {
        for columns in [2, 4, 6, 8, 10] {
            for ratio in [0., 1.] {
                let t = tensor(rows, columns, ratio);
                let view = QteaTensorView::from_bytes(t.as_bytes(), limits()).unwrap();
                assert_eq!(
                    view.metadata().byte_ledger(limits()).unwrap(),
                    view.byte_ledger()
                );
                assert_eq!(view.metadata(), meta(rows, columns, ratio));
                assert_eq!(
                    view.fitting_order().collect::<Vec<_>>(),
                    (0..columns).rev().collect::<Vec<_>>()
                );
                for r in 0..rows {
                    for c in 0..columns {
                        assert_eq!(
                            view.trit(r, c).unwrap(),
                            [Trit::NEG, Trit::ZERO, Trit::POS][(r * columns + c) % 3]
                        );
                        assert!(view.base_weight(r, c).unwrap().is_finite());
                    }
                }
                assert_eq!(view.trit(rows, 0), Err(QteaError::InvalidIndex));
                assert_eq!(
                    view.group_parameters(columns / 2, 0),
                    Err(QteaError::InvalidIndex)
                );
                assert_eq!(
                    view.column_multiplier(columns),
                    Err(QteaError::InvalidIndex)
                );
                let mut written = Vec::new();
                t.write_to(&mut written).unwrap();
                assert_eq!(written, t.as_bytes());
                let read = QteaTensor::read_from(&mut Cursor::new(&written), limits()).unwrap();
                assert_eq!(read.as_bytes(), t.as_bytes());
                assert_eq!(read.view().content_digest(), view.content_digest());
                let owned = QteaTensor::from_owned_bytes(written, limits()).unwrap();
                assert_eq!(owned.as_bytes(), t.as_bytes());
                for record in view.residuals() {
                    assert_eq!(record.value_at_row(rows), Err(QteaError::InvalidIndex));
                    for r in 0..rows {
                        assert_eq!(
                            record.value_at_row(r).unwrap().is_some(),
                            r % 4 == record.column() % 4
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn every_truncation_trailing_data_integrity_and_rehashed_malformed_body_refuse() {
    let t = tensor(4, 4, 0.5);
    assert_eq!(t.as_bytes().len(), 536);
    for cut in 0..t.as_bytes().len() {
        assert!(
            QteaTensorView::from_bytes(&t.as_bytes()[..cut], limits()).is_err(),
            "cut {cut}"
        );
        assert!(QteaTensor::read_from(&mut Cursor::new(&t.as_bytes()[..cut]), limits()).is_err());
    }
    let mut trailing = t.as_bytes().to_vec();
    trailing.push(0);
    assert!(QteaTensorView::from_bytes(&trailing, limits()).is_err());
    assert!(QteaTensor::read_from(&mut Cursor::new(trailing), limits()).is_err());
    let mut broken = t.as_bytes().to_vec();
    broken[400] ^= 1;
    assert_eq!(
        QteaTensorView::from_bytes(&broken, limits()).unwrap_err(),
        QteaError::InvalidBody
    );
    let corruptions = [
        (384, vec![243]),
        (387, vec![201]), // invalid B3 and nonzero-semantic tail
        (388, 0_f32.to_bits().to_le_bytes().to_vec()), // alpha
        (392, f32::NAN.to_bits().to_le_bytes().to_vec()),
        (396, (-1_f32).to_bits().to_le_bytes().to_vec()),
        (452, f32::INFINITY.to_bits().to_le_bytes().to_vec()), // column-v
        (452, (-1_f32).to_bits().to_le_bytes().to_vec()),
        (468, 2_u32.to_le_bytes().to_vec()), // duplicate permutation value
        (484, 0_u32.to_le_bytes().to_vec()), // record outside salient prefix
        (494, 2_u32.to_le_bytes().to_vec()), // duplicate record column
        (492, vec![0x04]),
        (493, vec![0x7f]), // position padding / FP8 NaN
        (493, vec![0xff]),
        (488, (-1_f32).to_bits().to_le_bytes().to_vec()),
        (488, f32::from_bits(1).to_bits().to_le_bytes().to_vec()),
    ]; // decoded residual overflow
    for (offset, value) in corruptions {
        let mut bytes = t.as_bytes().to_vec();
        bytes[offset..offset + value.len()].copy_from_slice(&value);
        reseal(&mut bytes);
        assert!(
            QteaTensorView::from_bytes(&bytes, limits()).is_err(),
            "resealed offset {offset}"
        );
    }
    for offset in [8, 12, 16, 24, 32, 40, 48, 56, 100, 168, 304] {
        let mut bytes = t.as_bytes().to_vec();
        bytes[offset] ^= 1;
        reseal(&mut bytes);
        assert!(
            QteaTensorView::from_bytes(&bytes, limits()).is_err(),
            "header {offset}"
        );
    }
    let mut missing = t.as_bytes().to_vec();
    missing[104..136].fill(0);
    reseal(&mut missing);
    assert_eq!(
        QteaTensorView::from_bytes(&missing, limits()).unwrap_err(),
        QteaError::InvalidMetadata
    );
    let mut base_overflow = t.as_bytes().to_vec();
    base_overflow[388..392].copy_from_slice(&f32::MAX.to_le_bytes());
    base_overflow[452..456].copy_from_slice(&2_f32.to_le_bytes());
    reseal(&mut base_overflow);
    assert_eq!(
        QteaTensorView::from_bytes(&base_overflow, limits()).unwrap_err(),
        QteaError::InvalidBody
    );
    let mut addition_overflow = t.as_bytes().to_vec();
    addition_overflow[440..444].copy_from_slice(&f32::MAX.to_le_bytes());
    addition_overflow[488..492].copy_from_slice(&1e-35_f32.to_le_bytes());
    addition_overflow[493] = 0x7e;
    reseal(&mut addition_overflow);
    assert_eq!(
        QteaTensorView::from_bytes(&addition_overflow, limits()).unwrap_err(),
        QteaError::InvalidBody
    );
}

#[test]
fn selected_fp8_signed_zero_is_preserved_and_not_confused_with_a_skipped_row() {
    let alpha = [f32::from_bits(1); 4];
    let beta = [-0.0; 4];
    let records = [
        QteaResidualRef {
            column: 0,
            scale_multiplier: 1.,
            indices: &[0],
            codes: &[0x80],
        },
        QteaResidualRef {
            column: 1,
            scale_multiplier: 1.,
            indices: &[1],
            codes: &[0],
        },
    ];
    let t = QteaTensor::from_parts(
        meta(4, 2, 1.),
        &[Trit::NEG; 8],
        &[0.01; 2],
        &[0, 1],
        std::iter::once(QteaGroupRef {
            alpha: &alpha,
            beta: &beta,
        }),
        records.into_iter(),
        limits(),
    )
    .unwrap();
    let view = t.view();
    let record = view.residuals().next().unwrap();
    assert_eq!(
        record.value_at_row(0).unwrap().unwrap().to_bits(),
        (-0_f32).to_bits()
    );
    assert_eq!(record.value_at_row(1).unwrap(), None);
    assert_eq!(
        view.base_weight(0, 0).unwrap().to_bits(),
        (-0_f32).to_bits()
    );
}

struct HeaderOnly<'a> {
    bytes: &'a [u8],
    read: usize,
}
impl Read for HeaderOnly<'_> {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        assert!(
            self.read < 384,
            "receiver read body before resource admission"
        );
        let n = out.len().min(384 - self.read);
        out[..n].copy_from_slice(&self.bytes[self.read..self.read + n]);
        self.read += n;
        Ok(n)
    }
}

#[test]
fn caps_precede_body_read_and_owned_overreservation_is_not_hidden() {
    let t = tensor(4, 2, 0.);
    let bytes = t.as_bytes();
    for cap in [
        QteaLimits {
            max_tensor_bytes: bytes.len() - 1,
            ..limits()
        },
        QteaLimits {
            max_validation_bytes: 1,
            ..limits()
        },
    ] {
        let mut reader = HeaderOnly { bytes, read: 0 };
        assert!(QteaTensor::read_from(&mut reader, cap).is_err());
        assert_eq!(reader.read, 384);
    }
    let mut huge = bytes.to_vec();
    huge[32..40].copy_from_slice(&(1_u64 << 24).to_le_bytes());
    let mut reader = HeaderOnly {
        bytes: &huge,
        read: 0,
    };
    assert!(QteaTensor::read_from(&mut reader, limits()).is_err());
    let mut reserved = Vec::with_capacity(bytes.len() + 1024);
    reserved.extend_from_slice(bytes);
    let cap = QteaLimits {
        max_tensor_bytes: bytes.len(),
        max_validation_bytes: 2,
    };
    assert!(QteaTensorView::from_bytes(&reserved, cap).is_ok());
    assert_eq!(
        QteaTensor::from_owned_bytes(reserved, cap).unwrap_err(),
        QteaError::TensorLimit
    );
}

#[test]
fn resealed_nonzero_binding_is_a_different_identity_not_custody_attestation() {
    let t = tensor(4, 2, 0.);
    let mut bytes = t.as_bytes().to_vec();
    bytes[104] ^= 0x40;
    reseal(&mut bytes);
    let changed = QteaTensorView::from_bytes(&bytes, limits()).unwrap();
    assert_ne!(
        changed.metadata().bindings.weights,
        t.view().metadata().bindings.weights
    );
    assert_ne!(changed.content_digest(), t.view().content_digest());
}

struct FailingWriter;
impl Write for FailingWriter {
    fn write(&mut self, _: &[u8]) -> std::io::Result<usize> {
        Err(std::io::ErrorKind::Other.into())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}
#[test]
fn writer_failure_is_not_success_and_wrong_provider_shapes_refuse() {
    assert!(matches!(
        tensor(4, 2, 0.).write_to(&mut FailingWriter),
        Err(QteaError::Io(_))
    ));
    let m = meta(4, 2, 0.);
    let alpha = [1.; 3];
    let other = [0.; 4];
    let result = QteaTensor::from_parts(
        m,
        &[Trit::ZERO; 8],
        &[1.; 2],
        &[1, 0],
        std::iter::once(QteaGroupRef {
            alpha: &alpha,
            beta: &other,
        }),
        std::iter::empty(),
        limits(),
    );
    assert_eq!(result.unwrap_err(), QteaError::InvalidGeometry);
}

struct Lying<I> {
    inner: I,
    claimed: usize,
}
impl<I: Iterator> Iterator for Lying<I> {
    type Item = I::Item;
    fn next(&mut self) -> Option<Self::Item> {
        self.inner.next()
    }
    fn size_hint(&self) -> (usize, Option<usize>) {
        (self.claimed, Some(self.claimed))
    }
}
impl<I: Iterator> ExactSizeIterator for Lying<I> {
    fn len(&self) -> usize {
        self.claimed
    }
}

#[test]
fn writer_counts_actual_items_instead_of_trusting_exact_size_claims() {
    let a = [1.; 4];
    let b = [0.; 4];
    let group = QteaGroupRef {
        alpha: &a,
        beta: &b,
    };
    for actual in [0, 2] {
        let groups = Lying {
            inner: std::iter::repeat_n(group, actual),
            claimed: 1,
        };
        let result = QteaTensor::from_parts(
            meta(4, 2, 0.),
            &[Trit::ZERO; 8],
            &[1.; 2],
            &[0, 1],
            groups,
            std::iter::empty(),
            limits(),
        );
        assert_eq!(result.unwrap_err(), QteaError::InvalidGeometry);
    }
    let records = [
        QteaResidualRef {
            column: 0,
            scale_multiplier: 1.,
            indices: &[0],
            codes: &[0],
        },
        QteaResidualRef {
            column: 1,
            scale_multiplier: 1.,
            indices: &[0],
            codes: &[0],
        },
        QteaResidualRef {
            column: 1,
            scale_multiplier: 1.,
            indices: &[0],
            codes: &[0],
        },
    ];
    for actual in [1, 3] {
        let residuals = Lying {
            inner: records[..actual].iter().copied(),
            claimed: 2,
        };
        let result = QteaTensor::from_parts(
            meta(4, 2, 1.),
            &[Trit::ZERO; 8],
            &[1.; 2],
            &[0, 1],
            std::iter::once(group),
            residuals,
            limits(),
        );
        assert_eq!(result.unwrap_err(), QteaError::InvalidGeometry);
    }
}

#[test]
fn every_single_prefix_bit_mutation_is_rejected_or_yields_finite_checked_values() {
    let original = tensor(4, 4, 0.5);
    for offset in 0..original.as_bytes().len() - 32 {
        for bit in 0..8 {
            let mut changed = original.as_bytes().to_vec();
            changed[offset] ^= 1 << bit;
            reseal(&mut changed);
            if let Ok(view) = QteaTensorView::from_bytes(&changed, limits()) {
                let m = view.metadata();
                for row in 0..m.rows {
                    for column in 0..m.columns {
                        assert!(view.base_weight(row, column).unwrap().is_finite());
                    }
                }
                for record in view.residuals() {
                    for row in 0..m.rows {
                        if let Some(value) = record.value_at_row(row).unwrap() {
                            assert!(value.is_finite());
                            assert!(
                                (view.base_weight(row, record.column()).unwrap() + value)
                                    .is_finite()
                            );
                        }
                    }
                }
            }
        }
    }
}
