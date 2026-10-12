use std::cell::{Cell, RefCell};
use std::io::{Cursor, Read, Seek, SeekFrom, Write};
use std::rc::Rc;

use half::f16;
use tritium_format::{
    AdditiveTensor, SaltTensorImportError as Error, SaltTensorImportLimits, SaltTensorSpoolError,
    SaltTensorSpoolLimits, import_salt_v2_tensor, import_salt_v2_tensor_to_spool,
    salt_v2::SaltV2Codec,
    salt_v2_package::{
        SaltV2Package, SaltV2PackageReadError, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor,
        SaltV2Tile, SaltV2Transform, write_salt_v2_package,
    },
};
use tritium_schema::{
    AdditiveLayout, Basis, PlaneAllocation, PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw,
    ScalePrecision, Transport,
};

fn limits() -> SaltTensorImportLimits {
    SaltTensorImportLimits {
        max_source_bytes: 1_000_000,
        max_trits: 1_000_000,
        max_scales: 1_000_000,
        max_owned_bytes: 1_000_000,
    }
}

fn spool_limits() -> SaltTensorSpoolLimits {
    SaltTensorSpoolLimits {
        max_source_bytes: 1_000_000,
        max_trits: 1_000_000,
        max_scales: 1_000_000,
        max_working_bytes: 2_000_000,
        max_payload_bytes: 1_000_000,
    }
}

fn check_spooled<R: Read + Seek>(reader: &mut SaltV2PackageReader<R>, migrated: &AdditiveTensor) {
    let mut spool = Cursor::new(vec![42; 17]);
    spool.set_position(17);
    let staged = import_salt_v2_tensor_to_spool(
        reader,
        "weight",
        migrated.layout(),
        spool_limits(),
        &mut spool,
    )
    .unwrap();
    assert_eq!(
        staged.tile_counts.as_slice(),
        migrated.tile_counts().unwrap()
    );
    assert_eq!(staged.blob.offset, 17);
    let bytes = &spool.get_ref()[17..];
    assert_eq!(staged.blob.length, bytes.len() as u64);
    assert_eq!(bytes, migrated.encode_payload().unwrap());
    assert_eq!(staged.blob.id.as_bytes(), blake3::hash(bytes).as_bytes());
    assert_eq!(&spool.get_ref()[..17], &[42; 17]);
    assert_eq!(spool.position(), spool.get_ref().len() as u64);
}

fn layout(rows: u64, cols: u64) -> AdditiveLayout {
    AdditiveLayout {
        rows,
        cols,
        tile: 256,
        group: 128,
        max_planes: 3,
        allocation: PlaneAllocation::PerTile,
        codec: PlaneCodec::D2,
        transport: Transport::Raw,
        basis: Basis::Identity,
        law: ScaleLaw {
            anchor: ScaleAnchor::Group,
            relation: PlaneRelation::Free,
            precision: ScalePrecision::F16,
        },
    }
}

fn tensor(rows: usize, cols: usize, counts: &[usize]) -> SaltV2Tensor {
    fixture(&[rows as u64, cols as u64], counts, 128, false, false)
}

fn fixture(
    dims: &[u64],
    counts: &[usize],
    source_group: usize,
    varied_scales: bool,
    zero_trits: bool,
) -> SaltV2Tensor {
    let length = dims.iter().product::<u64>() as usize;
    assert_eq!(length.div_ceil(256), counts.len());
    SaltV2Tensor::new_with_layout(
        "weight",
        dims.to_vec(),
        SaltV2Transform::None,
        source_group,
        counts
            .iter()
            .enumerate()
            .map(|(tile, &count)| {
                let n = (length - tile * 256).min(256);
                SaltV2Tile::new(
                    (0..count)
                        .map(|p| {
                            SaltV2Plane::new_with_scale_group_size(
                                (0..n)
                                    .map(|i| {
                                        if zero_trits {
                                            0
                                        } else {
                                            [0, 1, -1, 1][(i + p) % 4]
                                        }
                                    })
                                    .collect(),
                                (0..n.div_ceil(source_group))
                                    .map(|g| {
                                        if varied_scales {
                                            f16::from_f32(0.25 + (tile * 4 + g + p) as f32 / 16.)
                                        } else {
                                            [f16::from_bits(1), f16::from_bits(0x3555), f16::MAX][p]
                                        }
                                    })
                                    .collect(),
                                source_group,
                            )
                            .unwrap()
                        })
                        .collect(),
                )
                .unwrap()
            })
            .collect(),
    )
    .unwrap()
}

fn encoded(source: &SaltV2Tensor, codec: SaltV2Codec) -> Vec<u8> {
    write_salt_v2_package(&SaltV2Package::new(codec, vec![source.clone()]).unwrap())
        .unwrap()
        .bytes
}

fn check(source: &SaltV2Tensor, migrated: &AdditiveTensor) {
    let width = migrated.layout().cols as usize;
    let group = usize::from(migrated.layout().group);
    let index = tritium_core::OwnedPerTileIndex::new(
        migrated.layout(),
        migrated.tile_counts().unwrap().to_vec(),
    )
    .unwrap();
    for p in 0..usize::from(migrated.plane_count()) {
        for row in 0..migrated.layout().rows as usize {
            for tile in index.as_index().tiles_in_row(p, row).unwrap() {
                let columns = tile.columns();
                for col in columns.clone() {
                    let flat = row * width + col;
                    let original = &source.tiles()[flat / 256].planes()[p];
                    assert_eq!(
                        migrated.trits()[tile.trit_offset() + col - columns.start],
                        original.trits()[flat % 256]
                    );
                    assert_eq!(
                        f16::from_f32(
                            migrated.scales()[tile.scale_offset() + (col - columns.start) / group]
                        )
                        .to_bits(),
                        original.scales()[(flat % 256) / source.scale_group_size()].to_bits(),
                    );
                }
            }
        }
    }
    assert_eq!(
        *migrated,
        AdditiveTensor::decode_per_tile_payload(
            migrated.layout(),
            migrated.tile_counts().unwrap(),
            &migrated.encode_payload().unwrap(),
        )
        .unwrap()
    );
}

#[test]
fn actual_seek_reader_imports_flattened_ragged_tensor_without_invented_planes() {
    let source = tensor(2, 257, &[1, 1, 1]);
    let bytes =
        write_salt_v2_package(&SaltV2Package::new(SaltV2Codec::B3, vec![source.clone()]).unwrap())
            .unwrap();
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(bytes.bytes)).unwrap();
    let migrated = import_salt_v2_tensor(&mut reader, "weight", layout(2, 257), limits())
        .expect("the actual ragged flattened source is exactly representable");
    assert_eq!(migrated.tile_counts(), Some(&[1, 1, 1, 1][..]));
    assert_eq!(migrated.trits().len(), 514);
    check(&source, &migrated);
    reader.verify_unchanged().unwrap();
}

#[test]
fn every_source_codec_and_scale_group_preserves_exact_ragged_and_variable_data() {
    let mut cases = 0;
    for source_codec in [SaltV2Codec::D2, SaltV2Codec::B3, SaltV2Codec::S34] {
        for source_group in [64, 128, 256] {
            for width in [
                1usize, 3, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 512, 513,
            ] {
                for mode in 0..3 {
                    // Four rows make every source's last tile S34-legal even
                    // when individual rows are not aligned to four coefficients.
                    let counts: Vec<_> = (0..(4 * width).div_ceil(256))
                        .map(|i| match mode {
                            0 => 1,
                            1 => 3,
                            _ if width.is_multiple_of(256) || 256usize.is_multiple_of(width) => {
                                1 + i % 3
                            }
                            _ => 2,
                        })
                        .collect();
                    let source = fixture(&[4, width as u64], &counts, source_group, false, false);
                    let bytes = encoded(&source, source_codec);
                    for group in [32, 128] {
                        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
                            let mut reader =
                                SaltV2PackageReader::new_strict(Cursor::new(bytes.as_slice()))
                                    .unwrap();
                            let before = reader.package_id();
                            let migrated = import_salt_v2_tensor(
                                &mut reader,
                                "weight",
                                AdditiveLayout {
                                    group,
                                    codec,
                                    ..layout(4, width as u64)
                                },
                                limits(),
                            )
                            .unwrap();
                            check(&source, &migrated);
                            check_spooled(&mut reader, &migrated);
                            reader.verify_unchanged().unwrap();
                            assert_eq!(before, reader.package_id());
                            drop(reader);
                            assert_eq!(migrated.layout().cols, width as u64);
                            cases += 1;
                        }
                    }
                }
            }
        }
    }
    assert_eq!(cases, 1512);
}

#[test]
fn incompatible_count_or_scale_partitions_reject_even_with_zero_weights() {
    for zero_trits in [false, true] {
        for (counts, varied, expected) in [
            (
                vec![1, 2, 3],
                false,
                "V2 count partition is not row-tile representable",
            ),
            (
                vec![1, 1, 1],
                true,
                "V2 scale partition is not subgroup representable",
            ),
        ] {
            let source = fixture(&[2, 257], &counts, 128, varied, zero_trits);
            let bytes = encoded(&source, SaltV2Codec::B3);
            let mut reader =
                SaltV2PackageReader::new_strict(Cursor::new(bytes.as_slice())).unwrap();
            let identity = reader.package_id();
            assert_eq!(
                import_salt_v2_tensor(&mut reader, "weight", layout(2, 257), limits()),
                Err(Error::Unsupported(expected))
            );
            assert_eq!(reader.package_id(), identity);
            reader.verify_unchanged().unwrap();
        }
    }
    // Aligned rows and G32 subgroups can exactly preserve these differing
    // original G64 scale bits, not merely the constant-scale special case.
    let source = fixture(&[2, 256], &[1, 3], 64, true, false);
    let mut reader =
        SaltV2PackageReader::new_strict(Cursor::new(encoded(&source, SaltV2Codec::D2))).unwrap();
    let migrated = import_salt_v2_tensor(
        &mut reader,
        "weight",
        AdditiveLayout {
            group: 32,
            ..layout(2, 256)
        },
        limits(),
    )
    .unwrap();
    check(&source, &migrated);
    assert_eq!(
        import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits()),
        Err(Error::Unsupported(
            "V2 scale partition is not subgroup representable"
        ))
    );
}

#[test]
fn exact_limits_include_owned_index_and_one_byte_short_refuses() {
    let source = tensor(2, 256, &[1, 3]);
    let bytes = encoded(&source, SaltV2Codec::D2);
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(bytes.as_slice())).unwrap();
    let migrated = import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits()).unwrap();
    let info = reader.tensor_info("weight").unwrap();
    let exact = SaltTensorImportLimits {
        max_source_bytes: info.encoded_payload_bytes() + info.encoded_scale_bytes(),
        max_trits: migrated.trits().len() as u64,
        max_scales: migrated.scales().len() as u64,
        max_owned_bytes: (migrated.trits().len()
            + 4 * migrated.scales().len()
            + migrated.view().auxiliary_buffer_len_bytes().unwrap())
            as u64,
    };
    check(
        &source,
        &import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), exact).unwrap(),
    );
    for bounded in [
        SaltTensorImportLimits {
            max_source_bytes: exact.max_source_bytes - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_trits: exact.max_trits - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_scales: exact.max_scales - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_owned_bytes: exact.max_owned_bytes - 1,
            ..exact
        },
        SaltTensorImportLimits {
            max_owned_bytes: 1,
            ..exact
        },
    ] {
        assert!(matches!(
            import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), bounded),
            Err(Error::Limit { .. })
        ));
        reader.verify_unchanged().unwrap();
    }
}

#[test]
fn rank_flattening_and_unadmitted_metadata_do_not_guess_semantics() {
    for dims in [vec![512], vec![2, 2, 128]] {
        let source = fixture(&dims, &[1, 3], 128, false, false);
        let mut reader =
            SaltV2PackageReader::new_strict(Cursor::new(encoded(&source, SaltV2Codec::B3)))
                .unwrap();
        let width = *dims.last().unwrap();
        let rows = dims.iter().product::<u64>() / width;
        let migrated =
            import_salt_v2_tensor(&mut reader, "weight", layout(rows, width), limits()).unwrap();
        check(&source, &migrated);
        assert_eq!(
            import_salt_v2_tensor(&mut reader, "weight", layout(rows + 1, width), limits()),
            Err(Error::Geometry)
        );
        assert!(matches!(
            import_salt_v2_tensor(&mut reader, "missing", layout(rows, width), limits()),
            Err(Error::Read(_))
        ));
    }
    let source = tensor(2, 256, &[1, 3]);
    let rotated = SaltV2Tensor::new_with_transform(
        "weight",
        source.dims().to_vec(),
        SaltV2Transform::SignedRht { seed: 7, domain: 9 },
        source.tiles().to_vec(),
    )
    .unwrap();
    let mut reader =
        SaltV2PackageReader::new_strict(Cursor::new(encoded(&rotated, SaltV2Codec::D2))).unwrap();
    assert_eq!(
        import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits()),
        Err(Error::Unsupported("V2 transform lacks a bound block"))
    );
}

struct SharedSource {
    cursor: Rc<RefCell<Cursor<Vec<u8>>>>,
    armed: Rc<Cell<bool>>,
    zero_seeks: Rc<Cell<usize>>,
    max_request: Rc<Cell<usize>>,
    fail_fill: bool,
    replacement: Option<Vec<u8>>,
}

impl Read for SharedSource {
    fn read(&mut self, output: &mut [u8]) -> std::io::Result<usize> {
        self.max_request
            .set(self.max_request.get().max(output.len()));
        if self.armed.get() && self.fail_fill && self.zero_seeks.get() >= 2 {
            return Err(std::io::Error::other("injected second-visit read failure"));
        }
        let n = output.len().min(7);
        self.cursor.borrow_mut().read(&mut output[..n])
    }
}

impl Seek for SharedSource {
    fn seek(&mut self, position: SeekFrom) -> std::io::Result<u64> {
        if self.armed.get() && position == SeekFrom::Start(0) {
            self.zero_seeks.set(self.zero_seeks.get() + 1);
            if self.zero_seeks.get() == 2
                && let Some(replacement) = &self.replacement
            {
                self.cursor
                    .borrow_mut()
                    .get_mut()
                    .copy_from_slice(replacement);
            }
        }
        self.cursor.borrow_mut().seek(position)
    }
}

fn changed(source: &SaltV2Tensor) -> SaltV2Tensor {
    let mut tiles = source.tiles().to_vec();
    let mut planes = tiles[0].planes().to_vec();
    let mut trits: Vec<_> = planes[0].trits().iter().map(|trit| trit.get()).collect();
    trits[0] = if trits[0] == 0 { 1 } else { 0 };
    planes[0] = SaltV2Plane::new_with_scale_group_size(
        trits,
        planes[0].scales().to_vec(),
        source.scale_group_size(),
    )
    .unwrap();
    tiles[0] = SaltV2Tile::new(planes).unwrap();
    SaltV2Tensor::new_with_layout(
        source.name(),
        source.dims().to_vec(),
        source.transform(),
        source.scale_group_size(),
        tiles,
    )
    .unwrap()
}

#[test]
fn short_reads_and_fill_pass_failures_do_not_return_unverified_output() {
    let source = tensor(2, 256, &[1, 3]);
    for action in ["unchanged", "fail-fill", "change-fill"] {
        let bytes = encoded(&source, SaltV2Codec::D2);
        let cursor = Rc::new(RefCell::new(Cursor::new(bytes)));
        let armed = Rc::new(Cell::new(false));
        let zero_seeks = Rc::new(Cell::new(0));
        let max_request = Rc::new(Cell::new(0));
        let mut reader = SaltV2PackageReader::new_strict(SharedSource {
            cursor: Rc::clone(&cursor),
            armed: Rc::clone(&armed),
            zero_seeks: Rc::clone(&zero_seeks),
            max_request: Rc::clone(&max_request),
            fail_fill: action == "fail-fill",
            replacement: (action == "change-fill")
                .then(|| encoded(&changed(&source), SaltV2Codec::D2)),
        })
        .unwrap();
        armed.set(true);
        let result = import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits());
        assert_eq!(
            zero_seeks.get(),
            2,
            "the mutation/fault must reach the fill visit"
        );
        match action {
            "unchanged" => check(&source, &result.unwrap()),
            "fail-fill" => assert!(matches!(
                result,
                Err(Error::Read(SaltV2PackageReadError::Io { .. }))
            )),
            _ => assert!(matches!(
                result,
                Err(Error::Read(SaltV2PackageReadError::SourceChanged(_)))
            )),
        }
        assert!(max_request.get() <= 64 * 1024);
    }
}

#[test]
fn spooled_source_fill_faults_and_changes_do_not_return_a_blob() {
    let source = tensor(2, 256, &[1, 3]);
    for action in ["unchanged", "fail-fill", "change-fill"] {
        let cursor = Rc::new(RefCell::new(Cursor::new(encoded(&source, SaltV2Codec::D2))));
        let armed = Rc::new(Cell::new(false));
        let zero_seeks = Rc::new(Cell::new(0));
        let mut reader = SaltV2PackageReader::new_strict(SharedSource {
            cursor,
            armed: armed.clone(),
            zero_seeks: zero_seeks.clone(),
            max_request: Rc::new(Cell::new(0)),
            fail_fill: action == "fail-fill",
            replacement: (action == "change-fill")
                .then(|| encoded(&changed(&source), SaltV2Codec::D2)),
        })
        .unwrap();
        armed.set(true);
        let mut spool = Cursor::new(vec![42; 17]);
        spool.set_position(17);
        let result = import_salt_v2_tensor_to_spool(
            &mut reader,
            "weight",
            layout(2, 256),
            spool_limits(),
            &mut spool,
        );
        assert_eq!(zero_seeks.get(), 2);
        if action == "unchanged" {
            result.unwrap();
        } else {
            assert!(matches!(
                result,
                Err(SaltTensorSpoolError::Import(Error::Read(_)))
            ));
        }
        assert_eq!(&spool.get_ref()[..17], &[42; 17]);
    }
}

#[test]
fn spooled_caps_partitions_and_append_admission_do_not_change_existing_bytes() {
    for (source, target, cap) in [
        (tensor(2, 257, &[1, 2, 3]), layout(2, 257), spool_limits()),
        (
            fixture(&[2, 257], &[1, 1, 1], 128, true, true),
            layout(2, 257),
            spool_limits(),
        ),
        (
            tensor(2, 256, &[1, 3]),
            layout(2, 256),
            SaltTensorSpoolLimits {
                max_source_bytes: 1,
                ..spool_limits()
            },
        ),
        (
            tensor(2, 256, &[1, 3]),
            layout(2, 256),
            SaltTensorSpoolLimits {
                max_trits: 1,
                ..spool_limits()
            },
        ),
        (
            tensor(2, 256, &[1, 3]),
            layout(2, 256),
            SaltTensorSpoolLimits {
                max_scales: 1,
                ..spool_limits()
            },
        ),
        (
            tensor(2, 256, &[1, 3]),
            layout(2, 256),
            SaltTensorSpoolLimits {
                max_working_bytes: 1,
                ..spool_limits()
            },
        ),
        (
            tensor(2, 256, &[1, 3]),
            layout(2, 256),
            SaltTensorSpoolLimits {
                max_payload_bytes: 1,
                ..spool_limits()
            },
        ),
    ] {
        let mut reader =
            SaltV2PackageReader::new_strict(Cursor::new(encoded(&source, SaltV2Codec::B3)))
                .unwrap();
        let mut spool = Cursor::new(vec![42; 17]);
        spool.set_position(17);
        assert!(
            import_salt_v2_tensor_to_spool(&mut reader, "weight", target, cap, &mut spool).is_err()
        );
        assert_eq!(spool.into_inner(), vec![42; 17]);
    }
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded(
        &tensor(2, 256, &[1, 3]),
        SaltV2Codec::B3,
    )))
    .unwrap();
    let mut spool = Cursor::new(vec![42; 17]);
    spool.set_position(3);
    assert!(matches!(
        import_salt_v2_tensor_to_spool(
            &mut reader,
            "weight",
            layout(2, 256),
            spool_limits(),
            &mut spool
        ),
        Err(SaltTensorSpoolError::Import(Error::Unsupported(
            "spool must append at its current end"
        )))
    ));
    assert_eq!(spool.into_inner(), vec![42; 17]);
}

#[test]
fn spool_exact_conservative_caps_pass_and_one_byte_short_refuses_before_writes() {
    let source = tensor(2, 256, &[1, 3]);
    let mut reader =
        SaltV2PackageReader::new_strict(Cursor::new(encoded(&source, SaltV2Codec::D2))).unwrap();
    let owned = import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits()).unwrap();
    let info = reader.tensor_info("weight").unwrap();
    let mut caps = SaltTensorSpoolLimits {
        max_source_bytes: info.encoded_payload_bytes() + info.encoded_scale_bytes(),
        max_trits: owned.trits().len() as u64,
        max_scales: owned.scales().len() as u64,
        max_payload_bytes: owned.encode_payload().unwrap().len() as u64,
        max_working_bytes: 1000,
    };
    let mut empty = Cursor::new(Vec::new());
    let error =
        import_salt_v2_tensor_to_spool(&mut reader, "weight", layout(2, 256), caps, &mut empty)
            .unwrap_err();
    let SaltTensorSpoolError::Import(Error::Limit {
        resource: "spool working bytes",
        required,
        ..
    }) = error
    else {
        panic!("exact working admission")
    };
    assert!(empty.into_inner().is_empty());
    caps.max_working_bytes = required;
    import_salt_v2_tensor_to_spool(
        &mut reader,
        "weight",
        layout(2, 256),
        caps,
        &mut Cursor::new(Vec::new()),
    )
    .unwrap();
    for cap in [
        SaltTensorSpoolLimits {
            max_source_bytes: caps.max_source_bytes - 1,
            ..caps
        },
        SaltTensorSpoolLimits {
            max_trits: caps.max_trits - 1,
            ..caps
        },
        SaltTensorSpoolLimits {
            max_scales: caps.max_scales - 1,
            ..caps
        },
        SaltTensorSpoolLimits {
            max_working_bytes: caps.max_working_bytes - 1,
            ..caps
        },
        SaltTensorSpoolLimits {
            max_payload_bytes: caps.max_payload_bytes - 1,
            ..caps
        },
    ] {
        let mut spool = Cursor::new(vec![42; 17]);
        spool.set_position(17);
        assert!(matches!(
            import_salt_v2_tensor_to_spool(&mut reader, "weight", layout(2, 256), cap, &mut spool),
            Err(SaltTensorSpoolError::Import(Error::Limit { .. }))
        ));
        assert_eq!(spool.into_inner(), vec![42; 17]);
    }
}

#[test]
fn spooled_variable_tiles_and_partial_words_are_not_padded_between_rows() {
    for (cols, counts) in [
        (1, vec![1]),
        (256, vec![1, 3]),
        (257, vec![1, 1, 1]),
        (512, vec![1, 3, 2, 1]),
        (3, vec![1]),
        (255, vec![3]),
    ] {
        let rows = if cols == 255 || cols == 3 { 1 } else { 2 };
        let source = tensor(rows, cols, &counts);
        for codec in [PlaneCodec::D2, PlaneCodec::B3] {
            let mut reader =
                SaltV2PackageReader::new_strict(Cursor::new(encoded(&source, SaltV2Codec::B3)))
                    .unwrap();
            let target = AdditiveLayout {
                codec,
                ..layout(rows as u64, cols as u64)
            };
            let owned = import_salt_v2_tensor(&mut reader, "weight", target, limits()).unwrap();
            check_spooled(&mut reader, &owned);
        }
    }
}

struct FaultSpool {
    cursor: Cursor<Vec<u8>>,
    mode: &'static str,
    wrote: usize,
}
impl Read for FaultSpool {
    fn read(&mut self, out: &mut [u8]) -> std::io::Result<usize> {
        assert!(out.len() <= 65536);
        if self.mode == "read" {
            return Err(std::io::Error::other("spool read fault"));
        }
        let len = out.len().min(7);
        self.cursor.read(&mut out[..len])
    }
}
impl Write for FaultSpool {
    fn write(&mut self, bytes: &[u8]) -> std::io::Result<usize> {
        if self.mode == "write" && self.wrote >= 7 {
            return Err(std::io::Error::other("spool write fault"));
        }
        if self.mode == "zero" {
            return Ok(0);
        }
        let len = bytes.len().min(7);
        let n = self.cursor.write(&bytes[..len])?;
        if self.mode == "corrupt" && n != 0 {
            let index = self.cursor.position() as usize - n;
            self.cursor.get_mut()[index] ^= 1;
        }
        self.wrote += n;
        Ok(n)
    }
    fn flush(&mut self) -> std::io::Result<()> {
        panic!("caller owns flush/sync")
    }
}
impl Seek for FaultSpool {
    fn seek(&mut self, at: SeekFrom) -> std::io::Result<u64> {
        if self.mode == "seek" && matches!(at, SeekFrom::Start(_)) {
            return Err(std::io::Error::other("spool seek fault"));
        }
        self.cursor.seek(at)
    }
}

#[test]
fn spool_io_or_corruption_returns_no_blob_and_preserves_prior_ranges() {
    for mode in ["short", "write", "zero", "read", "seek", "corrupt"] {
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded(
            &tensor(2, 256, &[1, 3]),
            SaltV2Codec::D2,
        )))
        .unwrap();
        let mut spool = FaultSpool {
            cursor: Cursor::new(vec![42; 17]),
            mode,
            wrote: 0,
        };
        spool.cursor.set_position(17);
        let result = import_salt_v2_tensor_to_spool(
            &mut reader,
            "weight",
            layout(2, 256),
            spool_limits(),
            &mut spool,
        );
        match mode {
            "short" => {
                result.unwrap();
            }
            "corrupt" => assert!(matches!(result, Err(SaltTensorSpoolError::DigestMismatch))),
            _ => assert!(matches!(result, Err(SaltTensorSpoolError::Io(_)))),
        }
        assert_eq!(&spool.cursor.get_ref()[..17], &[42; 17]);
    }
}

#[test]
fn actual_file_staging_uses_bounded_codec_buffers_and_cleans_up() {
    struct Guard(std::path::PathBuf);
    impl Drop for Guard {
        fn drop(&mut self) {
            let _ = std::fs::remove_file(&self.0);
        }
    }
    let path = std::env::temp_dir().join(format!("tritium-v2-spool-{}", std::process::id()));
    let mut file = std::fs::OpenOptions::new()
        .read(true)
        .write(true)
        .create_new(true)
        .open(&path)
        .unwrap();
    let guard = Guard(path.clone());
    file.write_all(&[42; 17]).unwrap();
    // Force both coefficient and scale buffers to flush before finalization.
    let source = tensor(1, 256 * 65536, &vec![3; 65536]);
    let bytes = encoded(&source, SaltV2Codec::B3);
    let mut reader = SaltV2PackageReader::new_strict(Cursor::new(bytes)).unwrap();
    let caps = SaltTensorSpoolLimits {
        max_source_bytes: 100_000_000,
        max_trits: 100_000_000,
        max_scales: 10_000_000,
        max_payload_bytes: 100_000_000,
        ..spool_limits()
    };
    let result = import_salt_v2_tensor_to_spool(
        &mut reader,
        "weight",
        layout(1, 256 * 65536),
        caps,
        &mut file,
    )
    .unwrap();
    drop(reader);
    file.seek(SeekFrom::Start(17)).unwrap();
    let mut actual = Vec::new();
    file.read_to_end(&mut actual).unwrap();
    assert_eq!(actual.len() as u64, result.blob.length);
    assert_eq!(blake3::hash(&actual).as_bytes(), result.blob.id.as_bytes());
    let decoded = AdditiveTensor::decode_per_tile_payload(
        layout(1, 256 * 65536),
        &result.tile_counts,
        &actual,
    )
    .unwrap();
    check(&source, &decoded);
    drop(file);
    drop(guard);
    assert!(!path.exists());
}

#[test]
fn staged_legacy_tensor_dense_and_assets_reopen_as_one_imported_typed_package() {
    use tritium_format::trit_model::{
        TritModelStageError, encode_model_manifest, read_trit_model, stage_trit_model_from_spool,
    };
    use tritium_format::{
        TritBlobKind, TritPackageReadLimits, TritSpoolBlob, write_trit_package_from_spool,
    };
    use tritium_schema::{
        BlobId, BlobReference, LegacyModelIdentity, Level, LevelFit, ManifestAsset, ModelManifest,
        PlaneCounts, SchemaId, TensorMeta, TensorStorage,
    };
    let source = tensor(2, 257, &[1, 1, 1]);
    let original = Rc::new(RefCell::new(Cursor::new(encoded(&source, SaltV2Codec::B3))));
    let mut reader = SaltV2PackageReader::new_strict(SharedSource {
        cursor: Rc::clone(&original),
        armed: Rc::new(Cell::new(false)),
        zero_seeks: Rc::new(Cell::new(0)),
        max_request: Rc::new(Cell::new(0)),
        fail_fill: false,
        replacement: None,
    })
    .unwrap();
    let legacy = reader.package_id().to_string();
    let mut spool = Cursor::new(Vec::new());
    let target = AdditiveLayout {
        codec: PlaneCodec::B3,
        ..layout(2, 257)
    };
    let staged =
        import_salt_v2_tensor_to_spool(&mut reader, "weight", target, spool_limits(), &mut spool)
            .unwrap();
    let reference = |id| BlobReference { id, shard: None };
    let additive = TensorMeta {
        name: "weight".into(),
        blob: reference(staged.blob.id),
        storage: TensorStorage::Additive {
            layout: target,
            planes: PlaneCounts::PerTile(staged.tile_counts),
        },
    };
    let mut blobs = vec![staged.blob];
    let dense_bytes = half::f16::from_f32(1.).to_bits().to_le_bytes();
    let dense = TritSpoolBlob {
        kind: TritBlobKind::DenseTensor(tritium_core::DType::F16),
        id: BlobId::from_bytes(*blake3::hash(&dense_bytes).as_bytes()),
        offset: spool.position(),
        length: 2,
    };
    spool.write_all(&dense_bytes).unwrap();
    let dense_meta = TensorMeta {
        name: "preserved".into(),
        blob: reference(dense.id),
        storage: TensorStorage::Dense {
            shape: vec![1],
            precision: ScalePrecision::F16,
        },
    };
    blobs.push(dense);
    let mut assets = Vec::new();
    for name in ["config.json", "tokenizer.json"] {
        let asset = TritSpoolBlob {
            kind: TritBlobKind::Asset(name.into()),
            id: BlobId::from_bytes(*blake3::hash(b"{}").as_bytes()),
            offset: spool.position(),
            length: 2,
        };
        spool.write_all(b"{}").unwrap();
        assets.push(ManifestAsset {
            name: name.into(),
            blob: reference(asset.id),
        });
        blobs.push(asset);
    }
    let manifest = ModelManifest {
        schema: SchemaId { major: 1, minor: 0 },
        architecture: "llama".into(),
        assets,
        levels: vec![Level {
            name: "imported".into(),
            fit: LevelFit::Imported,
            tensors: vec![additive, dense_meta],
        }],
        shards: vec![],
        legacy_ids: vec![LegacyModelIdentity {
            id: legacy.clone(),
            level: "imported".into(),
        }],
    };
    let receiving = TritPackageReadLimits {
        max_source_bytes: 1_000_000,
        max_manifest_bytes: 64 * 1024,
        max_directory_bytes: 64 * 1024,
    };
    let calls = Cell::new(0);
    let mut staged_model = stage_trit_model_from_spool(
        &manifest,
        &blobs,
        &mut spool,
        receiving,
        Cursor::new(Vec::new()),
        || {
            calls.set(calls.get() + 1);
            reader.verify_unchanged() // original strict source, once before staged success
        },
    )
    .unwrap();
    assert_eq!(calls.get(), 1);
    let mut output = Vec::new();
    let id = write_trit_package_from_spool(
        &encode_model_manifest(&manifest).unwrap(),
        &blobs,
        &mut spool,
        1_000_000,
        &mut output,
    )
    .unwrap();
    let model = read_trit_model(&output).unwrap();
    assert_eq!(staged_model.package_id(), id);
    assert_eq!(staged_model.manifest(), model.manifest());
    assert_eq!(
        staged_model.model_id("imported").unwrap(),
        model.model_id("imported").unwrap()
    );
    let mut staged_payload = Vec::new();
    staged_model
        .visit_tensor("imported", "weight", |chunk| -> Result<(), ()> {
            staged_payload.extend_from_slice(chunk);
            Ok(())
        })
        .unwrap();
    assert_eq!(
        staged_payload,
        model.tensor("imported", "weight").unwrap().1
    );
    staged_model.verify().unwrap();
    model.validate_complete_payload_inventory().unwrap();
    assert_eq!(model.package_id(), id);
    assert_eq!(model.manifest().legacy_ids[0].id, legacy);
    assert_eq!(model.manifest().levels[0].fit, LevelFit::Imported);
    assert_eq!(
        model.tensor("imported", "preserved").unwrap().1,
        &dense_bytes
    );
    let (meta, payload) = model.tensor("imported", "weight").unwrap();
    let TensorStorage::Additive {
        layout,
        planes: PlaneCounts::PerTile(counts),
    } = &meta.storage
    else {
        panic!("actual compact metadata")
    };
    let recovered = AdditiveTensor::decode_per_tile_payload(*layout, counts, payload).unwrap();
    check(&source, &recovered);
    let altered = encoded(&changed(&source), SaltV2Codec::B3);
    original.borrow_mut().get_mut().copy_from_slice(&altered);
    assert!(matches!(
        stage_trit_model_from_spool(
            &manifest,
            &blobs,
            &mut spool,
            receiving,
            Cursor::new(Vec::new()),
            || reader.verify_unchanged(),
        ),
        Err(TritModelStageError::Guard(
            SaltV2PackageReadError::PackageChanged
        ))
    ));
    // Tiny structural assets do not claim runnable architecture/Qwen/release gates.
}

#[test]
fn caller_terminal_package_check_still_covers_unselected_tensor_changes() {
    let source = tensor(2, 256, &[1, 3]);
    let other = SaltV2Tensor::new_with_layout(
        "other",
        source.dims().to_vec(),
        source.transform(),
        source.scale_group_size(),
        source.tiles().to_vec(),
    )
    .unwrap();
    let bytes = write_salt_v2_package(
        &SaltV2Package::new(SaltV2Codec::D2, vec![source.clone(), other.clone()]).unwrap(),
    )
    .unwrap()
    .bytes;
    let cursor = Rc::new(RefCell::new(Cursor::new(bytes)));
    let mut reader = SaltV2PackageReader::new_strict(SharedSource {
        cursor: Rc::clone(&cursor),
        armed: Rc::new(Cell::new(false)),
        zero_seeks: Rc::new(Cell::new(0)),
        max_request: Rc::new(Cell::new(0)),
        fail_fill: false,
        replacement: None,
    })
    .unwrap();
    let altered = write_salt_v2_package(
        &SaltV2Package::new(SaltV2Codec::D2, vec![source.clone(), changed(&other)]).unwrap(),
    )
    .unwrap()
    .bytes;
    cursor.borrow_mut().get_mut().copy_from_slice(&altered);
    let staged = import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits()).unwrap();
    check(&source, &staged);
    assert_eq!(
        reader.verify_unchanged(),
        Err(SaltV2PackageReadError::PackageChanged)
    );
    // A tensor return has never been a whole-package publication authorization.
}

#[test]
fn header_metadata_presence_and_payload_mutations_fail_closed_before_return() {
    let source = tensor(2, 256, &[1, 3]);
    let original = encoded(&source, SaltV2Codec::D2);
    let mut bad_header = original.clone();
    bad_header[0] ^= 0x80;
    let renamed = SaltV2Tensor::new_with_layout(
        "weighu",
        source.dims().to_vec(),
        source.transform(),
        source.scale_group_size(),
        source.tiles().to_vec(),
    )
    .unwrap();
    for replacement in [
        bad_header,
        encoded(&renamed, SaltV2Codec::D2),
        encoded(&tensor(2, 256, &[3, 1]), SaltV2Codec::D2),
        encoded(
            &fixture(&[2, 256], &[1, 3], 128, true, false),
            SaltV2Codec::D2,
        ),
        encoded(&changed(&source), SaltV2Codec::D2),
    ] {
        assert_eq!(replacement.len(), original.len());
        let cursor = Rc::new(RefCell::new(Cursor::new(original.clone())));
        let mut reader = SaltV2PackageReader::new_strict(SharedSource {
            cursor: Rc::clone(&cursor),
            armed: Rc::new(Cell::new(false)),
            zero_seeks: Rc::new(Cell::new(0)),
            max_request: Rc::new(Cell::new(0)),
            fail_fill: false,
            replacement: None,
        })
        .unwrap();
        cursor.borrow_mut().get_mut().copy_from_slice(&replacement);
        assert!(matches!(
            import_salt_v2_tensor(&mut reader, "weight", layout(2, 256), limits()),
            Err(Error::Read(SaltV2PackageReadError::SourceChanged(_))),
        ));
    }
}
