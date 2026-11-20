//! Exact seek-backed feedback, not a GPU/physical-memory/performance qualification.
use std::io::{self, Cursor, Read, Seek, SeekFrom, Write};
use tritium_quantize::qtea_reference::{
    ArithmeticError, SeekBackedFitLimits, TensorFit, TensorFitConfig,
};
use tritium_quantize::{CurvatureSourceId, InputGram, InputGramAccumulator};

fn input(dead: bool) -> InputGram {
    let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32]).unwrap();
    let mut accumulator = InputGramAccumulator::new_bound(4, source, 0).unwrap();
    let mut values = [0.; 16];
    for c in usize::from(dead)..4 {
        values[c * 4 + c] = 1.;
    }
    accumulator
        .accumulate_batch(&values, 4, None, None)
        .unwrap();
    accumulator.finish().unwrap()
}
fn recipe() -> TensorFitConfig {
    TensorFitConfig {
        group_size: 2,
        ..TensorFitConfig::default()
    }
}
fn limits() -> SeekBackedFitLimits {
    SeekBackedFitLimits {
        max_work_bytes: TensorFit::required_seek_work_bytes(4, 4, 2).unwrap(),
        max_scratch_bytes: TensorFit::required_seek_scratch_bytes(4, 4).unwrap(),
    }
}
fn fit(
    scratch: &mut (impl Read + Write + Seek),
    caps: SeekBackedFitLimits,
) -> Result<TensorFit, ArithmeticError> {
    TensorFit::fit_seek_backed(
        &[-1., 1., -2., 2.].repeat(4),
        4,
        4,
        &input(false),
        recipe(),
        scratch,
        caps,
    )
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum Fault {
    None,
    NoAccess,
    Read,
    Write,
    Seek,
    WrongPosition,
    Truncate,
    Nonfinite,
    Flush,
    Extent,
}
struct Stream {
    cursor: Cursor<Vec<u8>>,
    fault: Fault,
    calls: usize,
    chunk: usize,
}
impl Stream {
    fn new(fault: Fault) -> Self {
        Self {
            cursor: Cursor::new(Vec::new()),
            fault,
            calls: 0,
            chunk: usize::MAX,
        }
    }
    fn touch(&mut self) {
        self.calls += 1;
        assert!(
            self.fault != Fault::NoAccess,
            "must refuse before scratch access"
        );
    }
}
impl Read for Stream {
    fn read(&mut self, bytes: &mut [u8]) -> io::Result<usize> {
        self.touch();
        if self.fault == Fault::Read {
            return Err(io::ErrorKind::PermissionDenied.into());
        }
        if self.fault == Fault::Truncate {
            return Ok(0);
        }
        let n = bytes.len().min(self.chunk);
        let read = self.cursor.read(&mut bytes[..n])?;
        if self.fault == Fault::Nonfinite && read >= 4 {
            bytes[..4].copy_from_slice(&f32::NAN.to_le_bytes());
        }
        Ok(read)
    }
}
impl Write for Stream {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        self.touch();
        if self.fault == Fault::Write {
            return Err(io::ErrorKind::WriteZero.into());
        }
        self.cursor.write(&bytes[..bytes.len().min(self.chunk)])
    }
    fn flush(&mut self) -> io::Result<()> {
        self.touch();
        if self.fault == Fault::Flush {
            return Err(io::ErrorKind::Other.into());
        }
        self.cursor.flush()
    }
}
impl Seek for Stream {
    fn seek(&mut self, from: SeekFrom) -> io::Result<u64> {
        self.touch();
        if self.fault == Fault::Seek {
            return Err(io::ErrorKind::Other.into());
        }
        let value = self.cursor.seek(from)?;
        if self.fault == Fault::WrongPosition && matches!(from, SeekFrom::Start(_)) {
            return Ok(value + 1);
        }
        if self.fault == Fault::Extent && from == SeekFrom::End(0) && value > 0 {
            return Ok(value + 1);
        }
        Ok(value)
    }
}

#[test]
fn caps_inputs_and_geometry_refuse_before_scratch_access() {
    let caps = limits();
    let mut scratch = Stream::new(Fault::NoAccess);
    assert_eq!(
        fit(
            &mut scratch,
            SeekBackedFitLimits {
                max_work_bytes: caps.max_work_bytes - 1,
                ..caps
            }
        ),
        Err(ArithmeticError::ResourceLimit {
            required: caps.max_work_bytes,
            limit: caps.max_work_bytes - 1
        })
    );
    assert_eq!(
        fit(
            &mut scratch,
            SeekBackedFitLimits {
                max_scratch_bytes: 63,
                ..caps
            }
        ),
        Err(ArithmeticError::ScratchLimit {
            required: 64,
            limit: 63
        })
    );
    let mut bad = [-1., 1., -2., 2.].repeat(4);
    bad[3] = f32::NAN;
    assert_eq!(
        TensorFit::fit_seek_backed(&bad, 4, 4, &input(false), recipe(), &mut scratch, caps),
        Err(ArithmeticError::NonFiniteInput { index: 3 })
    );
    assert_eq!(scratch.calls, 0);
    for (r, c, g) in [
        (0, 4, 2),
        (3, 4, 2),
        (4, 0, 2),
        (4, 4, 3),
        (4, 4, 0),
        (4, 1 << 25, 2),
    ] {
        assert_eq!(
            TensorFit::required_seek_work_bytes(r, c, g),
            Err(ArithmeticError::InvalidShape)
        );
    }
}

#[test]
fn nonempty_scratch_is_not_overwritten_and_short_io_is_supported() {
    let mut occupied = Cursor::new(vec![1, 2, 3, 4]);
    occupied.set_position(2);
    assert_eq!(
        fit(&mut occupied, limits()),
        Err(ArithmeticError::ScratchNotEmpty)
    );
    assert_eq!(occupied.into_inner(), [1, 2, 3, 4]);
    let mut short = Stream::new(Fault::None);
    short.chunk = 3;
    let actual = fit(&mut short, limits()).unwrap();
    let expected = TensorFit::fit(
        &[-1., 1., -2., 2.].repeat(4),
        4,
        4,
        &input(false),
        recipe(),
        usize::MAX,
    )
    .unwrap();
    assert_eq!(actual, expected);
    assert_eq!(short.cursor.get_ref().len(), 64);
    assert!(short.calls > 20);
}

#[test]
fn seek_read_write_flush_extent_and_nonfinite_fail_without_result() {
    for (fault, expected) in [
        (
            Fault::Seek,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::Other,
            },
        ),
        (
            Fault::WrongPosition,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::InvalidData,
            },
        ),
        (
            Fault::Write,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::WriteZero,
            },
        ),
        (
            Fault::Read,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::PermissionDenied,
            },
        ),
        (
            Fault::Truncate,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::UnexpectedEof,
            },
        ),
        (Fault::Nonfinite, ArithmeticError::NonFiniteArithmetic),
        (
            Fault::Flush,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::Other,
            },
        ),
        (
            Fault::Extent,
            ArithmeticError::ScratchIo {
                kind: io::ErrorKind::InvalidData,
            },
        ),
    ] {
        let mut scratch = Stream::new(fault);
        assert_eq!(fit(&mut scratch, limits()), Err(expected));
        if matches!(
            fault,
            Fault::Read | Fault::Truncate | Fault::Nonfinite | Fault::Flush | Fault::Extent
        ) {
            assert_eq!(
                scratch.cursor.get_ref().len(),
                64,
                "late failure must exercise real scratch"
            );
        }
    }
}

#[test]
fn dead_channels_signed_zero_and_refusals_keep_original_inputs_and_identity() {
    let weights = [-0., 1., -2., 2.].repeat(4);
    let original = weights.iter().map(|v| f32::to_bits(*v)).collect::<Vec<_>>();
    for dead in [false, true] {
        for ratio in [0., 0.5, 1.] {
            for rounds in [0, 2] {
                let config = TensorFitConfig {
                    salient_ratio: ratio,
                    rescale_iterations: rounds,
                    ..recipe()
                };
                let gram = input(dead);
                let normal = TensorFit::fit(&weights, 4, 4, &gram, config, usize::MAX).unwrap();
                let mut scratch = Cursor::new(Vec::new());
                let actual = TensorFit::fit_seek_backed(
                    &weights,
                    4,
                    4,
                    &gram,
                    config,
                    &mut scratch,
                    limits(),
                )
                .unwrap();
                assert_eq!(normal, actual);
                let codec = tritium_format::qtea::QteaLimits {
                    max_tensor_bytes: 4096,
                    max_validation_bytes: 4,
                };
                assert_eq!(
                    normal.to_qtea_tensor(codec).unwrap().as_bytes(),
                    actual.to_qtea_tensor(codec).unwrap().as_bytes()
                );
            }
        }
    }
    assert_eq!(
        weights.iter().map(|v| f32::to_bits(*v)).collect::<Vec<_>>(),
        original
    );
    let mut scratch = Cursor::new(Vec::new());
    assert!(matches!(
        TensorFit::fit_seek_backed(
            &[1.; 16],
            4,
            4,
            &input(false),
            recipe(),
            &mut scratch,
            limits()
        ),
        Err(ArithmeticError::DegenerateRow { .. })
    ));
    assert!(scratch.get_ref().is_empty());
}

#[test]
fn late_numerical_failure_leaves_only_private_scratch_without_fallback() {
    let config = TensorFitConfig {
        group_size: 4,
        decay: f64::MAX,
        ..recipe()
    };
    let mut scratch = Cursor::new(Vec::new());
    let weights = [-3., -1., 1., 3.].repeat(4);
    let caps = SeekBackedFitLimits {
        max_work_bytes: TensorFit::required_seek_work_bytes(4, 4, 4).unwrap(),
        ..limits()
    };
    assert_eq!(
        TensorFit::fit_seek_backed(&weights, 4, 4, &input(false), config, &mut scratch, caps),
        Err(ArithmeticError::NonFiniteArithmetic)
    );
    assert_eq!(scratch.get_ref().len(), 64);
    assert_eq!(
        TensorFit::fit(&weights, 4, 4, &input(false), config, usize::MAX),
        Err(ArithmeticError::NonFiniteArithmetic)
    );
}
