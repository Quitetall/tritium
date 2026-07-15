use std::io::{self, Cursor, Read, Seek, SeekFrom};
use std::ops::Range;
use std::sync::{Arc, Mutex};

use half::{bf16, f16};
use tritium_format::{SafeTensors, SafeTensorsError, SafeTensorsReader};

#[derive(Debug)]
struct GuardedReader {
    inner: Cursor<Vec<u8>>,
    forbidden: Vec<Range<u64>>,
    trace: Arc<Mutex<ReadTrace>>,
    max_per_read: usize,
}

#[derive(Debug, Default)]
struct ReadTrace {
    ranges: Vec<Range<u64>>,
    max_request: usize,
}

impl Read for GuardedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        let start = self.inner.position();
        let request_len = buf.len().min(self.max_per_read);
        let requested_end = start.saturating_add(request_len as u64);
        if self
            .forbidden
            .iter()
            .any(|range| start < range.end && requested_end > range.start)
        {
            return Err(io::Error::new(
                io::ErrorKind::PermissionDenied,
                "attempted to read the unselected tensor payload",
            ));
        }
        let n = self.inner.read(&mut buf[..request_len])?;
        let mut trace = self.trace.lock().unwrap();
        trace.max_request = trace.max_request.max(buf.len());
        trace.ranges.push(start..start + n as u64);
        Ok(n)
    }
}

impl Seek for GuardedReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.inner.seek(pos)
    }
}

fn container(header: &str, data: Vec<u8>) -> Vec<u8> {
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend_from_slice(header.as_bytes());
    bytes.extend(data);
    bytes
}

#[test]
fn seek_reader_reads_only_selected_tensor_spans() {
    const SKIP_BYTES: usize = 256 * 1024;
    const TAIL_BYTES: usize = 128 * 1024;

    let mut data = vec![0xA5; SKIP_BYTES];
    for value in [1.0f32, -2.0, 0.5, -0.25] {
        data.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes());
    }
    for value in [9.0f32, -8.0] {
        data.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes());
    }
    for value in [3.0f32, -4.0, 0.125] {
        data.extend_from_slice(&value.to_le_bytes());
    }
    data.extend(std::iter::repeat_n(0x5A, TAIL_BYTES));
    let header = format!(
        r#"{{"skip":{{"dtype":"F32","shape":[{}],"data_offsets":[0,{SKIP_BYTES}]}},"a_bf16":{{"dtype":"BF16","shape":[2,2],"data_offsets":[{SKIP_BYTES},{}]}},"b_f16":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}},"c_f32":{{"dtype":"F32","shape":[3],"data_offsets":[{},{}]}},"tail":{{"dtype":"F32","shape":[{}],"data_offsets":[{},{}]}}}}"#,
        SKIP_BYTES / 4,
        SKIP_BYTES + 8,
        SKIP_BYTES + 8,
        SKIP_BYTES + 12,
        SKIP_BYTES + 12,
        SKIP_BYTES + 24,
        TAIL_BYTES / 4,
        SKIP_BYTES + 24,
        SKIP_BYTES + 24 + TAIL_BYTES,
    );
    let bytes = container(&header, data);
    let data_start = 8 + header.len() as u64;
    let trace = Arc::new(Mutex::new(ReadTrace::default()));
    let guarded = GuardedReader {
        inner: Cursor::new(bytes),
        forbidden: vec![
            data_start..data_start + SKIP_BYTES as u64,
            data_start + (SKIP_BYTES + 24) as u64
                ..data_start + (SKIP_BYTES + 24 + TAIL_BYTES) as u64,
        ],
        trace: Arc::clone(&trace),
        max_per_read: usize::MAX,
    };

    let mut tensors = SafeTensorsReader::new(guarded).expect("parse header");
    assert_eq!(tensors.len(), 5);
    assert_eq!(tensors.shape("a_bf16"), Some(&[2usize, 2][..]));
    assert_eq!(tensors.dtype("b_f16"), Some("F16"));
    // Later, earlier, then repeated reads prove every access uses an absolute seek.
    assert_eq!(tensors.tensor_f32("c_f32").unwrap(), vec![3.0, -4.0, 0.125]);
    assert_eq!(
        tensors.tensor_f32("a_bf16").unwrap(),
        vec![1.0, -2.0, 0.5, -0.25]
    );
    assert_eq!(tensors.tensor_f32("b_f16").unwrap(), vec![9.0, -8.0]);
    assert_eq!(tensors.tensor_f32("c_f32").unwrap(), vec![3.0, -4.0, 0.125]);

    let trace = trace.lock().unwrap();
    assert_eq!(
        trace.ranges,
        vec![
            0..8,
            8..data_start,
            data_start + (SKIP_BYTES + 12) as u64..data_start + (SKIP_BYTES + 24) as u64,
            data_start + SKIP_BYTES as u64..data_start + (SKIP_BYTES + 8) as u64,
            data_start + (SKIP_BYTES + 8) as u64..data_start + (SKIP_BYTES + 12) as u64,
            data_start + (SKIP_BYTES + 12) as u64..data_start + (SKIP_BYTES + 24) as u64,
        ]
    );
}

#[test]
fn seek_reader_bounds_payload_read_requests_and_handles_partial_reads() {
    const VALUES: usize = 40_000;
    let mut data = Vec::with_capacity(VALUES * 4);
    for value in 0..VALUES {
        data.extend_from_slice(&(value as f32).to_le_bytes());
    }
    let header = format!(
        r#"{{"x":{{"dtype":"F32","shape":[{VALUES}],"data_offsets":[0,{}]}}}}"#,
        data.len()
    );
    let trace = Arc::new(Mutex::new(ReadTrace::default()));
    let guarded = GuardedReader {
        inner: Cursor::new(container(&header, data)),
        forbidden: Vec::new(),
        trace: Arc::clone(&trace),
        // Force `read_exact` to tolerate a conforming reader that returns short reads.
        max_per_read: 7_919,
    };
    let mut tensors = SafeTensorsReader::new(guarded).unwrap();
    let values = tensors.tensor_f32("x").unwrap();
    assert_eq!(values.len(), VALUES);
    assert_eq!(values[0], 0.0);
    assert_eq!(values[VALUES - 1], (VALUES - 1) as f32);
    assert!(trace.lock().unwrap().max_request <= 64 * 1024);

    *trace.lock().unwrap() = ReadTrace::default();
    let mut visited = 0usize;
    tensors
        .visit_tensor_bytes("x", usize::MAX, |chunk| visited += chunk.len())
        .unwrap();
    assert_eq!(visited, VALUES * size_of::<f32>());
    assert!(trace.lock().unwrap().max_request <= 64 * 1024);
}

#[test]
fn seek_reader_rejects_oversized_headers_before_allocating() {
    let mut bytes = ((100_000_000u64) + 1).to_le_bytes().to_vec();
    bytes.extend_from_slice(b"{}");
    assert!(matches!(
        SafeTensorsReader::new(Cursor::new(bytes)),
        Err(SafeTensorsError::HeaderTooLarge { .. })
    ));
}

#[test]
fn seek_reader_reports_payload_io_and_bounds_errors() {
    let header = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#;
    let bytes = container(header, vec![0; 4]);
    let data_start = 8 + header.len() as u64;
    let guarded = GuardedReader {
        inner: Cursor::new(bytes),
        forbidden: std::iter::once(data_start..data_start + 4).collect(),
        trace: Arc::new(Mutex::new(ReadTrace::default())),
        max_per_read: usize::MAX,
    };
    let mut tensors = SafeTensorsReader::new(guarded).unwrap();
    assert!(matches!(
        tensors.tensor_f32("x"),
        Err(SafeTensorsError::Io { .. })
    ));

    assert!(matches!(
        SafeTensorsReader::new(Cursor::new(container(
            r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,8]}}"#,
            vec![0; 4],
        ))),
        Err(SafeTensorsError::OutOfBounds(_))
    ));
}

#[test]
fn seek_reader_matches_borrowed_scalar_and_zero_sized_tensors() {
    let header = r#"{"scalar":{"dtype":"F32","shape":[],"data_offsets":[0,4]},"empty":{"dtype":"F32","shape":[3,0,7],"data_offsets":[4,4]}}"#;
    let bytes = container(header, 1.25f32.to_le_bytes().to_vec());
    let borrowed = SafeTensors::parse(&bytes).unwrap();
    let mut seek = SafeTensorsReader::new(Cursor::new(&bytes)).unwrap();

    for name in ["scalar", "empty"] {
        assert_eq!(seek.shape(name), borrowed.shape(name));
        assert_eq!(seek.dtype(name), borrowed.dtype(name));
        assert_eq!(
            seek.tensor_f32(name).unwrap(),
            borrowed.tensor_f32(name).unwrap()
        );
    }
    assert_eq!(seek.tensor_f32("scalar").unwrap(), vec![1.25]);
    assert!(seek.tensor_f32("empty").unwrap().is_empty());
}

#[test]
fn seek_reader_visits_exact_stored_bytes_for_fp_and_non_widening_dtypes() {
    let bf16_bytes = [1.0f32, -2.0, 0.5]
        .into_iter()
        .flat_map(|value| bf16::from_f32(value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let f16_bytes = [9.0f32, -8.0]
        .into_iter()
        .flat_map(|value| f16::from_f32(value).to_bits().to_le_bytes())
        .collect::<Vec<_>>();
    let f32_bytes = [3.0f32, -4.0, 0.125]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    let u8_bytes = vec![0, 127, 255];
    let mut data = Vec::new();
    data.extend_from_slice(&bf16_bytes);
    data.extend_from_slice(&f16_bytes);
    data.extend_from_slice(&f32_bytes);
    data.extend_from_slice(&u8_bytes);
    let header = format!(
        r#"{{"a_bf16":{{"dtype":"BF16","shape":[3],"data_offsets":[0,{}]}},"b_f16":{{"dtype":"F16","shape":[2],"data_offsets":[{},{}]}},"c_f32":{{"dtype":"F32","shape":[3],"data_offsets":[{},{}]}},"d_u8":{{"dtype":"U8","shape":[3],"data_offsets":[{},{}]}}}}"#,
        bf16_bytes.len(),
        bf16_bytes.len(),
        bf16_bytes.len() + f16_bytes.len(),
        bf16_bytes.len() + f16_bytes.len(),
        bf16_bytes.len() + f16_bytes.len() + f32_bytes.len(),
        bf16_bytes.len() + f16_bytes.len() + f32_bytes.len(),
        data.len()
    );
    let trace = Arc::new(Mutex::new(ReadTrace::default()));
    let guarded = GuardedReader {
        inner: Cursor::new(container(&header, data)),
        forbidden: Vec::new(),
        trace: Arc::clone(&trace),
        max_per_read: usize::MAX,
    };
    let mut tensors = SafeTensorsReader::new(guarded).unwrap();
    *trace.lock().unwrap() = ReadTrace::default();

    for (name, dtype, shape, expected, chunk_cap) in [
        ("c_f32", "F32", &[3usize][..], &f32_bytes, 5usize),
        ("a_bf16", "BF16", &[3usize][..], &bf16_bytes, 1usize),
        ("c_f32", "F32", &[3usize][..], &f32_bytes, 2usize),
        ("b_f16", "F16", &[2usize][..], &f16_bytes, 3usize),
        ("d_u8", "U8", &[3usize][..], &u8_bytes, 2usize),
    ] {
        assert_eq!(tensors.dtype(name), Some(dtype));
        assert_eq!(tensors.shape(name), Some(shape));
        let mut observed = Vec::new();
        let mut chunk_lengths = Vec::new();
        tensors
            .visit_tensor_bytes(name, chunk_cap, |chunk| {
                assert!(!chunk.is_empty());
                assert!(chunk.len() <= chunk_cap);
                chunk_lengths.push(chunk.len());
                observed.extend_from_slice(chunk);
            })
            .unwrap();
        assert_eq!(&observed, expected);
        assert_eq!(chunk_lengths.iter().sum::<usize>(), expected.len());
    }

    assert!(trace.lock().unwrap().max_request <= 5);
    // A later widened read proves the visitor did not make subsequent access
    // depend on the underlying reader's current cursor.
    assert_eq!(tensors.tensor_f32("a_bf16").unwrap(), vec![1.0, -2.0, 0.5]);
}

#[test]
fn seek_reader_raw_visitor_rejects_missing_tensor_and_zero_chunk_cap() {
    let header = r#"{"x":{"dtype":"F32","shape":[1],"data_offsets":[0,4]},"empty":{"dtype":"F32","shape":[0],"data_offsets":[4,4]}}"#;
    let mut tensors = SafeTensorsReader::new(Cursor::new(container(header, vec![0; 4]))).unwrap();

    assert!(matches!(
        tensors.visit_tensor_bytes("missing", 1, |_| {}),
        Err(SafeTensorsError::NotFound(name)) if name == "missing"
    ));
    assert!(matches!(
        tensors.visit_tensor_bytes("x", 0, |_| {}),
        Err(SafeTensorsError::InvalidChunkSize { requested: 0 })
    ));
    let mut calls = 0;
    tensors
        .visit_tensor_bytes("empty", 1, |_| calls += 1)
        .unwrap();
    assert_eq!(calls, 0);
}

#[derive(Clone, Debug)]
struct SharedReader(Arc<Mutex<Cursor<Vec<u8>>>>);

impl Read for SharedReader {
    fn read(&mut self, buf: &mut [u8]) -> io::Result<usize> {
        self.0.lock().unwrap().read(buf)
    }
}

impl Seek for SharedReader {
    fn seek(&mut self, pos: SeekFrom) -> io::Result<u64> {
        self.0.lock().unwrap().seek(pos)
    }
}

#[test]
fn seek_reader_raw_visitor_reads_live_bytes_and_fails_on_truncation() {
    let header = r#"{"x":{"dtype":"F32","shape":[2],"data_offsets":[0,8]}}"#;
    let bytes = container(
        header,
        [1.0f32, 2.0]
            .into_iter()
            .flat_map(f32::to_le_bytes)
            .collect(),
    );
    let data_start = 8 + header.len();
    let shared = Arc::new(Mutex::new(Cursor::new(bytes)));
    let mut tensors = SafeTensorsReader::new(SharedReader(Arc::clone(&shared))).unwrap();

    let replacement = [3.0f32, 4.0]
        .into_iter()
        .flat_map(f32::to_le_bytes)
        .collect::<Vec<_>>();
    shared.lock().unwrap().get_mut()[data_start..].copy_from_slice(&replacement);
    let mut observed = Vec::new();
    tensors
        .visit_tensor_bytes("x", 3, |chunk| observed.extend_from_slice(chunk))
        .unwrap();
    assert_eq!(observed, replacement);

    shared.lock().unwrap().get_mut().truncate(data_start + 7);
    assert!(matches!(
        tensors.visit_tensor_bytes("x", 3, |_| {}),
        Err(SafeTensorsError::Io {
            kind: io::ErrorKind::UnexpectedEof,
            ..
        })
    ));
}
