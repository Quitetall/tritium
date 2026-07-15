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
