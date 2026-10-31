//! Full output-grid diagnostic with constant-window encoded fixture generation.
use super::*;
#[path = "../../../../../tests/support/encoded_head.rs"]
mod fixture;
use fixture::{EncodedHead, SCALE_BITS};

fn assert_bits(actual: &[f32], expected: &[f32], mode: &str) {
    assert_eq!(actual.len(), expected.len());
    let mismatch = actual
        .iter()
        .zip(expected)
        .position(|(a, b)| a.to_bits() != b.to_bits());
    assert!(
        mismatch.is_none(),
        "{mode}: first differing output {mismatch:?}"
    );
}

#[test]
#[ignore = "explicit large physical head parity/timing; not model or release qualification"]
fn physical_b3_head_encoded_stream_parity_and_timing() {
    const ROWS: usize = 248320;
    const COLS: usize = 5120;
    let seed = std::env::var("TRITIUM_HEAD_SEED")
        .ok()
        .map(|v| v.parse::<u64>().unwrap());
    let source = EncodedHead::new(ROWS, COLS, 3);
    let source = seed.map_or(source, |seed| {
        EncodedHead::new(ROWS, COLS, 3).counter_mixed(seed)
    });
    let TensorMetadata::Additive(metadata) = source.metadata() else {
        unreachable!()
    };
    let plan = Plan::from_metadata(metadata).unwrap();
    assert_eq!(source.encoded_bytes(), 822_435_840);
    assert_eq!(plan.bytes, 822_435_900);
    let production_groups_selected = plan.b3_groups_eligible();
    let mut digest = blake3::Hasher::new();
    let mut visited = 0u64;
    let mut maximum_window = 0;
    source
        .visit(&mut |chunk| {
            visited += chunk.len() as u64;
            maximum_window = maximum_window.max(chunk.len());
            digest.update(chunk);
            Ok(())
        })
        .unwrap();
    assert_eq!(visited, source.encoded_bytes());
    assert_eq!(maximum_window, TENSOR_STREAM_CHUNK_BYTES);
    let backend = CudaBackend::new(0).expect("requires admitted physical CUDA");
    let tensor = backend.upload_tensor_stream(&source).unwrap();
    assert_eq!(tensor.len_bytes(), plan.bytes);
    let packed = tensor
        .as_any()
        .downcast_ref::<CudaTensor>()
        .unwrap()
        .packed
        .as_ref()
        .unwrap();
    assert_eq!(packed.b3_groups_eligible, production_groups_selected);
    assert!(!packed.b3_rows_eligible);
    let function = function(&backend);
    let host = (0..COLS)
        .map(|i| ((i % 257) as f32 - 128.) / 97.)
        .collect::<Vec<_>>();
    let input = backend.stream.clone_htod(&host).unwrap();
    let mut output = backend.stream.alloc_zeros::<f32>(ROWS).unwrap();
    launch_kernel::<false>(&backend, packed, &input, &mut output, 1, ROWS, false);
    let expected = backend.stream.clone_dtoh(&output).unwrap();
    assert!(expected.iter().all(|v| v.is_finite()));

    // Independent CPU arithmetic on four selected original rows. This never
    // constructs the full head's decoded coefficient or scale master.
    let selected = [0usize, 1, 127, ROWS - 1];
    let layout = AdditiveLayout {
        rows: selected.len() as u64,
        ..source.layout
    };
    let source_ref = &source;
    let trits = (0..3)
        .flat_map(|plane| {
            selected.into_iter().flat_map(move |row| {
                (0..COLS).map(move |col| {
                    let index = row * COLS + col;
                    let code = source_ref.code_at(plane, index / 5) as usize;
                    let digit = (code / 3usize.pow((index % 5) as u32)) % 3;
                    Trit::from_i8(digit as i8 - 1).unwrap()
                })
            })
        })
        .collect();
    // COLS/group=40 is divisible by the five-scale cycle, so selected rows
    // retain the original per-plane/per-row scale phase.
    let scales = (0..selected.len() * 40 * 3)
        .map(|i| f16::from_bits(SCALE_BITS[i % 5]).to_f32())
        .collect();
    let oracle = AdditiveTensor::new(layout, 3, trits, scales).unwrap();
    let mut cpu = vec![0.; selected.len()];
    reference_ternary_matmul(
        &host,
        &oracle.view().bind_matmul(),
        1,
        &mut vec![0.; COLS],
        &mut cpu,
    )
    .unwrap();
    for (index, row) in selected.into_iter().enumerate() {
        assert_eq!(
            cpu[index].to_bits(),
            expected[row].to_bits(),
            "CPU row {row}"
        );
    }
    let mut public = vec![0.; ROWS];
    backend
        .matmul(TensorMatmul {
            act: &host,
            tensor: &*tensor,
            batch: 1,
            transformed_act: &mut vec![0.; COLS],
            out: &mut public,
        })
        .unwrap();
    assert_bits(&public, &expected, "public encoded-stream matmul");
    let device_tensor = tensor.as_any().downcast_ref::<CudaTensor>().unwrap();
    let ones = backend.stream.clone_htod(&[1.0f32]).unwrap();
    device_tensor
        .project_device(&backend, &input, None, &ones, &mut output, 1)
        .unwrap();
    assert_bits(
        &backend.stream.clone_dtoh(&output).unwrap(),
        &expected,
        "resident projection",
    );

    // M>1 must keep the scalar fallback despite the measured M=1 family.
    let mut host2 = host.clone();
    host2.extend(host.iter().map(|v| -*v * 0.5));
    let input2 = backend.stream.clone_htod(&host2).unwrap();
    let mut output2 = backend.stream.alloc_zeros::<f32>(2 * ROWS).unwrap();
    launch_kernel::<false>(&backend, packed, &input2, &mut output2, 2, 2 * ROWS, false);
    let expected2 = backend.stream.clone_dtoh(&output2).unwrap();
    assert!(expected2.iter().all(|v| v.is_finite()));
    device_tensor
        .project_device(&backend, &input2, None, &ones, &mut output2, 2)
        .unwrap();
    assert_bits(
        &backend.stream.clone_dtoh(&output2).unwrap(),
        &expected2,
        "resident M2 fallback",
    );
    let mut public2 = vec![0.; 2 * ROWS];
    backend
        .matmul(TensorMatmul {
            act: &host2,
            tensor: &*tensor,
            batch: 2,
            transformed_act: &mut vec![0.; 2 * COLS],
            out: &mut public2,
        })
        .unwrap();
    assert_bits(&public2, &expected2, "public M2 fallback");

    let mut samples: [Vec<Duration>; 2] = core::array::from_fn(|_| Vec::new());
    for round in 0..10 {
        for offset in 0..2 {
            let mode = (round + offset) % 2;
            backend.stream.synchronize().unwrap();
            let started = Instant::now();
            if mode == 0 {
                launch_kernel::<false>(&backend, packed, &input, &mut output, 1, ROWS, false);
            } else {
                launch(
                    &backend,
                    &function,
                    packed,
                    &input,
                    &mut output,
                    ROWS as u32,
                    1,
                );
            }
            backend.stream.synchronize().unwrap();
            if round >= 2 {
                samples[mode].push(started.elapsed());
            }
            let actual = backend.stream.clone_dtoh(&output).unwrap();
            assert_bits(
                &actual,
                &expected,
                if mode == 0 { "scalar" } else { "groups128" },
            );
        }
    }
    eprintln!(
        "B3_HEAD_SAMPLES {}",
        serde_json::json!({
            "qualified":false, "rows":ROWS, "cols":COLS, "planes":3, "batch":1,
        "fixture":if seed.is_some() { "canonical_b3_counter_mix_f16_cycle5_v1" } else { "canonical_b3_cycle73_plane19_f16_cycle5_v1" },
        "fixture_seed":seed,
            "canonical_payload_bytes":visited, "fixture_blake3":digest.finalize().to_hex().to_string(),
            "maximum_source_window":maximum_window, "native_payload_bytes":plan.bytes,
            "physical_device":backend.physical_device_id(), "cpu_reference_rows":selected,
            "full_scalar_and_public_parity":true, "dispatch_changed":production_groups_selected,
            "resident_projection_parity":true, "batch2_fallback_parity":true,
            "scalar":timing_series(&samples[0]), "groups128":timing_series(&samples[1]),
            "scope":"Private synthetic head, paired warm host launch+synchronize samples on shared hardware. No actual-model, CUDA-event-only duration, physical-memory or independent performance/release qualification."
        })
    );
}
