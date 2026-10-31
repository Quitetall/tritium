//! Bounded read-only traversal and private-output lifetime, not replay admission.
use super::*;
fn raw_big() -> AdditiveTensor {
    let w = big();
    AdditiveTensor::new(
        AdditiveLayout {
            transport: Transport::Raw,
            ..w.layout()
        },
        w.plane_count(),
        w.trits().to_vec(),
        w.scales().to_vec(),
    )
    .unwrap()
}

#[test]
fn visitor_failures_wrong_originals_io_drift_and_unwind_drop_outputs_without_state_change() {
    let _serial = SERIAL.lock().unwrap();
    let p = plan(3);
    let weights: Vec<_> = (0..3)
        .map(|_| tensor(0, PlaneCodec::D2, Basis::Identity, 0, 2))
        .collect();
    let input = Shared::new(Vec::new());
    let mut current = Paged::initialize(p, limits(), parents(&weights), input.clone()).unwrap();
    let before = input.contents();
    for mode in 0..7 {
        let drops = Rc::new(Cell::new(0));
        let result = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
            current.visit_weights(
                |i, _| {
                    if mode == 0 && i == 2 {
                        return Err("original provider refusal");
                    }
                    let mut w = weights[i].clone();
                    if mode == 6 && i == 2 {
                        let mut s = w.scales().to_vec();
                        s[0] = 0.25;
                        w.replace_scales(s).unwrap();
                    }
                    Ok(w)
                },
                |i, _, _| {
                    if mode == 1 && i == 2 {
                        return Err("original visitor refusal");
                    }
                    if mode == 2 && i == 0 {
                        input.fail_read.set(true);
                    }
                    if mode == 3 && i == 2 {
                        input.fail_seek.set(true);
                    }
                    if mode == 4 && i == 2 {
                        let n = input.contents().len();
                        input.inner.borrow_mut().get_mut()[n - 1] ^= 1;
                    }
                    assert!(!(mode == 5 && i == 2), "injected visitor unwind");
                    Ok(Resource(drops.clone()))
                },
            )
        }));
        input.fail_read.set(false);
        input.fail_seek.set(false);
        if mode == 4 {
            let n = before.len();
            input.inner.borrow_mut().get_mut()[n - 1] ^= 1;
        }
        if mode == 5 {
            assert!(result.is_err());
        } else {
            let error = match result.unwrap() {
                Ok(_) => panic!("mode{mode} must reject"),
                Err(e) => e,
            };
            match mode {
                0 => assert!(matches!(
                    error,
                    NativePvPagedError::Provider("original provider refusal")
                )),
                1 => assert!(matches!(
                    error,
                    NativePvPagedError::Provider("original visitor refusal")
                )),
                2 | 3 => assert!(matches!(error, NativePvPagedError::Io(_))),
                _ => assert!(matches!(error, NativePvPagedError::State(_))),
            }
        }
        assert_eq!(
            drops.get(),
            match mode {
                0 | 1 | 5 | 6 => 2,
                2 => 1,
                _ => 3,
            }
        );
        assert_eq!(input.contents(), before);
        assert_eq!(current.completed_updates(), 0);
    }
    assert_eq!(
        current
            .visit_weights(parents(&weights), |_, _, _| Ok(()))
            .unwrap()
            .len(),
        3
    );
}

#[test]
fn file_backed_visit_and_canonical_encoding_peak_at_one_owner_not_checkpoint_size() {
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let mut current = Paged::initialize(
        plan(8),
        limits(),
        |_, _| Ok::<_, &str>(raw_big()),
        files.create("state.tnpv"),
    )
    .unwrap();
    let payload = raw_big().encode_payload().unwrap();
    let encoded_bytes = payload.len();
    let expected = *blake3::hash(&payload).as_bytes();
    drop(payload);
    let checkpoint_bytes = current.size_ledger().logical.checkpoint_bytes;
    let (result, peak, retained) = measured(|| {
        current.visit_weights(
            |_, _| Ok::<_, &str>(raw_big()),
            |_, _, w| {
                let encoded = w.encode_payload().unwrap();
                assert_eq!(encoded.len(), encoded_bytes);
                Ok(*blake3::hash(&encoded).as_bytes())
            },
        )
    });
    let hashes = result.unwrap();
    assert_eq!(hashes, vec![expected; 8]);
    assert!(
        peak as u64 <= current.size_ledger().max_owner_payload_bytes + encoded_bytes as u64 + 65536
    );
    assert!((peak as u64) < checkpoint_bytes);
    assert!(retained < 65536);
    eprintln!(
        "PAGED_VISIT owners=8 checkpoint_bytes={checkpoint_bytes} encoded_owner_bytes={encoded_bytes} peak_live_allocator_requests={peak} retained_allocator_requests={retained}"
    );
}
