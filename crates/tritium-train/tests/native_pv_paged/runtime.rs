//! File-backed optimizer state with real packed consumer adjoints and root versions.
use super::*;
use std::sync::Arc;
use tritium_cpu::CpuBackend;
use tritium_runtime::TensorPayloadBudget;
use tritium_spec::{TensorView, TernaryBackend};
use tritium_train::pv_tuning::*;

#[test]
fn packed_file_controller_reduces_and_stages_one_owner_without_checkpoint_or_window_buffers() {
    let _serial = SERIAL.lock().unwrap();
    rayon::join(|| (), || ());
    let files = Files::new();
    let (mut parent, view, p) = super::publication::original(&files);
    let source_limits = NativeRefinementOwnerReadLimits {
        max_working_bytes: 2 << 20,
        extra_payload_bytes: 0,
    };
    let state =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, source_limits, |source| {
            Paged::initialize(
                p.clone(),
                limits(),
                |i, n| source.load(i, n),
                files.create("runtime-old.tnpv"),
            )
        })
        .unwrap();
    let roots =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, source_limits, |source| {
            let mut roots = Vec::new();
            for (i, n) in p.owners.iter().enumerate() {
                let w = source.load(i, n)?;
                let h: Arc<dyn tritium_spec::DeviceBuffer> = Arc::from(
                    CpuBackend::new()
                        .upload_tensor(TensorView::Additive(w.view()))
                        .unwrap(),
                );
                roots.push((n.clone(), h));
            }
            Ok::<_, NativePvChildError>(roots)
        })
        .unwrap();
    let budget = TensorPayloadBudget::new(8 << 20);
    let mut live =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, source_limits, |source| {
            NativeCpuPvPagedSession::bind(state, roots, budget.clone(), |i, n| source.load(i, n))
        })
        .unwrap();
    let rec = RecordingLimits {
        max_journal_bytes: 128 << 20,
        max_frame_bytes: 128 << 20,
        max_working_bytes: 16384,
    };
    let mut journal = live
        .new_recording(files.create("runtime.tnpj"), rec)
        .unwrap();
    let x = vec![0.125; 260];
    let dy = vec![0.25; 1024];
    let one = [NativePvContribution::Matmul {
        activations: &x,
        output_gradients: &dy,
        batch: 1,
    }];
    let contributions: Vec<&[NativePvContribution<'_>]> =
        p.owners.iter().map(|_| one.as_slice()).collect();
    let before = live.session().checkpoint_digest();
    let bytes = live.session().size_ledger().logical.checkpoint_bytes;
    let frame_bytes = live.session().recording_size_ledger().unwrap().frame_bytes;
    let admitted = budget.used_bytes();
    let point = live.checkpoint(&mut journal).unwrap();
    let rejected =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, source_limits, |source| {
            let ready = live.stage_recorded(
                (
                    files.create("refused-next.tnpv"),
                    files.create("refused-frame.tmp"),
                ),
                |i, n| source.load(i, n),
                |_, _, _, w| {
                    Ok((
                        vec![0.125; w.scales().len()],
                        vec![0.125; w.view().plane_trits(0).unwrap().len()],
                    ))
                },
                rec,
            )?;
            let mut f = OpenOptions::new()
                .read(true)
                .write(true)
                .open(files.0.join("parent.trit"))
                .unwrap();
            f.seek(SeekFrom::End(-1)).unwrap();
            let mut b = [0];
            f.read_exact(&mut b).unwrap();
            f.seek(SeekFrom::End(-1)).unwrap();
            f.write_all(&[b[0] ^ 1]).unwrap();
            Ok::<_, NativeCpuPvPagedError<NativePvChildError>>(ready)
        });
    assert!(matches!(
        rejected,
        Err(NativeRefinementOwnerSourceError::Source(_))
    ));
    let mut f = OpenOptions::new()
        .read(true)
        .write(true)
        .open(files.0.join("parent.trit"))
        .unwrap();
    f.seek(SeekFrom::End(-1)).unwrap();
    let mut b = [0];
    f.read_exact(&mut b).unwrap();
    f.seek(SeekFrom::End(-1)).unwrap();
    f.write_all(&[b[0] ^ 1]).unwrap();
    drop(f);
    assert_eq!(budget.used_bytes(), admitted);
    assert_eq!(live.checkpoint(&mut journal).unwrap(), point);
    std::fs::remove_file(files.0.join("refused-next.tnpv")).unwrap();
    std::fs::remove_file(files.0.join("refused-frame.tmp")).unwrap();
    let (result, peak, retained) = measured(|| {
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, source_limits, |source| {
            live.stage_from_consumers_recorded(
                (
                    files.create("runtime-next.tnpv"),
                    files.create("runtime-frame.tmp"),
                ),
                |i, n| source.load(i, n),
                &contributions,
                NativeCpuPvPagedStepLimits {
                    recording: rec,
                    max_consumer_working_bytes: 8 << 20,
                },
            )
        })
    });
    let ready = result.unwrap();
    assert_eq!(live.session().checkpoint_digest(), before);
    assert_eq!(live.session().completed_updates(), 0);
    assert!(
        (peak as u64) < bytes,
        "runtime buffered checkpoint/window {peak}"
    );
    assert!(
        (peak as u64) < frame_bytes,
        "runtime buffered whole gradient frame {peak}"
    );
    let staged = budget.used_bytes() - admitted;
    assert!(retained as u64 <= staged + 65536);
    let (next, stats) = ready.commit(&mut live, &mut journal).unwrap();
    assert_eq!(stats.completed_updates, 1);
    assert_eq!(live.session().completed_updates(), 0);
    live = next;
    let (_, summary) = live.finish_recording(journal).unwrap();
    let mut expected = live.into_session().into_checkpoint();
    assert_eq!(budget.used_bytes(), 0);
    let (mut replayed, proof) =
        NativeRefinementOwnerSource::with(&mut parent, &view, &p, source_limits, |source| {
            replay_native_pv_paged_journal(
                NativePvPagedJournalReplay {
                    plan: p.clone(),
                    source: File::open(files.0.join("runtime.tnpj")).unwrap(),
                    expected_journal: summary.journal(),
                    checkpoint: &mut expected,
                },
                |i, n| source.load(i, n),
                |i| Ok(files.create(&format!("runtime-audit-{i}.tnpv"))),
                NativePvPagedReplayLimits {
                    max_journal_bytes: 128 << 20,
                    max_working_bytes: 16384,
                    paging: limits(),
                },
            )
        })
        .unwrap();
    proof.verify_paged(&mut replayed).unwrap();
    assert_eq!(proof.summary(), &summary);
    eprintln!(
        "NATIVE_CPU_PAGED_RUNTIME owners=8 checkpoint_bytes={bytes} frame_bytes={frame_bytes} packed_admitted_before={admitted} packed_staged_bytes={staged} peak_live_allocator_requests={peak} retained_allocator_requests={retained}"
    );
}
