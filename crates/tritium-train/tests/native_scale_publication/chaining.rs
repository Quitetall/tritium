//! Exact immediate-edge admission/replay; no transitive-history qualification.
use super::*;

#[test]
fn actual_refined_parent_starts_fresh_rounds_and_replays_each_immediate_edge() {
    let root = Scratch::new("chain");
    let mut original = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut parent = publish(
        &mut original,
        &root.0.join("first"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("round-1"),
    )
    .unwrap();
    let norm = tensor_bytes(&mut parent.model, "refined", "norm");
    let frozen = tensor_bytes(&mut parent.model, "refined", "frozen.weight");
    let names: Vec<_> = parent
        .model
        .manifest()
        .assets
        .iter()
        .map(|v| v.name.clone())
        .collect();
    let assets: Vec<_> = names
        .iter()
        .map(|n| asset_bytes(&mut parent.model, n))
        .collect();
    for round in 2..=3 {
        let record =
            NativeScaleChildRecordView::from_child(parent.evidence.log(), &mut parent.model)
                .unwrap();
        record.verify_child(&mut parent.model).unwrap();
        assert!(
            NativeRefinementParentView::from_refined_record(&mut parent.model, "full", &record)
                .is_err()
        );
        let view =
            NativeRefinementParentView::from_refined_record(&mut parent.model, "refined", &record)
                .unwrap();
        assert_eq!(
            view.binding().package.as_bytes(),
            parent.model.package_id().as_bytes()
        );
        assert_eq!(view.binding().evidence, *record.reference());
        assert_eq!(view.binding().source, original.plan.parent.source);
        let mut plan = original.plan.clone();
        plan.parent = view.binding().clone();
        plan.child_level = "refined".into();
        let decoded = decode_native_scale_parents(
            &mut parent.model,
            &view,
            &plan,
            NativeScaleParentReadLimits {
                max_working_bytes: 2_000_000,
                extra_payload_bytes: 0,
            },
        )
        .unwrap();
        let fresh = NativeScaleRefinementSession::new(plan.clone(), decoded.clone()).unwrap();
        assert_eq!(fresh.completed_updates(), 0);
        assert!(
            NativeScaleRefinementSession::resume(
                plan.clone(),
                decoded.clone(),
                &original.checkpoint
            )
            .is_err()
        );
        let (session, journal, checkpoint, replay) = trained(&plan, &decoded, 0.125);
        let previous = parent.model.package_id();
        let mut child = publish_native_scale_child(
            &session,
            NativeScaleChildSource {
                parent: &mut parent.model,
                parent_evidence: &view,
                replay: &replay,
                journal: &mut Cursor::new(&journal),
                machine: "synthetic-chain",
                validation: &validation(),
            },
            &mut Cursor::new(Vec::new()),
            &mut DirectoryTensorSink::new(root.0.join(format!("round-{round}"))),
            limits(),
            recorder(&format!("round-{round}")),
        )
        .unwrap();
        assert_ne!(child.model.package_id(), previous);
        assert_eq!(
            child.receipt.refinement().started().plan.parent,
            *view.binding()
        );
        verify_native_scale_child(
            NativeScaleChildVerification {
                parent: &mut parent.model,
                parent_evidence: &view,
                child: &mut child.model,
                log: child.evidence.log(),
                plan: &plan,
                journal: &mut Cursor::new(&journal),
                checkpoint: &checkpoint,
            },
            NativeScaleParentReadLimits {
                max_working_bytes: 2_000_000,
                extra_payload_bytes: 0,
            },
            journal_limits(),
        )
        .unwrap();
        assert!(
            verify_native_scale_child(
                NativeScaleChildVerification {
                    parent: &mut original.parent,
                    parent_evidence: &original.view,
                    child: &mut child.model,
                    log: child.evidence.log(),
                    plan: &plan,
                    journal: &mut Cursor::new(&journal),
                    checkpoint: &checkpoint
                },
                NativeScaleParentReadLimits {
                    max_working_bytes: 2_000_000,
                    extra_payload_bytes: 0
                },
                journal_limits()
            )
            .is_err()
        );
        assert_eq!(tensor_bytes(&mut child.model, "refined", "norm"), norm);
        assert_eq!(
            tensor_bytes(&mut child.model, "refined", "frozen.weight"),
            frozen
        );
        for (name, expected) in names.iter().zip(&assets) {
            assert_eq!(asset_bytes(&mut child.model, name), *expected);
        }
        assert_eq!(*original.bytes.borrow(), original.original);
        parent = child;
    }
}

#[test]
fn standalone_record_cannot_rebind_to_a_different_compatible_model_container() {
    use tritium_format::trit_model::encode_model_manifest;
    use tritium_format::{TritBlob, TritBlobKind, write_trit_package};
    let root = Scratch::new("chain-rebind");
    let mut fixture = Fixture::new(
        ScalePrecision::F32,
        Basis::Identity,
        PlaneCodec::B3,
        "full",
        false,
    );
    let mut child = publish(
        &mut fixture,
        &root.0.join("child"),
        &mut Cursor::new(Vec::new()),
        limits(),
        recorder("child"),
    )
    .unwrap();
    let record =
        NativeScaleChildRecordView::from_child(child.evidence.log(), &mut child.model).unwrap();
    let mut manifest = child.model.manifest().clone();
    manifest.architecture = "different-compatible-architecture".into();
    let mut payloads = Vec::new();
    for meta in &manifest.levels[0].tensors {
        let bytes = tensor_bytes(&mut child.model, "refined", &meta.name);
        let kind = match &meta.storage {
            TensorStorage::Additive { .. } => TritBlobKind::AdditiveTensor,
            TensorStorage::Dense { precision, .. } => TritBlobKind::DenseTensor(match precision {
                ScalePrecision::F32 => tritium_core::DType::F32,
                ScalePrecision::Bf16 => tritium_core::DType::BF16,
                _ => tritium_core::DType::F16,
            }),
            _ => panic!("unsupported fixture"),
        };
        payloads.push((kind, bytes));
    }
    for asset in &manifest.assets {
        payloads.push((
            TritBlobKind::Asset(asset.name.clone()),
            asset_bytes(&mut child.model, &asset.name),
        ));
    }
    let blobs: Vec<_> = payloads
        .iter()
        .map(|(kind, bytes)| TritBlob {
            kind: kind.clone(),
            bytes,
        })
        .collect();
    let bytes = write_trit_package(&encode_model_manifest(&manifest).unwrap(), &blobs).unwrap();
    let mut other = TritModelReader::open(Cursor::new(bytes), receiving()).unwrap();
    // A fresh standalone observation can be byte/record-consistent in its narrow scope.
    NativeScaleChildRecordView::from_child(child.evidence.log(), &mut other).unwrap();
    // The old observation must never silently bind those different actual bytes.
    assert!(record.verify_child(&mut other).is_err());
    assert!(
        NativeRefinementParentView::from_refined_record(&mut other, "refined", &record).is_err()
    );
    record.verify_child(&mut child.model).unwrap();
    let path = root.0.join("child/model.trit");
    let mut damaged = fs::read(&path).unwrap();
    let end = damaged.len() - 1;
    damaged[end] ^= 1;
    fs::write(&path, damaged).unwrap();
    assert!(record.verify_child(&mut child.model).is_err());
}
