//! Actual installed Python conversion -> Rust P/V publication -> installed receiver.
use std::{
    fs,
    io::Cursor,
    path::{Path, PathBuf},
    process::{Command, Stdio},
    time::{Duration, Instant},
};
use tritium_core::reference_embed;
use tritium_evidence::*;
use tritium_format::{AdditiveTensor, TritPackageReadLimits, trit_model::TritModelReader};
use tritium_quantize::engine::DirectoryTensorSink;
use tritium_schema::*;
use tritium_train::pv_tuning::*;

struct Scratch(PathBuf);
impl Drop for Scratch {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn run(mut command: Command) {
    let mut child = command.stdin(Stdio::null()).spawn().unwrap();
    let start = Instant::now();
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "installed Python P/V probe failed");
            break;
        }
        if start.elapsed() > Duration::from_secs(240) {
            child.kill().unwrap();
            child.wait().unwrap();
            panic!("Python P/V probe exceeded240seconds");
        }
        std::thread::sleep(Duration::from_millis(20));
    }
}
fn receiving() -> TritPackageReadLimits {
    TritPackageReadLimits {
        max_source_bytes: 16_000_000,
        max_manifest_bytes: 2_000_000,
        max_directory_bytes: 2_000_000,
    }
}
fn child_limits() -> NativePvChildLimits {
    NativePvChildLimits {
        max_spool_bytes: 16_000_000,
        max_working_bytes: 64_000_000,
        max_evidence_bytes: 16_000_000,
        max_companion_bytes: 16_000_000,
        receiving: receiving(),
    }
}
fn validation() -> [NativeRefinementValidation; 1] {
    [NativeRefinementValidation {
        content: [30; 32],
        parent_loss: 2.0,
        child_loss: 3.0,
    }]
}
fn recorder(pv: bool) -> EvidenceRecorder {
    EvidenceRecorder::new(
        if pv {
            "actual-python-pv"
        } else {
            "actual-python-scale"
        },
        [
            NATIVE_REFINEMENT_STARTED_REGISTRATION,
            NATIVE_REFINEMENT_CHECKPOINTED_REGISTRATION,
            NATIVE_REFINEMENT_COMPLETED_REGISTRATION,
            if pv {
                NATIVE_PV_REPLAY_OBSERVED_REGISTRATION
            } else {
                NATIVE_SCALE_REPLAY_OBSERVED_REGISTRATION
            },
        ],
    )
    .unwrap()
}
fn publish_pv(
    root: &Path,
    reader: &mut TritModelReader<fs::File>,
    view: &NativeRefinementParentView,
    plan: &NativeRefinementPlan,
    parents: &[AdditiveTensor],
) {
    let mut s = NativePvRefinementSession::new(plan.clone(), parents.to_vec()).unwrap();
    let jl = NativePvJournalLimits {
        max_journal_bytes: 16_000_000,
        max_working_bytes: 64_000_000,
    };
    let mut journal = NativePvJournalWriter::new(Cursor::new(Vec::new()), &s, jl).unwrap();
    let scales: Vec<_> = parents
        .iter()
        .map(|w| vec![0.125; w.scales().len()])
        .collect();
    let weights: Vec<_> = parents
        .iter()
        .map(|w| vec![0.25; w.view().plane_trits(0).unwrap().len()])
        .collect();
    for _ in &plan.training {
        s.step_recorded(
            &scales.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &weights.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut journal,
        )
        .unwrap();
    }
    assert!(
        parents
            .iter()
            .zip(s.weights())
            .any(|(a, b)| a.trits() != b.trits()),
        "fixture must exercise actual code updates"
    );
    let (journal, summary) = journal.finish(&s).unwrap();
    let journal = journal.into_inner();
    let checkpoint = s.checkpoint_bytes().unwrap();
    let replay = replay_native_pv_journal(
        plan.clone(),
        parents.to_vec(),
        Cursor::new(&journal),
        summary.journal(),
        &checkpoint,
        jl,
    )
    .unwrap();
    publish_native_pv_child(
        &s,
        NativePvChildSource {
            parent: reader,
            parent_evidence: view,
            replay: &replay,
            journal: &mut Cursor::new(&journal),
            machine: "synthetic-python-corpus",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(root.join("child")),
        child_limits(),
        recorder(true),
    )
    .unwrap();
    let mut expected = Vec::new();
    for (name, weight) in plan.owners.iter().zip(s.weights()) {
        let rows = weight.layout().rows as usize;
        let cols = weight.layout().cols as usize;
        assert!(
            rows * cols <= 250_000,
            "only bounded tiny test expectations may be materialized"
        );
        let mut values = vec![0.0; rows * cols];
        reference_embed(
            &weight.view().bind_gather(),
            &(0..rows).collect::<Vec<_>>(),
            &mut values,
        )
        .unwrap();
        expected.push(serde_json::json!({"name":name,"rows":rows,"cols":cols,"values":values}));
    }
    fs::write(
        root.join("expected.json"),
        serde_json::to_vec(&expected).unwrap(),
    )
    .unwrap();
}
fn publish_scale(
    root: &Path,
    reader: &mut TritModelReader<fs::File>,
    view: &NativeRefinementParentView,
    plan: &NativeRefinementPlan,
    parents: Vec<AdditiveTensor>,
) {
    let mut plan = plan.clone();
    let NativeRecoveryRecipe::AlternatingPv {
        continuous_optimizer,
        ..
    } = plan.recipe
    else {
        panic!("PV fixture")
    };
    plan.recipe = NativeRecoveryRecipe::ScaleOnly {
        optimizer: continuous_optimizer,
    };
    plan.child_level = "scale-child".into();
    let mut s = NativeScaleRefinementSession::new(plan.clone(), parents.clone()).unwrap();
    let jl = NativeScaleJournalLimits {
        max_journal_bytes: 16_000_000,
        max_working_bytes: 64_000_000,
    };
    let mut journal = NativeScaleJournalWriter::new(Cursor::new(Vec::new()), &s, jl).unwrap();
    let gradients: Vec<_> = parents
        .iter()
        .map(|w| vec![0.125; w.scales().len()])
        .collect();
    for _ in &plan.training {
        s.step_recorded(
            &gradients.iter().map(Vec::as_slice).collect::<Vec<_>>(),
            &mut journal,
        )
        .unwrap();
    }
    let (journal, summary) = journal.finish(&s).unwrap();
    let journal = journal.into_inner();
    let checkpoint = s.checkpoint_bytes().unwrap();
    let replay = replay_native_scale_journal(
        plan,
        parents,
        Cursor::new(&journal),
        summary.journal(),
        &checkpoint,
        jl,
    )
    .unwrap();
    publish_native_scale_child(
        &s,
        NativeScaleChildSource {
            parent: reader,
            parent_evidence: view,
            replay: &replay,
            journal: &mut Cursor::new(journal),
            machine: "synthetic-python-corpus",
            validation: &validation(),
        },
        &mut Cursor::new(Vec::new()),
        &mut DirectoryTensorSink::new(root.join("scale")),
        child_limits(),
        recorder(false),
    )
    .unwrap();
}

#[test]
#[ignore = "requires this checkout's installed wheel plus pinned Torch/Transformers/pytest; explicit corpus probe"]
fn actual_pv_publisher_installed_python_descriptor_round_trip() {
    let python = std::env::var_os("TRITIUM_PV_TEST_PYTHON")
        .expect("explicit installed candidate interpreter required");
    let repo = Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("../..")
        .canonicalize()
        .unwrap();
    let receiver = repo.join("crates/tritium-py/tests/test_native_pv_loading.py");
    let root =
        Scratch(std::env::temp_dir().join(format!("tritium-pv-python-{}", std::process::id())));
    fs::create_dir(&root.0).unwrap();
    let mut prepare = Command::new(&python);
    prepare.args(["-I","-B","-c","import runpy,sys;runpy.run_path(sys.argv[1],run_name='test_native_pv_loading')['_prepare'](sys.argv[2])"]).arg(&receiver).arg(&root.0).current_dir(&repo);
    run(prepare);
    let mut cases = fs::read_dir(&root.0)
        .unwrap()
        .map(|e| e.unwrap().path())
        .collect::<Vec<_>>();
    cases.sort();
    assert_eq!(cases.len(), 9);
    for case in cases {
        let bytes = fs::read(case.join("plan.json")).unwrap();
        assert!(bytes.len() < 1_000_000);
        let plan: NativeRefinementPlan = serde_json::from_slice(&bytes).unwrap();
        let text = fs::read_to_string(case.join("parent/evidence.jsonl")).unwrap();
        assert!(text.len() < 16_000_000);
        let log = verify_jsonl(&text).unwrap();
        let mut reader = TritModelReader::open(
            fs::File::open(case.join("parent/model.trit")).unwrap(),
            receiving(),
        )
        .unwrap();
        let receipt = QuantizationReceiptView::fold(&log).unwrap().remove(0);
        let view = NativeRefinementParentView::from_quantization(
            &mut reader,
            &plan.parent.level,
            &receipt,
        )
        .unwrap();
        let parents = decode_native_pv_parents(
            &mut reader,
            &view,
            &plan,
            NativePvParentReadLimits {
                max_working_bytes: 64_000_000,
                extra_payload_bytes: 0,
            },
        )
        .unwrap();
        publish_pv(&case, &mut reader, &view, &plan, &parents);
        publish_scale(&case, &mut reader, &view, &plan, parents);
    }
    let mut tests = Command::new(python);
    tests
        .args([
            "-I",
            "-B",
            "-m",
            "pytest",
            "-q",
            "-ra",
            "-p",
            "no:cacheprovider",
        ])
        .arg(&receiver)
        .env("TRITIUM_PV_FIXTURES", &root.0)
        .env("TRITIUM_TEST_INSTALLED_WHEEL", "1")
        .current_dir(repo);
    run(tests);
}
