//! Synthetic metadata lifecycle only; no model identity or qualification claim.
use serde::{Deserialize, Serialize};
use tritium_evidence::{
    EventRegistration, EvidenceEvent, EvidenceRecorder, VerifiedLog, verify_jsonl_with_registry,
};

#[derive(Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
struct MetadataProbe {
    label: String,
    counts: Vec<u8>,
}
impl EvidenceEvent for MetadataProbe {
    const SCHEMA_ID: &'static str = "tritium.test.synthetic_metadata_memory";
    const VERSION: u16 = 1;
}
const REGISTRATION: EventRegistration = EventRegistration {
    schema: MetadataProbe::SCHEMA_ID,
    version: MetadataProbe::VERSION,
};

fn recorded(count: usize) -> EvidenceRecorder {
    let mut recorder = EvidenceRecorder::new("synthetic-metadata-memory", [REGISTRATION]).unwrap();
    recorder
        .emit(
            "metadata",
            None,
            0,
            &MetadataProbe {
                label: "synthetic-only: é 💡 \\\"".into(),
                counts: vec![1; count],
            },
        )
        .unwrap();
    recorder
}
// Reproduce the old lifecycle through public interfaces, including cold wire
// verification and registry checking. No original log/model data is modified.
fn retained_tree_reference(recorder: EvidenceRecorder) -> (String, VerifiedLog) {
    let jsonl = recorder.to_jsonl().unwrap();
    let log = verify_jsonl_with_registry(&jsonl, [REGISTRATION]).unwrap();
    std::hint::black_box(recorder.events()); // old tree must coexist with parse
    (jsonl, log)
}

fn report_high_water_rss() {
    #[cfg(target_os = "linux")]
    {
        let status = std::fs::read_to_string("/proc/self/status").unwrap();
        let high_water = status
            .lines()
            .find(|line| line.starts_with("VmHWM:"))
            .unwrap();
        println!("scope=synthetic-metadata-only process_memory={high_water}");
    }
}

#[test]
fn sealing_and_borrowed_decode_preserve_wire_root_and_strict_payload_checks() {
    for count in [0, 1, 23, 24, 255, 256] {
        let expected = recorded(count);
        let root = expected.root_digest().unwrap();
        let (wire, log) = retained_tree_reference(expected);
        let seal = recorded(count).seal().unwrap();
        assert_eq!(seal.jsonl(), wire);
        assert_eq!(seal.log(), &log);
        assert_eq!(seal.log().root_digest, root);
        let owned: MetadataProbe = serde_json::from_value(log.events[0].payload.clone()).unwrap();
        let borrowed: MetadataProbe =
            Deserialize::deserialize(&seal.log().events[0].payload).unwrap();
        assert_eq!(borrowed, owned);
        let mut invalid = seal.log().events[0].payload.clone();
        invalid["unexpected"] = true.into();
        assert!(MetadataProbe::deserialize(&invalid).is_err());
        invalid.as_object_mut().unwrap().remove("unexpected");
        if count != 0 {
            invalid["counts"][0] = serde_json::json!(256);
            assert!(MetadataProbe::deserialize(&invalid).is_err());
        }
    }
}

#[test]
fn synthetic_metadata_memory_probe() {
    let count = std::env::var("TRITIUM_SYNTHETIC_METADATA_ITEMS")
        .map(|s| s.parse::<usize>().unwrap())
        .unwrap_or(128);
    assert!(
        count <= 2_097_152,
        "explicit probe is capped at 2 Mi entries"
    );
    let reference = std::env::var("TRITIUM_SYNTHETIC_METADATA_REFERENCE").as_deref() == Ok("1");
    let recorder = recorded(count);
    let root = recorder.root_digest().unwrap();
    let wire_hash = blake3::hash(recorder.to_jsonl().unwrap().as_bytes());
    let (jsonl, log, decoded): (String, VerifiedLog, MetadataProbe) = if reference {
        let (jsonl, log) = retained_tree_reference(recorder);
        let decoded = serde_json::from_value(log.events[0].payload.clone()).unwrap();
        (jsonl, log, decoded)
    } else {
        let seal = recorder.seal().unwrap();
        let decoded: MetadataProbe =
            Deserialize::deserialize(&seal.log().events[0].payload).unwrap();
        // Move through the public owned seal; do not clone its Value tree.
        assert_eq!(seal.log().root_digest, root);
        assert_eq!(blake3::hash(seal.jsonl().as_bytes()), wire_hash);
        assert_eq!(decoded.counts.len(), count);
        assert!(decoded.counts.iter().all(|&n| n == 1));
        println!(
            "scope=synthetic-metadata-only mode=candidate items={count} jsonl_bytes={} root={root} wire_blake3={wire_hash}",
            seal.jsonl().len()
        );
        report_high_water_rss();
        return;
    };
    assert_eq!(log.root_digest, root);
    assert_eq!(blake3::hash(jsonl.as_bytes()), wire_hash);
    assert_eq!(decoded.counts.len(), count);
    assert!(decoded.counts.iter().all(|&n| n == 1));
    println!(
        "scope=synthetic-metadata-only mode=retained-tree-reference items={count} jsonl_bytes={} root={root} wire_blake3={wire_hash}",
        jsonl.len()
    );
    report_high_water_rss();
}
