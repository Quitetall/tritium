use std::{env, fs, path::PathBuf};

use serde_json::Value;

const DISPATCHES: &[&str] = &[
    "linear_input_64",
    "linear_output_64",
    "linear_parameter_64",
    "linear_primary_input_64",
    "linear_rows_64",
    "optimizer_blocks_256",
    "packed_words_64",
    "rope_pairs_64",
    "single",
];
const REPEATS: &[&str] = &["once", "per_output"];

fn quoted(value: &str) -> String {
    format!("{value:?}")
}

fn string<'a>(object: &'a Value, field: &str) -> &'a str {
    object[field]
        .as_str()
        .unwrap_or_else(|| panic!("dispatch catalog {field} must be a string"))
}

fn identifier(value: &str) -> bool {
    let mut chars = value.chars();
    chars
        .next()
        .is_some_and(|first| first == '_' || first.is_ascii_alphabetic())
        && chars.all(|ch| ch == '_' || ch.is_ascii_alphanumeric())
}

fn exact_fields(value: &Value, expected: &[&str], context: &str) {
    let object = value
        .as_object()
        .unwrap_or_else(|| panic!("{context} must be an object"));
    assert_eq!(object.len(), expected.len(), "unknown {context} field");
    assert!(
        expected.iter().all(|field| object.contains_key(*field)),
        "missing {context} field"
    );
}

fn main() {
    tritium_build_info::emit_source_identity();
    let manifest_dir = PathBuf::from(env::var("CARGO_MANIFEST_DIR").expect("manifest dir"));
    let catalog_path = manifest_dir.join("data/training/v1/webgpu-dispatch-v1.json");
    let manifest_path = manifest_dir.join("data/training/v1/manifest.json");
    println!("cargo:rerun-if-changed={}", catalog_path.display());
    println!("cargo:rerun-if-changed={}", manifest_path.display());
    let bytes = fs::read(&catalog_path).expect("read WebGPU dispatch catalog");
    let value: Value = serde_json::from_slice(&bytes).expect("parse dispatch catalog");
    exact_fields(&value, &["schema_id", "schema_version", "forms"], "catalog");
    assert_eq!(value["schema_id"], "tritium.webgpu_dispatch_catalog");
    assert_eq!(value["schema_version"], 1);
    let forms = value["forms"].as_array().expect("dispatch forms array");
    assert_eq!(forms.len(), 57, "frozen WebGPU dispatch form count");

    let manifest: Value =
        serde_json::from_slice(&fs::read(&manifest_path).expect("read portable training manifest"))
            .expect("parse portable training manifest");
    let mut expected_forms = std::collections::BTreeSet::new();
    for operation in manifest["operations"]
        .as_array()
        .expect("manifest operations")
    {
        let id = string(operation, "id");
        let category = string(operation, "category");
        if category == "lifecycle" {
            continue;
        }
        if category == "optimizer" {
            expected_forms.insert((id.to_owned(), "step".to_owned()));
        } else {
            expected_forms.insert((id.to_owned(), "forward".to_owned()));
            if operation["vjp"] == "first_order" {
                expected_forms.insert((id.to_owned(), "vjp".to_owned()));
            }
        }
    }

    let mut generated = String::from(
        "pub(crate) static PORTABLE_DISPATCH_FORMS_V1: &[PortableDispatchFormV1] = &[\n",
    );
    let mut module_ids = std::collections::BTreeSet::new();
    let mut actual_forms = std::collections::BTreeSet::new();
    for form in forms {
        exact_fields(form, &["operation", "execution", "stages"], "form");
        let operation = string(form, "operation");
        let execution = string(form, "execution");
        assert!(
            actual_forms.insert((operation.to_owned(), execution.to_owned())),
            "duplicate WebGPU dispatch form {operation}|{execution}"
        );
        generated.push_str(&format!(
            "PortableDispatchFormV1 {{ operation: {}, execution: {}, stages: &[",
            quoted(operation),
            quoted(execution),
        ));
        let stages = form["stages"].as_array().expect("stages must be an array");
        assert!(!stages.is_empty(), "dispatch form stages must not be empty");
        for stage in stages {
            exact_fields(
                stage,
                &["moduleId", "entryPoint", "selector", "dispatch", "repeat"],
                "stage",
            );
            let module_id = string(stage, "moduleId");
            assert!(identifier(module_id), "invalid WebGPU module identifier");
            assert!(
                manifest_dir.join(format!("src/{module_id}.wgsl")).is_file(),
                "missing WebGPU module {module_id}"
            );
            module_ids.insert(module_id.to_owned());
            let entry_point = string(stage, "entryPoint");
            assert!(
                identifier(entry_point),
                "invalid WebGPU entry-point identifier"
            );
            let selector = match &stage["selector"] {
                Value::Null => "None".to_owned(),
                Value::Number(number) => {
                    let value = number
                        .as_u64()
                        .filter(|value| u32::try_from(*value).is_ok())
                        .expect("dispatch selector must be null or a u32");
                    format!("Some({value})")
                }
                _ => panic!("dispatch selector must be null or a u32"),
            };
            assert_eq!(
                module_id == "pointwise",
                selector != "None",
                "only pointwise stages carry selectors"
            );
            let dispatch = string(stage, "dispatch");
            assert!(DISPATCHES.contains(&dispatch), "unknown dispatch geometry");
            let repeat = string(stage, "repeat");
            assert!(REPEATS.contains(&repeat), "unknown dispatch repeat policy");
            generated.push_str(&format!(
                "PortableDispatchStageV1 {{ module_id: {}, entry_point: {}, selector: {}, dispatch: {}, repeat: {} }},",
                quoted(module_id),
                quoted(entry_point),
                selector,
                quoted(dispatch),
                quoted(repeat),
            ));
        }
        generated.push_str("] },\n");
    }
    assert_eq!(
        actual_forms, expected_forms,
        "WebGPU dispatch forms drifted from manifest"
    );
    generated.push_str("];\n\n");
    generated.push_str("pub(crate) fn portable_shader_source_v1(module_id: &str) -> Option<&'static str> { match module_id {\n");
    for module_id in module_ids {
        generated.push_str(&format!(
            "{} => Some(include_str!(concat!(env!(\"CARGO_MANIFEST_DIR\"), \"/src/{}.wgsl\"))),\n",
            quoted(&module_id),
            module_id,
        ));
    }
    generated.push_str("_ => None, } }\n");
    let output =
        PathBuf::from(env::var("OUT_DIR").expect("out dir")).join("portable_dispatch_catalog.rs");
    fs::write(output, generated).expect("write generated WebGPU dispatch catalog");
}
