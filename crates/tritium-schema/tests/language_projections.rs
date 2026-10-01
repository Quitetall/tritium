#![cfg(feature = "schema-gen")]

use std::path::Path;
use std::process::Command;

#[test]
fn generated_language_types_are_current_and_cover_wide_integers_and_tagged_variants() {
    let status = Command::new(env!("CARGO_BIN_EXE_tritium-schema-projections"))
        .arg("--check")
        .status()
        .expect("run the public schema projection check");
    assert!(status.success(), "generated schema projections are stale");

    let root = Path::new(env!("CARGO_MANIFEST_DIR")).join("../../schemas");
    let typescript = std::fs::read_to_string(root.join("typescript/v1.d.ts"))
        .expect("generated TypeScript projection exists");
    let python =
        std::fs::read_to_string(root.join("python/v1.pyi")).expect("generated Python stub exists");

    assert!(typescript.contains("export interface AdditiveLayout"));
    assert!(typescript.contains("readonly rows: number;"));
    assert!(typescript.contains("Number.MAX_SAFE_INTEGER"));
    assert!(typescript.contains("export type Basis ="));
    assert!(typescript.contains("SignedRht:"));
    assert!(python.contains("class AdditiveLayout(TypedDict)"));
    assert!(python.contains("rows: int"));
    assert!(python.contains("seq: int"));
    assert!(python.contains("SignedRht"));
}
