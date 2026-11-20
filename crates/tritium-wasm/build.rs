fn main() {
    println!("cargo:rerun-if-env-changed=TRITIUM_WASM_PHYSICAL_DEVICE");
    tritium_build_info::emit_source_identity();
}
