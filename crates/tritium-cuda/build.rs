// Compiles the CUDA kernel to PTX via nvcc, but only when the `cuda` feature is
// enabled (Cargo sets CARGO_FEATURE_CUDA). With the feature off this is a no-op,
// so cpu-only builds never need a CUDA toolkit. Implementation in progress.
fn main() {
    if std::env::var_os("CARGO_FEATURE_CUDA").is_none() {
        return;
    }
    // v0.10 Wave C: invoke nvcc here to emit tq2_0_add.ptx into OUT_DIR.
}
