//! `tritium inspect <PATH.gguf>`: parse a GGUF container and render a summary.
//!
//! The actual rendering is split from the I/O so it can be unit-tested against an
//! in-memory [`GgufFile`]: [`render_inspect`] takes a parsed file and returns the
//! human-facing report as a `String`, while [`run`] does the file read + parse and
//! prints the result. All failures flow through [`anyhow::Result`] so a missing,
//! short, or corrupt file yields a clean error and a non-zero exit — never a panic.

use std::fmt::Write as _;
use std::path::Path;

use anyhow::Context as _;
use tritium_format::{GgufFile, GgufValue, read_gguf};

/// Map a raw ggml type-id to a short human-readable name.
///
/// Known ternary and common float/int ids get a proper label; anything else is
/// rendered as `type N` so the output stays informative without guessing layout.
fn ggml_type_name(ggml_type: u32) -> String {
    match ggml_type {
        0 => "F32".to_owned(),
        1 => "F16".to_owned(),
        tritium_format::GGML_TYPE_Q2_0 => "Q2_0".to_owned(),
        tritium_format::GGML_TYPE_TQ1_0 => "TQ1_0".to_owned(),
        tritium_format::GGML_TYPE_TQ2_0 => "TQ2_0".to_owned(),
        other => format!("type {other}"),
    }
}

/// Format a tensor's dimensions as `[a, b, c]` (ggml order, fastest-varying first).
fn format_dims(dims: &[u64]) -> String {
    let mut s = String::from("[");
    for (i, d) in dims.iter().enumerate() {
        if i > 0 {
            s.push_str(", ");
        }
        // `write!` to a String is infallible; ignore the Result.
        let _ = write!(s, "{d}");
    }
    s.push(']');
    s
}

/// Render the inspection report for an already-parsed [`GgufFile`].
///
/// The output lists the GGUF version, metadata key count, the
/// `general.architecture` value when present, the resolved tensor-data alignment,
/// and a table of every tensor (name, dims, type name, byte offset, byte size).
/// Pure and deterministic — the entry point for unit tests.
#[must_use]
pub(crate) fn render_inspect(file: &GgufFile) -> String {
    let mut out = String::new();

    let _ = writeln!(out, "GGUF version:   {}", file.version);
    let _ = writeln!(out, "Metadata count: {}", file.metadata.len());

    match file
        .get_metadata("general.architecture")
        .and_then(GgufValue::as_str)
    {
        Some(arch) => {
            let _ = writeln!(out, "Architecture:   {arch}");
        }
        None => {
            let _ = writeln!(out, "Architecture:   (unset)");
        }
    }

    let _ = writeln!(out, "Alignment:      {}", file.alignment());
    let _ = writeln!(out, "Tensor count:   {}", file.tensors.len());

    // Column widths: size the name column to the widest tensor name (with a
    // sensible floor) so the table stays aligned without wrapping a wide name.
    let name_w = file
        .tensors
        .iter()
        .map(|t| t.name.len())
        .max()
        .unwrap_or(0)
        .max("NAME".len());
    let dims_w = file
        .tensors
        .iter()
        .map(|t| format_dims(&t.dims).len())
        .max()
        .unwrap_or(0)
        .max("DIMS".len());
    let type_w = file
        .tensors
        .iter()
        .map(|t| ggml_type_name(t.ggml_type).len())
        .max()
        .unwrap_or(0)
        .max("TYPE".len());

    let _ = writeln!(out, "\nTensors:");
    let _ = writeln!(
        out,
        "  {:<name_w$}  {:<dims_w$}  {:<type_w$}  {:>12}  {:>12}",
        "NAME", "DIMS", "TYPE", "OFFSET", "N_BYTES",
    );
    for t in &file.tensors {
        let _ = writeln!(
            out,
            "  {:<name_w$}  {:<dims_w$}  {:<type_w$}  {:>12}  {:>12}",
            t.name,
            format_dims(&t.dims),
            ggml_type_name(t.ggml_type),
            t.offset,
            t.n_bytes,
        );
    }

    out
}

/// Read `path`, parse it as GGUF, and print the inspection report to stdout.
///
/// # Errors
/// Returns an [`anyhow::Error`] (with file-path context) if the file cannot be
/// read or fails to parse as a GGUF container. The caller maps this to a non-zero
/// exit; nothing here panics on malformed or truncated input.
pub(crate) fn run(path: &Path) -> anyhow::Result<()> {
    let bytes = std::fs::read(path)
        .with_context(|| format!("failed to read GGUF file `{}`", path.display()))?;
    let file = read_gguf(&bytes)
        .with_context(|| format!("failed to parse `{}` as GGUF", path.display()))?;
    print!("{}", render_inspect(&file));
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::BTreeMap;
    use tritium_format::TensorInfo;

    /// Build an in-memory `GgufFile` for rendering tests (no byte parsing needed).
    fn sample_file() -> GgufFile {
        let mut metadata = BTreeMap::new();
        metadata.insert(
            "general.architecture".to_owned(),
            GgufValue::String("llama".to_owned()),
        );
        metadata.insert("general.alignment".to_owned(), GgufValue::U32(32));

        GgufFile::new(
            3,
            metadata,
            vec![
                TensorInfo::new(
                    "blk.0.weight".to_owned(),
                    vec![256, 4],
                    tritium_format::GGML_TYPE_TQ2_0,
                    0,
                    264,
                ),
                TensorInfo::new(
                    "blk.0.qh".to_owned(),
                    vec![256],
                    tritium_format::GGML_TYPE_TQ1_0,
                    264,
                    54,
                ),
                TensorInfo::new("output_norm.bias".to_owned(), vec![4], 0, 320, 16),
                TensorInfo::new("token_embd.weight".to_owned(), vec![8], 1, 336, 16),
                TensorInfo::new("mystery".to_owned(), vec![1], 99, 352, 0),
            ],
            0,
        )
    }

    #[test]
    fn type_names_map_correctly() {
        assert_eq!(ggml_type_name(0), "F32");
        assert_eq!(ggml_type_name(1), "F16");
        assert_eq!(ggml_type_name(34), "TQ1_0");
        assert_eq!(ggml_type_name(35), "TQ2_0");
        assert_eq!(ggml_type_name(42), "Q2_0");
        assert_eq!(ggml_type_name(99), "type 99");
    }

    #[test]
    fn dims_format_is_bracketed_csv() {
        assert_eq!(format_dims(&[]), "[]");
        assert_eq!(format_dims(&[256]), "[256]");
        assert_eq!(format_dims(&[256, 4]), "[256, 4]");
    }

    #[test]
    fn render_contains_header_fields() {
        let report = render_inspect(&sample_file());
        assert!(report.contains("GGUF version:   3"), "{report}");
        assert!(report.contains("Architecture:   llama"), "{report}");
        assert!(report.contains("Alignment:      32"), "{report}");
        assert!(report.contains("Metadata count: 2"), "{report}");
        assert!(report.contains("Tensor count:   5"), "{report}");
    }

    #[test]
    fn render_contains_tensor_names_and_types() {
        let report = render_inspect(&sample_file());
        // Names.
        assert!(report.contains("blk.0.weight"), "{report}");
        assert!(report.contains("blk.0.qh"), "{report}");
        assert!(report.contains("output_norm.bias"), "{report}");
        assert!(report.contains("token_embd.weight"), "{report}");
        assert!(report.contains("mystery"), "{report}");
        // Type names mapped from ggml ids.
        assert!(report.contains("TQ2_0"), "{report}");
        assert!(report.contains("TQ1_0"), "{report}");
        assert!(report.contains("F32"), "{report}");
        assert!(report.contains("F16"), "{report}");
        assert!(report.contains("type 99"), "{report}");
        // Dims rendered.
        assert!(report.contains("[256, 4]"), "{report}");
    }

    #[test]
    fn render_handles_missing_architecture() {
        let mut file = sample_file();
        file.metadata.remove("general.architecture");
        let report = render_inspect(&file);
        assert!(report.contains("Architecture:   (unset)"), "{report}");
    }

    #[test]
    fn run_on_missing_file_errors_cleanly() {
        let err = run(Path::new("/nonexistent/definitely/not/here.gguf"))
            .expect_err("missing file must error");
        // Error is descriptive and mentions the path; no panic occurred.
        let msg = format!("{err:#}");
        assert!(msg.contains("failed to read GGUF file"), "{msg}");
    }

    #[test]
    fn run_on_corrupt_file_errors_cleanly() {
        // A real, readable file that is not valid GGUF must parse-error, not panic.
        let mut tmp = std::env::temp_dir();
        tmp.push(format!("tritium-cli-corrupt-{}.gguf", std::process::id()));
        std::fs::write(&tmp, b"not a gguf file at all").expect("write temp");
        let err = run(&tmp).expect_err("corrupt file must error");
        let _ = std::fs::remove_file(&tmp);
        let msg = format!("{err:#}");
        assert!(msg.contains("failed to parse"), "{msg}");
    }
}
