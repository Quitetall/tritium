//! `tritium list-backends`: enumerate every registered [`TernaryBackend`].
//!
//! Discovery is handled by [`tritium_runtime::Registry`], which collects backends
//! self-registered through `linkme`. The CLI crate links `tritium_cpu` (see the
//! `use tritium_cpu as _;` at the crate root) so its registration is present, which
//! is what makes the `cpu` backend appear here. Rendering is split from printing so
//! it can be unit-tested against a list of [`DeviceCaps`].

use std::fmt::Write as _;

use tritium_spec::DeviceCaps;

/// Render the list-backends report for a slice of [`DeviceCaps`].
///
/// Each backend gets its family name, device name, feature flags, and (when known)
/// total memory. Pure and deterministic — the entry point for unit tests.
#[must_use]
pub(crate) fn render_backends(caps: &[DeviceCaps]) -> String {
    let mut out = String::new();

    if caps.is_empty() {
        out.push_str("No backends registered.\n");
        return out;
    }

    let _ = writeln!(out, "Registered backends ({}):", caps.len());
    for c in caps {
        let _ = writeln!(out, "  {} — {}", c.backend, c.device_name);

        let features = if c.features.is_empty() {
            "(none)".to_owned()
        } else {
            c.features.join(", ")
        };
        let _ = writeln!(out, "      features: {features}");

        if c.total_memory_bytes > 0 {
            let _ = writeln!(out, "      memory:   {} bytes", c.total_memory_bytes);
        }
    }

    out
}

/// Initialise the backend registry and print the list of registered backends.
///
/// Never fails: an empty registry is a valid (if unusual) outcome and is reported
/// as such. Registry initialisation may write per-backend diagnostics to stderr.
pub(crate) fn run() {
    let registry = tritium_runtime::Registry::init();
    print!("{}", render_backends(&registry.enumerate()));
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn empty_registry_renders_message() {
        let report = render_backends(&[]);
        assert!(report.contains("No backends registered"), "{report}");
    }

    #[test]
    fn renders_backend_name_device_and_features() {
        let caps = vec![
            DeviceCaps::new("cpu", "x86_64 (avx2)")
                .with_features(vec!["avx2".to_owned(), "fma".to_owned()])
                .with_memory(16 * 1024 * 1024 * 1024),
            DeviceCaps::new("cuda", "NVIDIA RTX 4090"),
        ];
        let report = render_backends(&caps);
        assert!(report.contains("Registered backends (2)"), "{report}");
        assert!(report.contains("cpu — x86_64 (avx2)"), "{report}");
        assert!(report.contains("avx2, fma"), "{report}");
        assert!(report.contains("17179869184 bytes"), "{report}");
        assert!(report.contains("cuda — NVIDIA RTX 4090"), "{report}");
        // A backend with no features shows the (none) placeholder.
        assert!(report.contains("(none)"), "{report}");
    }

    #[test]
    fn live_registry_includes_cpu() {
        // The crate links `tritium_cpu`, so the live registry must surface it.
        let registry = tritium_runtime::Registry::init();
        let report = render_backends(&registry.enumerate());
        assert!(
            report.contains("cpu"),
            "live report should list cpu: {report}"
        );
    }
}
