//! Source-bound producer for ADR-0035 HESTIA Gate-C evidence.

use std::path::Path;

use anyhow::{Context, bail};
use serde::Serialize;
use sha2::{Digest, Sha256};
use tritium_cuda::train::CudaTrainBackendV1;
use tritium_spec::{TrainReceiptV1, TrainingVectorSetV3};
use tritium_testkit::{TrainingConformanceReport, run_training_conformance};
use tritium_train::{
    CpuTrainBackendV1,
    gradcheck::{GradCheckCfg, hestia_gate_c_report},
};

const OPERATION: &str = "graph.hestia_relax";

#[derive(Debug, Serialize)]
struct GateReceipt {
    schema: &'static str,
    result: &'static str,
    release: String,
    source_revision: String,
    gradcheck: GradcheckReceipt,
    portable_cpu: PortableReceipt,
    portable_cuda: PortableReceipt,
}

#[derive(Debug, Serialize)]
struct GradcheckReceipt {
    suite: &'static str,
    result: &'static str,
    inputs: [&'static str; 2],
    max_relative_error: f32,
    tolerance: f32,
}

#[derive(Debug, Serialize)]
struct PortableReceipt {
    backend: &'static str,
    result: &'static str,
    manifest_version: u32,
    operation: &'static str,
    vector_digest: String,
    case_count: usize,
    physical_device: String,
    driver: String,
}

pub(crate) fn seal(
    release: &str,
    source_revision: &str,
    output: &Path,
    cuda_device: usize,
) -> anyhow::Result<()> {
    validate_envelope(release, source_revision, output)?;
    let receipt = measure(release, source_revision, cuda_device)?;
    let mut bytes = serde_json::to_vec_pretty(&receipt)?;
    bytes.push(b'\n');
    crate::salt::publish_immutable(output, &bytes)?;
    print!(
        "{}",
        String::from_utf8(bytes).expect("receipt JSON is UTF-8")
    );
    Ok(())
}

fn validate_envelope(release: &str, source_revision: &str, output: &Path) -> anyhow::Result<()> {
    if release.is_empty() {
        bail!("release must be nonempty");
    }
    if source_revision.len() != 40
        || !source_revision
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        bail!("source revision must be 40 lowercase hexadecimal characters");
    }
    if output.as_os_str().is_empty() || output.file_name().is_none() || output.is_symlink() {
        bail!("output must name a non-symlink receipt file");
    }
    Ok(())
}

fn measure(
    release: &str,
    source_revision: &str,
    cuda_device: usize,
) -> anyhow::Result<GateReceipt> {
    require_source_identity("CPU", CpuTrainBackendV1::source_identity(), source_revision)?;
    require_source_identity(
        "CUDA",
        CudaTrainBackendV1::source_identity(),
        source_revision,
    )?;
    let vectors = TrainingVectorSetV3::parse_json(TrainingVectorSetV3::canonical_json())
        .context("parse canonical portable-training V3 corpus")?;
    let hestia_case_count = vectors
        .cases()
        .iter()
        .filter(|case| case.operation == OPERATION)
        .count();
    if hestia_case_count != 5 {
        bail!("portable V3 corpus must contain exactly five HESTIA cases");
    }
    let vector_digest = format!(
        "sha256:{}",
        crate::hex::hex_digest(&Sha256::digest(TrainingVectorSetV3::canonical_json()))
    );

    let gradcheck = measured_gradcheck()?;
    let cpu_backend = CpuTrainBackendV1::new();
    let cpu = run_training_conformance(&cpu_backend, &vectors);
    require_conformance("CPU", &cpu, source_revision)?;
    let cpu_device = physical_device(&cpu, OPERATION)?;

    let cuda_backend = CudaTrainBackendV1::new(cuda_device)
        .with_context(|| format!("open CUDA device {cuda_device}"))?;
    let cuda = run_training_conformance(&cuda_backend, &vectors);
    require_conformance("CUDA", &cuda, source_revision)?;
    let cuda_device_id = physical_device(&cuda, OPERATION)?;
    let driver_version = cuda_backend.cuda_driver_version();
    if driver_version == 0 {
        bail!("CUDA driver version is unavailable");
    }

    Ok(GateReceipt {
        schema: "tritium.stage7-hestia-gate-c.v1",
        result: "pass",
        release: release.to_owned(),
        source_revision: source_revision.to_owned(),
        gradcheck,
        portable_cpu: PortableReceipt {
            backend: "cpu",
            result: "pass",
            manifest_version: 3,
            operation: OPERATION,
            vector_digest: vector_digest.clone(),
            case_count: hestia_case_count,
            physical_device: cpu_device,
            driver: "rust-reference".to_owned(),
        },
        portable_cuda: PortableReceipt {
            backend: "cuda",
            result: "pass",
            manifest_version: 3,
            operation: OPERATION,
            vector_digest,
            case_count: hestia_case_count,
            physical_device: cuda_device_id,
            driver: format!("cuda-driver-api:{driver_version}"),
        },
    })
}

fn require_source_identity(
    label: &str,
    source_identity: &str,
    source_revision: &str,
) -> anyhow::Result<()> {
    let expected = format!("source-git:{source_revision}");
    if source_identity != expected {
        bail!("{label} backend was not built from exact clean {expected}");
    }
    Ok(())
}

fn measured_gradcheck() -> anyhow::Result<GradcheckReceipt> {
    let cfg = GradCheckCfg::default();
    let report = hestia_gate_c_report()
        .map_err(|failure| anyhow::anyhow!("HESTIA analytic gradcheck failed: {failure:?}"))?;
    if report.checked_inputs != [0, 2] || report.checked_elements != 16 {
        bail!("HESTIA analytic gradcheck coverage differs from Gate-C");
    }
    Ok(GradcheckReceipt {
        suite: "tritium-train/gradcheck_hestia",
        result: "pass",
        inputs: ["weight", "temperature"],
        max_relative_error: report.max_relative_error,
        tolerance: cfg.tol.relative,
    })
}

fn require_conformance(
    label: &str,
    report: &TrainingConformanceReport,
    source_revision: &str,
) -> anyhow::Result<()> {
    if !report.is_ok() {
        bail!(
            "{label} portable V3 conformance failed: {:?}",
            report.failed
        );
    }
    let expected_source = format!("+source-git:{source_revision}");
    let receipts: Vec<&TrainReceiptV1> = report
        .passed
        .iter()
        .filter_map(|case| case.receipt.as_ref())
        .collect();
    if receipts.is_empty()
        || receipts
            .iter()
            .any(|receipt| !receipt.backend_build.ends_with(&expected_source))
    {
        bail!("{label} backend was not built from exact clean source-git:{source_revision}");
    }
    Ok(())
}

fn physical_device(report: &TrainingConformanceReport, operation: &str) -> anyhow::Result<String> {
    report
        .passed
        .iter()
        .filter_map(|case| case.receipt.as_ref())
        .find(|receipt| receipt.operation == operation)
        .and_then(|receipt| receipt.physical_device.clone())
        .filter(|device| !device.is_empty())
        .context("HESTIA conformance emitted no physical device identity")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn envelope_rejects_noncanonical_revision_before_measurement() {
        assert!(validate_envelope("1.1.0-rc.0", "not-a-revision", Path::new("x.json")).is_err());
    }

    #[test]
    fn measured_gradcheck_records_exact_gate_coverage() {
        let receipt = measured_gradcheck().unwrap();
        assert_eq!(receipt.inputs, ["weight", "temperature"]);
        assert_eq!(receipt.tolerance, 2e-3);
        assert!(receipt.max_relative_error <= receipt.tolerance);
    }
}
