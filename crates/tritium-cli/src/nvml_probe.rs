//! Fail-closed NVIDIA hardware evidence collected through `nvidia-smi`/NVML.

use std::process::Command;

use anyhow::{Context, ensure};
use serde::Serialize;
use tritium_cuda::CudaDeviceIdentity;

const MIB_BYTES: u64 = 1024 * 1024;
const STDERR_LIMIT_BYTES: usize = 2048;
const REBOOT_HINT: &str = "The NVIDIA kernel module and userspace/NVML library may be out of sync; a reboot is required after a driver update before retrying.";
const GPU_QUERY_ARGS: [&str; 2] = [
    "--query-gpu=pci.bus_id,name,driver_version,memory.total,memory.used",
    "--format=csv,noheader,nounits",
];
const DRIVER_VERSION_LABELS: [&str; 2] = ["Driver Version:", "KMD Version:"];
const CUDA_VERSION_LABELS: [&str; 2] = ["CUDA Version:", "CUDA UMD Version:"];

/// NVML-backed physical-device identity and memory evidence for one CUDA device.
#[derive(Clone, Debug, Eq, PartialEq, Serialize)]
pub(crate) struct NvmlGpuSnapshot {
    /// CUDA ordinal whose stable PCI identity selected this physical GPU.
    pub(crate) cuda_ordinal: usize,
    /// Normalized PCI address in `dddd:bb:dd.f` form.
    pub(crate) pci_bus_id: String,
    /// Exact device name reported by both CUDA and NVML.
    pub(crate) name: String,
    /// NVIDIA driver branch reported by NVML (for example, `610.43.03`).
    pub(crate) driver_version: String,
    /// CUDA Driver API compatibility reported by NVML, encoded as CUDA does
    /// (for example, `13010` for CUDA 13.1).
    pub(crate) nvml_cuda_driver_version: u32,
    /// Physical framebuffer capacity, converted from MiB with checked arithmetic.
    pub(crate) total_memory_bytes: u64,
    /// Physical framebuffer usage at the time of the query.
    pub(crate) used_memory_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct QueriedGpu {
    pci_bus_id: String,
    name: String,
    driver_version: String,
    total_memory_bytes: u64,
    used_memory_bytes: u64,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmiSystemVersions {
    driver_version: String,
    cuda_driver_version: u32,
}

#[derive(Clone, Debug, Eq, PartialEq)]
struct SmiOutput {
    success: bool,
    status: String,
    stdout: Vec<u8>,
    stderr: Vec<u8>,
}

/// Query NVML and select the one physical GPU matching a CUDA identity.
///
/// This is intentionally fail-closed: unavailable NVML, ambiguous PCI identity,
/// name drift, malformed memory data, or CUDA/NVML driver-version disagreement
/// all reject the evidence instead of silently falling back to CUDA-only data.
pub(crate) fn probe_cuda_device(
    cuda_identity: &CudaDeviceIdentity,
) -> anyhow::Result<NvmlGpuSnapshot> {
    probe_with_runner(cuda_identity, run_nvidia_smi)
}

fn probe_with_runner(
    cuda_identity: &CudaDeviceIdentity,
    mut run: impl FnMut(&[&str]) -> anyhow::Result<SmiOutput>,
) -> anyhow::Result<NvmlGpuSnapshot> {
    let gpu_output = run(&GPU_QUERY_ARGS).context("run NVML GPU query")?;
    require_success("NVML GPU query", &gpu_output)?;
    let gpus = parse_gpu_query(&gpu_output.stdout)?;

    let system_output = run(&[]).context("run NVML system-version query")?;
    require_success("NVML system-version query", &system_output)?;
    let system_versions = parse_system_versions(&system_output.stdout)?;

    let cuda_pci_bus_id =
        normalize_pci_bus_id(&cuda_identity.pci_bus_id).context("normalize CUDA PCI bus id")?;
    let mut matches = gpus.iter().filter(|gpu| gpu.pci_bus_id == cuda_pci_bus_id);
    let gpu = matches.next().with_context(|| {
        let observed = gpus
            .iter()
            .map(|gpu| gpu.pci_bus_id.as_str())
            .collect::<Vec<_>>()
            .join(", ");
        format!(
            "NVML reported no GPU at CUDA PCI address {cuda_pci_bus_id}; observed [{}]",
            observed
        )
    })?;
    ensure!(
        matches.next().is_none(),
        "NVML reported multiple GPUs at CUDA PCI address {cuda_pci_bus_id}; refusing ambiguous hardware evidence"
    );
    ensure!(
        gpu.name == cuda_identity.device_name.trim(),
        "CUDA/NVML device-name mismatch at {cuda_pci_bus_id}: CUDA reported {:?}, NVML reported {:?}",
        cuda_identity.device_name,
        gpu.name
    );
    ensure!(
        gpu.driver_version == system_versions.driver_version,
        "NVML driver-version mismatch: GPU query reported {}, system summary reported {}. {REBOOT_HINT}",
        gpu.driver_version,
        system_versions.driver_version
    );

    let cuda_version = decode_cuda_driver_version(cuda_identity.cuda_driver_version)?;
    let nvml_version = decode_cuda_driver_version(system_versions.cuda_driver_version)?;
    ensure!(
        cuda_version == nvml_version,
        "CUDA/NVML CUDA Driver API mismatch: CUDA reported {}.{} ({}), NVML reported {}.{} ({}). {REBOOT_HINT}",
        cuda_version.0,
        cuda_version.1,
        cuda_identity.cuda_driver_version,
        nvml_version.0,
        nvml_version.1,
        system_versions.cuda_driver_version
    );

    Ok(NvmlGpuSnapshot {
        cuda_ordinal: cuda_identity.ordinal,
        pci_bus_id: gpu.pci_bus_id.clone(),
        name: gpu.name.clone(),
        driver_version: gpu.driver_version.clone(),
        nvml_cuda_driver_version: system_versions.cuda_driver_version,
        total_memory_bytes: gpu.total_memory_bytes,
        used_memory_bytes: gpu.used_memory_bytes,
    })
}

fn run_nvidia_smi(args: &[&str]) -> anyhow::Result<SmiOutput> {
    let output = Command::new("nvidia-smi")
        .args(args)
        .env("LC_ALL", "C")
        .output()
        .with_context(|| format!("start nvidia-smi with arguments {args:?}. {REBOOT_HINT}"))?;
    Ok(SmiOutput {
        success: output.status.success(),
        status: output.status.to_string(),
        stdout: output.stdout,
        stderr: output.stderr,
    })
}

fn require_success(operation: &str, output: &SmiOutput) -> anyhow::Result<()> {
    ensure!(
        output.success,
        "{operation} failed with status {}: {}. {REBOOT_HINT}",
        output.status,
        bounded_stderr(&output.stderr)
    );
    Ok(())
}

fn bounded_stderr(stderr: &[u8]) -> String {
    let end = stderr.len().min(STDERR_LIMIT_BYTES);
    let mut message = String::from_utf8_lossy(&stderr[..end]).trim().to_owned();
    if message.is_empty() {
        message.push_str("nvidia-smi produced no stderr");
    }
    const SUFFIX: &str = " [stderr truncated]";
    if stderr.len() > end || message.len() > STDERR_LIMIT_BYTES {
        let limit = STDERR_LIMIT_BYTES - SUFFIX.len();
        let boundary = (0..=limit)
            .rev()
            .find(|&index| message.is_char_boundary(index))
            .unwrap_or(0);
        message.truncate(boundary);
        message.truncate(message.trim_end().len());
        message.push_str(SUFFIX);
    }
    message
}

fn parse_gpu_query(stdout: &[u8]) -> anyhow::Result<Vec<QueriedGpu>> {
    let text = std::str::from_utf8(stdout).context("NVML GPU query output is not UTF-8")?;
    let mut gpus = Vec::new();
    for (line_index, raw_line) in text.lines().enumerate() {
        let line = raw_line.trim_end_matches('\r');
        if line.trim().is_empty() {
            continue;
        }
        let fields = parse_csv_record(line)
            .with_context(|| format!("parse NVML GPU query CSV line {}", line_index + 1))?;
        ensure!(
            fields.len() == 5,
            "NVML GPU query line {} has {} fields; expected exactly 5",
            line_index + 1,
            fields.len()
        );

        let pci_bus_id = normalize_pci_bus_id(&fields[0])
            .with_context(|| format!("parse PCI bus id on NVML line {}", line_index + 1))?;
        let name = fields[1].trim().to_owned();
        ensure!(
            !name.is_empty(),
            "NVML GPU query line {} has an empty device name",
            line_index + 1
        );
        let driver_version = parse_nvidia_driver_version(&fields[2])
            .with_context(|| format!("parse driver version on NVML line {}", line_index + 1))?;
        let total_memory_bytes = parse_mib_bytes(&fields[3], "total memory")
            .with_context(|| format!("parse memory on NVML line {}", line_index + 1))?;
        let used_memory_bytes = parse_mib_bytes(&fields[4], "used memory")
            .with_context(|| format!("parse memory on NVML line {}", line_index + 1))?;
        ensure!(
            used_memory_bytes <= total_memory_bytes,
            "NVML GPU query line {} reports used memory {} above total memory {}",
            line_index + 1,
            used_memory_bytes,
            total_memory_bytes
        );
        gpus.push(QueriedGpu {
            pci_bus_id,
            name,
            driver_version,
            total_memory_bytes,
            used_memory_bytes,
        });
    }
    ensure!(!gpus.is_empty(), "NVML GPU query returned no GPU rows");
    Ok(gpus)
}

fn parse_system_versions(stdout: &[u8]) -> anyhow::Result<SmiSystemVersions> {
    let text = std::str::from_utf8(stdout).context("NVML system summary is not UTF-8")?;
    let driver = labeled_token(text, &DRIVER_VERSION_LABELS)?;
    let driver_version = parse_nvidia_driver_version(driver)
        .context("parse driver version from NVML system summary")?;
    let cuda = labeled_token(text, &CUDA_VERSION_LABELS)?;
    let cuda_driver_version = parse_cuda_version_text(cuda)
        .context("parse CUDA Driver API version from NVML system summary")?;
    Ok(SmiSystemVersions {
        driver_version,
        cuda_driver_version,
    })
}

fn labeled_token<'a>(text: &'a str, labels: &[&str]) -> anyhow::Result<&'a str> {
    let mut matched = None;
    for &label in labels {
        let mut matches = text.match_indices(label);
        if let Some((index, _)) = matches.next() {
            ensure!(
                matches.next().is_none(),
                "NVML system summary contains {label:?} more than once"
            );
            ensure!(
                matched.is_none(),
                "NVML system summary contains more than one accepted version label from {labels:?}"
            );
            matched = Some((label, index));
        }
    }
    let (label, index) =
        matched.with_context(|| format!("NVML system summary is missing one of {labels:?}"))?;
    let suffix = &text[index + label.len()..];
    let token = suffix
        .trim_start()
        .split(|character: char| character.is_whitespace() || character == '|')
        .next()
        .unwrap_or_default();
    ensure!(
        !token.is_empty(),
        "NVML system summary has no value after {label:?}"
    );
    Ok(token)
}

fn parse_cuda_version_text(raw: &str) -> anyhow::Result<u32> {
    let mut components = raw.trim().split('.');
    let major = parse_decimal_component(components.next(), "CUDA major version")?;
    let minor = parse_decimal_component(components.next(), "CUDA minor version")?;
    if let Some(patch) = components.next() {
        ensure!(
            parse_decimal_component(Some(patch), "CUDA patch version")? == 0,
            "NVML CUDA version has a non-zero patch component"
        );
    }
    ensure!(
        components.next().is_none(),
        "NVML CUDA version has too many components"
    );
    ensure!(major > 0, "NVML CUDA major version must be non-zero");
    ensure!(minor < 100, "NVML CUDA minor version exceeds CUDA encoding");
    major
        .checked_mul(1000)
        .and_then(|value| {
            minor
                .checked_mul(10)
                .and_then(|minor| value.checked_add(minor))
        })
        .context("NVML CUDA version exceeds u32")
}

fn decode_cuda_driver_version(version: u32) -> anyhow::Result<(u32, u32)> {
    ensure!(
        version != 0,
        "CUDA Driver API version is unavailable (zero)"
    );
    ensure!(
        version.is_multiple_of(10),
        "CUDA Driver API version {version} is not in CUDA integer encoding"
    );
    let major = version / 1000;
    let minor = (version % 1000) / 10;
    ensure!(major > 0, "CUDA Driver API major version is zero");
    ensure!(
        minor < 100,
        "CUDA Driver API minor version exceeds CUDA integer encoding"
    );
    Ok((major, minor))
}

fn parse_nvidia_driver_version(raw: &str) -> anyhow::Result<String> {
    let version = raw.trim();
    let components = version.split('.').collect::<Vec<_>>();
    ensure!(
        (2..=4).contains(&components.len()),
        "NVIDIA driver version must have two to four numeric components"
    );
    for component in components {
        ensure!(
            !component.is_empty() && component.bytes().all(|byte| byte.is_ascii_digit()),
            "NVIDIA driver version contains a non-numeric component"
        );
        component
            .parse::<u32>()
            .context("NVIDIA driver version component exceeds u32")?;
    }
    Ok(version.to_owned())
}

fn parse_decimal_component(raw: Option<&str>, label: &str) -> anyhow::Result<u32> {
    let raw = raw.with_context(|| format!("missing {label}"))?;
    ensure!(
        !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()),
        "{label} is not an unsigned decimal integer"
    );
    raw.parse::<u32>()
        .with_context(|| format!("{label} exceeds u32"))
}

fn parse_mib_bytes(raw: &str, label: &str) -> anyhow::Result<u64> {
    let raw = raw.trim();
    ensure!(
        !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_digit()),
        "NVML {label} is not an unsigned integer MiB value"
    );
    let mib = raw
        .parse::<u64>()
        .with_context(|| format!("NVML {label} exceeds u64 MiB"))?;
    mib.checked_mul(MIB_BYTES)
        .with_context(|| format!("NVML {label} overflows bytes"))
}

fn normalize_pci_bus_id(raw: &str) -> anyhow::Result<String> {
    let components = raw.trim().split(':').collect::<Vec<_>>();
    ensure!(
        components.len() == 3,
        "PCI bus id must contain domain, bus, and device/function"
    );
    let domain = parse_hex_component(components[0], "PCI domain", u16::MAX.into())?;
    let bus = parse_hex_component(components[1], "PCI bus", u8::MAX.into())?;
    let device_function = components[2].split('.').collect::<Vec<_>>();
    ensure!(
        device_function.len() == 2,
        "PCI bus id must contain exactly one function separator"
    );
    let device = parse_hex_component(device_function[0], "PCI device", 0x1f)?;
    let function = parse_hex_component(device_function[1], "PCI function", 0x7)?;
    Ok(format!("{domain:04x}:{bus:02x}:{device:02x}.{function:x}"))
}

fn parse_hex_component(raw: &str, label: &str, maximum: u32) -> anyhow::Result<u32> {
    ensure!(
        !raw.is_empty() && raw.bytes().all(|byte| byte.is_ascii_hexdigit()),
        "{label} is not an unsigned hexadecimal integer"
    );
    let value = u32::from_str_radix(raw, 16).with_context(|| format!("{label} exceeds u32"))?;
    ensure!(value <= maximum, "{label} exceeds {maximum:#x}");
    Ok(value)
}

fn parse_csv_record(line: &str) -> anyhow::Result<Vec<String>> {
    #[derive(Clone, Copy, Debug, Eq, PartialEq)]
    enum State {
        Unquoted,
        Quoted,
        QuoteClosed,
    }

    let mut state = State::Unquoted;
    let mut fields = Vec::new();
    let mut field = String::new();
    for character in line.chars() {
        match (state, character) {
            (State::Unquoted, ',') => {
                fields.push(field.trim().to_owned());
                field.clear();
            }
            (State::Unquoted, '"') if field.trim().is_empty() => {
                field.clear();
                state = State::Quoted;
            }
            (State::Unquoted, '"') => anyhow::bail!("quote inside an unquoted CSV field"),
            (State::Unquoted, character) => field.push(character),
            (State::Quoted, '"') => state = State::QuoteClosed,
            (State::Quoted, character) => field.push(character),
            (State::QuoteClosed, '"') => {
                field.push('"');
                state = State::Quoted;
            }
            (State::QuoteClosed, ',') => {
                fields.push(field.trim().to_owned());
                field.clear();
                state = State::Unquoted;
            }
            (State::QuoteClosed, character) if character.is_whitespace() => {}
            (State::QuoteClosed, _) => {
                anyhow::bail!("non-whitespace data after a quoted CSV field")
            }
        }
    }
    ensure!(state != State::Quoted, "unterminated quoted CSV field");
    fields.push(field.trim().to_owned());
    Ok(fields)
}

#[cfg(test)]
mod tests {
    use std::collections::VecDeque;

    use super::*;

    fn identity() -> CudaDeviceIdentity {
        CudaDeviceIdentity {
            ordinal: 3,
            device_name: "NVIDIA GeForce RTX 4090".to_owned(),
            pci_bus_id: "0000:01:00.0".to_owned(),
            cuda_driver_version: 13_010,
        }
    }

    fn successful(stdout: &str) -> SmiOutput {
        SmiOutput {
            success: true,
            status: "exit status: 0".to_owned(),
            stdout: stdout.as_bytes().to_vec(),
            stderr: Vec::new(),
        }
    }

    fn query_output() -> SmiOutput {
        successful("00000000:01:00.0, NVIDIA GeForce RTX 4090, 610.43.03, 24564, 1234\n")
    }

    fn summary_output() -> SmiOutput {
        successful("| NVIDIA-SMI 610.43.03  Driver Version: 610.43.03  CUDA Version: 13.1 |\n")
    }

    fn probe_with_outputs(
        identity: &CudaDeviceIdentity,
        outputs: impl IntoIterator<Item = SmiOutput>,
    ) -> anyhow::Result<NvmlGpuSnapshot> {
        let mut outputs = outputs.into_iter().collect::<VecDeque<_>>();
        probe_with_runner(identity, |_| {
            outputs
                .pop_front()
                .context("test did not inject enough nvidia-smi outputs")
        })
    }

    #[test]
    fn parses_query_csv_and_converts_mib_to_bytes() {
        let rows =
            parse_gpu_query(b"00000000:AF:1F.7, \"NVIDIA, Test \"\"GPU\"\"\", 610.43.03, 2, 1\r\n")
                .unwrap();
        assert_eq!(
            rows,
            vec![QueriedGpu {
                pci_bus_id: "0000:af:1f.7".to_owned(),
                name: "NVIDIA, Test \"GPU\"".to_owned(),
                driver_version: "610.43.03".to_owned(),
                total_memory_bytes: 2 * MIB_BYTES,
                used_memory_bytes: MIB_BYTES,
            }]
        );
    }

    #[test]
    fn rejects_malformed_query_fields_memory_and_accounting() {
        assert!(parse_gpu_query(b"0000:01:00.0, GPU, 610.43, 1\n").is_err());
        assert!(parse_gpu_query(b"0000:01:00.0, GPU, 610.43, 1 MiB, 0\n").is_err());
        assert!(parse_gpu_query(b"0000:01:00.0, GPU, 610.43, 1, 2\n").is_err());
        assert!(parse_gpu_query(b"0000:01:00.0, \"GPU, 610.43, 1, 0\n").is_err());
    }

    #[test]
    fn normalizes_pci_bus_ids_and_rejects_out_of_range_fields() {
        assert_eq!(
            normalize_pci_bus_id("00000000:AF:01.2").unwrap(),
            "0000:af:01.2"
        );
        assert_eq!(
            normalize_pci_bus_id("abcd:00:1f.7").unwrap(),
            "abcd:00:1f.7"
        );
        assert!(normalize_pci_bus_id("00010000:01:00.0").is_err());
        assert!(normalize_pci_bus_id("0000:100:00.0").is_err());
        assert!(normalize_pci_bus_id("0000:01:20.0").is_err());
        assert!(normalize_pci_bus_id("0000:01:00.8").is_err());
    }

    #[test]
    fn selects_unique_exact_cuda_identity_and_versions() {
        let snapshot = probe_with_outputs(&identity(), [query_output(), summary_output()]).unwrap();
        assert_eq!(
            snapshot,
            NvmlGpuSnapshot {
                cuda_ordinal: 3,
                pci_bus_id: "0000:01:00.0".to_owned(),
                name: "NVIDIA GeForce RTX 4090".to_owned(),
                driver_version: "610.43.03".to_owned(),
                nvml_cuda_driver_version: 13_010,
                total_memory_bytes: 24_564 * MIB_BYTES,
                used_memory_bytes: 1_234 * MIB_BYTES,
            }
        );
    }

    #[test]
    fn parses_nvidia_610_kmd_and_cuda_umd_version_labels() {
        let versions = parse_system_versions(
            b"| NVIDIA-SMI 610.43.03  KMD Version: 610.43.03  CUDA UMD Version: 13.3 |\n",
        )
        .unwrap();
        assert_eq!(
            versions,
            SmiSystemVersions {
                driver_version: "610.43.03".to_owned(),
                cuda_driver_version: 13_030,
            }
        );
    }

    #[test]
    fn rejects_missing_duplicate_and_name_mismatched_cuda_identity() {
        let missing = successful("0000:02:00.0, NVIDIA GeForce RTX 4090, 610.43.03, 2, 1\n");
        assert!(
            probe_with_outputs(&identity(), [missing, summary_output()])
                .unwrap_err()
                .to_string()
                .contains("no GPU")
        );

        let duplicate = successful(
            "0000:01:00.0, NVIDIA GeForce RTX 4090, 610.43.03, 2, 1\n00000000:01:00.0, NVIDIA GeForce RTX 4090, 610.43.03, 2, 1\n",
        );
        assert!(
            probe_with_outputs(&identity(), [duplicate, summary_output()])
                .unwrap_err()
                .to_string()
                .contains("multiple GPUs")
        );

        let wrong_name = successful("0000:01:00.0, NVIDIA A100, 610.43.03, 2, 1\n");
        assert!(
            probe_with_outputs(&identity(), [wrong_name, summary_output()])
                .unwrap_err()
                .to_string()
                .contains("device-name mismatch")
        );
    }

    #[test]
    fn rejects_cuda_api_and_driver_branch_mismatch() {
        let wrong_cuda =
            successful("| NVIDIA-SMI 610.43.03  Driver Version: 610.43.03  CUDA Version: 13.2 |\n");
        let error = probe_with_outputs(&identity(), [query_output(), wrong_cuda])
            .unwrap_err()
            .to_string();
        assert!(error.contains("CUDA Driver API mismatch"));
        assert!(error.contains("reboot is required"));

        let wrong_branch =
            successful("| NVIDIA-SMI 610.44.00  Driver Version: 610.44.00  CUDA Version: 13.1 |\n");
        let error = probe_with_outputs(&identity(), [query_output(), wrong_branch])
            .unwrap_err()
            .to_string();
        assert!(error.contains("driver-version mismatch"));
        assert!(error.contains("reboot is required"));
    }

    #[test]
    fn bounds_command_failure_stderr_and_includes_reboot_hint() {
        let failure = SmiOutput {
            success: false,
            status: "exit status: 18".to_owned(),
            stdout: Vec::new(),
            stderr: vec![b'x'; STDERR_LIMIT_BYTES + 100],
        };
        let error = probe_with_outputs(&identity(), [failure])
            .unwrap_err()
            .to_string();
        assert!(error.contains("exit status: 18"));
        assert!(error.contains("[stderr truncated]"));
        assert!(error.contains("reboot is required"));
        assert!(error.len() < STDERR_LIMIT_BYTES + 512);
    }

    #[test]
    fn rejects_missing_or_ambiguous_system_version_labels() {
        assert!(parse_system_versions(b"Driver Version: 610.43.03\n").is_err());
        assert!(
            parse_system_versions(
                b"Driver Version: 610.43.03 CUDA Version: 13.1 CUDA Version: 13.1"
            )
            .is_err()
        );
        assert!(parse_system_versions(b"Driver Version: 610.43.03 CUDA Version: 13.1.1").is_err());
        assert!(
            parse_system_versions(
                b"Driver Version: 610.43.03 KMD Version: 610.43.03 CUDA UMD Version: 13.3"
            )
            .is_err()
        );
    }
}
