//! Bounded read-only preparation through the ordinary retained text loader.
use anyhow::{Context as _, ensure};
use clap::{Args, ValueEnum};
use serde::Serialize;
use std::{
    io::{self, Write},
    path::PathBuf,
};

#[derive(Args, Debug)]
pub(crate) struct InspectTritArgs {
    #[arg(long)]
    model: PathBuf,
    /// Actual target level, not the original bundle profile name.
    #[arg(long)]
    level: String,
    /// Pure aggregate streaming costs; no GPU/context initialization or runtime admission.
    #[arg(long, value_enum, requires = "max_context")]
    backend: Option<InspectionBackend>,
    /// Refuse backend-declared dense emulation in the structural plan.
    #[arg(long, requires = "backend")]
    native_only: bool,
    /// Per-matrix backend payload ceiling, not aggregate or physical memory.
    #[arg(long, requires = "backend")]
    max_tensor_payload_bytes: Option<u64>,
    #[arg(long, default_value_t = 2 * 1024 * 1024 * 1024u64)]
    max_package_bytes: u64,
    #[arg(long, default_value_t = 16 * 1024 * 1024u64)]
    max_manifest_bytes: u64,
    #[arg(long, default_value_t = 16 * 1024 * 1024u64)]
    max_directory_bytes: u64,
    #[arg(long, default_value_t = 16 * 1024 * 1024u64)]
    max_config_bytes: u64,
    #[arg(long, default_value_t = 64 * 1024 * 1024u64)]
    max_tensor_bytes: u64,
    #[arg(long, default_value_t = 256 * 1024 * 1024u64)]
    max_decoded_tensor_bytes: u64,
    /// Entire serialized report ceiling including its newline; not process RSS.
    #[arg(long, default_value_t = 1024 * 1024usize)]
    max_output_bytes: usize,
    #[arg(long, requires_all = ["max_segment_tokens", "max_host_cache_payload_bytes"])]
    max_context: Option<usize>,
    #[arg(long, requires = "max_context")]
    max_segment_tokens: Option<usize>,
    #[arg(long, requires = "max_context")]
    max_host_cache_payload_bytes: Option<u64>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, ValueEnum)]
enum InspectionBackend {
    Cpu,
    Cuda,
}
impl InspectionBackend {
    fn plan(
        self,
        metadata: tritium_spec::TensorMetadata<'_>,
        encoding: tritium_spec::TensorStreamEncoding,
    ) -> Result<Option<tritium_spec::TensorStreamPlan>, tritium_spec::BackendError> {
        match self {
            Self::Cpu => tritium_spec::TernaryBackend::tensor_stream_plan(
                &tritium_cpu::CpuBackend,
                metadata,
                encoding,
            ),
            #[cfg(feature = "cuda")]
            Self::Cuda => tritium_cuda::CudaBackend::plan_tensor_stream(metadata, encoding),
            #[cfg(not(feature = "cuda"))]
            Self::Cuda => Err(tritium_spec::BackendError::Backend(
                "CUDA inspection requires a build with --features cuda".into(),
            )),
        }
    }
    const fn name(self) -> &'static str {
        match self {
            Self::Cpu => "cpu",
            Self::Cuda => "cuda",
        }
    }
}

#[derive(Serialize)]
struct ResourcePlan {
    payload_bytes: u64,
    source_workspace_bytes: u64,
    host_workspace_bytes: u64,
    device_workspace_bytes: u64,
    retained_host_bytes: u64,
}
impl From<&tritium_nn::StreamedModelLoadPlan> for ResourcePlan {
    fn from(plan: &tritium_nn::StreamedModelLoadPlan) -> Self {
        Self {
            payload_bytes: plan.payload_bytes(),
            source_workspace_bytes: plan.source_workspace_bytes(),
            host_workspace_bytes: plan.host_workspace_bytes(),
            device_workspace_bytes: plan.device_workspace_bytes(),
            retained_host_bytes: plan.retained_host_bytes(),
        }
    }
}

#[derive(Serialize)]
struct Report<'a> {
    status: &'static str,
    source: &'a tritium_schema::LoadedSourceIdentity,
    architecture: &'a str,
    layer_count: usize,
    vocab_size: usize,
    source_bytes: u64,
    tensors: &'a [tritium_nn::TensorLoadPlan],
    encoded_tensor_bytes_sum: u64,
    logical_decoded_tensor_bytes_sum: u64,
    maximum_tensor_working_payload_bytes: u64,
    tokenizer_encoded_bytes: u64,
    logical_host_cache_payload_bytes: Option<u64>,
    backend_payload_bytes: Option<u64>,
    backend: Option<&'static str>,
    resource_plan: Option<ResourcePlan>,
    backend_initialized: bool,
    physical_memory: &'static str,
    weights_uploaded: bool,
}
struct BoundedOutput {
    bytes: Vec<u8>,
    maximum: usize,
}
impl Write for BoundedOutput {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        let size = self
            .bytes
            .len()
            .checked_add(bytes.len())
            .ok_or_else(|| io::Error::other("report size overflow"))?;
        if size > self.maximum {
            return Err(io::Error::other(
                "inspection report exceeds --max-output-bytes",
            ));
        }
        self.bytes
            .try_reserve_exact(bytes.len())
            .map_err(io::Error::other)?;
        self.bytes.extend_from_slice(bytes);
        Ok(bytes.len())
    }
    fn flush(&mut self) -> io::Result<()> {
        Ok(())
    }
}
pub(crate) fn run(args: InspectTritArgs) -> anyhow::Result<()> {
    ensure!(
        args.max_output_bytes > 0,
        "--max-output-bytes must be positive"
    );
    ensure!(
        args.backend != Some(InspectionBackend::Cuda) || cfg!(feature = "cuda"),
        "CUDA inspection requires a build with --features cuda"
    );
    let execution = match (
        args.max_context,
        args.max_segment_tokens,
        args.max_host_cache_payload_bytes,
    ) {
        (None, None, None) => None,
        (Some(max_context), Some(max_segment_tokens), Some(max_host_cache_payload_bytes)) => {
            Some(tritium_nn::StreamLimits {
                max_context,
                max_segment_tokens,
                max_host_cache_payload_bytes,
            })
        }
        _ => anyhow::bail!("inspection stream limits must be supplied together"),
    };
    let receiving = tritium_format::TritPackageReadLimits {
        max_source_bytes: args.max_package_bytes,
        max_manifest_bytes: usize::try_from(args.max_manifest_bytes)?,
        max_directory_bytes: usize::try_from(args.max_directory_bytes)?,
    };
    let limits = tritium_nn::Qwen35TritReadLimits {
        max_config_bytes: args.max_config_bytes,
        max_tensor_bytes: args.max_tensor_bytes,
        max_decoded_tensor_bytes: args.max_decoded_tensor_bytes,
    };
    let streamed = args
        .backend
        .map(|backend| {
            tritium_nn::inspect_trit_file_streamed(
                &args.model,
                &args.level,
                receiving,
                limits,
                execution
                    .ok_or_else(|| anyhow::anyhow!("backend inspection requires stream limits"))?,
                tritium_schema::TensorUploadPolicy {
                    native_only: args.native_only,
                    max_payload_bytes: args.max_tensor_payload_bytes,
                },
                |metadata, encoding| backend.plan(metadata, encoding),
            )
            .map_err(anyhow::Error::from)
        })
        .transpose()?;
    let source_only = if streamed.is_none() {
        Some(tritium_nn::inspect_trit_file(
            &args.model,
            &args.level,
            receiving,
            limits,
            execution,
        )?)
    } else {
        None
    };
    let plan = streamed
        .as_ref()
        .map(|p| p.source_plan())
        .or(source_only.as_ref())
        .expect("exactly one inspection mode");
    let report = Report {
        status: if streamed.is_some() {
            "resource-planned-not-runtime-admitted"
        } else {
            "source-prepared-not-runtime-admitted"
        },
        source: plan.source(),
        architecture: plan.architecture().name(),
        layer_count: plan.architecture().layer_count(),
        vocab_size: plan.architecture().vocab_size(),
        source_bytes: plan.source_bytes(),
        tensors: plan.tensors(),
        encoded_tensor_bytes_sum: plan.encoded_tensor_bytes_sum(),
        logical_decoded_tensor_bytes_sum: plan.logical_decoded_tensor_bytes_sum(),
        maximum_tensor_working_payload_bytes: plan.maximum_tensor_working_payload_bytes(),
        tokenizer_encoded_bytes: plan.tokenizer_encoded_bytes(),
        logical_host_cache_payload_bytes: plan
            .cache_admission()
            .map(|c| c.logical_host_cache_payload_bytes()),
        backend_payload_bytes: streamed.as_ref().map(|p| p.payload_bytes()),
        backend: args.backend.map(InspectionBackend::name),
        resource_plan: streamed.as_ref().map(ResourcePlan::from),
        backend_initialized: false,
        physical_memory: if streamed.is_some() {
            "unknown-not-covered-by-logical-plan"
        } else {
            "unknown-not-covered-by-source-plan"
        },
        weights_uploaded: false,
    };
    let mut output = BoundedOutput {
        bytes: Vec::new(),
        maximum: args.max_output_bytes,
    };
    serde_json::to_writer(&mut output, &report).context("bounded inspection JSON")?;
    output.write_all(b"\n")?;
    io::stdout().lock().write_all(&output.bytes)?;
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn output_ceiling_includes_newline_and_never_exceeds_bound() {
        let mut output = BoundedOutput {
            bytes: Vec::new(),
            maximum: 3,
        };
        output.write_all(b"{}").unwrap();
        output.write_all(b"\n").unwrap();
        assert!(output.write_all(b"x").is_err());
        assert_eq!(output.bytes, b"{}\n");
    }
}
