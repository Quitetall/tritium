//! Thin explicit file receiver; orchestration and publication stay in SALT.
use crate::convert_plan::{NamedPath, parse_named_path, paths};
use anyhow::{Context, Result, ensure};
use clap::{Args, Subcommand};
use std::{io::Write, path::PathBuf};
use tritium_quantize::engine::read_original_file_bounded;
use tritium_salt::{
    QteaCampaignFiles, QteaCampaignInspection, QteaFileCampaignRequest, inspect_qtea_file_campaign,
    run_qtea_file_campaign_to_directory,
};

#[derive(Args, Debug)]
pub(crate) struct Inputs {
    /// Strict tritium.qtea.file-campaign.v1 request; no implicit recipes or metrics.
    #[arg(long)]
    request: PathBuf,
    /// Repeat every actual original safetensors shard.
    #[arg(long, required = true)]
    source_shard: Vec<PathBuf>,
    /// Exact named original asset, NAME=PATH; repeat as needed.
    #[arg(long,value_parser=parse_named_path)]
    asset: Vec<NamedPath>,
    /// Exact native Gram/cache input, OWNER=PATH; repeat every matrix.
    #[arg(long,required=true,value_parser=parse_named_path)]
    curvature: Vec<NamedPath>,
    /// Positive request and selected-inspection input JSON byte cap (not RSS).
    #[arg(long, default_value_t = 16_777_216)]
    max_json_bytes: usize,
    /// Positive generated JSON output capacity cap (not model/cache RSS).
    #[arg(long, default_value_t = 16_777_216)]
    max_report_bytes: usize,
}
#[derive(Subcommand, Debug)]
pub(crate) enum QteaCommand {
    /// Inspect actual source/curvature/work and caller-estimate admission; no fitting.
    Inspect {
        #[command(flatten)]
        inputs: Box<Inputs>,
    },
    /// Revalidate an explicit selected inspection, fit and publish an unqualified model.
    Run {
        #[command(flatten)]
        inputs: Box<Inputs>,
        /// Selected inspection JSON from inspect; data, not independent approval.
        #[arg(long)]
        inspection: PathBuf,
        /// New private spool filename; parent must exist, removed after work.
        #[arg(long)]
        spool: PathBuf,
        /// New candidate directory; never overwrite existing paths.
        #[arg(long)]
        out: PathBuf,
    },
}
impl Inputs {
    fn read(&self) -> Result<(QteaFileCampaignRequest, QteaCampaignFiles)> {
        ensure!(
            self.max_json_bytes > 0 && self.max_report_bytes > 0,
            "JSON caps must be positive"
        );
        let bytes = read_original_file_bounded(&self.request, self.max_json_bytes as u64)?;
        Ok((
            QteaFileCampaignRequest::from_slice(&bytes, self.max_json_bytes)?,
            QteaCampaignFiles {
                source_shards: self.source_shard.clone(),
                assets: paths(&self.asset)?,
                curvature: paths(&self.curvature)?,
            },
        ))
    }
}
pub(crate) fn run(command: QteaCommand) -> Result<()> {
    let bytes = match command {
        QteaCommand::Inspect { inputs } => {
            let (r, f) = inputs.read()?;
            inspect_qtea_file_campaign(&r, &f)?.to_json(inputs.max_report_bytes)?
        }
        QteaCommand::Run {
            inputs,
            inspection,
            spool,
            out,
        } => {
            let (r, f) = inputs.read()?;
            let selected = QteaCampaignInspection::from_slice(
                &read_original_file_bounded(&inspection, inputs.max_json_bytes as u64)?,
                inputs.max_json_bytes,
            )?;
            let result = run_qtea_file_campaign_to_directory(&r, &f, &selected, &spool, &out)?;
            result.to_json(inputs.max_report_bytes).context(
                "candidate committed; failed to encode publication observations (not rollback)",
            )?
        }
    };
    write_observations(&mut std::io::stdout().lock(), &bytes)?;
    Ok(())
}
fn write_observations(output: &mut impl Write, bytes: &[u8]) -> Result<()> {
    output
        .write_all(bytes)
        .and_then(|_| output.write_all(b"\n"))
        .context("write unqualified QTEA observations")
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn json_and_newline_failures_are_returned_without_panicking() {
        struct FailAfter(usize);
        impl Write for FailAfter {
            fn write(&mut self, b: &[u8]) -> std::io::Result<usize> {
                if self.0 == 0 {
                    return Err(std::io::ErrorKind::BrokenPipe.into());
                }
                let n = b.len().min(self.0);
                self.0 -= n;
                Ok(n)
            }
            fn flush(&mut self) -> std::io::Result<()> {
                Ok(())
            }
        }
        for bytes_before_failure in [0, 2] {
            let e = write_observations(&mut FailAfter(bytes_before_failure), b"{}").unwrap_err();
            assert_eq!(
                e.downcast_ref::<std::io::Error>().unwrap().kind(),
                std::io::ErrorKind::BrokenPipe
            );
        }
        let mut out = Vec::new();
        write_observations(&mut out, b"{}").unwrap();
        assert_eq!(out, b"{}\n");
    }
}
