//! Admitted experimental synthesis producer for the separate ADR0157 Python reference.
//! Not a model benchmark, producer attestation or qualification receipt.

use sha2::{Digest, Sha256};
use std::{
    error::Error,
    fs::{self, File, OpenOptions},
    io::{self, Write},
    path::PathBuf,
    sync::Arc,
};
use tritium_cpu::qtea::matmul_into;
use tritium_format::qtea::{QteaLimits, QteaTensor};
use tritium_quantize::qtea_reference::{TensorFit, TensorFitConfig};
use tritium_quantize::{CurvatureSourceId, InputGramAccumulator};
use tritium_salt::{
    AdmittedSolverPlan, ContentId, FrontierOrdering, FrontierProfile, FrontierProfileId,
    FrontierResourceEstimate, FrontierSolver, FrontierSolverError, FrontierStage,
    FrontierStageRequest, QteaTensorOrigin, QteaTensorWorkPlan, ResourceVector, SolverDescriptor,
    SolverFamily, SolverId, SolverRegistry, SolverRequest, SolverTrust,
};

// Planning-only synthetic fixture: no builtin installer, inventory or measurements.
#[derive(Debug)]
struct SyntheticPlanner {
    descriptor: SolverDescriptor,
    estimate: FrontierResourceEstimate,
}
impl FrontierSolver for SyntheticPlanner {
    fn descriptor(&self) -> &SolverDescriptor {
        &self.descriptor
    }
    fn estimate(&self, _: &SolverRequest) -> Result<FrontierResourceEstimate, FrontierSolverError> {
        Ok(self.estimate)
    }
}

fn synthetic_admission(
    work: &QteaTensorWorkPlan<'_>,
) -> Result<(AdmittedSolverPlan, FrontierStageRequest), Box<dyn Error>> {
    let id = SolverId::new("qtea-salient-residual.v1")?;
    let descriptor = SolverDescriptor::new(
        id.clone(),
        SolverFamily::QteaSalientResidual,
        SolverTrust::Experimental,
        1,
    )?;
    let m = work.memory();
    let resources = ResourceVector::new(
        u64::try_from(m.total_logical_bytes)?,
        0,
        0,
        u64::try_from(work.byte_ledger().total_bytes)?,
        0,
        u64::try_from(m.total_logical_bytes - m.borrowed_input_bytes)?,
        0,
        None,
    );
    let estimate = FrontierResourceEstimate::new(
        resources,
        ContentId::of_bytes(b"synthetic conformance machine label; not inventory"),
        ContentId::of_bytes(b"synthetic logical estimate; not observations"),
    )?;
    let profile = FrontierProfile::new(
        FrontierProfileId::new("test.qtea-admitted-conformance")?,
        FrontierOrdering::Fixed,
        vec![id.clone()],
        SolverTrust::Experimental,
        ResourceVector::new(
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            u64::MAX,
            None,
        ),
        None,
        false,
    )?;
    let mut registry = SolverRegistry::new();
    registry.register(Arc::new(SyntheticPlanner {
        descriptor,
        estimate,
    }))?;
    let meta = work.metadata();
    let plan = registry.plan(
        &id,
        SolverTrust::Experimental,
        &profile.request(u64::try_from(meta.rows)?, u64::try_from(meta.columns)?)?,
    )?;
    let request = FrontierStageRequest::new(
        ContentId::of_bytes(b"synthetic conformance run"),
        work.origin().source_id,
        0,
        FrontierStage::Fit,
        &profile,
        &plan,
        work.input_id(),
    )?;
    Ok((plan, request))
}

fn new_file(path: PathBuf) -> io::Result<File> {
    OpenOptions::new().write(true).create_new(true).open(path)
}

fn sha256_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut text = String::with_capacity(64);
    for &byte in Sha256::digest(bytes).iter() {
        text.push(char::from(HEX[usize::from(byte >> 4)]));
        text.push(char::from(HEX[usize::from(byte & 15)]));
    }
    text
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let destination = PathBuf::from(args.next().ok_or_else(|| {
        io::Error::new(
            io::ErrorKind::InvalidInput,
            "usage: qtea_conformance NEW_DIRECTORY",
        )
    })?);
    if args.next().is_some() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidInput,
            "exactly one directory required",
        )
        .into());
    }
    // Never overwrite or repair an earlier run. A failed run may leave partial
    // unpublished synthetic output, which is not successful conformance evidence.
    fs::create_dir(&destination)?;
    let limits = QteaLimits {
        max_tensor_bytes: 1 << 20,
        max_validation_bytes: 4096,
    };
    let cases = [
        (4, 8, 2, 0.),
        (4, 8, 2, 0.25),
        (4, 8, 2, 1.),
        (8, 12, 4, 0.),
        (8, 12, 4, 0.25),
        (8, 12, 4, 1.),
        (20, 6, 2, 0.34),
        (32, 18, 6, 0.5),
    ];
    for (case, (rows, columns, width, ratio)) in cases.into_iter().enumerate() {
        let weights = (0..rows)
            .flat_map(|r| {
                (0..columns).map(move |c| {
                    (((r + 1) * (c + 3)) % 17) as f32 * 0.35 - 2.8
                        + c as f32 * 0.19
                        + r as f32 * 0.07
                })
            })
            .collect::<Vec<_>>();
        let mut inputs = vec![0.; columns * columns];
        for column in 0..columns {
            inputs[column * columns + column] = 1. + column as f32 * 0.05;
        }
        inputs[1] = 0.4;
        inputs[columns + 1] = 0.8;
        // Declared synthetic identities, explicitly not authenticated model data.
        let source = CurvatureSourceId::new([1; 32], [2; 32], [3; 32])?;
        let mut builder = InputGramAccumulator::new_bound(columns, source, 0)?;
        builder.accumulate_batch(&inputs, columns, None, None)?;
        let gram = builder.finish()?;
        let numerical = TensorFit::plan(
            &weights,
            rows,
            columns,
            &gram,
            TensorFitConfig {
                group_size: width,
                salient_ratio: ratio,
                ..TensorFitConfig::default()
            },
            TensorFit::required_work_bytes(rows, columns)?,
        )?;
        let stem = format!("case-{case:03}");
        let owner = format!("{stem}.weight");
        let mut original = Sha256::new();
        for value in &weights {
            original.update(value.to_bits().to_le_bytes());
        }
        let work = QteaTensorWorkPlan::new(
            numerical,
            QteaTensorOrigin {
                source_id: ContentId::of_bytes(b"synthetic qtea conformance source artifact v1"),
                owner: &owner,
                tensor_index: case as u32,
                source_tensor_digest: original.finalize().into(),
            },
            limits,
            1 << 24,
        )?;
        let (plan, request) = synthetic_admission(&work)?;
        let work = work.admit(&plan, &request, plan.content_id())?;
        let tensor_path = destination.join(format!("{stem}.qtea"));
        let mut file = new_file(tensor_path.clone())?;
        let admitted_result = work.fit_to(&mut file)?;
        let result = admitted_result.work_result();
        file.sync_all()?;
        drop(file);
        println!(
            "work_input={} tensor={} stage_output={}",
            result.input_id(),
            result.tensor_id(),
            admitted_result.output_id()
        );
        // Forward uses only the actual reopened file, not retained fitter state.
        let cold = QteaTensor::read_from(&mut File::open(&tensor_path)?, limits)?;
        assert_eq!(result.tensor_id(), ContentId::of_bytes(cold.as_bytes()));
        assert_eq!(result.wire_digest(), cold.view().content_digest());
        let sha256 = sha256_hex(cold.as_bytes());
        for batch in [0, 1, 3] {
            let input = (0..batch * columns)
                .map(|i| (i % 7) as f32 * 0.125 - 0.375)
                .collect::<Vec<_>>();
            let mut output = vec![123.; batch * rows];
            let mut staging = vec![456.; batch * rows];
            matmul_into(cold.view(), &input, batch, &mut output, &mut staging)?;
            let input_bits = input.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let output_bits = output.iter().map(|v| v.to_bits()).collect::<Vec<_>>();
            let mut observation = new_file(destination.join(format!("{stem}.batch-{batch}.json")))?;
            writeln!(
                observation,
                "{{\"schema\":\"tritium.qtea.cpu-observation.v1\",\"tensor_sha256\":\"{sha256}\",\"batch\":{batch},\"input_bits\":{input_bits:?},\"output_bits\":{output_bits:?}}}"
            )?;
            observation.sync_all()?;
        }
    }
    println!("Produced8 synthetic native tensors and24 CPU observations; NOT qualification");
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::sha256_hex;

    #[test]
    fn actual_observation_digest_encoder_matches_sha256_known_answers() {
        assert_eq!(
            sha256_hex(b""),
            "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855"
        );
        assert_eq!(
            sha256_hex(b"abc"),
            "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad"
        );
    }
}
