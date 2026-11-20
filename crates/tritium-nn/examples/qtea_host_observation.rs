//! Replay supplied synthetic inputs through the actual packed model projection.
//! Output can be checked by the separate Python wire/forward reference. This is
//! not an authenticated producer, real-model evaluator or qualification receipt.
use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};
use std::{
    fs::File,
    io::{self, Read},
};
use tritium_format::qtea::{QteaLimits, QteaTensor};
use tritium_nn::{HostQteaLinear, Projection};

const LIMIT: usize = 1 << 20;
const SCHEMA: &str = "tritium.qtea.cpu-observation.v1";

#[derive(Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
struct Observation {
    schema: String,
    tensor_sha256: String,
    batch: usize,
    input_bits: Vec<u32>,
    output_bits: Vec<u32>,
}

fn invalid(message: &str) -> io::Error {
    io::Error::new(io::ErrorKind::InvalidData, message)
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    if args.len() != 2 {
        return Err(
            invalid("usage: qtea_host_observation TENSOR SUPPLIED_OBSERVATION_JSON").into(),
        );
    }
    let tensor = QteaTensor::read_from(
        &mut File::open(&args[0])?,
        QteaLimits {
            max_tensor_bytes: LIMIT,
            max_validation_bytes: LIMIT,
        },
    )?;
    let digest: String = Sha256::digest(tensor.as_bytes())
        .iter()
        .map(|b| format!("{b:02x}"))
        .collect();
    let mut json = Vec::new();
    File::open(&args[1])?
        .take(LIMIT as u64 + 1)
        .read_to_end(&mut json)?;
    if json.len() > LIMIT {
        return Err(invalid("observation exceeds byte cap").into());
    }
    let mut observation: Observation = serde_json::from_slice(&json)?;
    if observation.schema != SCHEMA || observation.tensor_sha256 != digest {
        return Err(invalid("observation schema or tensor digest mismatch").into());
    }
    let matrix = HostQteaLinear::from_tensor(tensor, LIMIT);
    let count = observation
        .batch
        .checked_mul(matrix.rows())
        .filter(|n| n.checked_mul(4).is_some_and(|bytes| bytes <= LIMIT))
        .ok_or_else(|| invalid("output geometry exceeds cap"))?;
    if observation.batch.checked_mul(matrix.columns()) != Some(observation.input_bits.len()) {
        return Err(invalid("input geometry mismatch").into());
    }
    let input: Vec<_> = observation
        .input_bits
        .iter()
        .map(|bits| f32::from_bits(*bits))
        .collect();
    let mut output = Vec::new();
    output.try_reserve_exact(count)?;
    output.resize(count, 0.);
    let projection = Projection::HostQtea(std::sync::Arc::new(matrix));
    projection.forward(
        &tritium_cpu::CpuBackend::new(),
        &input,
        observation.batch,
        &mut output,
    )?;
    observation.output_bits = output.iter().map(|value| value.to_bits()).collect();
    serde_json::to_writer(io::stdout().lock(), &observation)?;
    Ok(())
}
