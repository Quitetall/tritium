//! Load a DFlash drafter (`z-lab/*-DFlash` layout: `config.json` and one
//! `model.safetensors`) onto a resident Qwen3.6 executor.

use std::fs::File;
use std::io::BufReader;
use std::path::Path;

use tritium_cuda::{DFlashConfig, Qwen35Resident};
use tritium_format::SafeTensorsReader;
use tritium_spec::BackendError;

use crate::NnError;

fn config_error(message: impl Into<String>) -> NnError {
    NnError::InvalidArtifact(message.into())
}

/// Read a DFlash drafter's `config.json` into the executor's [`DFlashConfig`].
///
/// # Errors
/// Rejects a missing or malformed config.
pub fn dflash_config(dir: &Path) -> Result<DFlashConfig, NnError> {
    let text = std::fs::read_to_string(dir.join("config.json"))
        .map_err(|error| config_error(format!("read drafter config: {error}")))?;
    let json: serde_json::Value = serde_json::from_str(&text)
        .map_err(|error| config_error(format!("parse drafter config: {error}")))?;
    let usize_of = |key: &str| {
        json[key]
            .as_u64()
            .map(|value| value as usize)
            .ok_or_else(|| config_error(format!("drafter config lacks {key}")))
    };
    let head_dim = usize_of("head_dim")?;
    if head_dim != 128 {
        return Err(config_error("drafter head_dim must be 128"));
    }
    let dflash = &json["dflash_config"];
    let target_layers = dflash["target_layer_ids"]
        .as_array()
        .ok_or_else(|| config_error("drafter config lacks dflash_config.target_layer_ids"))?
        .iter()
        .map(|id| id.as_u64().map(|id| id as usize))
        .collect::<Option<Vec<_>>>()
        .ok_or_else(|| config_error("bad target_layer_ids"))?;
    let sliding = json["layer_types"]
        .as_array()
        .ok_or_else(|| config_error("drafter config lacks layer_types"))?
        .iter()
        .map(|kind| kind.as_str() == Some("sliding_attention"))
        .collect();
    Ok(DFlashConfig {
        hidden: usize_of("hidden_size")?,
        heads: usize_of("num_attention_heads")?,
        kv_heads: usize_of("num_key_value_heads")?,
        intermediate: usize_of("intermediate_size")?,
        target_layers,
        block_size: usize_of("block_size")?,
        mask_token: dflash["mask_token_id"]
            .as_u64()
            .ok_or_else(|| config_error("drafter config lacks mask_token_id"))?
            as u32,
        rope_theta: json["rope_theta"].as_f64().unwrap_or(10_000.0) as f32,
        rms_eps: json["rms_norm_eps"].as_f64().unwrap_or(1e-6) as f32,
        sliding_window: json["sliding_window"].as_u64().unwrap_or(0) as usize,
        sliding,
    })
}

/// Load the drafter in `dir` onto `executor` (see
/// [`Qwen35Resident::attach_dflash`]), optionally with a smaller verify block.
///
/// # Errors
/// Rejects a malformed drafter or returns the executor's error.
pub fn attach_dflash_drafter(
    executor: &mut Qwen35Resident,
    dir: &Path,
    block_size: Option<usize>,
) -> Result<(), NnError> {
    let mut config = dflash_config(dir)?;
    if let Some(block) = block_size {
        config.block_size = block;
    }
    let file = File::open(dir.join("model.safetensors"))
        .map_err(|error| config_error(format!("open drafter weights: {error}")))?;
    let mut reader = SafeTensorsReader::new(BufReader::new(file))
        .map_err(|error| config_error(format!("read drafter weights: {error}")))?;
    let mut fetch = |name: &str| -> Result<(Vec<f32>, Vec<usize>), BackendError> {
        let shape = reader
            .shape(name)
            .ok_or_else(|| BackendError::InvalidInput(format!("drafter lacks tensor {name}")))?
            .to_vec();
        let values = reader
            .tensor_f32(name)
            .map_err(|error| BackendError::InvalidInput(format!("read drafter {name}: {error}")))?;
        Ok((values, shape))
    };
    executor.attach_dflash(config, &mut fetch)?;
    Ok(())
}
