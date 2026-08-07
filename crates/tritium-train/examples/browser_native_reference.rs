//! Produce exact native CPU artifact for physical-browser training qualification.

use std::error::Error;
use std::fs::OpenOptions;
use std::io::Write as _;
use std::path::Path;

use half::f16;
use tritium_format::salt_v2::SaltV2Codec;
use tritium_format::salt_v2_package::{
    SaltV2Package, SaltV2Plane, SaltV2Tensor, SaltV2Tile, write_salt_v2_package,
};
use tritium_spec::{
    TrainAttributeV1, TrainAttributeValueV1, TrainBackendV1, TrainBufferDataMutV1,
    TrainBufferDataRefV1, TrainExecutionV1, TrainNamedBufferMutV1, TrainNamedBufferRefV1,
    TrainOutputV1, TrainReceiptV1, TrainRequestV1, TrainingOpManifestV2,
};
use tritium_train::CpuTrainBackendV1;

const WIDTH: usize = 256;
const PLANES: usize = 2;

struct NativeReference {
    artifact: Vec<u8>,
    reloaded: Vec<u8>,
    export_receipt: TrainReceiptV1,
    reload_receipt: TrainReceiptV1,
}

fn execute_f32(
    backend: &CpuTrainBackendV1,
    operation: &'static str,
    execution: TrainExecutionV1,
    inputs: &[TrainNamedBufferRefV1<'_>],
    attributes: &[TrainAttributeV1],
    output_name: &'static str,
    output_shape: &[u64],
) -> Result<(Vec<f32>, TrainReceiptV1), Box<dyn Error>> {
    let request = TrainRequestV1::new(operation, execution, inputs, attributes);
    let output_len = output_shape
        .iter()
        .try_fold(1_u64, |product, dimension| product.checked_mul(*dimension))
        .ok_or("native output shape overflows")?;
    let output_len = usize::try_from(output_len)?;
    let mut values = vec![0.0_f32; output_len];
    let receipt = {
        let mut buffers = [TrainNamedBufferMutV1::new(
            output_name,
            output_shape,
            TrainBufferDataMutV1::F32(&mut values),
        )];
        let mut output = TrainOutputV1::new(&mut buffers);
        backend.execute(request, &mut output)?
    };
    Ok((values, receipt))
}

fn execute_bytes(
    backend: &CpuTrainBackendV1,
    operation: &'static str,
    execution: TrainExecutionV1,
    input_name: &'static str,
    input: &[u8],
    output_name: &'static str,
) -> Result<(Vec<u8>, TrainReceiptV1), Box<dyn Error>> {
    let shape = [u64::try_from(input.len())?];
    let inputs = [TrainNamedBufferRefV1::new(
        input_name,
        &shape,
        TrainBufferDataRefV1::Bytes(input),
    )];
    let attributes = [TrainAttributeV1::new(
        "format",
        TrainAttributeValueV1::Text("salt_v2_package_v1"),
    )];
    let request = TrainRequestV1::new(operation, execution, &inputs, &attributes);
    let mut values = vec![0_u8; input.len()];
    let receipt = {
        let mut buffers = [TrainNamedBufferMutV1::new(
            output_name,
            &shape,
            TrainBufferDataMutV1::Bytes(&mut values),
        )];
        let mut output = TrainOutputV1::new(&mut buffers);
        backend.execute(request, &mut output)?
    };
    Ok((values, receipt))
}

fn fit_package(parameter: &[f32]) -> Result<Vec<u8>, Box<dyn Error>> {
    if parameter.len() != WIDTH || parameter.iter().any(|value| !value.is_finite()) {
        return Err("native browser reference parameter is malformed".into());
    }
    let mut residual = parameter.to_vec();
    let mut planes = Vec::with_capacity(PLANES);
    for _ in 0..PLANES {
        let mut sum = 0.0_f32;
        for value in &residual {
            sum += value.abs();
        }
        let scale_f16 = f16::from_f32(sum / WIDTH as f32);
        let scale = scale_f16.to_f32();
        if !scale.is_finite() || (sum != 0.0 && scale == 0.0) {
            return Err("native browser reference scale is not representable".into());
        }
        let mut trits = Vec::with_capacity(WIDTH);
        for value in &mut residual {
            let ratio = *value / scale;
            let trit = if ratio >= 0.5 {
                1_i8
            } else if ratio <= -0.5 {
                -1_i8
            } else {
                0_i8
            };
            trits.push(trit);
            let contribution = scale * f32::from(trit);
            *value -= contribution;
        }
        planes.push(SaltV2Plane::new(
            trits,
            vec![scale_f16; WIDTH.div_ceil(128)],
        )?);
    }
    let tile = SaltV2Tile::new(planes)?;
    let tensor = SaltV2Tensor::new("weight", vec![1, WIDTH as u64], vec![tile])?;
    let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor])?;
    Ok(write_salt_v2_package(&package)?.bytes)
}

fn produce_reference() -> Result<NativeReference, Box<dyn Error>> {
    let backend = CpuTrainBackendV1::new();
    let weight = (0..WIDTH)
        .map(|index| (index as i32 % 9 - 4) as f32 / 8.0)
        .collect::<Vec<_>>();
    let shape = [1, WIDTH as u64];
    let salt_attributes = [
        TrainAttributeV1::new("rows", TrainAttributeValueV1::U64(1)),
        TrainAttributeV1::new("cols", TrainAttributeValueV1::U64(WIDTH as u64)),
        TrainAttributeV1::new("planes", TrainAttributeValueV1::U64(PLANES as u64)),
    ];
    let salt_inputs = [TrainNamedBufferRefV1::new(
        "weight",
        &shape,
        TrainBufferDataRefV1::F32(&weight),
    )];
    let (quantized, _) = execute_f32(
        &backend,
        "graph.salt_ste",
        TrainExecutionV1::Forward,
        &salt_inputs,
        &salt_attributes,
        "result",
        &shape,
    )?;

    let target = vec![0.0_f32; WIDTH];
    let mse_forward_inputs = [
        TrainNamedBufferRefV1::new("prediction", &shape, TrainBufferDataRefV1::F32(&quantized)),
        TrainNamedBufferRefV1::new("target", &shape, TrainBufferDataRefV1::F32(&target)),
    ];
    let (loss, _) = execute_f32(
        &backend,
        "loss.mse",
        TrainExecutionV1::Forward,
        &mse_forward_inputs,
        &[],
        "result",
        &[],
    )?;
    if loss.len() != 1 || !loss[0].is_finite() {
        return Err("native browser reference loss is malformed".into());
    }
    let grad_output = [1.0_f32];
    let mse_inputs = [
        TrainNamedBufferRefV1::new("prediction", &shape, TrainBufferDataRefV1::F32(&quantized)),
        TrainNamedBufferRefV1::new("target", &shape, TrainBufferDataRefV1::F32(&target)),
        TrainNamedBufferRefV1::new("grad_output", &[], TrainBufferDataRefV1::F32(&grad_output)),
    ];
    let (grad_quantized, _) = execute_f32(
        &backend,
        "loss.mse",
        TrainExecutionV1::Vjp,
        &mse_inputs,
        &[],
        "grad_prediction",
        &shape,
    )?;

    let salt_vjp_inputs = [
        TrainNamedBufferRefV1::new("weight", &shape, TrainBufferDataRefV1::F32(&weight)),
        TrainNamedBufferRefV1::new(
            "grad_output",
            &shape,
            TrainBufferDataRefV1::F32(&grad_quantized),
        ),
    ];
    let (gradient, _) = execute_f32(
        &backend,
        "graph.salt_ste",
        TrainExecutionV1::Vjp,
        &salt_vjp_inputs,
        &salt_attributes,
        "grad_weight",
        &shape,
    )?;

    let step_inputs = [
        TrainNamedBufferRefV1::new("parameter", &shape, TrainBufferDataRefV1::F32(&weight)),
        TrainNamedBufferRefV1::new("gradient", &shape, TrainBufferDataRefV1::F32(&gradient)),
    ];
    let step_attributes = [
        TrainAttributeV1::new("step", TrainAttributeValueV1::U64(1)),
        TrainAttributeV1::new("lr", TrainAttributeValueV1::F32(0.1)),
    ];
    let (updated, _) = execute_f32(
        &backend,
        "optimizer.sgd",
        TrainExecutionV1::Step,
        &step_inputs,
        &step_attributes,
        "parameter",
        &shape,
    )?;

    let package = fit_package(&updated)?;
    let (artifact, export_receipt) = execute_bytes(
        &backend,
        "lifecycle.export",
        TrainExecutionV1::Export,
        "package",
        &package,
        "artifact",
    )?;
    let (reloaded, reload_receipt) = execute_bytes(
        &backend,
        "lifecycle.reload",
        TrainExecutionV1::Reload,
        "artifact",
        &artifact,
        "package",
    )?;
    if package != artifact || artifact != reloaded {
        return Err("native SALT export/reload changed artifact bytes".into());
    }
    Ok(NativeReference {
        artifact,
        reloaded,
        export_receipt,
        reload_receipt,
    })
}

fn write_new(path: &Path, bytes: &[u8]) -> Result<(), Box<dyn Error>> {
    let mut file = OpenOptions::new().write(true).create_new(true).open(path)?;
    file.write_all(bytes)?;
    file.sync_all()?;
    Ok(())
}

fn hex(bytes: &[u8]) -> String {
    const DIGITS: &[u8; 16] = b"0123456789abcdef";
    let mut output = String::with_capacity(bytes.len() * 2);
    for byte in bytes {
        output.push(DIGITS[(byte >> 4) as usize] as char);
        output.push(DIGITS[(byte & 0x0f) as usize] as char);
    }
    output
}

fn main() -> Result<(), Box<dyn Error>> {
    let mut args = std::env::args_os().skip(1);
    let artifact_path = args
        .next()
        .ok_or("usage: browser_native_reference ARTIFACT_PATH RELOADED_PATH")?;
    let reloaded_path = args
        .next()
        .ok_or("usage: browser_native_reference ARTIFACT_PATH RELOADED_PATH")?;
    if args.next().is_some() {
        return Err("usage: browser_native_reference ARTIFACT_PATH RELOADED_PATH".into());
    }
    let reference = produce_reference()?;
    write_new(Path::new(&artifact_path), &reference.artifact)?;
    write_new(Path::new(&reloaded_path), &reference.reloaded)?;
    let physical_device = reference
        .reload_receipt
        .physical_device
        .as_deref()
        .ok_or("native CPU receipt omitted physical device")?;
    if reference.export_receipt.backend_build != reference.reload_receipt.backend_build
        || reference.export_receipt.backend_id != reference.reload_receipt.backend_id
        || reference.export_receipt.physical_device.as_deref() != Some(physical_device)
        || reference.export_receipt.manifest_digest != TrainingOpManifestV2::digest()
        || reference.reload_receipt.manifest_digest != TrainingOpManifestV2::digest()
    {
        return Err("native lifecycle receipt identity drifted".into());
    }
    println!("backend_id={}", reference.reload_receipt.backend_id);
    println!(
        "backend_build_hex={}",
        hex(reference.reload_receipt.backend_build.as_bytes())
    );
    println!("physical_device_hex={}", hex(physical_device.as_bytes()));
    println!(
        "manifest_digest={}",
        hex(&reference.reload_receipt.manifest_digest)
    );
    println!("export_operation={}", reference.export_receipt.operation);
    println!(
        "export_input_digest={}",
        hex(&reference.export_receipt.input_digest)
    );
    println!(
        "export_output_digest={}",
        hex(&reference.export_receipt.output_digest)
    );
    println!(
        "export_peak_resident_bytes={}",
        reference.export_receipt.peak_resident_bytes
    );
    println!(
        "export_scratch_bytes={}",
        reference.export_receipt.scratch_bytes
    );
    println!(
        "export_host_transfers={}",
        reference.export_receipt.host_transfers
    );
    println!(
        "export_device_resident={}",
        reference.export_receipt.device_resident
    );
    println!("reload_operation={}", reference.reload_receipt.operation);
    println!(
        "reload_input_digest={}",
        hex(&reference.reload_receipt.input_digest)
    );
    println!(
        "reload_output_digest={}",
        hex(&reference.reload_receipt.output_digest)
    );
    println!(
        "reload_peak_resident_bytes={}",
        reference.reload_receipt.peak_resident_bytes
    );
    println!(
        "reload_scratch_bytes={}",
        reference.reload_receipt.scratch_bytes
    );
    println!(
        "reload_host_transfers={}",
        reference.reload_receipt.host_transfers
    );
    println!(
        "reload_device_resident={}",
        reference.reload_receipt.device_resident
    );
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn native_reference_runs_one_step_and_round_trips_exact_salt() {
        let reference = produce_reference().expect("native reference");
        assert_eq!(reference.artifact, reference.reloaded);
        assert_eq!(reference.export_receipt.operation, "lifecycle.export");
        assert_eq!(reference.reload_receipt.operation, "lifecycle.reload");
        assert!(reference.export_receipt.device_resident);
        assert!(reference.reload_receipt.device_resident);
        assert_eq!(reference.export_receipt.host_transfers, 0);
        assert_eq!(reference.reload_receipt.host_transfers, 0);
        assert_eq!(
            hex(&reference.artifact),
            concat!(
                "54534c5432504b470100020001000004e0000000000000000600000002000000",
                "0001000000000000010000000000000068000000000000000800000000000000",
                "0000000000000000000000000000000000000000000000007765696768740100",
                "00000000000000010000000000006c4f751ac908e553ee6c4f751ac908e553ee",
                "6c4f751ac908e553ee6c4f751ac908e553ee6c4f751ac908e553ee6c4f751ac9",
                "08795a4dc0199159d21de85a4dc0199159d21de85a4dc0199159d21de85a4dc0",
                "199159d21de85a4dc0199159d21de85a4dc01991597873347334b82eb82e0000",
            )
        );
    }
}
