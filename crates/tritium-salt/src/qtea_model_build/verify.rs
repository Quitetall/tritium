use super::*;

/// Check actual package/observation bindings, not original-byte custody, rerun
/// fitting, opaque admission authority, physical measurements or qualification.
/// Caller keeps the retained source immutable and performs its terminal verify.
/// # Errors
/// Missing/oversized/malformed record, source/owner/asset/shape/precision or
/// actual tensor content/numerical fingerprint/request mismatch.
pub fn verify_qtea_model_build_record<R: Read + Seek>(
    reader: &mut QteaModelReader<R>,
    max_record_bytes: usize,
) -> Result<QteaModelBuildRecord, QteaModelBuildError> {
    if reader.asset_bytes("qtea-fit.json").map_err(cause)? > max_record_bytes as u64 {
        return Err(fail("QTEA fit record receiver cap exceeded"));
    }
    let raw = reader.asset("qtea-fit.json").map_err(cause)?;
    let record: QteaModelBuildRecord = serde_json::from_slice(&raw).map_err(cause)?;
    if record.schema != "tritium.qtea.model-build-observation.v1"
        || record.qualification
        || record.campaign.as_bytes() == &[0; 32]
        || tritium_evidence::original_source_id(
            &record.architecture,
            &record.source_tensors,
            &record.source_assets,
        )
        .map_err(cause)?
            != record.source_model
        || reader.manifest().source_model != *record.source_model.as_bytes()
    {
        return Err(fail("QTEA build record source/scope mismatch"));
    }
    let manifest = reader.manifest().clone();
    if record
        .source_assets
        .iter()
        .any(|a| a.name == "qtea-fit.json")
    {
        return Err(fail(
            "reserved fit asset cannot be an original source asset",
        ));
    }
    if manifest.tensors.len() != record.source_tensors.len()
        || manifest.assets.len() != record.source_assets.len()
    {
        return Err(fail("QTEA build record complete inventory mismatch"));
    }
    for asset in &record.source_assets {
        let actual = if asset.name == "config.json" {
            Some(manifest.config)
        } else {
            manifest.assets.get(&asset.name).copied()
        };
        if actual != Some(asset.content) {
            return Err(fail("QTEA original asset binding mismatch"));
        }
    }
    let mut fits = BTreeMap::new();
    for fit in &record.fits {
        if fits.insert(fit.name.as_str(), fit).is_some() {
            return Err(fail("duplicate QTEA fit observation"));
        }
    }
    for (index, original) in record.source_tensors.iter().enumerate() {
        let name = &original.source.name;
        let shape = &original.source.shape;
        let tensor = manifest
            .tensors
            .get(name)
            .ok_or_else(|| fail("missing QTEA build owner"))?;
        match tensor {
            QteaModelTensor::Qtea {
                blob,
                rows,
                columns,
            } => {
                if shape.as_slice() != [*rows, *columns] {
                    return Err(fail("QTEA original matrix shape mismatch"));
                }
                let fit = fits
                    .remove(name.as_str())
                    .ok_or_else(|| fail("missing QTEA fit observation"))?;
                let actual = reader.qtea(name).map_err(cause)?;
                let view = actual.view();
                let meta = view.metadata();
                let numerical = tritium_quantize::qtea_reference::numerical_input_digest(meta)
                    .map_err(cause)?;
                let input_id = crate::qtea_tensor_work::work_input_id(
                    QteaTensorOrigin {
                        source_id: record.campaign,
                        owner: name,
                        tensor_index: u32::try_from(index).map_err(cause)?,
                        source_tensor_digest: *original.content.as_bytes(),
                    },
                    numerical,
                );
                let estimate = crate::FrontierResourceEstimate::new(
                    fit.estimate,
                    fit.machine_id,
                    fit.evidence_id,
                )
                .map_err(cause)?;
                let plan_id = crate::frontier_v3::admitted_plan_id(
                    fit.request.solver(),
                    fit.request.solver_request(),
                    estimate,
                );
                let output_id = crate::frontier_v3::qtea_output_id(
                    fit.request.content_id(),
                    fit.plan_id,
                    estimate,
                    fit.input_id,
                    fit.tensor_id,
                );
                if fit.source_tensor_index as usize != index
                    || fit.input_id != input_id
                    || fit.plan_id != plan_id
                    || fit.output_id != output_id
                    || fit.request.solver()
                        != crate::QteaReferenceAdapter::new()
                            .map_err(cause)?
                            .descriptor()
                    || fit.request.solver_request().rows() != *rows
                    || fit.request.solver_request().columns() != *columns
                    || fit.estimate != fit.request.estimate()
                    || fit.machine_id.as_bytes() == &[0; 32]
                    || fit.evidence_id.as_bytes() == &[0; 32]
                    || fit.blob_id != *blob
                    || fit.tensor_id != ContentId::of_bytes(actual.as_bytes())
                    || fit.recipe_digest != view.recipe_digest()
                    || fit.weights_digest != meta.bindings.weights
                    || fit.gram_digest != meta.bindings.gram
                    || fit.activation_cache != meta.bindings.activation_cache
                    || fit.token_stream != meta.bindings.token_stream
                    || fit.input_id != fit.request.input_id()
                    || fit.request.source_id() != record.campaign
                    || fit.request.stage() != crate::FrontierStage::Fit
                    || fit.request.solver().id().as_str()
                        != crate::FRONTIER_QTEA_REFERENCE_SOLVER_ID
                    || !matches!(
                        fit.curvature.as_str(),
                        "captured_full_matmul" | "native_supplied"
                    )
                {
                    return Err(fail(
                        "QTEA fitted bytes/numerical/request observation mismatch",
                    ));
                }
            }
            QteaModelTensor::Vector {
                blob,
                length,
                precision,
            } => {
                if shape.as_slice() != [*length]
                    || *blob != original.content
                    || *precision != original.source.precision
                {
                    return Err(fail("QTEA preserved vector/original binding mismatch"));
                }
                reader.vector(name).map_err(cause)?;
            }
            QteaModelTensor::Preserved {
                blob,
                shape: actual,
                precision,
            } => {
                if actual != shape
                    || *blob != original.content
                    || *precision != original.source.precision
                {
                    return Err(fail("QTEA preserved state/original binding mismatch"));
                }
                reader.preserved(name).map_err(cause)?;
            }
        }
    }
    if !fits.is_empty() {
        return Err(fail("unreferenced QTEA fit observations"));
    }
    reader.verify().map_err(cause)?;
    Ok(record)
}
