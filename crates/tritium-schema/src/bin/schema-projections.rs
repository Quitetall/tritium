use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use tritium_schema::{
    AdditiveLayout, AdmittedLaw, Basis, BlobId, BlobReference, DirectFitLevel, DirectFitLimits,
    DirectFitPlan, DirectFitProfile, EvidenceEnvelope, EvidenceReference, JointGroupFitOptions,
    LadderFitOptions, LadderScaleRefit, LayoutError, LegacyModelIdentity, Level, LevelFit,
    LoadedSourceIdentity, ManifestAsset, ModelId, ModelLoaded, ModelManifest, ModelSemanticMeta,
    ModelShard, NamedAssetDigest, NamedTensorDigest, PackageId, PlaneAllocation, PlaneCodec,
    PlaneCounts, PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision, SchemaId,
    SemanticTensorDigest, TensorCaps, TensorExecution, TensorMeta, TensorSemanticMeta,
    TensorStorage, TensorUploadPolicy, TensorUploaded, Transport, UnknownReason, Verdict,
};

fn main() -> Result<(), Box<dyn Error>> {
    let check = match env::args().nth(1).as_deref() {
        None => false,
        Some("--check") => true,
        Some(other) => return Err(format!("unknown argument {other:?}; expected --check").into()),
    };
    if env::args().nth(2).is_some() {
        return Err("accepts at most one argument (--check)".into());
    }

    let workspace = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../..");
    let output_dir = workspace.join("schemas/json/v1");
    if !check {
        fs::create_dir_all(&output_dir)?;
    }

    macro_rules! project {
        ($type:ty, $name:literal) => {
            write_projection::<$type>(&output_dir, $name, check)?;
        };
    }

    project!(AdditiveLayout, "additive-layout");
    project!(AdmittedLaw, "admitted-law");
    project!(Basis, "basis");
    project!(BlobId, "blob-id");
    project!(EvidenceEnvelope<serde_json::Value>, "evidence-envelope");
    project!(LayoutError, "layout-error");
    project!(ModelId, "model-id");
    project!(tritium_schema::OriginalSourceId, "original-source-id");
    project!(tritium_schema::CalibrationCacheId, "calibration-cache-id");
    project!(
        tritium_schema::OriginalSalienceCaptureProfile,
        "original-salience-capture-profile"
    );
    project!(
        tritium_schema::OriginalSalienceCaptureLimits,
        "original-salience-capture-limits"
    );
    project!(
        tritium_schema::OriginalSalienceCaptureInput,
        "original-salience-capture-input"
    );
    project!(
        tritium_schema::OriginalSalienceCapturePair,
        "original-salience-capture-pair"
    );
    project!(
        tritium_schema::OriginalSalienceCapturePlan,
        "original-salience-capture-plan"
    );
    project!(
        tritium_schema::OriginalSalienceCaptureStarted,
        "original-salience-capture-started"
    );
    project!(
        tritium_schema::OriginalSalienceCaptureOutput,
        "original-salience-capture-output"
    );
    project!(
        tritium_schema::OriginalSalienceCaptureCompleted,
        "original-salience-capture-completed"
    );
    project!(
        tritium_schema::CalibrationCaptureProfile,
        "calibration-capture-profile"
    );
    project!(
        tritium_schema::CalibrationCaptureLimits,
        "calibration-capture-limits"
    );
    project!(
        tritium_schema::CalibrationCaptureInput,
        "calibration-capture-input"
    );
    project!(
        tritium_schema::CalibrationCapturePlan,
        "calibration-capture-plan"
    );
    project!(
        tritium_schema::CalibrationCaptureStarted,
        "calibration-capture-started"
    );
    project!(
        tritium_schema::CalibrationCaptureCompleted,
        "calibration-capture-completed"
    );
    project!(
        tritium_schema::CalibrationCacheLimits,
        "calibration-cache-limits"
    );
    project!(
        tritium_schema::CalibrationCacheRequest,
        "calibration-cache-request"
    );
    project!(
        tritium_schema::CalibrationCacheBinding,
        "calibration-cache-binding"
    );
    project!(PackageId, "package-id");
    project!(
        tritium_schema::CalibrationCacheManifest,
        "calibration-cache-manifest"
    );
    project!(PlaneAllocation, "plane-allocation");
    project!(PlaneCodec, "plane-codec");
    project!(PlaneRelation, "plane-relation");
    project!(ScaleAnchor, "scale-anchor");
    project!(ScaleLaw, "scale-law");
    project!(ScalePrecision, "scale-precision");
    project!(SchemaId, "schema-id");
    project!(SemanticTensorDigest, "semantic-tensor-digest");
    project!(TensorCaps, "tensor-caps");
    project!(TensorExecution, "tensor-execution");
    project!(TensorUploadPolicy, "tensor-upload-policy");
    project!(TensorUploaded, "tensor-uploaded");
    project!(Transport, "transport");
    project!(UnknownReason, "unknown-reason");
    project!(Verdict, "verdict");
    project!(BlobReference, "blob-reference");
    project!(EvidenceReference, "evidence-reference");
    project!(LegacyModelIdentity, "legacy-model-identity");
    project!(Level, "level");
    project!(LevelFit, "level-fit");
    project!(ManifestAsset, "manifest-asset");
    project!(ModelManifest, "model-manifest");
    project!(ModelShard, "model-shard");
    project!(PlaneCounts, "plane-counts");
    project!(TensorMeta, "tensor-meta");
    project!(TensorStorage, "tensor-storage");
    project!(TensorSemanticMeta, "tensor-semantic-meta");
    project!(ModelSemanticMeta, "model-semantic-meta");
    project!(NamedAssetDigest, "named-asset-digest");
    project!(NamedTensorDigest, "named-tensor-digest");
    project!(LoadedSourceIdentity, "loaded-source-identity");
    project!(ModelLoaded, "model-loaded");
    project!(DirectFitPlan, "direct-fit-plan");
    project!(DirectFitProfile, "direct-fit-profile");
    project!(DirectFitLevel, "direct-fit-level");
    project!(DirectFitLimits, "direct-fit-limits");
    project!(LadderFitOptions, "ladder-fit-options");
    project!(LadderScaleRefit, "ladder-scale-refit");
    project!(JointGroupFitOptions, "joint-group-fit-options");
    project!(tritium_schema::WorkPrecision, "work-precision");
    project!(
        tritium_schema::JointFeedbackFitOptions,
        "joint-feedback-fit-options"
    );
    project!(tritium_schema::FitInputDigest, "fit-input-digest");
    project!(tritium_schema::CalibrationSource, "calibration-source");
    project!(
        tritium_schema::CalibrationPartition,
        "calibration-partition"
    );
    project!(tritium_schema::CalibrationOutput, "calibration-output");
    project!(tritium_schema::CalibrationConsumer, "calibration-consumer");
    project!(tritium_schema::CalibrationPlan, "calibration-plan");
    project!(tritium_schema::CalibrationFactor, "calibration-factor");
    project!(tritium_schema::CalibrationFactors, "calibration-factors");
    project!(tritium_schema::DirectFitMetric, "direct-fit-metric");
    project!(tritium_schema::DirectFitOutput, "direct-fit-output");
    project!(tritium_schema::DirectLevelsFitted, "direct-levels-fitted");
    project!(tritium_schema::CalibratedFitPlan, "calibrated-fit-plan");
    project!(
        tritium_schema::CalibratedLevelsFitted,
        "calibrated-levels-fitted"
    );
    project!(tritium_schema::QuantSourceTensor, "quant-source-tensor");
    project!(tritium_schema::QuantAction, "quant-action");
    project!(tritium_schema::QuantTensorPlan, "quant-tensor-plan");
    project!(tritium_schema::QuantLimits, "quant-limits");
    project!(tritium_schema::QuantPlan, "quant-plan");
    project!(tritium_schema::SalienceInverseAxis, "salience-inverse-axis");
    project!(tritium_schema::SalienceNumerics, "salience-numerics");
    project!(
        tritium_schema::SalienceStatisticsKind,
        "salience-statistics-kind"
    );
    project!(
        tritium_schema::OriginalCapturedSalienceSourceStageManifest,
        "original-captured-salience-source-stage-manifest"
    );
    project!(
        tritium_schema::OriginalCapturedSalienceConversionManifest,
        "original-captured-salience-conversion-manifest"
    );
    project!(
        tritium_schema::SalienceMomentBinding,
        "salience-moment-binding"
    );
    project!(
        tritium_schema::SalienceTransformOperation,
        "salience-transform-operation"
    );
    project!(
        tritium_schema::SalienceTransformLimits,
        "salience-transform-limits"
    );
    project!(
        tritium_schema::SalienceTransformPlan,
        "salience-transform-plan"
    );
    project!(
        tritium_schema::SalienceOperationOutput,
        "salience-operation-output"
    );
    project!(
        tritium_schema::SalienceTransformObservation,
        "salience-transform-observation"
    );
    project!(
        tritium_schema::SalienceTransformStarted,
        "salience-transform-started"
    );
    project!(
        tritium_schema::SalienceTransformCompleted,
        "salience-transform-completed"
    );
    project!(
        tritium_schema::SalienceSourceStageManifest,
        "salience-source-stage-manifest"
    );
    project!(tritium_schema::SalienceStageLimits, "salience-stage-limits");
    project!(
        tritium_schema::SalienceConversionManifest,
        "salience-conversion-manifest"
    );
    project!(
        tritium_schema::CapturedSalienceConversionManifest,
        "captured-salience-conversion-manifest"
    );
    project!(
        tritium_schema::SalienceConversionLimits,
        "salience-conversion-limits"
    );
    project!(
        tritium_schema::CalibratedQuantAction,
        "calibrated-quant-action"
    );
    project!(
        tritium_schema::CalibratedQuantTensorPlan,
        "calibrated-quant-tensor-plan"
    );
    project!(tritium_schema::CalibratedQuantPlan, "calibrated-quant-plan");
    project!(
        tritium_schema::CalibratedQuantizationStarted,
        "calibrated-quantization-started"
    );
    project!(
        tritium_schema::CalibratedQuantizationCompleted,
        "calibrated-quantization-completed"
    );
    project!(tritium_schema::QuantSourceBinding, "quant-source-binding");
    project!(tritium_schema::QuantOutputLevel, "quant-output-level");
    project!(tritium_schema::QuantizationStarted, "quantization-started");
    project!(
        tritium_schema::QuantizationCompleted,
        "quantization-completed"
    );

    Ok(())
}

fn write_projection<T: JsonSchema>(
    directory: &Path,
    name: &str,
    check: bool,
) -> Result<(), Box<dyn Error>> {
    let path = directory.join(format!("{name}.schema.json"));
    let schema = schemars::schema_for!(T);
    let rendered = format!("{}\n", serde_json::to_string_pretty(&schema)?);
    if check {
        let existing = fs::read_to_string(&path)?;
        if existing != rendered {
            return Err(format!(
                "{} is stale; rerun the projection generator",
                path.display()
            )
            .into());
        }
    } else {
        fs::write(&path, rendered)?;
    }
    Ok(())
}
