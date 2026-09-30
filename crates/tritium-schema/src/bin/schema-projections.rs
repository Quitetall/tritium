use std::{
    env,
    error::Error,
    fs,
    path::{Path, PathBuf},
};

use schemars::JsonSchema;
use tritium_schema::{
    AdditiveLayout, AdmittedLaw, Basis, BlobId, LayoutError, ModelId, PackageId, PlaneAllocation,
    PlaneCodec, PlaneRelation, ScaleAnchor, ScaleLaw, ScalePrecision, SchemaId,
    SemanticTensorDigest, Transport, UnknownReason, Verdict,
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
    project!(LayoutError, "layout-error");
    project!(ModelId, "model-id");
    project!(PackageId, "package-id");
    project!(PlaneAllocation, "plane-allocation");
    project!(PlaneCodec, "plane-codec");
    project!(PlaneRelation, "plane-relation");
    project!(ScaleAnchor, "scale-anchor");
    project!(ScaleLaw, "scale-law");
    project!(ScalePrecision, "scale-precision");
    project!(SchemaId, "schema-id");
    project!(SemanticTensorDigest, "semantic-tensor-digest");
    project!(Transport, "transport");
    project!(UnknownReason, "unknown-reason");
    project!(Verdict, "verdict");

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
