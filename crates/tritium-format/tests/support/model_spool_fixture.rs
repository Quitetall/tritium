//! Prepared source inventory outside reader/writer allocation measurement.
use tritium_format::trit_model::decode_model_manifest;
use tritium_format::{TritSpoolBlob, read_trit_package};
use tritium_schema::ModelManifest;

pub(super) fn prepared(bytes: &[u8]) -> (ModelManifest, Vec<TritSpoolBlob>) {
    let package = read_trit_package(bytes).unwrap();
    (
        decode_model_manifest(package.manifest()).unwrap(),
        package
            .blobs()
            .iter()
            .map(|blob| TritSpoolBlob {
                kind: blob.kind.clone(),
                id: blob.id,
                offset: blob.offset,
                length: blob.length,
            })
            .collect(),
    )
}
