//! Canonical selected-owner encoding and preserved-payload staging for all adapters.
use super::*;
pub(in crate::pv_tuning::native_scale) struct Staged {
    pub(super) blobs: Vec<TritSpoolBlob>,
    pub(super) selected: Vec<NativeRefinementTensorOutput>,
    pub(super) outputs: Vec<NativeRefinementTensorOutput>,
}
pub(in crate::pv_tuning::native_scale) fn spool_bytes<R: Read + Seek>(
    parent: &TritModelReader<R>,
    plan: &NativeRefinementPlan,
) -> Result<u64> {
    let level = parent
        .manifest()
        .levels
        .iter()
        .find(|l| l.name == plan.parent.level)
        .ok_or(NativeScaleChildError::Binding(
            "native child parent level absent",
        ))?;
    let mut bytes = 0;
    for meta in &level.tensors {
        kind(&meta.storage)?;
        bytes = add(bytes, parent.tensor_byte_len(&level.name, &meta.name)?)?;
    }
    for asset in &parent.manifest().assets {
        bytes = add(bytes, parent.asset_byte_len(&asset.name)?)?;
    }
    Ok(bytes)
}
pub(in crate::pv_tuning::native_scale) fn encode_owner<S: Write + Seek>(
    spool: &mut S,
    weight: &tritium_format::AdditiveTensor,
    length: u64,
) -> Result<TritSpoolBlob> {
    let bytes = weight.encode_payload()?;
    if bytes.len() as u64 != length {
        return Err(NativeScaleChildError::Binding(
            "native child encoded tensor length differs",
        ));
    }
    let offset = spool.seek(SeekFrom::End(0))?;
    for chunk in bytes.chunks(8192) {
        spool.write_all(chunk)?;
    }
    Ok(TritSpoolBlob {
        kind: TritBlobKind::AdditiveTensor,
        id: blob(&bytes),
        offset,
        length,
    })
}
pub(in crate::pv_tuning::native_scale) fn copy_preserved<R: Read + Seek, S: Read + Write + Seek>(
    plan: &NativeRefinementPlan,
    parent: &mut TritModelReader<R>,
    spool: &mut S,
    selected_blobs: &[TritSpoolBlob],
    expected_bytes: u64,
) -> Result<Staged> {
    if selected_blobs.len() != plan.owners.len() {
        return Err(NativeScaleChildError::Binding(
            "native child selected inventory differs",
        ));
    }
    let level = parent
        .manifest()
        .levels
        .iter()
        .position(|l| l.name == plan.parent.level)
        .ok_or(NativeScaleChildError::Binding(
            "native child parent level absent",
        ))?;
    let mut blobs = Vec::new();
    let mut selected = Vec::new();
    let mut outputs = Vec::new();
    for i in 0..parent.manifest().levels[level].tensors.len() {
        let meta = &parent.manifest().levels[level].tensors[i];
        let (name, original, storage) = (meta.name.clone(), meta.blob.id, kind(&meta.storage)?);
        let length = parent.tensor_byte_len(&plan.parent.level, &name)?;
        let (offset, content) = if let Ok(index) = plan.owners.binary_search(&name) {
            let b = &selected_blobs[index];
            if b.length != length {
                return Err(NativeScaleChildError::Binding(
                    "native child selected length differs",
                ));
            }
            selected.push(NativeRefinementTensorOutput {
                name: name.clone(),
                content: b.id,
            });
            (b.offset, b.id)
        } else {
            let offset = spool.seek(SeekFrom::End(0))?;
            parent.visit_tensor(&plan.parent.level, &name, |chunk| spool.write_all(chunk))?;
            (offset, original)
        };
        blobs.push(TritSpoolBlob {
            kind: storage,
            id: content,
            offset,
            length,
        });
        outputs.push(NativeRefinementTensorOutput { name, content });
    }
    for i in 0..parent.manifest().assets.len() {
        let a = &parent.manifest().assets[i];
        let (name, id) = (a.name.clone(), a.blob.id);
        let length = parent.asset_byte_len(&name)?;
        let offset = spool.seek(SeekFrom::End(0))?;
        parent.visit_asset(&name, |chunk| spool.write_all(chunk))?;
        blobs.push(TritSpoolBlob {
            kind: TritBlobKind::Asset(name),
            id,
            offset,
            length,
        });
    }
    if spool.seek(SeekFrom::End(0))? != expected_bytes {
        return Err(NativeScaleChildError::Binding(
            "native child spool length differs",
        ));
    }
    spool.flush()?;
    Ok(Staged {
        blobs,
        selected,
        outputs,
    })
}
