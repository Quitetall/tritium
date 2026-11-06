//! Shared checked scalar staging for distinct cache profiles; no identity domain.
#[derive(Debug)]
pub(crate) enum ScalarError {
    Allocation,
    Length,
}
pub(crate) fn reserved<T>(count: usize) -> Result<Vec<T>, ScalarError> {
    let mut values = Vec::new();
    values
        .try_reserve_exact(count)
        .map_err(|_| ScalarError::Allocation)?;
    Ok(values)
}
pub(crate) fn check_len(bytes: &[u8], count: usize, width: usize) -> Result<(), ScalarError> {
    if width == 0 || count.checked_mul(width) != Some(bytes.len()) {
        return Err(ScalarError::Length);
    }
    Ok(())
}
pub(crate) fn encode<T, const N: usize>(
    values: &[T],
    convert: impl Fn(&T) -> [u8; N],
) -> Result<Vec<u8>, ScalarError> {
    if N == 0 {
        return Err(ScalarError::Length);
    }
    let mut bytes = reserved(values.len().checked_mul(N).ok_or(ScalarError::Allocation)?)?;
    for value in values {
        bytes.extend_from_slice(&convert(value));
    }
    Ok(bytes)
}
pub(crate) fn decode<T, const N: usize>(
    bytes: &[u8],
    count: usize,
    convert: impl Fn([u8; N]) -> T,
) -> Result<Vec<T>, ScalarError> {
    check_len(bytes, count, N)?;
    let mut values = reserved(count)?;
    values.extend(bytes.as_chunks::<N>().0.iter().map(|chunk| convert(*chunk)));
    Ok(values)
}
pub(crate) fn digest<T, const N: usize>(
    values: &[T],
    convert: impl Fn(&T) -> [u8; N],
) -> tritium_schema::BlobId {
    let mut hash = blake3::Hasher::new();
    for value in values {
        hash.update(&convert(value));
    }
    tritium_schema::BlobId::from_bytes(*hash.finalize().as_bytes())
}
