//! Dense, host-packed, or CUDA-resident token table shared by embedding gather
//! and a tied LM head.

use core::mem::size_of;
use std::sync::Arc;

use rayon::prelude::*;
use tritium_format::PackedSaltRow;
use tritium_spec::TernaryBackend;

use crate::error::NnError;
use crate::layers::HostSaltV2Linear;
use crate::layers::packed_salt::PackedSaltMatrix;
#[cfg(feature = "cuda")]
use crate::layers::projection::{salt_v2_cuda_backend, salt_v2_forward_exact};

#[derive(Clone, Debug)]
enum Storage {
    Dense(Vec<f32>),
    Salt(PackedSaltMatrix),
    HostSaltV2(Arc<HostSaltV2Linear>),
    #[cfg(feature = "cuda")]
    SaltV2(Arc<tritium_cuda::SaltV2ResidentTensor>),
}

/// One token table used for both input embedding gather and exact tied-head logits.
///
/// Dense fp/BitNet models retain their original fp32 table. SALT models retain one
/// host-packed matrix or one physically encoded CUDA allocation; gather and
/// unembedding execute against that same storage, so tying never creates a second
/// `vocab × hidden` allocation.
#[derive(Clone, Debug)]
pub struct TokenEmbedding {
    rows: usize,
    cols: usize,
    storage: Storage,
}

impl TokenEmbedding {
    /// Build a token table around compact host SALT V2 storage.
    ///
    /// # Errors
    /// Returns [`NnError::Shape`] when the matrix has empty geometry.
    pub fn from_host_salt_v2(tensor: Arc<HostSaltV2Linear>) -> Result<Self, NnError> {
        let rows = tensor.rows();
        let cols = tensor.columns();
        if rows == 0 || cols == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: rows.saturating_mul(cols),
            });
        }
        Ok(Self {
            rows,
            cols,
            storage: Storage::HostSaltV2(tensor),
        })
    }

    /// Build a token table around an already-published CUDA SALT V2 allocation.
    ///
    /// The [`Arc`] is also suitable for a tied [`Projection::SaltV2`](super::Projection),
    /// so embedding and unembedding can share the same physical allocation.
    ///
    /// # Errors
    /// [`NnError::Shape`] if the resident handle has an empty vocabulary or hidden width.
    #[cfg(feature = "cuda")]
    pub(crate) fn from_salt_v2_resident(
        tensor: Arc<tritium_cuda::SaltV2ResidentTensor>,
    ) -> Result<Self, NnError> {
        let rows = tensor.rows();
        let cols = tensor.columns();
        if rows == 0 || cols == 0 {
            return Err(NnError::Shape {
                expected: 1,
                got: rows.saturating_mul(cols),
            });
        }
        Ok(Self {
            rows,
            cols,
            storage: Storage::SaltV2(tensor),
        })
    }

    pub(crate) fn from_packed_matrix(
        matrix: PackedSaltMatrix,
        rows: usize,
        cols: usize,
    ) -> Result<Self, NnError> {
        if matrix.n_out() != rows || matrix.k_in() != cols {
            return Err(NnError::Shape {
                expected: rows.saturating_mul(cols),
                got: matrix.n_out().saturating_mul(matrix.k_in()),
            });
        }
        Ok(Self {
            rows,
            cols,
            storage: Storage::Salt(matrix),
        })
    }

    /// Build a dense fp32 token table.
    ///
    /// # Errors
    /// [`NnError::Shape`] if dimensions are zero, overflow, or disagree with `values`.
    pub fn from_dense(values: Vec<f32>, rows: usize, cols: usize) -> Result<Self, NnError> {
        let expected = rows.checked_mul(cols).ok_or(NnError::Shape {
            expected: usize::MAX,
            got: values.len(),
        })?;
        if rows == 0 || cols == 0 || values.len() != expected {
            return Err(NnError::Shape {
                expected: expected.max(1),
                got: values.len(),
            });
        }
        Ok(Self {
            rows,
            cols,
            storage: Storage::Dense(values),
        })
    }

    /// Build a token table from validated packed SALT rows.
    ///
    /// # Errors
    /// [`NnError::Shape`] if matrix geometry disagrees with the rows, or
    /// [`NnError::Backend`] if a packed plane contains a non-finite scale.
    pub fn from_packed_salt(
        rows_data: Vec<PackedSaltRow>,
        rows: usize,
        cols: usize,
    ) -> Result<Self, NnError> {
        Ok(Self {
            rows,
            cols,
            storage: Storage::Salt(PackedSaltMatrix::new(rows_data, rows, cols)?),
        })
    }

    /// Vocabulary rows.
    #[must_use]
    pub const fn rows(&self) -> usize {
        self.rows
    }

    /// Hidden columns.
    #[must_use]
    pub const fn cols(&self) -> usize {
        self.cols
    }

    /// Borrow dense values when this is an fp32 table.
    ///
    /// Packed SALT storage deliberately returns `None`; callers must use gather or
    /// exact unembedding instead of silently materializing a dense copy.
    #[must_use]
    pub fn as_dense(&self) -> Option<&[f32]> {
        match &self.storage {
            Storage::Dense(values) => Some(values),
            Storage::Salt(_) | Storage::HostSaltV2(_) => None,
            #[cfg(feature = "cuda")]
            Storage::SaltV2(_) => None,
        }
    }

    /// Whether this table retains additive SALT storage.
    ///
    /// The historical method name covers both host-packed SALT V1 rows and a
    /// physically encoded resident SALT V2 tensor.
    #[must_use]
    pub const fn is_packed_salt(&self) -> bool {
        match self.storage {
            Storage::Salt(_) | Storage::HostSaltV2(_) => true,
            #[cfg(feature = "cuda")]
            Storage::SaltV2(_) => true,
            Storage::Dense(_) => false,
        }
    }

    /// Number of host-side sparse residual planes. Dense fp32 and resident SALT V2
    /// return zero; the resident physical codec does not expose V1 sparse-plane metadata.
    #[must_use]
    pub const fn sparse_plane_count(&self) -> usize {
        match &self.storage {
            Storage::Dense(_) => 0,
            Storage::Salt(matrix) => matrix.sparse_plane_count(),
            Storage::HostSaltV2(_) => 0,
            #[cfg(feature = "cuda")]
            Storage::SaltV2(_) => 0,
        }
    }

    /// Retained payload and metadata bytes, excluding this wrapper.
    /// Cloned packed tables share arenas, so summing this value across clones double-counts them.
    #[must_use]
    pub fn resident_bytes(&self) -> usize {
        match &self.storage {
            Storage::Dense(values) => values.capacity().saturating_mul(size_of::<f32>()),
            Storage::Salt(matrix) => matrix.resident_bytes(),
            Storage::HostSaltV2(tensor) => tensor.resident_bytes(),
            #[cfg(feature = "cuda")]
            Storage::SaltV2(tensor) => {
                usize::try_from(tensor.allocation_receipt().steady_resident_bytes())
                    .unwrap_or(usize::MAX)
            }
        }
    }

    /// Gather token rows into `out`, which must have `tokens.len() × cols` elements.
    ///
    /// # Errors
    /// [`NnError::Shape`] for a wrong output length or [`NnError::MissingTensor`]
    /// for an out-of-vocabulary token.
    pub fn gather(&self, tokens: &[u32], out: &mut [f32]) -> Result<(), NnError> {
        match &self.storage {
            Storage::Salt(matrix) => matrix.gather(tokens, out),
            Storage::HostSaltV2(tensor) => tensor.gather_rows(tokens, out),
            #[cfg(feature = "cuda")]
            Storage::SaltV2(_) => Err(NnError::Backend(
                "resident SALT V2 embedding gather requires the backend-aware gather_with_backend"
                    .into(),
            )),
            Storage::Dense(values) => {
                let expected = tokens.len().checked_mul(self.cols).ok_or(NnError::Shape {
                    expected: usize::MAX,
                    got: out.len(),
                })?;
                if out.len() != expected {
                    return Err(NnError::Shape {
                        expected,
                        got: out.len(),
                    });
                }
                out.par_chunks_mut(self.cols)
                    .zip(tokens.par_iter())
                    .try_for_each(|(dst, &token)| {
                        let row = token as usize;
                        if row >= self.rows {
                            return Err(NnError::MissingTensor(format!("token_embd row {row}")));
                        }
                        let src = values
                            .get(row * self.cols..(row + 1) * self.cols)
                            .ok_or_else(|| {
                                NnError::MissingTensor(format!("token_embd row {row}"))
                            })?;
                        dst.copy_from_slice(src);
                        Ok::<(), NnError>(())
                    })
            }
        }
    }

    /// Gather token rows, dispatching resident SALT V2 storage through its owning
    /// CUDA backend while retaining the existing host behavior for dense/SALT V1.
    ///
    /// # Errors
    /// The errors documented by [`Self::gather`], or [`NnError::Backend`] when a
    /// resident table is paired with a non-CUDA/different-context backend.
    pub fn gather_with_backend(
        &self,
        backend: &dyn TernaryBackend,
        tokens: &[u32],
        out: &mut [f32],
    ) -> Result<(), NnError> {
        #[cfg(not(feature = "cuda"))]
        let _ = backend;
        match &self.storage {
            #[cfg(feature = "cuda")]
            Storage::SaltV2(tensor) => {
                let expected = tokens.len().checked_mul(self.cols).ok_or(NnError::Shape {
                    expected: usize::MAX,
                    got: out.len(),
                })?;
                if out.len() != expected {
                    return Err(NnError::Shape {
                        expected,
                        got: out.len(),
                    });
                }
                if let Some(row) = tokens
                    .iter()
                    .copied()
                    .map(u64::from)
                    .find(|&row| row >= self.rows as u64)
                {
                    return Err(NnError::MissingTensor(format!("token_embd row {row}")));
                }
                let cuda = salt_v2_cuda_backend(backend)?;
                let _receipt = cuda.salt_v2_gather_rows(tensor, tokens, out)?;
                Ok(())
            }
            Storage::Dense(_) | Storage::Salt(_) | Storage::HostSaltV2(_) => {
                self.gather(tokens, out)
            }
        }
    }

    /// Compute exact tied-head logits in global-K accumulation order.
    ///
    /// Packed weights are reconstructed one 256-element block at a time. No A8
    /// activation quantization is applied on this path.
    ///
    /// # Errors
    /// [`NnError::Shape`] if `hidden` or `logits` has the wrong length.
    pub fn unembed_exact(&self, hidden: &[f32], logits: &mut [f32]) -> Result<(), NnError> {
        if hidden.len() != self.cols {
            return Err(NnError::Shape {
                expected: self.cols,
                got: hidden.len(),
            });
        }
        if logits.len() != self.rows {
            return Err(NnError::Shape {
                expected: self.rows,
                got: logits.len(),
            });
        }
        match &self.storage {
            Storage::Salt(matrix) => matrix.project_exact(hidden, logits),
            Storage::HostSaltV2(tensor) => tensor.forward(hidden, 1, logits),
            #[cfg(feature = "cuda")]
            Storage::SaltV2(_) => Err(NnError::Backend(
                "resident SALT V2 unembedding requires the backend-aware unembed_exact_with_backend"
                    .into(),
            )),
            Storage::Dense(values) => {
                logits.par_iter_mut().enumerate().for_each(|(row, slot)| {
                    let weights = &values[row * self.cols..(row + 1) * self.cols];
                    let mut acc = 0.0f32;
                    for col in 0..self.cols {
                        acc += hidden[col] * weights[col];
                    }
                    *slot = acc;
                });
                Ok(())
            }
        }
    }

    /// Compute tied-head logits, dispatching resident SALT V2 storage through
    /// the same exact CUDA projection path used by [`Projection::SaltV2`](super::Projection).
    ///
    /// # Errors
    /// The errors documented by [`Self::unembed_exact`], or [`NnError::Backend`]
    /// when a resident table is paired with a non-CUDA/different-context backend.
    pub fn unembed_exact_with_backend(
        &self,
        backend: &dyn TernaryBackend,
        hidden: &[f32],
        logits: &mut [f32],
    ) -> Result<(), NnError> {
        #[cfg(not(feature = "cuda"))]
        let _ = backend;
        match &self.storage {
            #[cfg(feature = "cuda")]
            Storage::SaltV2(tensor) => salt_v2_forward_exact(backend, tensor, hidden, 1, logits),
            Storage::Dense(_) | Storage::Salt(_) | Storage::HostSaltV2(_) => {
                self.unembed_exact(hidden, logits)
            }
        }
    }
}

#[cfg(test)]
mod host_tests {
    use std::io::Cursor;
    use std::sync::Arc;

    use half::f16;
    use tritium_format::{
        salt_v2::SaltV2Codec,
        salt_v2_package::{
            SaltV2Package, SaltV2PackageReader, SaltV2Plane, SaltV2Tensor, SaltV2Tile,
            write_salt_v2_package,
        },
    };

    use super::TokenEmbedding;
    use crate::layers::HostSaltV2Linear;

    #[test]
    fn host_salt_v2_table_gathers_and_unembeds_from_shared_storage() {
        let plane = SaltV2Plane::new(vec![1, 0, -1, 1], vec![f16::from_f32(0.5)]).unwrap();
        let tensor = SaltV2Tensor::new(
            "token_embd.weight",
            vec![2, 2],
            vec![SaltV2Tile::new(vec![plane]).unwrap()],
        )
        .unwrap();
        let package = SaltV2Package::new(SaltV2Codec::B3, vec![tensor]).unwrap();
        let encoded = write_salt_v2_package(&package).unwrap();
        let mut reader = SaltV2PackageReader::new_strict(Cursor::new(encoded.bytes)).unwrap();
        let resident =
            Arc::new(HostSaltV2Linear::from_reader(&mut reader, "token_embd.weight").unwrap());
        let embedding = TokenEmbedding::from_host_salt_v2(Arc::clone(&resident)).unwrap();

        let mut gathered = [0.0; 4];
        embedding.gather(&[1, 0], &mut gathered).unwrap();
        assert_eq!(gathered, [-0.5, 0.5, 0.5, 0.0]);

        let mut logits = [0.0; 2];
        embedding.unembed_exact(&[2.0, -1.0], &mut logits).unwrap();
        assert_eq!(logits, [1.0, -1.5]);
        assert!(embedding.as_dense().is_none());
        assert_eq!(Arc::strong_count(&resident), 2);
    }
}

#[cfg(all(test, feature = "cuda"))]
mod cuda_tests {
    use std::sync::Arc;

    use half::f16;
    use tritium_format::salt_v2::SaltV2Codec;
    use tritium_format::salt_v2_package::{SaltV2Plane, SaltV2Tensor, SaltV2Tile};

    use super::TokenEmbedding;
    use crate::layers::Projection;

    #[test]
    fn resident_salt_v2_table_gathers_and_unembeds_through_its_cuda_backend() {
        let cuda = match tritium_cuda::CudaBackend::new(0) {
            Ok(backend) => backend,
            Err(error) => {
                eprintln!("skipping resident SALT V2 embedding test: no CUDA device ({error})");
                return;
            }
        };
        let plane = SaltV2Plane::new(vec![1, 0, -1, 0, 1, 1, -1, -1, 0], vec![f16::from_f32(0.5)])
            .expect("valid resident embedding plane");
        let tensor = SaltV2Tensor::new(
            "token_embd.weight",
            vec![3, 3],
            vec![SaltV2Tile::new(vec![plane]).expect("valid resident embedding tile")],
        )
        .expect("valid resident embedding tensor");
        let resident = Arc::new(
            cuda.upload_salt_v2(&tensor, SaltV2Codec::D2)
                .expect("upload resident embedding"),
        );
        let embedding = TokenEmbedding::from_salt_v2_resident(Arc::clone(&resident))
            .expect("build resident embedding");
        let tied_head = Projection::SaltV2(Arc::clone(&resident));

        assert_eq!(Arc::strong_count(&resident), 3);
        assert!(embedding.as_dense().is_none());
        assert_eq!((embedding.rows(), embedding.cols()), (3, 3));

        let mut gathered = vec![f32::NAN; 9];
        embedding
            .gather_with_backend(&cuda, &[2, 0, 2], &mut gathered)
            .expect("gather resident rows");
        assert_eq!(
            gathered,
            vec![-0.5, -0.5, 0.0, 0.5, 0.0, -0.5, -0.5, -0.5, 0.0]
        );

        let hidden = [2.0, -1.0, 4.0];
        let mut logits = vec![f32::NAN; 3];
        embedding
            .unembed_exact_with_backend(&cuda, &hidden, &mut logits)
            .expect("unembed through resident table");
        assert_eq!(logits, vec![-1.0, 1.5, -0.5]);

        let mut projection_logits = vec![f32::NAN; 3];
        tied_head
            .forward(&cuda, &hidden, 1, &mut projection_logits)
            .expect("forward through shared resident head");
        assert_eq!(projection_logits, logits);

        assert!(matches!(
            embedding.gather_with_backend(&cuda, &[0], &mut [0.0; 2]),
            Err(crate::NnError::Shape {
                expected: 3,
                got: 2
            })
        ));
        assert!(matches!(
            embedding.gather_with_backend(&cuda, &[3], &mut [0.0; 3]),
            Err(crate::NnError::MissingTensor(_))
        ));
        assert!(matches!(
            tied_head.forward(&cuda, &hidden, 1, &mut [0.0; 2]),
            Err(crate::NnError::Shape {
                expected: 3,
                got: 2
            })
        ));

        let error = embedding
            .gather(&[0], &mut [0.0; 3])
            .expect_err("host-only gather must reject resident storage");
        assert!(error.to_string().contains("backend-aware"));
    }
}
