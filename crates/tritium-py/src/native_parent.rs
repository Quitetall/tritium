//! Distinct native descriptors behind one actual-byte immediate-parent seam.
use crate::{model_conversion::NativeModelConversion, model_refinement::NativeModelRefinement};
use pyo3::{exceptions::PyTypeError, prelude::*};
use std::{fs::File, path::Path};
use tritium_evidence::NativeRefinementParentView;
use tritium_format::trit_model::TritModelReader;

#[derive(Clone)]
pub(crate) enum NativeTrainingParent {
    Conversion(NativeModelConversion),
    Refined(NativeModelRefinement),
}
impl From<NativeModelConversion> for NativeTrainingParent {
    fn from(v: NativeModelConversion) -> Self {
        Self::Conversion(v)
    }
}
impl From<NativeModelRefinement> for NativeTrainingParent {
    fn from(v: NativeModelRefinement) -> Self {
        Self::Refined(v)
    }
}
impl NativeTrainingParent {
    pub(crate) fn extract(value: &Bound<'_, PyAny>) -> PyResult<Self> {
        if let Ok(v) = value.extract::<PyRef<'_, NativeModelConversion>>() {
            return Ok(Self::Conversion((*v).clone()));
        }
        if let Ok(v) = value.extract::<PyRef<'_, NativeModelRefinement>>() {
            return Ok(Self::Refined((*v).clone()));
        }
        Err(PyTypeError::new_err(
            "actual NativeModelConversion or NativeModelRefinement parent required",
        ))
    }
    pub(crate) fn root(&self) -> &Path {
        match self {
            Self::Conversion(v) => v.root(),
            Self::Refined(v) => v.root(),
        }
    }
    pub(crate) fn refinement_parent_view(
        &self,
        level: &str,
    ) -> PyResult<(TritModelReader<File>, NativeRefinementParentView)> {
        match self {
            Self::Conversion(v) => v.refinement_parent_view(level),
            Self::Refined(v) => v.refinement_parent_view(level),
        }
    }
    pub(crate) fn admission(&self) -> &'static str {
        match self {
            Self::Conversion(_) => "conversion-record-consistent-unqualified",
            Self::Refined(_) => "refined-record-consistent-unqualified",
        }
    }
}
