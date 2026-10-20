//! Shared HDF5 type mapping for convert (dataset tiles, attrs, and coordinate variables).
//!
//! [`map_hdf5_element_dtype`] is used when planning datasets; [`element_dtype_from_hdf5_dataset`]
//! is the infallible probe for 1-D coordinate arrays.

use hdf5_metno::types::{FloatSize, IntSize, TypeDescriptor};

use crate::utils::dtype::ElementDtype;

use super::ConvertError;

/// Map an HDF5 type descriptor to a supported v1 element dtype, if any.
#[must_use]
pub(crate) fn element_dtype_from_hdf5_descriptor(desc: &TypeDescriptor) -> Option<ElementDtype> {
    match desc {
        TypeDescriptor::Float(FloatSize::U4) => Some(ElementDtype::F32),
        TypeDescriptor::Float(FloatSize::U8) => Some(ElementDtype::F64),
        TypeDescriptor::Boolean
        | TypeDescriptor::Integer(IntSize::U1)
        | TypeDescriptor::Unsigned(IntSize::U1) => Some(ElementDtype::U8),
        TypeDescriptor::Integer(IntSize::U2) => Some(ElementDtype::I16),
        TypeDescriptor::Integer(IntSize::U4) => Some(ElementDtype::I32),
        TypeDescriptor::Integer(IntSize::U8) => Some(ElementDtype::I64),
        TypeDescriptor::Unsigned(IntSize::U2) => Some(ElementDtype::U16),
        TypeDescriptor::Unsigned(IntSize::U4) => Some(ElementDtype::U32),
        TypeDescriptor::Unsigned(IntSize::U8) => Some(ElementDtype::U64),
        _ => None,
    }
}

/// Element dtype for a dataset, if the HDF5 type is supported on the v1 wire.
#[must_use]
pub(crate) fn element_dtype_from_hdf5_dataset(ds: &hdf5_metno::Dataset) -> Option<ElementDtype> {
    let desc = ds.dtype().ok()?.to_descriptor().ok()?;
    element_dtype_from_hdf5_descriptor(&desc)
}

/// # Errors
///
/// Returns [`ConvertError::Hdf5`] or [`ConvertError::UnsupportedDtype`].
pub(crate) fn map_hdf5_element_dtype(
    ds: &hdf5_metno::Dataset,
    name: &str,
) -> Result<ElementDtype, ConvertError> {
    let td = ds.dtype()?;
    let desc = td.to_descriptor()?;
    element_dtype_from_hdf5_descriptor(&desc).ok_or(ConvertError::UnsupportedDtype {
        name: name.to_owned(),
        detail: format!("{desc:?}"),
    })
}
