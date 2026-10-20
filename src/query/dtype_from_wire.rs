//! Map catalog wire dtype tags to [`ElementDtype`] with query validation errors.

use crate::catalog::DATASET_DTYPE_TAG_V1;
use crate::utils::dtype::ElementDtype;

use super::types::TetError;

impl ElementDtype {
    /// Parse a catalog wire `dtype` tag.
    ///
    /// # Errors
    ///
    /// Returns [`TetError::Validation`] for unsupported tags.
    pub fn from_wire(dtype: u32) -> Result<Self, TetError> {
        Self::try_from_wire_tag(dtype).ok_or_else(|| {
            let tags = DATASET_DTYPE_TAG_V1;
            TetError::Validation(format!(
                "unsupported dataset dtype {dtype} (supported: f32={}, f64={}, i32={}, i64={}, u8={}, u16={}, i16={}, u32={}, f16={}, u64={})",
                tags.f32,
                tags.f64,
                tags.i32,
                tags.i64,
                tags.u8,
                tags.u16,
                tags.i16,
                tags.u32,
                tags.f16,
                tags.u64
            ))
        })
    }
}
