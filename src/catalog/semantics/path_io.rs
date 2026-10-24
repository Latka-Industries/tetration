//! Shared path / summary helpers for apply + wizard (+ declare).

use std::fs;
use std::path::Path;

use crate::catalog::{CatalogError, TetFileSummaryV1, read_tet_summary_v1};

/// Refuse sealed `.tet.d` packages / generation paths (mutable `.tet` only).
pub(crate) fn ensure_mutable_tet(path: &Path, what: &str) -> Result<(), CatalogError> {
    if path.is_dir() || path.join("PROTOCOL").is_file() || ancestor_is_package(path) {
        return Err(CatalogError::InvalidSemantics(format!(
            "{what} targets a mutable .tet file; a sealed .tet.d generation is immutable — use the mutable .tet, then seal a new generation"
        )));
    }
    Ok(())
}

pub(crate) fn ancestor_is_package(path: &Path) -> bool {
    let mut cur = path.parent();
    while let Some(dir) = cur {
        if dir.join("PROTOCOL").is_file() {
            return true;
        }
        cur = dir.parent();
    }
    false
}

pub(crate) fn read_summary_from_path(path: &Path) -> Result<TetFileSummaryV1, CatalogError> {
    let data = fs::read(path)?;
    read_tet_summary_v1(&data)
}
