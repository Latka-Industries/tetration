//! Shared Zarr v2 / v3 dtype name tables for convert and export.

use super::dtype::ElementDtype;

/// Zarr v3 `data_type` name for an element dtype.
#[must_use]
pub fn element_dtype_zarr_v3_name(elem: ElementDtype) -> &'static str {
    match elem {
        ElementDtype::F32 => "float32",
        ElementDtype::F64 => "float64",
        ElementDtype::I32 => "int32",
        ElementDtype::I64 => "int64",
        ElementDtype::U8 => "uint8",
        ElementDtype::U16 => "uint16",
        ElementDtype::I16 => "int16",
        ElementDtype::U32 => "uint32",
        ElementDtype::F16 => "float16",
        ElementDtype::U64 => "uint64",
    }
}

/// Parse a Zarr v3 `data_type` tag (string or `{ "name": … }`) into [`ElementDtype`].
///
/// Returns `None` when the tag is missing or unsupported.
#[must_use]
pub fn element_dtype_from_zarr_v3(data_type: Option<&serde_json::Value>) -> Option<ElementDtype> {
    let tag = data_type.and_then(|v| v.as_str()).or_else(|| {
        data_type
            .and_then(|v| v.get("name"))
            .and_then(|v| v.as_str())
    })?;
    match tag {
        "float16" => Some(ElementDtype::F16),
        "float32" => Some(ElementDtype::F32),
        "float64" => Some(ElementDtype::F64),
        "uint8" | "bool" | "int8" => Some(ElementDtype::U8),
        "uint16" => Some(ElementDtype::U16),
        "int16" => Some(ElementDtype::I16),
        "int32" => Some(ElementDtype::I32),
        "int64" => Some(ElementDtype::I64),
        "uint32" => Some(ElementDtype::U32),
        "uint64" => Some(ElementDtype::U64),
        _ => None,
    }
}

/// Result of parsing a Zarr v2 NumPy-style dtype string.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ZarrV2DtypeParse {
    /// Little-endian (or endian-neutral) dtype we support.
    Ok(ElementDtype),
    /// Big-endian dtype (`>` prefix) — refused for import.
    BigEndian,
    /// Unrecognized dtype string.
    Unsupported,
}

/// Parse a Zarr v2 dtype string (`<f4`, `|u1`, …).
#[must_use]
pub fn parse_zarr_v2_dtype(dtype: &str) -> ZarrV2DtypeParse {
    if dtype.starts_with('>') {
        return ZarrV2DtypeParse::BigEndian;
    }
    match dtype {
        "<f2" => ZarrV2DtypeParse::Ok(ElementDtype::F16),
        "<f4" => ZarrV2DtypeParse::Ok(ElementDtype::F32),
        "<f8" => ZarrV2DtypeParse::Ok(ElementDtype::F64),
        "|u1" | "<u1" | "|b1" | "|i1" | "<i1" => ZarrV2DtypeParse::Ok(ElementDtype::U8),
        "<u2" => ZarrV2DtypeParse::Ok(ElementDtype::U16),
        "<i2" => ZarrV2DtypeParse::Ok(ElementDtype::I16),
        "<i4" => ZarrV2DtypeParse::Ok(ElementDtype::I32),
        "<i8" => ZarrV2DtypeParse::Ok(ElementDtype::I64),
        "<u4" => ZarrV2DtypeParse::Ok(ElementDtype::U32),
        "<u8" => ZarrV2DtypeParse::Ok(ElementDtype::U64),
        _ => ZarrV2DtypeParse::Unsupported,
    }
}
