//! `s3://` single-file convert: download the whole object, then the existing importer (THI-502).
//!
//! Zarr prefixes are rejected. Chunked Zarr-on-S3 reads are a later issue (THI-516).

use std::path::Path;

#[cfg(feature = "tetration-s3")]
use super::convert_to_tet_with_options;
use super::sniff::{
    CsvConvertInput, Hdf5ConvertInput, NetcdfConvertInput, NumpyConvertInput, ParquetConvertInput,
    ZarrConvertInput,
};
use super::{ConvertError, ConvertOptions, ConvertProgress, ConvertReport};
use crate::catalog::store::{TetOpenTarget, parse_tet_uri};

/// `true` when `input` is an `s3://` URI.
#[must_use]
pub fn is_s3_convert_uri(input: &str) -> bool {
    input.trim().starts_with("s3://")
}

/// Bucket and object key accepted as a single-file convert source.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3ConvertObject {
    /// Bucket from `s3://bucket/key`.
    pub bucket: String,
    /// Object key after the bucket.
    pub key: String,
    /// Final path segment, used as the temp file name so the importer stem matches a local file.
    pub file_name: String,
}

/// Parse `s3://bucket/key` into a single-file convert source.
///
/// # Errors
///
/// Malformed URI, a trailing slash, a `.zarr` name, or an extension the single-file importers do not take.
pub fn parse_s3_convert_object(uri: &str) -> Result<S3ConvertObject, ConvertError> {
    let parsed = parse_tet_uri(uri.trim()).map_err(|err| ConvertError::S3(err.to_string()))?;
    let TetOpenTarget::S3 { bucket, key, .. } = parsed.target else {
        return Err(ConvertError::S3(format!("not an s3 URI: {uri}")));
    };
    if key.ends_with('/') {
        return Err(ConvertError::S3ZarrNotSupported {
            uri: uri.trim().to_string(),
        });
    }
    let file_name = key
        .rsplit('/')
        .next()
        .filter(|name| !name.is_empty() && *name != "." && *name != "..")
        .ok_or_else(|| ConvertError::S3(format!("s3 object key has no file name: {key}")))?
        .to_string();
    let ext = Path::new(&file_name)
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_ascii_lowercase();
    if ZarrConvertInput::EXTENSIONS.contains(&ext.as_str()) {
        return Err(ConvertError::S3ZarrNotSupported {
            uri: uri.trim().to_string(),
        });
    }
    if !single_file_extension(&ext) {
        return Err(ConvertError::S3(format!(
            "unsupported s3 convert object `{file_name}` (single-file extensions only; Zarr-on-S3 is later)"
        )));
    }
    Ok(S3ConvertObject {
        bucket,
        key,
        file_name,
    })
}

fn single_file_extension(ext: &str) -> bool {
    Hdf5ConvertInput::EXTENSIONS.contains(&ext)
        || NetcdfConvertInput::EXTENSIONS.contains(&ext)
        || NumpyConvertInput::NPY_EXTENSIONS.contains(&ext)
        || NumpyConvertInput::NPZ_EXTENSIONS.contains(&ext)
        || CsvConvertInput::CSV_EXTENSIONS.contains(&ext)
        || CsvConvertInput::TSV_EXTENSIONS.contains(&ext)
        || ParquetConvertInput::EXTENSIONS.contains(&ext)
}

/// Download `uri` to a temp file and convert it with the local importer.
///
/// Credentials and the HTTP endpoint come from the environment:
/// `TETRATION_S3_ENDPOINT` (required), `TETRATION_S3_ACCESS_KEY`, `TETRATION_S3_SECRET_KEY`,
/// `TETRATION_S3_REGION` (default `us-east-1`). The bucket is the one in the URI.
///
/// The report's `input` is the `s3://` URI. The temp copy is removed when this returns.
///
/// # Errors
///
/// Missing feature, bad URI, Zarr key, missing endpoint, store I/O, or importer failure.
pub fn convert_s3_object_to_tet(
    uri: &str,
    output: &Path,
    options: ConvertOptions,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let object = parse_s3_convert_object(uri)?;
    #[cfg(not(feature = "tetration-s3"))]
    {
        let _ = (object, output, options, progress);
        Err(ConvertError::S3FeatureDisabled)
    }
    #[cfg(feature = "tetration-s3")]
    {
        convert_s3_object_to_tet_enabled(&object, uri.trim(), output, options, progress)
    }
}

#[cfg(feature = "tetration-s3")]
fn convert_s3_object_to_tet_enabled(
    object: &S3ConvertObject,
    uri: &str,
    output: &Path,
    options: ConvertOptions,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    use crate::catalog::store::{ReadAt, S3Config, S3Store};

    let endpoint = std::env::var("TETRATION_S3_ENDPOINT")
        .ok()
        .filter(|s| !s.is_empty())
        .ok_or(ConvertError::S3ConfigMissing("TETRATION_S3_ENDPOINT"))?;
    let access_key_id =
        std::env::var("TETRATION_S3_ACCESS_KEY").unwrap_or_else(|_| "tetration".to_string());
    let secret_access_key =
        std::env::var("TETRATION_S3_SECRET_KEY").unwrap_or_else(|_| "tetration-secret".to_string());
    let region = std::env::var("TETRATION_S3_REGION").unwrap_or_else(|_| "us-east-1".to_string());
    let store = S3Store::connect(S3Config {
        endpoint,
        bucket: object.bucket.clone(),
        prefix: String::new(),
        region,
        access_key_id,
        secret_access_key,
    })
    .map_err(|err| ConvertError::S3(err.to_string()))?;
    let bytes = store
        .read_all(&object.key)
        .map_err(|err| ConvertError::S3(err.to_string()))?;

    let scratch = temp_scratch_dir()?;
    let staged = scratch.join(&object.file_name);
    std::fs::write(&staged, &bytes).map_err(|err| ConvertError::S3(err.to_string()))?;
    let mut report = convert_to_tet_with_options(&staged, output, options, progress);
    let _ = std::fs::remove_dir_all(&scratch);
    if let Ok(report) = report.as_mut() {
        report.input = uri.to_string();
    }
    report
}

#[cfg(feature = "tetration-s3")]
fn temp_scratch_dir() -> Result<std::path::PathBuf, ConvertError> {
    use std::time::{SystemTime, UNIX_EPOCH};

    let nanos = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_nanos());
    let dir = std::env::temp_dir().join(format!("tet-convert-s3-{}-{nanos}", std::process::id()));
    std::fs::create_dir_all(&dir).map_err(|err| ConvertError::S3(err.to_string()))?;
    Ok(dir)
}
