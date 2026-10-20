//! `s3://` convert.
//!
//! A single object (`.h5`, `.nc`, `.npy`, `.csv`, `.parquet`, …) is downloaded whole, then
//! passed to the existing importer. A Zarr v2 or v3 prefix is read one metadata object and
//! one chunk object at a time (THI-516). Credentials stay in `TETRATION_S3_*`.

use std::path::Path;

#[cfg(feature = "tetration-s3")]
use std::sync::{Arc, OnceLock};

#[cfg(feature = "tetration-s3")]
use super::convert_to_tet_with_options;
use super::sniff::{
    CsvConvertInput, Hdf5ConvertInput, NetcdfConvertInput, NumpyConvertInput, ParquetConvertInput,
    ZarrConvertInput,
};
#[cfg(feature = "tetration-s3")]
use super::zarr_io::{ZarrObjects, zarr_child_names_from_keys};
use super::{ConvertError, ConvertOptions, ConvertProgress, ConvertReport};
use crate::catalog::store::{GenerationPin, TetOpenTarget, parse_tet_uri};

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

enum S3ConvertKind {
    File(S3ConvertObject),
    Zarr { bucket: String, prefix: String },
}

/// Parse `s3://bucket/key` into a single-file convert source.
///
/// Zarr prefixes (trailing slash, a `.zarr` name, or a key with no extension) are rejected
/// here. [`convert_s3_object_to_tet`] reads those stores chunk by chunk.
///
/// # Errors
///
/// Malformed URI, a Zarr prefix, or an extension the single-file importers do not take.
pub fn parse_s3_convert_object(uri: &str) -> Result<S3ConvertObject, ConvertError> {
    let (bucket, key) = s3_bucket_key(uri)?;
    if zarr_convert_candidate(&key) {
        return Err(ConvertError::S3(format!(
            "s3 URI `{uri}` is a Zarr prefix; convert_s3_object_to_tet reads it one chunk at a time"
        )));
    }
    parse_single_file(&bucket, &key, uri)
}

fn s3_bucket_key(uri: &str) -> Result<(String, String), ConvertError> {
    let parsed = parse_tet_uri(uri.trim()).map_err(|err| ConvertError::S3(err.to_string()))?;
    let TetOpenTarget::S3 {
        bucket,
        key,
        generation,
    } = parsed.target
    else {
        return Err(ConvertError::S3(format!("not an s3 URI: {uri}")));
    };
    if matches!(generation, GenerationPin::Explicit(_)) {
        return Err(ConvertError::S3(
            "s3 convert does not accept ?generation= (that query pins a tet package)".into(),
        ));
    }
    Ok((bucket, key))
}

fn classify_s3_convert(uri: &str) -> Result<S3ConvertKind, ConvertError> {
    let (bucket, key) = s3_bucket_key(uri)?;
    if key.trim_end_matches('/').is_empty() {
        return Err(ConvertError::S3(format!(
            "s3 convert URI needs a key: {uri}"
        )));
    }
    if zarr_convert_candidate(&key) {
        return Ok(S3ConvertKind::Zarr {
            bucket,
            prefix: key.trim_end_matches('/').to_string(),
        });
    }
    Ok(S3ConvertKind::File(parse_single_file(&bucket, &key, uri)?))
}

/// Trailing slash, `*.zarr`, or a final segment with no `.` (a store prefix, not `file.ext`).
fn zarr_convert_candidate(key: &str) -> bool {
    if key.ends_with('/') {
        return true;
    }
    let name = key.rsplit('/').next().unwrap_or(key);
    if name.is_empty() {
        return true;
    }
    if Path::new(name)
        .extension()
        .and_then(|ext| ext.to_str())
        .is_some_and(|ext| ext.eq_ignore_ascii_case("zarr"))
    {
        return true;
    }
    !name.contains('.')
}

fn parse_single_file(bucket: &str, key: &str, uri: &str) -> Result<S3ConvertObject, ConvertError> {
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
        return Err(ConvertError::S3(format!(
            "s3 URI `{uri}` is a Zarr prefix; convert_s3_object_to_tet reads it one chunk at a time"
        )));
    }
    if !single_file_extension(&ext) {
        return Err(ConvertError::S3(format!(
            "unsupported s3 convert object `{file_name}` (single-file extensions: .h5, .nc, .npy, .npz, .csv, .tsv, .parquet)"
        )));
    }
    Ok(S3ConvertObject {
        bucket: bucket.to_owned(),
        key: key.to_owned(),
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

/// Convert one `s3://` object or Zarr prefix into a `.tet`.
///
/// A single file is downloaded whole, then imported. A Zarr v2 or v3 prefix fetches metadata,
/// then each chunk object as the corresponding tile is written. The output matches a local
/// convert of the same store. Credentials and the HTTP endpoint come from the environment:
/// `TETRATION_S3_ENDPOINT` (required), `TETRATION_S3_ACCESS_KEY`, `TETRATION_S3_SECRET_KEY`,
/// `TETRATION_S3_REGION` (default `us-east-1`). The bucket is the one in the URI.
///
/// The report's `input` is the `s3://` URI.
///
/// # Errors
///
/// Missing feature, bad URI, missing endpoint, a prefix that is not Zarr, store I/O, or importer failure.
pub fn convert_s3_object_to_tet(
    uri: &str,
    output: &Path,
    options: ConvertOptions,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    match classify_s3_convert(uri)? {
        S3ConvertKind::Zarr { bucket, prefix } => {
            convert_s3_zarr(&bucket, &prefix, uri.trim(), output, options, progress)
        }
        S3ConvertKind::File(object) => {
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
    }
}

fn convert_s3_zarr(
    bucket: &str,
    prefix: &str,
    uri: &str,
    output: &Path,
    options: ConvertOptions,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    #[cfg(not(feature = "tetration-s3"))]
    {
        let _ = (bucket, prefix, uri, output, options, progress);
        Err(ConvertError::S3FeatureDisabled)
    }
    #[cfg(feature = "tetration-s3")]
    {
        convert_s3_zarr_enabled(bucket, prefix, uri, output, options, progress)
    }
}

#[cfg(feature = "tetration-s3")]
fn convert_s3_zarr_enabled(
    bucket: &str,
    prefix: &str,
    uri: &str,
    output: &Path,
    options: ConvertOptions,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    use super::shared::chunk_target_bytes;
    use super::zarr::convert_zarr_from_objects;
    use crate::catalog::store::S3Store;

    let config = s3_config(bucket, prefix)?;
    let store = S3Store::connect(config)?;
    if !s3_prefix_is_zarr(&store)? {
        return Err(ConvertError::S3(format!(
            "{uri} is not a Zarr v2 or v3 store (no zarr.json, .zgroup, or .zarray at the prefix)"
        )));
    }
    let objects = Arc::new(S3ZarrObjects {
        store,
        uri: uri.trim_end_matches('/').to_owned(),
        keys: OnceLock::new(),
    });
    let chunk_bytes = chunk_target_bytes(options.chunk_mib)?;
    let mut report = convert_zarr_from_objects(
        Path::new(uri),
        objects,
        output,
        options.parallel_jobs,
        options.strict,
        chunk_bytes,
        progress,
    )?;
    uri.clone_into(&mut report.input);
    Ok(report)
}

#[cfg(feature = "tetration-s3")]
fn s3_config(bucket: &str, prefix: &str) -> Result<crate::catalog::store::S3Config, ConvertError> {
    use crate::catalog::store::S3Config;

    S3Config::from_env(bucket.to_owned(), prefix.to_owned()).map_err(|err| {
        if err.to_string().contains("TETRATION_S3_ENDPOINT") {
            ConvertError::S3ConfigMissing("TETRATION_S3_ENDPOINT")
        } else {
            ConvertError::S3(err.to_string())
        }
    })
}

#[cfg(feature = "tetration-s3")]
fn s3_prefix_is_zarr(store: &crate::catalog::store::S3Store) -> Result<bool, ConvertError> {
    use crate::catalog::store::ReadAt;

    for key in ["zarr.json", ".zgroup", ".zarray"] {
        if store
            .object_meta(key)
            .map_err(|err| ConvertError::S3(err.to_string()))?
            .is_some()
        {
            return Ok(true);
        }
    }
    Ok(false)
}

#[cfg(feature = "tetration-s3")]
struct S3ZarrObjects {
    store: crate::catalog::store::S3Store,
    uri: String,
    keys: OnceLock<Vec<String>>,
}

#[cfg(feature = "tetration-s3")]
impl S3ZarrObjects {
    fn listed_keys(&self) -> Result<&Vec<String>, ConvertError> {
        use crate::catalog::store::ReadAt;

        if let Some(keys) = self.keys.get() {
            return Ok(keys);
        }
        let keys = self
            .store
            .list_prefix("")
            .map_err(|err| ConvertError::Zarr(format!("listing {}: {err}", self.uri)))?;
        Ok(self.keys.get_or_init(|| keys))
    }
}

#[cfg(feature = "tetration-s3")]
impl ZarrObjects for S3ZarrObjects {
    fn read_bytes(&self, rel: &str) -> Result<Vec<u8>, ConvertError> {
        use crate::catalog::store::ReadAt;

        self.store
            .read_all(rel)
            .map_err(|err| ConvertError::Zarr(format!("{}: {err}", self.location(rel))))
    }

    fn is_file(&self, rel: &str) -> Result<bool, ConvertError> {
        use crate::catalog::store::ReadAt;

        Ok(self
            .store
            .object_meta(rel)
            .map_err(|err| ConvertError::Zarr(err.to_string()))?
            .is_some())
    }

    fn child_names(&self, rel: &str, v3: bool) -> Result<Vec<String>, ConvertError> {
        Ok(zarr_child_names_from_keys(self.listed_keys()?, rel, v3))
    }

    fn location(&self, rel: &str) -> String {
        if rel.is_empty() {
            self.uri.clone()
        } else {
            format!("{}/{}", self.uri.trim_end_matches('/'), rel)
        }
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
    use crate::catalog::store::{ReadAt, S3Store};

    let config = s3_config(&object.bucket, "")?;
    let store = S3Store::connect(config)?;
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
