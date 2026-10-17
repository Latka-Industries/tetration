//! Parse tetration open targets (paths, `file://`, `memory://`, recognize `s3://`).
//!
//! Local suffixes `.tet` / `.tet.d` are matched case-insensitively so labels like
//! `ds@./Foo.TET` still split correctly.

use std::path::PathBuf;

use thiserror::Error;

/// Which sealed generation a package reader should pin.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GenerationPin {
    /// Read `CURRENT` once and pin that generation.
    Current,
    /// Pin an explicit generation id.
    Explicit(u64),
}

/// Parsed open target (no I/O).
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum TetOpenTarget {
    /// Standalone layout-v1 `.tet` (mmap path).
    LocalTet(PathBuf),
    /// `.tet.d` package root for [`crate::catalog::LocalStore`].
    LocalPackage {
        /// Package directory (usually ends with `.tet.d`).
        root: PathBuf,
        /// Which generation to pin when opening.
        generation: GenerationPin,
    },
    /// Object key inside a caller-supplied [`crate::catalog::PackageStore`] (tests / embedders).
    StorePackage {
        /// Which generation to pin when opening.
        generation: GenerationPin,
    },
    /// `s3://bucket/prefix` package root. [`parse_s3_package_uri`] turns this into a
    /// store prefix; `open_s3_package` (feature `tetration-s3`) pins and range-reads it.
    S3 {
        /// Bucket name from `s3://bucket/…`.
        bucket: String,
        /// Object key prefix / path after the bucket.
        key: String,
        /// Which generation to pin when opening.
        generation: GenerationPin,
    },
}

/// Optional CLI-style label (`dataset@uri`) plus target.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ParsedTetUri {
    /// Text before `@` when present (dataset hint; unused for I/O).
    pub label: Option<String>,
    /// Resolved open target (local file, package, memory, or S3 shape).
    pub target: TetOpenTarget,
}

/// URI / path parse failures.
#[derive(Debug, Error, PartialEq, Eq)]
pub enum TetUriError {
    #[error("empty tet URI")]
    Empty,
    #[error("unsupported URI scheme: {0}")]
    UnsupportedScheme(String),
    #[error("invalid s3 URI (need s3://bucket/key): {0}")]
    InvalidS3(String),
    #[error("invalid memory URI (use memory://current or memory://g/<id>): {0}")]
    InvalidMemory(String),
    #[error("invalid generation id: {0}")]
    InvalidGeneration(String),
    #[error("not an s3 package URI (need s3://bucket/prefix): {0}")]
    NotS3Package(String),
}

/// Package root inside an S3 bucket, parsed from `s3://` with no I/O.
///
/// `prefix` is the package root passed to `S3Config` (empty means the bucket root).
/// `?generation=N` pins that generation; otherwise the reader pins `CURRENT` once.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct S3PackageLocation {
    /// Text before `@` when the URI is `label@s3://…`.
    pub label: Option<String>,
    /// Bucket name.
    pub bucket: String,
    /// Package root inside the bucket. Empty is the bucket root.
    pub prefix: String,
    /// Which generation to pin when opening.
    pub generation: GenerationPin,
}

/// Parse `s3://bucket/prefix` or `label@s3://bucket/prefix?generation=N`.
///
/// The prefix is the package root (`CURRENT`, `generations/…`), not a single object key.
/// A trailing slash is stripped. Credentials are not in the URI.
///
/// # Errors
///
/// The input is not an `s3://` URI, or the prefix has an empty, `.`, or `..` segment.
pub fn parse_s3_package_uri(input: &str) -> Result<S3PackageLocation, TetUriError> {
    let parsed = parse_tet_uri(input)?;
    let TetOpenTarget::S3 {
        bucket,
        key,
        generation,
    } = parsed.target
    else {
        return Err(TetUriError::NotS3Package(input.trim().to_string()));
    };
    let prefix = normalize_package_prefix(&key)
        .map_err(|_| TetUriError::InvalidS3(input.trim().to_string()))?;
    Ok(S3PackageLocation {
        label: parsed.label,
        bucket,
        prefix,
        generation,
    })
}

/// Parse a path or URI into a [`ParsedTetUri`]. Does not touch the network or filesystem.
///
/// # Errors
///
/// Returns [`TetUriError`] for empty input or malformed schemes.
pub fn parse_tet_uri(input: &str) -> Result<ParsedTetUri, TetUriError> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        return Err(TetUriError::Empty);
    }

    let (label, rest) = match trimmed.split_once('@') {
        Some((before, after)) if !before.is_empty() && looks_like_scheme_or_path(after) => {
            (Some(before.to_string()), after)
        }
        _ => (None, trimmed),
    };

    let target = if let Some(rest) = rest.strip_prefix("s3://") {
        parse_s3(rest)?
    } else if let Some(rest) = rest.strip_prefix("memory://") {
        parse_memory(rest)?
    } else if let Some(rest) = rest.strip_prefix("file://") {
        parse_file_url(rest)?
    } else if rest.contains("://") {
        let scheme = rest.split("://").next().unwrap_or("").to_string();
        return Err(TetUriError::UnsupportedScheme(scheme));
    } else {
        parse_local_path(rest)?
    };

    Ok(ParsedTetUri { label, target })
}

/// Package root inside a bucket. Empty is the bucket root. Same segment rules as store keys.
pub(crate) fn normalize_package_prefix(key: &str) -> Result<String, TetUriError> {
    let prefix = key.trim_matches('/');
    if prefix.is_empty() {
        return Ok(String::new());
    }
    super::validate_store_key(prefix).map_err(|_| TetUriError::InvalidS3(key.to_string()))?;
    Ok(prefix.to_string())
}

fn looks_like_scheme_or_path(s: &str) -> bool {
    s.contains("://")
        || s.starts_with('/')
        || s.starts_with('.')
        || has_ascii_suffix_ignore_case(s, ".tet")
        || has_ascii_suffix_ignore_case(s, ".tet.d")
}

fn has_ascii_suffix_ignore_case(s: &str, suffix: &str) -> bool {
    s.len() >= suffix.len() && s[s.len() - suffix.len()..].eq_ignore_ascii_case(suffix)
}

fn parse_s3(rest: &str) -> Result<TetOpenTarget, TetUriError> {
    let (path, generation) = split_generation_query(rest)?;
    if path.is_empty() || path.starts_with('/') {
        return Err(TetUriError::InvalidS3(rest.to_string()));
    }
    let (bucket, key) = match path.split_once('/') {
        Some((bucket, key)) => (bucket, key),
        None => (path, ""),
    };
    if bucket.is_empty() {
        return Err(TetUriError::InvalidS3(rest.to_string()));
    }
    Ok(TetOpenTarget::S3 {
        bucket: bucket.to_string(),
        key: key.to_string(),
        generation,
    })
}

fn parse_memory(rest: &str) -> Result<TetOpenTarget, TetUriError> {
    let (path, generation) = split_generation_query(rest)?;
    let generation = if path.is_empty() || path == "current" {
        GenerationPin::Current
    } else if let Some(id) = path.strip_prefix("g/") {
        GenerationPin::Explicit(parse_generation_id(id)?)
    } else if matches!(generation, GenerationPin::Explicit(_)) {
        generation
    } else {
        return Err(TetUriError::InvalidMemory(rest.to_string()));
    };
    Ok(TetOpenTarget::StorePackage { generation })
}

fn parse_file_url(rest: &str) -> Result<TetOpenTarget, TetUriError> {
    // file:///abs/path or file://localhost/abs/path — keep it simple: strip optional host.
    let path = if let Some(after) = rest.strip_prefix("localhost") {
        after
    } else {
        rest
    };
    let path = percent_decode_light(path);
    parse_local_path(&path)
}

fn parse_local_path(path: &str) -> Result<TetOpenTarget, TetUriError> {
    let (path, generation) = split_generation_query(path)?;
    let pb = PathBuf::from(path);
    let trimmed = path.trim_end_matches('/');
    if has_ascii_suffix_ignore_case(trimmed, ".tet.d") {
        Ok(TetOpenTarget::LocalPackage {
            root: pb,
            generation,
        })
    } else {
        Ok(TetOpenTarget::LocalTet(pb))
    }
}

fn split_generation_query(s: &str) -> Result<(&str, GenerationPin), TetUriError> {
    if let Some((path, query)) = s.split_once('?') {
        for part in query.split('&') {
            if let Some(v) = part.strip_prefix("generation=") {
                return Ok((path, GenerationPin::Explicit(parse_generation_id(v)?)));
            }
        }
        Ok((path, GenerationPin::Current))
    } else {
        Ok((s, GenerationPin::Current))
    }
}

fn parse_generation_id(s: &str) -> Result<u64, TetUriError> {
    s.trim()
        .parse::<u64>()
        .map_err(|_| TetUriError::InvalidGeneration(s.to_string()))
}

fn percent_decode_light(s: &str) -> String {
    // Enough for spaces in paths; full URL decode is not required for this PR.
    s.replace("%20", " ")
}
