//! S3-compatible [`PackageStore`](super::PackageStore) (feature `tetration-s3`).
//!
//! Conditional publish uses HTTP `If-None-Match` / `If-Match` (decision K). MinIO speaks
//! that pair; point [`S3Config::endpoint`] at a local MinIO for tests.

use std::fmt;
use std::future::Future;
use std::sync::Arc;

use futures::StreamExt;
use object_store::aws::{AmazonS3, AmazonS3Builder, S3ConditionalPut};
use object_store::path::Path;
use object_store::{ObjectStore, ObjectStoreExt, PutMode, PutOptions, UpdateVersion};

use thiserror::Error;

use super::pin::{PinnedPackedTet, pin_packed_tet};
use super::uri::parse_s3_package_uri;
use super::{
    CatalogError, ObjectMeta, PublishGeneration, ReadAt, WritePart, parse_current_bytes,
    validate_store_key,
};

const DIR_MARKER: &str = ".tet-dir";
const CURRENT_KEY: &str = "CURRENT";

/// Body plus ETag for the `CURRENT` pointer.
type CurrentObject = (Vec<u8>, Option<String>);

/// Connection settings for an S3-compatible bucket.
///
/// The secret access key is omitted from [`Debug`].
pub struct S3Config {
    /// HTTP(S) endpoint, for example `http://127.0.0.1:9000`.
    pub endpoint: String,
    /// Bucket that holds the package.
    pub bucket: String,
    /// Package root inside the bucket. Empty means the bucket root.
    pub prefix: String,
    /// SigV4 region. MinIO accepts `us-east-1`.
    pub region: String,
    /// Access key id.
    pub access_key_id: String,
    /// Secret access key.
    pub secret_access_key: String,
}

impl S3Config {
    /// Endpoint, region, and keys from `TETRATION_S3_*`. `bucket` and `prefix` are arguments.
    ///
    /// `TETRATION_S3_ENDPOINT` is required. Access key defaults to `tetration`, secret to
    /// `tetration-secret`, region to `us-east-1` (the MinIO test defaults).
    ///
    /// # Errors
    ///
    /// [`CatalogError::Package`] when `TETRATION_S3_ENDPOINT` is unset.
    pub fn from_env(
        bucket: impl Into<String>,
        prefix: impl Into<String>,
    ) -> Result<Self, CatalogError> {
        let endpoint = std::env::var("TETRATION_S3_ENDPOINT").unwrap_or_default();
        if endpoint.is_empty() {
            return Err(CatalogError::Package("TETRATION_S3_ENDPOINT is unset"));
        }
        Ok(Self {
            endpoint,
            bucket: bucket.into(),
            prefix: prefix.into(),
            region: std::env::var("TETRATION_S3_REGION").unwrap_or_else(|_| "us-east-1".into()),
            access_key_id: std::env::var("TETRATION_S3_ACCESS_KEY")
                .unwrap_or_else(|_| "tetration".into()),
            secret_access_key: std::env::var("TETRATION_S3_SECRET_KEY")
                .unwrap_or_else(|_| "tetration-secret".into()),
        })
    }
}

impl fmt::Debug for S3Config {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Config")
            .field("endpoint", &self.endpoint)
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .field("region", &self.region)
            .finish_non_exhaustive()
    }
}

/// Package store backed by an S3-compatible API.
///
/// Keys on the store traits are package-relative. They are joined onto [`S3Config::prefix`].
pub struct S3Store {
    runtime: tokio::runtime::Runtime,
    store: Arc<AmazonS3>,
    prefix: String,
    endpoint: String,
    bucket: String,
}

impl fmt::Debug for S3Store {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("S3Store")
            .field("endpoint", &self.endpoint)
            .field("bucket", &self.bucket)
            .field("prefix", &self.prefix)
            .finish_non_exhaustive()
    }
}

impl S3Store {
    /// Open a bucket. Does not create the bucket.
    ///
    /// # Errors
    ///
    /// Invalid endpoint, prefix, or client configuration.
    pub fn connect(config: S3Config) -> Result<Self, CatalogError> {
        if !config.endpoint.starts_with("http://") && !config.endpoint.starts_with("https://") {
            return Err(CatalogError::Package(
                "s3 endpoint must start with http:// or https://",
            ));
        }
        let prefix = normalize_prefix(&config.prefix)?;
        let allow_http = config.endpoint.starts_with("http://");
        let region = if config.region.is_empty() {
            "us-east-1"
        } else {
            config.region.as_str()
        };
        let store = AmazonS3Builder::new()
            .with_endpoint(&config.endpoint)
            .with_bucket_name(&config.bucket)
            .with_region(region)
            .with_access_key_id(&config.access_key_id)
            .with_secret_access_key(&config.secret_access_key)
            .with_allow_http(allow_http)
            .with_virtual_hosted_style_request(false)
            .with_conditional_put(S3ConditionalPut::ETagMatch)
            .build()
            .map_err(store_io)?;
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_io()
            .enable_time()
            .build()?;
        Ok(Self {
            runtime,
            store: Arc::new(store),
            prefix,
            endpoint: config.endpoint,
            bucket: config.bucket,
        })
    }

    /// ETag of `key`, if the object exists.
    ///
    /// # Errors
    ///
    /// Invalid key or backend I/O.
    #[cfg(test)]
    pub(crate) fn e_tag(&self, key: &str) -> Result<Option<String>, CatalogError> {
        let path = self.locate(key)?;
        let store = Arc::clone(&self.store);
        self.wait(async move {
            match store.head(&path).await {
                Ok(meta) => Ok(meta.e_tag),
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(err) => Err(store_io(err)),
            }
        })
    }

    /// Put `bytes` at `key` only when the current ETag is `e_tag`.
    ///
    /// A rejected precondition leaves the existing object in place and returns
    /// [`CatalogError::InvalidCurrent`].
    ///
    /// # Errors
    ///
    /// [`CatalogError::InvalidCurrent`] when the ETag does not match, or backend I/O.
    pub(crate) fn put_if_match(
        &self,
        key: &str,
        bytes: &[u8],
        e_tag: &str,
    ) -> Result<(), CatalogError> {
        self.put_mode(
            key,
            bytes,
            PutMode::Update(UpdateVersion {
                e_tag: Some(e_tag.to_string()),
                version: None,
            }),
        )
    }

    fn put_mode(&self, key: &str, bytes: &[u8], mode: PutMode) -> Result<(), CatalogError> {
        let path = self.locate(key)?;
        let exists_at = std::path::PathBuf::from(key);
        let store = Arc::clone(&self.store);
        let payload: object_store::PutPayload = bytes.to_vec().into();
        let opts = PutOptions::from(mode);
        self.wait(async move {
            match store.put_opts(&path, payload, opts).await {
                Ok(_) => Ok(()),
                Err(object_store::Error::AlreadyExists { .. }) => {
                    Err(CatalogError::PackagePathExists(exists_at))
                }
                Err(object_store::Error::Precondition { .. }) => Err(CatalogError::InvalidCurrent),
                Err(err) => Err(store_io(err)),
            }
        })
    }

    fn wait<F, T>(&self, fut: F) -> T
    where
        F: Future<Output = T>,
    {
        self.runtime.block_on(fut)
    }

    fn locate(&self, key: &str) -> Result<Path, CatalogError> {
        validate_store_key(key)?;
        let full = join_prefix(&self.prefix, key);
        Path::parse(full).map_err(|_| CatalogError::Package("invalid store key"))
    }

    /// Delete every object under this store prefix, including directory markers.
    ///
    /// # Errors
    ///
    /// Backend I/O.
    #[cfg(test)]
    pub(crate) fn clear(&self) -> Result<(), CatalogError> {
        for key in self.list_rel("", true)? {
            self.remove_object(&key)?;
        }
        Ok(())
    }

    fn list_rel(
        &self,
        user_prefix: &str,
        include_markers: bool,
    ) -> Result<Vec<String>, CatalogError> {
        let trimmed = user_prefix.trim_end_matches('/');
        if !trimmed.is_empty() {
            validate_store_key(trimmed)?;
        }
        let list_at = if trimmed.is_empty() {
            if self.prefix.is_empty() {
                None
            } else {
                Some(
                    Path::parse(&self.prefix)
                        .map_err(|_| CatalogError::Package("invalid store key"))?,
                )
            }
        } else {
            Some(self.locate(trimmed)?)
        };
        let store = Arc::clone(&self.store);
        let package_prefix = self.prefix.clone();
        let found = self.wait(async move {
            let mut out = Vec::new();
            let mut stream = store.list(list_at.as_ref());
            while let Some(item) = stream.next().await {
                let meta = item.map_err(store_io)?;
                if let Some(rel) = relative_key(&package_prefix, meta.location.as_ref()) {
                    out.push(rel);
                }
            }
            Ok::<Vec<String>, CatalogError>(out)
        })?;
        let mut out: Vec<String> = found
            .into_iter()
            .filter(|rel| include_markers || !is_dir_marker(rel))
            .filter(|rel| prefix_match(rel, trimmed))
            .collect();
        out.sort();
        out.dedup();
        Ok(out)
    }
}

impl ReadAt for S3Store {
    fn read_at(&self, key: &str, offset: u64, len: u64) -> Result<Vec<u8>, CatalogError> {
        let path = self.locate(key)?;
        let store = Arc::clone(&self.store);
        self.wait(async move {
            let meta = match store.head(&path).await {
                Ok(meta) => meta,
                Err(object_store::Error::NotFound { .. }) => {
                    return Err(CatalogError::Package("store object missing"));
                }
                Err(err) => return Err(store_io(err)),
            };
            if offset > meta.size {
                return Err(CatalogError::Package("read_at offset past end of object"));
            }
            let take = len.min(meta.size - offset);
            if take == 0 {
                return Ok(Vec::new());
            }
            let bytes = store
                .get_range(&path, offset..offset + take)
                .await
                .map_err(store_io)?;
            Ok(bytes.to_vec())
        })
    }

    fn object_meta(&self, key: &str) -> Result<Option<ObjectMeta>, CatalogError> {
        let path = self.locate(key)?;
        let store = Arc::clone(&self.store);
        self.wait(async move {
            match store.head(&path).await {
                Ok(meta) => Ok(Some(ObjectMeta { len: meta.size })),
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(err) => Err(store_io(err)),
            }
        })
    }

    fn list_prefix(&self, prefix: &str) -> Result<Vec<String>, CatalogError> {
        self.list_rel(prefix, false)
    }
}

impl WritePart for S3Store {
    fn create_dir_exclusive(&self, key: &str) -> Result<(), CatalogError> {
        validate_store_key(key)?;
        let dir = key.trim_end_matches('/').to_string();
        let existing = self.list_rel(&dir, true)?;
        if existing
            .iter()
            .any(|rel| rel == &dir || rel.starts_with(&format!("{dir}/")))
        {
            return Err(CatalogError::PackagePathExists(std::path::PathBuf::from(
                dir,
            )));
        }
        let marker = format!("{dir}/{DIR_MARKER}");
        match self.put_mode(&marker, b"", PutMode::Create) {
            Ok(()) => Ok(()),
            Err(CatalogError::PackagePathExists(_)) => Err(CatalogError::PackagePathExists(
                std::path::PathBuf::from(dir),
            )),
            Err(err) => Err(err),
        }
    }

    fn write_part(&self, key: &str, bytes: &[u8]) -> Result<(), CatalogError> {
        self.put_mode(key, bytes, PutMode::Overwrite)
    }

    fn remove_object(&self, key: &str) -> Result<(), CatalogError> {
        let path = self.locate(key)?;
        let store = Arc::clone(&self.store);
        self.wait(async move {
            match store.delete(&path).await {
                Ok(()) => Ok(()),
                Err(object_store::Error::NotFound { .. }) => Ok(()),
                Err(err) => Err(store_io(err)),
            }
        })
    }

    fn remove_prefix(&self, prefix: &str) -> Result<(), CatalogError> {
        if prefix.is_empty() {
            return Err(CatalogError::Package("refusing to remove store root"));
        }
        let keys = self.list_rel(prefix, true)?;
        for key in keys {
            self.remove_object(&key)?;
        }
        Ok(())
    }

    fn rename_prefix(&self, from_prefix: &str, to_prefix: &str) -> Result<(), CatalogError> {
        let from = from_prefix.trim_end_matches('/');
        let to = to_prefix.trim_end_matches('/');
        validate_store_key(from)?;
        validate_store_key(to)?;
        let keys = self.list_rel(from, true)?;
        if keys.is_empty() {
            return Err(CatalogError::Package("rename_prefix source missing"));
        }
        let mut plan = Vec::with_capacity(keys.len());
        for key in keys {
            let new_key = if key == from {
                to.to_string()
            } else {
                format!("{to}{}", &key[from.len()..])
            };
            if self.object_meta(&new_key)?.is_some() {
                return Err(CatalogError::PackagePathExists(std::path::PathBuf::from(
                    new_key,
                )));
            }
            plan.push((key, new_key));
        }
        let store = Arc::clone(&self.store);
        let prefix = self.prefix.clone();
        self.wait(async move {
            for (from_key, to_key) in &plan {
                let src = object_path(&prefix, from_key)?;
                let dst = object_path(&prefix, to_key)?;
                store.copy(&src, &dst).await.map_err(store_io)?;
            }
            for (from_key, _) in &plan {
                let src = object_path(&prefix, from_key)?;
                match store.delete(&src).await {
                    Ok(()) | Err(object_store::Error::NotFound { .. }) => {}
                    Err(err) => return Err(store_io(err)),
                }
            }
            Ok(())
        })
    }
}

impl PublishGeneration for S3Store {
    fn publish_generation(
        &self,
        generation: u64,
        expected_current: Option<u64>,
    ) -> Result<(), CatalogError> {
        let body = format!("{generation}\n");
        match self.get_current()? {
            None => {
                if expected_current.is_some() {
                    return Err(CatalogError::InvalidCurrent);
                }
                match self.put_mode(CURRENT_KEY, body.as_bytes(), PutMode::Create) {
                    Ok(()) => Ok(()),
                    Err(CatalogError::PackagePathExists(_)) => Err(CatalogError::InvalidCurrent),
                    Err(err) => Err(err),
                }
            }
            Some((bytes, e_tag)) => {
                let actual = parse_current_bytes(&bytes)?;
                if Some(actual) != expected_current {
                    return Err(CatalogError::InvalidCurrent);
                }
                let Some(e_tag) = e_tag else {
                    return Err(CatalogError::Package("conditional put requires an etag"));
                };
                self.put_if_match(CURRENT_KEY, body.as_bytes(), &e_tag)
            }
        }
    }

    fn current_generation(&self) -> Result<Option<u64>, CatalogError> {
        match self.read_all(CURRENT_KEY) {
            Ok(bytes) => parse_current_bytes(&bytes).map(Some),
            Err(CatalogError::Package("store object missing")) => Ok(None),
            Err(err) => Err(err),
        }
    }
}

impl S3Store {
    fn get_current(&self) -> Result<Option<CurrentObject>, CatalogError> {
        let path = self.locate(CURRENT_KEY)?;
        let store = Arc::clone(&self.store);
        self.wait(async move {
            match store.get(&path).await {
                Ok(got) => {
                    let e_tag = got.meta.e_tag.clone();
                    let bytes = got.bytes().await.map_err(store_io)?.to_vec();
                    Ok(Some((bytes, e_tag)))
                }
                Err(object_store::Error::NotFound { .. }) => Ok(None),
                Err(err) => Err(store_io(err)),
            }
        })
    }
}

fn store_io(err: object_store::Error) -> CatalogError {
    CatalogError::Io(std::io::Error::other(err))
}

/// A pinned `s3://` package. The generation is fixed at open and is not re-read from `CURRENT`.
#[derive(Debug)]
pub struct OpenedS3Package {
    store: S3Store,
    pin: PinnedPackedTet,
}

impl OpenedS3Package {
    /// Connected store. Keys are package-relative.
    #[must_use]
    pub fn store(&self) -> &S3Store {
        &self.store
    }

    /// Packed object pinned for this handle.
    #[must_use]
    pub fn pin(&self) -> &PinnedPackedTet {
        &self.pin
    }
}

/// Failure opening an `s3://` package URI.
#[derive(Debug, Error)]
pub enum S3OpenError {
    /// The URI is not `s3://bucket/prefix`.
    #[error(transparent)]
    Uri(#[from] super::uri::TetUriError),
    /// Connect, pin, or object-store I/O failed.
    #[error(transparent)]
    Store(#[from] CatalogError),
}

/// Connect to `s3://bucket/prefix` and pin one generation.
///
/// Credentials come from `TETRATION_S3_*` (`TETRATION_S3_ENDPOINT` is required). The URI may be
/// `label@s3://bucket/prefix` or end with `?generation=N`. This reads `CURRENT` at most once
/// when no generation is given, then range-reads through [`PinnedPackedTet`].
///
/// # Errors
///
/// Malformed URI, missing `TETRATION_S3_ENDPOINT`, connect failure, or a generation that is
/// not sealed.
pub fn open_s3_package(uri: &str) -> Result<OpenedS3Package, S3OpenError> {
    let location = parse_s3_package_uri(uri)?;
    let store = S3Store::connect(S3Config::from_env(location.bucket, location.prefix)?)?;
    let pin = pin_packed_tet(&store, location.generation)?;
    Ok(OpenedS3Package { store, pin })
}

fn normalize_prefix(prefix: &str) -> Result<String, CatalogError> {
    let trimmed = prefix.trim_matches('/');
    if trimmed.is_empty() {
        return Ok(String::new());
    }
    validate_store_key(trimmed)?;
    Ok(trimmed.to_string())
}

fn join_prefix(prefix: &str, key: &str) -> String {
    if prefix.is_empty() {
        key.to_string()
    } else {
        format!("{prefix}/{key}")
    }
}

fn object_path(prefix: &str, key: &str) -> Result<Path, CatalogError> {
    Path::parse(join_prefix(prefix, key)).map_err(|_| CatalogError::Package("invalid store key"))
}

fn relative_key(prefix: &str, full: &str) -> Option<String> {
    if prefix.is_empty() {
        return Some(full.to_string());
    }
    full.strip_prefix(&format!("{prefix}/"))
        .filter(|rel| !rel.is_empty())
        .map(str::to_string)
}

fn is_dir_marker(rel: &str) -> bool {
    rel.rsplit_once('/')
        .is_some_and(|(_, name)| name == DIR_MARKER)
        || rel == DIR_MARKER
}

fn prefix_match(rel: &str, prefix: &str) -> bool {
    if prefix.is_empty() {
        true
    } else {
        rel == prefix || rel.starts_with(&format!("{prefix}/"))
    }
}
