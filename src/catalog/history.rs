//! Optional file footer at EOF (`THST`): history rows and optional `metadata` JSON.

use std::collections::BTreeMap;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use argus::{LeReader, LeWriter};
use serde::{Deserialize, Deserializer, Serialize, Serializer};

use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};

use super::CatalogError;
use super::metadata::{self, MetadataLimitsV1, TetMetadataV1};

/// On-disk `THST` history footer tail (layout v1).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct HistoryFooterWireV1 {
    /// EOF magic (`THST`).
    pub magic: [u8; 4],
    /// Footer format version (must match on read).
    pub version: u32,
    /// Byte length of the fixed tail (`history_json_len` + `history_version` + magic).
    pub tail_len: usize,
}

impl HistoryFooterWireV1 {
    /// Layout v1 wire constants for the history footer.
    pub const DEFAULT: Self = Self {
        magic: *b"THST",
        version: 1,
        tail_len: 16,
    };
}

/// One provenance row in the `THST` footer `history` array.
///
/// New files use JSON objects. Legacy files may store `[op, source, at]` triples; readers accept both.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct HistoryEvent {
    pub op: String,
    pub source: String,
    pub at: String,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub parents: Vec<String>,
    #[serde(default, skip_serializing_if = "BTreeMap::is_empty")]
    pub params: BTreeMap<String, String>,
}

impl HistoryEvent {
    /// Build a row with the current Unix timestamp (`at` as decimal seconds).
    #[must_use]
    pub fn new(op: impl Into<String>, source: impl Into<String>) -> Self {
        Self {
            op: op.into(),
            source: source.into(),
            at: unix_timestamp_now(),
            parents: Vec::new(),
            params: BTreeMap::new(),
        }
    }

    /// Legacy tuple view for callers that used `(op, source, at)`.
    #[must_use]
    pub fn as_tuple(&self) -> (&str, &str, &str) {
        (&self.op, &self.source, &self.at)
    }
}

/// Legacy alias for the pre-struct history row type.
pub type HistoryEventV1 = HistoryEvent;

/// Byte range of footer `metadata` spilled before the `THST` JSON (inline `metadata` omitted).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct MetadataBlobRefV1 {
    pub offset: u64,
    pub len: u64,
}

/// Parsed `THST` JSON payload (history + optional metadata).
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct FooterBlobV1 {
    pub history: Vec<HistoryEvent>,
    pub metadata: Option<TetMetadataV1>,
    pub metadata_ref: Option<MetadataBlobRefV1>,
}

impl FooterBlobV1 {
    /// Validate history rows and inline metadata (not the spilled blob).
    ///
    /// # Errors
    ///
    /// Returns [`CatalogError::InvalidWriteSpec`] when limits are exceeded.
    pub fn validate(&self) -> Result<(), CatalogError> {
        let limits = MetadataLimitsV1::DEFAULT;
        if self.history.len() > limits.history_events {
            return Err(CatalogError::InvalidWriteSpec(
                "history exceeds history_events limit",
            ));
        }
        for ev in &self.history {
            ev.validate(limits)?;
        }
        if let Some(meta) = &self.metadata {
            meta.validate()?;
        }
        if self.metadata.is_some() && self.metadata_ref.is_some() {
            return Err(CatalogError::InvalidWriteSpec(
                "footer metadata and metadata_ref are mutually exclusive",
            ));
        }
        Ok(())
    }
}

impl HistoryEvent {
    fn validate(&self, limits: MetadataLimitsV1) -> Result<(), CatalogError> {
        metadata::validate_attr_string(&self.op, "history op", limits.attr_string_bytes)?;
        metadata::validate_attr_string(&self.source, "history source", limits.attr_string_bytes)?;
        metadata::validate_attr_string(&self.at, "history at", limits.attr_string_bytes)?;
        if self.parents.len() > limits.history_parents {
            return Err(CatalogError::InvalidWriteSpec(
                "history parents exceeds history_parents limit",
            ));
        }
        for p in &self.parents {
            metadata::validate_attr_string(p, "history parent", limits.attr_string_bytes)?;
        }
        if self.params.len() > limits.history_params {
            return Err(CatalogError::InvalidWriteSpec(
                "history params exceeds history_params limit",
            ));
        }
        for (k, v) in &self.params {
            metadata::validate_attr_string(k, "history param key", limits.attr_string_bytes)?;
            metadata::validate_attr_string(v, "history param value", limits.attr_string_bytes)?;
        }
        Ok(())
    }
}

impl Serialize for FooterBlobV1 {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        FooterBlobWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for FooterBlobV1 {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let wire = FooterBlobWire::deserialize(deserializer)?;
        Ok(wire.into_footer())
    }
}

#[derive(Serialize, Deserialize)]
struct FooterBlobWire {
    #[serde(default)]
    history: Vec<HistoryEventWire>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata: Option<TetMetadataV1>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    metadata_ref: Option<MetadataBlobRefV1>,
}

#[derive(Deserialize)]
#[serde(untagged)]
enum HistoryEventWire {
    Triple([String; 3]),
    Event(HistoryEvent),
}

impl HistoryEventWire {
    fn into_event(self) -> HistoryEvent {
        match self {
            Self::Triple([op, source, at]) => HistoryEvent {
                op,
                source,
                at,
                parents: Vec::new(),
                params: BTreeMap::new(),
            },
            Self::Event(ev) => ev,
        }
    }
}

impl Serialize for HistoryEventWire {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        match self {
            Self::Event(ev) => ev.serialize(serializer),
            Self::Triple(t) => t.serialize(serializer),
        }
    }
}

impl FooterBlobWire {
    fn from(blob: &FooterBlobV1) -> Self {
        Self {
            history: blob
                .history
                .iter()
                .map(|ev| HistoryEventWire::Event(ev.clone()))
                .collect(),
            metadata: blob.metadata.clone(),
            metadata_ref: blob.metadata_ref,
        }
    }

    fn into_footer(self) -> FooterBlobV1 {
        FooterBlobV1 {
            history: self
                .history
                .into_iter()
                .map(HistoryEventWire::into_event)
                .collect(),
            metadata: self.metadata,
            metadata_ref: self.metadata_ref,
        }
    }
}

/// Append a convert event footer and set [`SUPERBLOCK_FLAG_HISTORY_FOOTER`] on the superblock.
///
/// Merges with an existing footer when present (preserves `metadata`).
///
/// # Errors
///
/// Returns [`CatalogError`] when JSON encoding or I/O fails.
pub fn append_convert_history(
    path: &Path,
    source: &str,
) -> Result<Vec<HistoryEvent>, CatalogError> {
    let mut blob = read_footer_blob_from_path(path).unwrap_or_default();
    let event = HistoryEvent::new("convert", source);
    blob.history.push(event.clone());
    write_footer_blob(path, &blob)?;
    Ok(vec![event])
}

/// Unix seconds since epoch as a decimal string (history `at` field).
#[must_use]
pub fn unix_timestamp_now() -> String {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map_or_else(|_| "0".to_owned(), |d| d.as_secs().to_string())
}

/// Append history rows (merges with existing footer `metadata` when present).
///
/// # Errors
///
/// Returns [`CatalogError`] when JSON encoding or I/O fails.
pub fn append_history_events(path: &Path, events: &[HistoryEvent]) -> Result<(), CatalogError> {
    if events.is_empty() {
        return Ok(());
    }
    let mut blob = read_footer_blob_from_path(path).unwrap_or_default();
    blob.history.extend_from_slice(events);
    write_footer_blob(path, &blob)
}

/// Write footer JSON (`history` + optional `metadata`), replacing any prior footer.
///
/// When inline JSON exceeds [`MetadataLimitsV1::footer_json_bytes`], spills `metadata` to a raw
/// UTF-8 blob immediately before the footer JSON and sets `metadata_ref`.
///
/// # Errors
///
/// Returns [`CatalogError`] when validation, JSON encoding, or I/O fails.
pub fn write_footer_blob(path: &Path, blob: &FooterBlobV1) -> Result<(), CatalogError> {
    if blob.history.is_empty() && blob.metadata.is_none() && blob.metadata_ref.is_none() {
        return Ok(());
    }
    blob.validate()?;

    let inline_json = encode_footer_json(blob)?;
    if metadata::validate_footer_json_len(&inline_json).is_ok() {
        return rewrite_footer_bytes(path, inline_json);
    }

    let meta = blob
        .metadata
        .as_ref()
        .ok_or(CatalogError::InvalidWriteSpec(
            "footer JSON exceeds limit and there is no metadata to spill",
        ))?;
    let meta_json =
        serde_json::to_vec(meta).map_err(|_| CatalogError::InvalidWriteSpec("metadata JSON"))?;
    metadata::validate_metadata_blob_len(&meta_json)?;

    let spill_blob = FooterBlobV1 {
        history: blob.history.clone(),
        metadata: None,
        metadata_ref: None,
    };
    let spill_footer_json = encode_footer_json(&spill_blob)?;
    if metadata::validate_footer_json_len(&spill_footer_json).is_ok() {
        return rewrite_footer_bytes_with_spill(path, spill_footer_json, Some(&meta_json));
    }

    Err(CatalogError::InvalidWriteSpec(
        "footer JSON exceeds footer_json_bytes limit even without inline metadata",
    ))
}

fn encode_footer_json(blob: &FooterBlobV1) -> Result<Vec<u8>, CatalogError> {
    serde_json::to_vec(blob).map_err(|_| CatalogError::InvalidWriteSpec("footer JSON"))
}

fn read_footer_blob_from_path(path: &Path) -> Result<FooterBlobV1, CatalogError> {
    let data = std::fs::read(path)?;
    let sb = SuperblockV1::from_bytes(
        data.get(..SUPERBLOCK_V1_LEN)
            .ok_or(CatalogError::TooShort {
                need: SUPERBLOCK_V1_LEN,
                got: data.len(),
            })?
            .try_into()
            .map_err(|_| CatalogError::InvalidWriteSpec("superblock length"))?,
    )
    .map_err(CatalogError::Layout)?;
    read_footer_blob(&data, sb.flags)
}

/// Parse the `THST` footer when present.
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_footer_blob(data: &[u8], flags: u32) -> Result<FooterBlobV1, CatalogError> {
    let Some(footer_len) = history_footer_len(data)? else {
        return Ok(FooterBlobV1::default());
    };
    if flags & SUPERBLOCK_FLAG_HISTORY_FOOTER == 0 {
        return Ok(FooterBlobV1::default());
    }
    let json_start = data
        .len()
        .checked_sub(footer_len)
        .ok_or(CatalogError::TooShort {
            need: footer_len,
            got: data.len(),
        })?;
    let wire = HistoryFooterWireV1::DEFAULT;
    let mut blob: FooterBlobV1 =
        serde_json::from_slice(&data[json_start..data.len() - wire.tail_len])
            .map_err(|_| CatalogError::InvalidWriteSpec("footer JSON parse"))?;
    if let Some(spill) = blob.metadata_ref {
        blob.metadata = Some(read_metadata_spill(data, spill)?);
        blob.metadata_ref = None;
    }
    Ok(blob)
}

fn read_metadata_spill(
    data: &[u8],
    spill: MetadataBlobRefV1,
) -> Result<TetMetadataV1, CatalogError> {
    let offset = usize::try_from(spill.offset).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "metadata_ref.offset",
        value: spill.offset,
    })?;
    let len = usize::try_from(spill.len).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "metadata_ref.len",
        value: spill.len,
    })?;
    let end = offset
        .checked_add(len)
        .ok_or(CatalogError::InvalidWriteSpec("metadata_ref overflow"))?;
    if end > data.len() {
        return Err(CatalogError::InvalidWriteSpec("metadata_ref out of bounds"));
    }
    let meta: TetMetadataV1 = serde_json::from_slice(&data[offset..end])
        .map_err(|_| CatalogError::InvalidWriteSpec("metadata spill JSON parse"))?;
    meta.validate()?;
    Ok(meta)
}

/// Parse history rows from the footer (empty when absent).
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_history(data: &[u8], flags: u32) -> Result<Vec<HistoryEvent>, CatalogError> {
    Ok(read_footer_blob(data, flags)?.history)
}

/// Parse optional metadata from the footer (resolves `metadata_ref` spill when present).
///
/// # Errors
///
/// Returns [`CatalogError`] when the footer is present but malformed.
pub fn read_metadata(data: &[u8], flags: u32) -> Result<TetMetadataV1, CatalogError> {
    Ok(read_footer_blob(data, flags)?.metadata.unwrap_or_default())
}

fn rewrite_footer_bytes(path: &Path, json: Vec<u8>) -> Result<(), CatalogError> {
    metadata::validate_footer_json_len(&json)?;
    rewrite_footer_bytes_with_spill(path, json, None)
}

fn rewrite_footer_bytes_with_spill(
    path: &Path,
    footer_json: Vec<u8>,
    metadata_spill: Option<&[u8]>,
) -> Result<(), CatalogError> {
    let data = std::fs::read(path)?;
    let sb = SuperblockV1::from_bytes(
        data.get(..SUPERBLOCK_V1_LEN)
            .ok_or(CatalogError::TooShort {
                need: SUPERBLOCK_V1_LEN,
                got: data.len(),
            })?
            .try_into()
            .map_err(|_| CatalogError::InvalidWriteSpec("superblock length"))?,
    )
    .map_err(CatalogError::Layout)?;

    let payload_end = usize::try_from(payload_file_len(&data, sb.flags)?).map_err(|_| {
        CatalogError::TooLargeForPlatform {
            field: "payload_end",
            value: u64::MAX,
        }
    })?;

    let (footer_bytes, spill_bytes) = if let Some(spill) = metadata_spill {
        metadata::validate_metadata_blob_len(spill)?;
        let offset = u64::try_from(payload_end).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "metadata_spill.offset",
            value: u64::MAX,
        })?;
        let len = u64::try_from(spill.len()).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "metadata_spill.len",
            value: u64::MAX,
        })?;
        let stub = FooterBlobV1 {
            history: serde_json::from_slice::<FooterBlobWire>(&footer_json)
                .map_err(|_| CatalogError::InvalidWriteSpec("footer JSON parse"))?
                .history
                .into_iter()
                .map(HistoryEventWire::into_event)
                .collect(),
            metadata: None,
            metadata_ref: Some(MetadataBlobRefV1 { offset, len }),
        };
        let encoded = encode_footer_json(&stub)?;
        metadata::validate_footer_json_len(&encoded)?;
        (encoded, Some(spill.to_vec()))
    } else {
        (footer_json, None)
    };

    let json_len =
        u64::try_from(footer_bytes.len()).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "footer_json_len",
            value: u64::MAX,
        })?;

    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(payload_end as u64)?;
    f.seek(SeekFrom::End(0))?;
    if let Some(spill) = spill_bytes {
        f.write_all(&spill)?;
    }
    f.write_all(&footer_bytes)?;
    let wire = HistoryFooterWireV1::DEFAULT;
    let mut tail = [0u8; HistoryFooterWireV1::DEFAULT.tail_len];
    {
        let mut w = LeWriter::new(&mut tail);
        w.put_u64(json_len);
        w.put_u32(wire.version);
        w.put_bytes(&wire.magic);
        debug_assert_eq!(w.position(), wire.tail_len);
    }
    f.write_all(&tail)?;

    let mut sb_bytes = [0u8; SUPERBLOCK_V1_LEN];
    f.seek(SeekFrom::Start(0))?;
    f.read_exact(&mut sb_bytes)?;
    let mut sb = SuperblockV1::from_bytes(&sb_bytes).map_err(CatalogError::Layout)?;
    sb.flags |= SUPERBLOCK_FLAG_HISTORY_FOOTER;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&sb.to_bytes())?;
    f.sync_all()?;
    Ok(())
}

/// File length used for chunk payload bounds (excludes optional history footer and metadata spill).
///
/// # Errors
///
/// Returns [`CatalogError`] when a history footer is declared but malformed.
pub fn payload_file_len(data: &[u8], flags: u32) -> Result<u64, CatalogError> {
    let len = u64::try_from(data.len()).map_err(|_| CatalogError::TooLargeForPlatform {
        field: "file_len",
        value: u64::MAX,
    })?;
    if flags & SUPERBLOCK_FLAG_HISTORY_FOOTER == 0 {
        return Ok(len);
    }
    let footer = history_footer_len(data)?.ok_or(CatalogError::InvalidWriteSpec(
        "superblock history flag set but footer missing",
    ))?;
    let json_start = data
        .len()
        .checked_sub(footer)
        .ok_or(CatalogError::InvalidWriteSpec(
            "history footer longer than file",
        ))?;
    let wire = HistoryFooterWireV1::DEFAULT;
    let json_end = data.len() - wire.tail_len;
    let spill_len = serde_json::from_slice::<FooterBlobWire>(&data[json_start..json_end])
        .ok()
        .and_then(|w| w.metadata_ref)
        .map_or(0, |r| usize::try_from(r.len).unwrap_or(usize::MAX));
    let suffix = footer
        .checked_add(spill_len)
        .ok_or(CatalogError::InvalidWriteSpec("footer spill overflow"))?;
    len.checked_sub(
        u64::try_from(suffix).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "footer_suffix",
            value: u64::MAX,
        })?,
    )
    .ok_or(CatalogError::InvalidWriteSpec(
        "history footer longer than file",
    ))
}

fn history_footer_len(data: &[u8]) -> Result<Option<usize>, CatalogError> {
    let wire = HistoryFooterWireV1::DEFAULT;
    if data.len() < wire.tail_len {
        return Ok(None);
    }
    let tail = data.len() - wire.tail_len;
    let mut r = LeReader::new(&data[tail..tail + wire.tail_len]);
    let json_len = r.take_u64();
    let version = r.take_u32();
    let magic = r.take_4();
    if magic != wire.magic {
        return Ok(None);
    }
    if version != wire.version {
        return Err(CatalogError::InvalidWriteSpec(
            "unsupported history footer version",
        ));
    }
    let json_len_usize =
        usize::try_from(json_len).map_err(|_| CatalogError::TooLargeForPlatform {
            field: "history_json_len",
            value: json_len,
        })?;
    let total = wire
        .tail_len
        .checked_add(json_len_usize)
        .ok_or(CatalogError::InvalidWriteSpec("history footer overflow"))?;
    if total > data.len() {
        return Err(CatalogError::InvalidWriteSpec(
            "history footer out of bounds",
        ));
    }
    Ok(Some(total))
}

/// Test helper: write footer JSON bytes (validates inline size cap).
#[cfg(test)]
pub fn rewrite_footer_bytes_for_test(path: &Path, json: Vec<u8>) -> Result<(), CatalogError> {
    rewrite_footer_bytes(path, json)
}
