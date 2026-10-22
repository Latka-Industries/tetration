//! Embed zone maps + secondary indexes into a sealed `.tet` (0.7.0).
//!
//! Contract: [`docs/embedded_indexes.md`](../../docs/embedded_indexes.md).

use std::path::Path;

use crate::layout::{SUPERBLOCK_V1_LEN, SuperblockV1};

use super::CatalogError;
use super::history::{
    self, FooterBlobV1, read_embedded_secondary_index_bytes, read_embedded_zone_maps_bytes,
    rewrite_footer_with_index_spills,
};
use super::secondary_index::{PreparedSecondaryIndex, prepare_secondary_index};
use super::zone::{
    ChunkZoneStat, GenerationZoneMaps, ZONE_MAPS_VERSION, encode_zone_maps, sha256_hex,
    zone_maps_json_for_packed,
};

/// Standalone / non-package generation id written into embedded ZONE/INDEX JSON.
pub const EMBEDDED_STANDALONE_GENERATION: u64 = 0;

/// Result of embedding indexes into a sealed image.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EmbeddedIndexes {
    /// SHA-256 of the file bytes **before** zone/index spills were appended.
    pub packed_sha256: String,
    /// Pretty `ZONE.json` bytes that were embedded (and may be mirrored as a sidecar).
    pub zone_maps_json: Vec<u8>,
    /// Hex SHA-256 of [`Self::zone_maps_json`].
    pub zone_maps_sha256: String,
    /// Optional pretty `INDEX.json` when at least one kind qualifies.
    pub secondary_index: PreparedSecondaryIndex,
}

/// Embed zone maps (and optional `INDEX.json`) into `path` as `history_version = 2` footer spills.
///
/// Hashes the current file as `packed_sha256` (after stripping any prior zone/index spills), builds
/// the same documents as package seal, then rewrites the footer. Re-embed is allowed (append /
/// `tet index`): prior zone/index spills are replaced; history/metadata are kept.
///
/// # Errors
///
/// I/O or catalog parse failures.
pub fn embed_indexes_in_tet(path: &Path, generation: u64) -> Result<EmbeddedIndexes, CatalogError> {
    embed_indexes_in_tet_with_zones(path, generation, None)
}

/// Like [`embed_indexes_in_tet`], but reuses zone stats collected during the write (no second
/// full-file zone scan). Secondary INDEX kinds still run against the sealed bytes.
///
/// # Errors
///
/// I/O or catalog parse failures.
pub fn embed_indexes_in_tet_with_zones(
    path: &Path,
    generation: u64,
    precomputed_zones: Option<Vec<ChunkZoneStat>>,
) -> Result<EmbeddedIndexes, CatalogError> {
    // Drop prior zone/index so `packed_sha256` matches a fresh v1-footer seal (spec).
    clear_embedded_indexes(path)?;

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

    let packed_sha256 = sha256_hex(&data);
    let (zone_maps_json, zone_maps_sha256) = if let Some(chunks) = precomputed_zones {
        let doc = GenerationZoneMaps {
            version: ZONE_MAPS_VERSION,
            generation,
            packed_sha256: packed_sha256.clone(),
            chunks,
        };
        let bytes = encode_zone_maps(&doc)?;
        let sha = sha256_hex(&bytes);
        (bytes, sha)
    } else {
        zone_maps_json_for_packed(&data, generation, &packed_sha256)?
    };
    let secondary_index = prepare_secondary_index(&data, generation, &packed_sha256)?;

    let logical = history::read_footer_blob(&data, sb.flags).unwrap_or_else(|_| FooterBlobV1 {
        history: Vec::new(),
        metadata: None,
        metadata_ref: None,
    });

    rewrite_footer_with_index_spills(
        path,
        &logical,
        Some(zone_maps_json.as_slice()),
        secondary_index.bytes.as_deref(),
    )?;

    Ok(EmbeddedIndexes {
        packed_sha256,
        zone_maps_json,
        zone_maps_sha256,
        secondary_index,
    })
}

/// Drop embedded ZONE/INDEX spills, rewriting a `history_version = 1` footer (metadata/history kept).
///
/// Used before re-embed, by `tet index --clear`, and by benches that need a no-pushdown baseline.
///
/// # Errors
///
/// I/O or catalog parse failures.
pub fn clear_embedded_indexes(path: &Path) -> Result<(), CatalogError> {
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

    if read_embedded_zone_maps_bytes(&data, sb.flags)?.is_none()
        && read_embedded_secondary_index_bytes(&data, sb.flags)?.is_none()
    {
        return Ok(());
    }

    let mut logical = history::read_footer_blob(&data, sb.flags).unwrap_or_else(|_| FooterBlobV1 {
        history: Vec::new(),
        metadata: None,
        metadata_ref: None,
    });
    // `write_footer_blob` no-ops on a fully empty blob; keep a row so spills actually truncate.
    if logical.history.is_empty() && logical.metadata.is_none() && logical.metadata_ref.is_none() {
        logical
            .history
            .push(history::HistoryEvent::new("index", "clear"));
    }
    history::write_footer_blob(path, &logical)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::catalog::{
        CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, RawArrayWrite, payload_file_len,
        read_tet_summary_v1, write_raw_array_file,
    };
    use crate::layout::SUPERBLOCK_FLAG_HISTORY_FOOTER;

    #[test]
    fn embed_indexes_round_trip_spills_and_payload_bounds() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("embed.tet");
        let shape = [4u64];
        let chunk_shape = [2u64];
        let values = [1.0f32, 2.0, 3.0, 4.0];
        let data = bytemuck::cast_slice(&values);
        write_raw_array_file(
            &path,
            &RawArrayWrite {
                name: "data",
                dtype: DATASET_DTYPE_TAG_V1.f32,
                shape: &shape,
                chunk_shape: &chunk_shape,
                chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
                data,
                file_execution: None,
            },
        )
        .unwrap();

        let before = std::fs::read(&path).unwrap();
        let before_sha = sha256_hex(&before);

        let embedded = embed_indexes_in_tet(&path, EMBEDDED_STANDALONE_GENERATION).unwrap();
        assert_eq!(embedded.packed_sha256, before_sha);
        assert!(!embedded.zone_maps_json.is_empty());

        let after = std::fs::read(&path).unwrap();
        assert!(after.len() > before.len());
        let sb = SuperblockV1::from_bytes(after[..SUPERBLOCK_V1_LEN].try_into().unwrap()).unwrap();
        assert_ne!(sb.flags & SUPERBLOCK_FLAG_HISTORY_FOOTER, 0);

        let zone = read_embedded_zone_maps_bytes(&after, sb.flags)
            .unwrap()
            .expect("zone spill");
        assert_eq!(zone, embedded.zone_maps_json);

        let summary = read_tet_summary_v1(&after).unwrap();
        assert_eq!(summary.datasets.len(), 1);
        let payload_end = payload_file_len(&after, sb.flags).unwrap();
        assert_eq!(payload_end, before.len() as u64);

        // Re-embed replaces spills (append / tet index).
        let again = embed_indexes_in_tet(&path, EMBEDDED_STANDALONE_GENERATION).unwrap();
        let after2 = std::fs::read(&path).unwrap();
        let sb2 =
            SuperblockV1::from_bytes(after2[..SUPERBLOCK_V1_LEN].try_into().unwrap()).unwrap();
        let zone2 = read_embedded_zone_maps_bytes(&after2, sb2.flags)
            .unwrap()
            .expect("zone after re-embed");
        assert_eq!(zone2, again.zone_maps_json);
        assert!(!again.zone_maps_json.is_empty());
    }
}
