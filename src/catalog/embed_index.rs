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
use super::zone::{sha256_hex, zone_maps_json_for_packed};

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
/// Hashes the current file as `packed_sha256`, builds the same documents as package seal, then
/// rewrites the footer. The file must not already carry embedded zone/index refs.
///
/// # Errors
///
/// I/O, catalog parse, or an image that already has embed refs.
pub fn embed_indexes_in_tet(path: &Path, generation: u64) -> Result<EmbeddedIndexes, CatalogError> {
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

    if read_embedded_zone_maps_bytes(&data, sb.flags)?.is_some()
        || read_embedded_secondary_index_bytes(&data, sb.flags)?.is_some()
    {
        return Err(CatalogError::InvalidWriteSpec(
            "tet already has embedded zone/index spills; refuse in-place re-embed",
        ));
    }

    let packed_sha256 = sha256_hex(&data);
    let (zone_maps_json, zone_maps_sha256) =
        zone_maps_json_for_packed(&data, generation, &packed_sha256)?;
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

        // Second embed must refuse.
        assert!(embed_indexes_in_tet(&path, 0).is_err());
    }
}
