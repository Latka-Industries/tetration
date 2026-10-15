//! On-disk layout v1: fixed superblock at offset 0.
//!
//! See `docs/layout_v1.md` for the byte-level spec.

use std::fs::{File, OpenOptions};
use std::io::{self, Write};
use std::path::Path;

use argus::{LeReader, LeWriter};
use memmap2::{Mmap, MmapOptions};
use serde::Serialize;
use thiserror::Error;

/// File magic: ASCII `TETR`.
pub const MAGIC: &[u8; 4] = b"TETR";

/// Length of layout v1 superblock in bytes.
pub const SUPERBLOCK_V1_LEN: usize = 32;

/// On-disk layout version described by `docs/layout_v1.md`.
pub const LAYOUT_VERSION_V1: u32 = 1;

/// Superblock `flags`: optional [`catalog::history`] footer at EOF (`THST` magic).
pub const SUPERBLOCK_FLAG_HISTORY_FOOTER: u32 = 1 << 0;

/// Parsed layout-v1 superblock (fields mirror the spec table).
#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct SuperblockV1 {
    pub layout_version: u32,
    pub dataset_count: u32,
    pub flags: u32,
    pub chunk_index_offset: u64,
    pub chunk_index_length: u64,
}

impl SuperblockV1 {
    /// Superblock for an empty container: no datasets, no chunk index bytes.
    #[must_use]
    pub fn empty_file() -> Self {
        Self {
            layout_version: LAYOUT_VERSION_V1,
            dataset_count: 0,
            flags: 0,
            chunk_index_offset: SUPERBLOCK_V1_LEN as u64,
            chunk_index_length: 0,
        }
    }

    /// Serialize to exactly [`SUPERBLOCK_V1_LEN`] bytes (little-endian).
    #[must_use]
    pub fn to_bytes(&self) -> [u8; SUPERBLOCK_V1_LEN] {
        let mut buf = [0u8; SUPERBLOCK_V1_LEN];
        let mut w = LeWriter::new(&mut buf);
        w.put_bytes(MAGIC);
        w.put_u32(self.layout_version);
        w.put_u32(self.dataset_count);
        w.put_u32(self.flags);
        w.put_u64(self.chunk_index_offset);
        w.put_u64(self.chunk_index_length);
        debug_assert_eq!(w.position(), SUPERBLOCK_V1_LEN);
        buf
    }

    /// Parse a validated 32-byte superblock (caller must check magic/version separately if needed).
    pub(crate) fn from_bytes(data: &[u8; SUPERBLOCK_V1_LEN]) -> Result<Self, LayoutError> {
        let mut r = LeReader::new(data.as_slice());
        let m = r.take_4();
        if &m != MAGIC {
            return Err(LayoutError::BadMagic(m));
        }
        let layout_version = r.take_u32();
        if layout_version != LAYOUT_VERSION_V1 {
            return Err(LayoutError::UnsupportedVersion(layout_version));
        }
        Ok(Self {
            layout_version,
            dataset_count: r.take_u32(),
            flags: r.take_u32(),
            chunk_index_offset: r.take_u64(),
            chunk_index_length: r.take_u64(),
        })
    }
}

/// Superblock parse and layout validation failures.
#[derive(Debug, Error)]
pub enum LayoutError {
    #[error("file too short for superblock: need {need} bytes, got {got}")]
    TooShort { need: usize, got: usize },
    #[error("bad magic: expected TETR, got {0:?}")]
    BadMagic([u8; 4]),
    #[error("unsupported layout_version: {0} (only {LAYOUT_VERSION_V1} is supported)")]
    UnsupportedVersion(u32),
    #[error(
        "chunk index region is out of bounds for file length {file_len}: offset {offset}, length {length}"
    )]
    IndexOutOfBounds {
        file_len: u64,
        offset: u64,
        length: u64,
    },
}

/// Parse and validate layout v1 superblock from the start of `data`.
///
/// # Errors
///
/// Returns [`LayoutError`] when the buffer is too short, magic or version is wrong, or the
/// chunk index region described by the superblock does not fit within `data`.
pub fn read_superblock_v1(data: &[u8]) -> Result<SuperblockV1, LayoutError> {
    let mut r = LeReader::require(data, "superblock", SUPERBLOCK_V1_LEN).map_err(|e| {
        LayoutError::TooShort {
            need: e.need,
            got: e.got,
        }
    })?;
    let m = r.take_4();
    if &m != MAGIC {
        return Err(LayoutError::BadMagic(m));
    }
    let layout_version = r.take_u32();
    if layout_version != LAYOUT_VERSION_V1 {
        return Err(LayoutError::UnsupportedVersion(layout_version));
    }
    let sb = SuperblockV1 {
        layout_version,
        dataset_count: r.take_u32(),
        flags: r.take_u32(),
        chunk_index_offset: r.take_u64(),
        chunk_index_length: r.take_u64(),
    };
    let file_len = data.len() as u64;
    validate_index_bounds(file_len, sb.chunk_index_offset, sb.chunk_index_length)?;
    Ok(sb)
}

fn validate_index_bounds(file_len: u64, offset: u64, length: u64) -> Result<(), LayoutError> {
    let end = offset
        .checked_add(length)
        .ok_or(LayoutError::IndexOutOfBounds {
            file_len,
            offset,
            length,
        })?;
    if end > file_len {
        return Err(LayoutError::IndexOutOfBounds {
            file_len,
            offset,
            length,
        });
    }
    Ok(())
}

/// Create a new `.tet` file containing only a valid empty v1 superblock (truncates if present).
///
/// # Errors
///
/// Propagates I/O errors from opening, writing, or syncing the file.
pub fn create_empty_v1_file(path: &Path) -> io::Result<()> {
    let sb = SuperblockV1::empty_file();
    let mut f = OpenOptions::new()
        .write(true)
        .create(true)
        .truncate(true)
        .open(path)?;
    f.write_all(&sb.to_bytes())?;
    f.sync_all()?;
    Ok(())
}

/// Memory-map an existing file for read-only access.
///
/// # Errors
///
/// Propagates I/O errors from opening the file or creating the memory map.
pub fn mmap_file_read(path: &Path) -> io::Result<Mmap> {
    let file = File::open(path)?;
    unsafe { MmapOptions::new().map(&file) }
}

/// Open a path, mmap it, and parse the v1 superblock.
///
/// # Errors
///
/// Returns [`LayoutOpenError::Io`] on open/mmap failure, or [`LayoutOpenError::Layout`] when the
/// mapped bytes are not a valid v1 superblock.
pub fn open_superblock_v1(path: &Path) -> Result<(Mmap, SuperblockV1), LayoutOpenError> {
    let mmap = mmap_file_read(path)?;
    let sb = read_superblock_v1(&mmap).map_err(LayoutOpenError::Layout)?;
    Ok((mmap, sb))
}

/// Path open / mmap failures wrapping [`LayoutError`] from superblock parse.
#[derive(Debug, Error)]
pub enum LayoutOpenError {
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Layout(#[from] LayoutError),
}
