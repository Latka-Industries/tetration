//! Individual repair implementations.

use std::fs::{self, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::Path;

use crate::catalog::{
    CatalogError, DEFAULT_PUBLISH_LOCK_STALE, TetPackage, format_generation_id, payload_file_len,
};
use crate::layout::{SUPERBLOCK_FLAG_HISTORY_FOOTER, SUPERBLOCK_V1_LEN, SuperblockV1};

use super::RepairActionResult;

/// Apply one repair code to `path`.
///
/// # Errors
///
/// Returns [`CatalogError`] on I/O failure or unsupported code.
pub fn apply_repair_code(
    path: &Path,
    code: &str,
    dry_run: bool,
) -> Result<RepairActionResult, CatalogError> {
    match code {
        "footer_invalid" => strip_invalid_footer(path, dry_run),
        "quarantine_publish" => quarantine_stale_publish_locks(path, dry_run),
        other => Ok(RepairActionResult {
            code: other.to_owned(),
            applied: false,
            dry_run,
            message: format!("unknown repair code: {other}"),
        }),
    }
}

/// Quarantine every stale `publishing/<G>/` lock under a package root.
fn quarantine_stale_publish_locks(
    path: &Path,
    dry_run: bool,
) -> Result<RepairActionResult, CatalogError> {
    let package = TetPackage::open(path)?;
    let mut quarantined = Vec::new();
    let mut would = Vec::new();
    let dir = package.publishing_dir();
    if dir.is_dir() {
        for entry in fs::read_dir(&dir)? {
            let entry = entry?;
            let name = entry.file_name();
            let Some(name_s) = name.to_str() else {
                continue;
            };
            if name_s.contains("quarantine") || !entry.path().is_dir() {
                continue;
            }
            let Ok(g) = name_s.parse::<u64>() else {
                continue;
            };
            if !package.publish_lock_is_stale(g, DEFAULT_PUBLISH_LOCK_STALE)? {
                continue;
            }
            if dry_run {
                would.push(format_generation_id(g));
            } else {
                let dest = package.quarantine_publish_lock(g, DEFAULT_PUBLISH_LOCK_STALE)?;
                quarantined.push(dest.display().to_string());
            }
        }
    }

    if dry_run {
        return Ok(RepairActionResult {
            code: "quarantine_publish".to_owned(),
            applied: false,
            dry_run: true,
            message: if would.is_empty() {
                "no stale publish locks to quarantine".to_owned()
            } else {
                format!("would quarantine publishing/{{{}}}", would.join(","))
            },
        });
    }

    Ok(RepairActionResult {
        code: "quarantine_publish".to_owned(),
        applied: !quarantined.is_empty(),
        dry_run: false,
        message: if quarantined.is_empty() {
            "no stale publish locks to quarantine".to_owned()
        } else {
            format!("quarantined: {}", quarantined.join(", "))
        },
    })
}

/// Truncate after chunk payloads and clear the history-footer superblock flag.
fn strip_invalid_footer(path: &Path, dry_run: bool) -> Result<RepairActionResult, CatalogError> {
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

    let payload_end = payload_file_len(&data, sb.flags)?;

    if dry_run {
        return Ok(RepairActionResult {
            code: "footer_invalid".to_owned(),
            applied: false,
            dry_run: true,
            message: format!(
                "would truncate to {payload_end} bytes and clear history footer flag (from {} bytes)",
                data.len()
            ),
        });
    }

    let mut f = OpenOptions::new().read(true).write(true).open(path)?;
    f.set_len(payload_end)?;
    f.seek(SeekFrom::Start(0))?;
    let mut sb_bytes = [0u8; SUPERBLOCK_V1_LEN];
    f.read_exact(&mut sb_bytes)?;
    let mut sb = SuperblockV1::from_bytes(&sb_bytes).map_err(CatalogError::Layout)?;
    sb.flags &= !SUPERBLOCK_FLAG_HISTORY_FOOTER;
    f.seek(SeekFrom::Start(0))?;
    f.write_all(&sb.to_bytes())?;
    f.sync_all()?;

    Ok(RepairActionResult {
        code: "footer_invalid".to_owned(),
        applied: true,
        dry_run: false,
        message: format!("truncated to {payload_end} bytes and cleared history footer flag"),
    })
}
