//! `.tet.d` package health checks (writer protocol v2 / THI-466).

use std::fs;
use std::path::{Path, PathBuf};

use crate::catalog::package::sha256_file_hex;
use crate::catalog::zone::sha256_hex;
use crate::catalog::{
    CatalogError, DEFAULT_PUBLISH_LOCK_STALE, GenerationManifest, PACKAGE_PROTOCOL_VERSION,
    SECONDARY_INDEX_FILE, SECONDARY_INDEX_OMITTED, SECONDARY_INDEX_PRESENT, TetPackage,
    ZONE_MAPS_FILE, ZONE_MAPS_REQUIRED, format_generation_id, known_index_entries,
    secondary_index_for_packed, zone_maps_for_packed,
};

use super::options::VerifyOptions;
use super::report::{
    TetVerifyReport, VerifyFinding, VerifySummary, err_finding, ok_finding, warn_finding,
};
use super::run::verify_tet_file_with_options;

/// `true` when `path` looks like a package root (`PROTOCOL` present).
#[must_use]
pub fn looks_like_tet_package(path: &Path) -> bool {
    path.is_dir() && path.join("PROTOCOL").is_file()
}

/// Verify a `.tet.d` package (PROTOCOL, CURRENT, generations, manifests, packed images).
///
/// # Errors
///
/// Returns [`CatalogError::NotAPackage`] / open failures when `path` is not a package.
pub fn verify_tet_package(path: &Path) -> Result<TetVerifyReport, CatalogError> {
    verify_tet_package_with_options(path, VerifyOptions::default())
}

/// Package verify with explicit [`VerifyOptions`] (forwarded to each `packed.tet`).
///
/// # Errors
///
/// Same as [`verify_tet_package`].
pub fn verify_tet_package_with_options(
    path: &Path,
    opts: VerifyOptions,
) -> Result<TetVerifyReport, CatalogError> {
    let path_s = path.display().to_string();
    let package = match TetPackage::open(path) {
        Ok(p) => p,
        Err(e) => {
            return Ok(TetVerifyReport::from_fatal(
                Some(path_s),
                0,
                "protocol",
                e.to_string(),
            ));
        }
    };

    let mut findings = Vec::new();
    let mut current_summary: Option<VerifySummary> = None;
    findings.push(ok_finding(
        "protocol",
        Some(format!(
            "protocol_version={}",
            package.protocol().protocol_version
        )),
    ));

    let current = match package.current() {
        Ok(c) => c,
        Err(e) => {
            findings.push(err_finding("current", e.to_string()));
            return Ok(finish_package(path_s, findings, None));
        }
    };

    match current {
        None => {
            if sealed_generation_dirs_present(&package) {
                // Empty package may omit CURRENT; sealed gens without a pointer are corrupt.
                findings.push(err_finding(
                    "current",
                    "CURRENT absent but sealed generations/ present".to_owned(),
                ));
            } else {
                findings.push(ok_finding(
                    "current",
                    Some("absent (empty package)".to_owned()),
                ));
            }
        }
        Some(g) => {
            findings.push(ok_finding("current", Some(format!("generation={g}"))));
            match verify_sealed_generation(&package, g, opts, /*is_current*/ true) {
                Ok((extra, summary)) => {
                    findings.extend(extra);
                    current_summary = summary;
                }
                Err(fatal) => {
                    findings.extend(fatal);
                    return Ok(finish_package(path_s, findings, None));
                }
            }
        }
    }

    // Prior sealed generations (retain current-1 by default policy); soft-check all present.
    for entry in visit_generation_dirs(&package) {
        match entry {
            GenDirEntry::Sealed(g) if current == Some(g) => {}
            GenDirEntry::Sealed(g) => match verify_sealed_generation(&package, g, opts, false) {
                Ok((extra, _)) => findings.extend(extra),
                Err(fatal) => findings.extend(fatal),
            },
            GenDirEntry::Unexpected(name_s) => findings.push(warn_finding(
                "generations",
                format!("unexpected generations/ entry: {name_s}"),
            )),
        }
    }

    findings.extend(check_publishing_locks(&package));
    findings.extend(check_staging_orphans(&package));

    Ok(finish_package(path_s, findings, current_summary))
}

enum GenDirEntry {
    Sealed(u64),
    Unexpected(String),
}

fn visit_generation_dirs(package: &TetPackage) -> Vec<GenDirEntry> {
    let Ok(entries) = fs::read_dir(package.generations_dir()) else {
        return Vec::new();
    };
    let mut out = Vec::new();
    for entry in entries.flatten() {
        if !entry.path().is_dir() {
            continue;
        }
        let name = entry.file_name();
        let Some(name_s) = name.to_str() else {
            continue;
        };
        if name_s.contains("quarantine") {
            continue;
        }
        match name_s.parse::<u64>() {
            Ok(g) => out.push(GenDirEntry::Sealed(g)),
            Err(_) => out.push(GenDirEntry::Unexpected(name_s.to_owned())),
        }
    }
    out
}

fn sealed_generation_dirs_present(package: &TetPackage) -> bool {
    visit_generation_dirs(package)
        .iter()
        .any(|e| matches!(e, GenDirEntry::Sealed(_)))
}

fn finish_package(
    path_s: String,
    findings: Vec<VerifyFinding>,
    summary: Option<VerifySummary>,
) -> TetVerifyReport {
    TetVerifyReport {
        ok: true,
        path: Some(path_s),
        file_len: 0,
        findings,
        recommendations: Vec::new(),
        summary,
    }
    .finalize()
}

fn generation_label(is_current: bool, generation: u64) -> String {
    if is_current {
        "current_generation".to_owned()
    } else {
        format!("generation_{}", format_generation_id(generation))
    }
}

fn packed_check(is_current: bool) -> &'static str {
    if is_current {
        "current_packed"
    } else {
        "generation_packed"
    }
}

fn manifest_check(is_current: bool) -> &'static str {
    if is_current {
        "current_manifest"
    } else {
        "generation_manifest"
    }
}

fn verify_sealed_generation(
    package: &TetPackage,
    generation: u64,
    opts: VerifyOptions,
    is_current: bool,
) -> Result<(Vec<VerifyFinding>, Option<VerifySummary>), Vec<VerifyFinding>> {
    let prefix = generation_label(is_current, generation);
    let (packed, findings) = require_packed(package, generation, is_current, Vec::new())?;
    let (manifest, findings) = require_manifest(package, generation, is_current, findings)?;
    let findings = check_manifest_ids(&manifest, generation, findings)?;
    let findings = check_packed_len(&packed, &manifest, is_current, &prefix, findings)?;
    let findings = check_packed_sha(&packed, &manifest, is_current, findings)?;
    let findings = check_zone_maps(package, generation, &manifest, &packed, opts, findings)?;
    let findings =
        check_secondary_indexes(package, generation, &manifest, &packed, opts, findings)?;
    check_packed_layout(&packed, opts, &prefix, findings)
}

fn require_packed(
    package: &TetPackage,
    generation: u64,
    is_current: bool,
    mut findings: Vec<VerifyFinding>,
) -> Result<(PathBuf, Vec<VerifyFinding>), Vec<VerifyFinding>> {
    let packed = package.packed_tet_path(generation);
    if !packed.is_file() {
        findings.push(err_finding(
            packed_check(is_current),
            format!(
                "generations/{}/packed.tet missing",
                format_generation_id(generation)
            ),
        ));
        return Err(findings);
    }
    findings.push(ok_finding(
        packed_check(is_current),
        Some(packed.display().to_string()),
    ));
    Ok((packed, findings))
}

fn require_manifest(
    package: &TetPackage,
    generation: u64,
    is_current: bool,
    mut findings: Vec<VerifyFinding>,
) -> Result<(GenerationManifest, Vec<VerifyFinding>), Vec<VerifyFinding>> {
    match package.read_manifest(generation) {
        Ok(m) => Ok((m, findings)),
        Err(e) => {
            findings.push(err_finding(manifest_check(is_current), e.to_string()));
            Err(findings)
        }
    }
}

fn check_manifest_ids(
    manifest: &GenerationManifest,
    generation: u64,
    mut findings: Vec<VerifyFinding>,
) -> Result<Vec<VerifyFinding>, Vec<VerifyFinding>> {
    if manifest.generation != generation {
        findings.push(err_finding(
            "manifest_generation",
            format!(
                "MANIFEST generation {} != directory {}",
                manifest.generation, generation
            ),
        ));
        return Err(findings);
    }
    if manifest.protocol_version != PACKAGE_PROTOCOL_VERSION {
        findings.push(err_finding(
            "manifest_protocol",
            format!(
                "MANIFEST protocol_version {} != {}",
                manifest.protocol_version, PACKAGE_PROTOCOL_VERSION
            ),
        ));
        return Err(findings);
    }
    Ok(findings)
}

fn check_zone_maps(
    package: &TetPackage,
    generation: u64,
    manifest: &GenerationManifest,
    packed: &Path,
    opts: VerifyOptions,
    mut findings: Vec<VerifyFinding>,
) -> Result<Vec<VerifyFinding>, Vec<VerifyFinding>> {
    let zone_path = package.generation_dir(generation).join(ZONE_MAPS_FILE);
    match manifest.zone_maps.as_deref() {
        Some(ZONE_MAPS_REQUIRED) => {
            if !zone_path.is_file() {
                findings.push(err_finding(
                    "zone_maps",
                    format!(
                        "generations/{}/{} missing (required)",
                        format_generation_id(generation),
                        ZONE_MAPS_FILE
                    ),
                ));
                return Err(findings);
            }
            let bytes = match fs::read(&zone_path) {
                Ok(b) => b,
                Err(e) => {
                    findings.push(err_finding("zone_maps", e.to_string()));
                    return Err(findings);
                }
            };
            let actual = sha256_hex(&bytes);
            match &manifest.zone_maps_sha256 {
                Some(expected) if expected == &actual => {}
                Some(expected) => {
                    findings.push(err_finding(
                        "zone_maps_sha256",
                        format!("{ZONE_MAPS_FILE} sha256 {actual} != MANIFEST {expected}"),
                    ));
                    return Err(findings);
                }
                None => {
                    findings.push(err_finding(
                        "zone_maps_sha256",
                        "MANIFEST zone_maps is required but zone_maps_sha256 is absent".to_owned(),
                    ));
                    return Err(findings);
                }
            }
            let doc = match serde_json::from_slice::<crate::catalog::GenerationZoneMaps>(&bytes) {
                Ok(d) => d,
                Err(e) => {
                    findings.push(err_finding("zone_maps", e.to_string()));
                    return Err(findings);
                }
            };
            if doc.generation != generation {
                findings.push(err_finding(
                    "zone_maps",
                    format!(
                        "ZONE.json generation {} != directory {generation}",
                        doc.generation
                    ),
                ));
                return Err(findings);
            }
            if let Some(expected) = &manifest.packed_sha256
                && expected != &doc.packed_sha256
            {
                findings.push(err_finding(
                    "zone_maps",
                    "ZONE.json packed_sha256 does not match MANIFEST".to_owned(),
                ));
                return Err(findings);
            }
            if opts.deep_decode {
                let packed_bytes = match fs::read(packed) {
                    Ok(b) => b,
                    Err(e) => {
                        findings.push(err_finding("zone_maps_stale", e.to_string()));
                        return Err(findings);
                    }
                };
                let fresh = match zone_maps_for_packed(
                    &packed_bytes,
                    generation,
                    doc.packed_sha256.as_str(),
                ) {
                    Ok(z) => z,
                    Err(e) => {
                        findings.push(err_finding("zone_maps_stale", e.to_string()));
                        return Err(findings);
                    }
                };
                if fresh.chunks != doc.chunks {
                    findings.push(err_finding(
                        "zone_maps_stale",
                        "ZONE.json does not match a recompute of packed.tet".to_owned(),
                    ));
                    return Err(findings);
                }
            }
            findings.push(ok_finding("zone_maps", Some("required".to_owned())));
            Ok(findings)
        }
        Some(other) => {
            findings.push(err_finding(
                "zone_maps",
                format!("unsupported MANIFEST zone_maps value {other:?}"),
            ));
            Err(findings)
        }
        None => {
            findings.push(warn_finding(
                "zone_maps",
                "advisory: generation sealed without zone maps".to_owned(),
            ));
            Ok(findings)
        }
    }
}

fn check_secondary_indexes(
    package: &TetPackage,
    generation: u64,
    manifest: &GenerationManifest,
    packed: &Path,
    opts: VerifyOptions,
    mut findings: Vec<VerifyFinding>,
) -> Result<Vec<VerifyFinding>, Vec<VerifyFinding>> {
    let index_path = package
        .generation_dir(generation)
        .join(SECONDARY_INDEX_FILE);
    match manifest.secondary_indexes.as_deref() {
        Some(SECONDARY_INDEX_PRESENT) => {
            check_present_secondary_index(generation, &index_path, manifest, packed, opts, findings)
        }
        Some(SECONDARY_INDEX_OMITTED) => {
            if index_path.is_file() {
                findings.push(err_finding(
                    "secondary_indexes",
                    format!(
                        "generations/{}/{} present but MANIFEST secondary_indexes is omitted",
                        format_generation_id(generation),
                        SECONDARY_INDEX_FILE
                    ),
                ));
                return Err(findings);
            }
            findings.push(ok_finding(
                "secondary_indexes",
                Some(SECONDARY_INDEX_OMITTED.to_owned()),
            ));
            Ok(findings)
        }
        Some(other) => {
            findings.push(err_finding(
                "secondary_indexes",
                format!("unsupported MANIFEST secondary_indexes value {other:?}"),
            ));
            Err(findings)
        }
        None => {
            findings.push(warn_finding(
                "secondary_indexes",
                "advisory: generation sealed without secondary indexes".to_owned(),
            ));
            Ok(findings)
        }
    }
}

fn check_present_secondary_index(
    generation: u64,
    index_path: &Path,
    manifest: &GenerationManifest,
    packed: &Path,
    opts: VerifyOptions,
    mut findings: Vec<VerifyFinding>,
) -> Result<Vec<VerifyFinding>, Vec<VerifyFinding>> {
    if !index_path.is_file() {
        findings.push(err_finding(
            "secondary_indexes",
            format!(
                "generations/{}/{} missing (present)",
                format_generation_id(generation),
                SECONDARY_INDEX_FILE
            ),
        ));
        return Err(findings);
    }
    let bytes = match fs::read(index_path) {
        Ok(b) => b,
        Err(e) => {
            findings.push(err_finding("secondary_indexes", e.to_string()));
            return Err(findings);
        }
    };
    let actual = sha256_hex(&bytes);
    match &manifest.secondary_indexes_sha256 {
        Some(expected) if expected == &actual => {}
        Some(expected) => {
            findings.push(err_finding(
                "secondary_indexes_sha256",
                format!("{SECONDARY_INDEX_FILE} sha256 {actual} != MANIFEST {expected}"),
            ));
            return Err(findings);
        }
        None => {
            findings.push(err_finding(
                "secondary_indexes_sha256",
                "MANIFEST secondary_indexes is present but secondary_indexes_sha256 is absent"
                    .to_owned(),
            ));
            return Err(findings);
        }
    }
    let value: serde_json::Value = match serde_json::from_slice(&bytes) {
        Ok(v) => v,
        Err(e) => {
            findings.push(err_finding("secondary_indexes", e.to_string()));
            return Err(findings);
        }
    };
    let version = value.get("version").and_then(serde_json::Value::as_u64);
    if version != Some(u64::from(crate::catalog::SECONDARY_INDEX_VERSION)) {
        findings.push(warn_finding(
            "secondary_indexes_version",
            format!(
                "advisory: {SECONDARY_INDEX_FILE} version {version:?} is not {}; query falls back",
                crate::catalog::SECONDARY_INDEX_VERSION
            ),
        ));
        return Ok(findings);
    }
    let doc = match serde_json::from_value::<crate::catalog::GenerationSecondaryIndex>(value) {
        Ok(d) => d,
        Err(e) => {
            findings.push(err_finding("secondary_indexes", e.to_string()));
            return Err(findings);
        }
    };
    if doc.generation != generation {
        findings.push(err_finding(
            "secondary_indexes",
            format!(
                "{SECONDARY_INDEX_FILE} generation {} != directory {generation}",
                doc.generation
            ),
        ));
        return Err(findings);
    }
    if let Some(expected) = &manifest.packed_sha256
        && expected != &doc.packed_sha256
    {
        findings.push(err_finding(
            "secondary_indexes",
            format!("{SECONDARY_INDEX_FILE} packed_sha256 does not match MANIFEST"),
        ));
        return Err(findings);
    }
    if opts.deep_decode
        && let Err(detail) = secondary_index_matches_recompute(packed, generation, &doc)
    {
        findings.push(err_finding("secondary_indexes_stale", detail));
        return Err(findings);
    }
    findings.push(ok_finding(
        "secondary_indexes",
        Some(SECONDARY_INDEX_PRESENT.to_owned()),
    ));
    Ok(findings)
}

fn secondary_index_matches_recompute(
    packed: &Path,
    generation: u64,
    doc: &crate::catalog::GenerationSecondaryIndex,
) -> Result<(), String> {
    let packed_bytes = fs::read(packed).map_err(|e| e.to_string())?;
    let fresh = secondary_index_for_packed(&packed_bytes, generation, doc.packed_sha256.as_str())
        .map_err(|e| e.to_string())?;
    let fresh_known = fresh.as_ref().map(known_index_entries).unwrap_or_default();
    if fresh_known != known_index_entries(doc) {
        return Err(format!(
            "{SECONDARY_INDEX_FILE} does not match a recompute of packed.tet"
        ));
    }
    Ok(())
}

fn check_packed_len(
    packed: &Path,
    manifest: &GenerationManifest,
    is_current: bool,
    prefix: &str,
    mut findings: Vec<VerifyFinding>,
) -> Result<Vec<VerifyFinding>, Vec<VerifyFinding>> {
    let packed_len = fs::metadata(packed).map_or(0, |m| m.len());
    if packed_len != manifest.packed_byte_len {
        findings.push(err_finding(
            "manifest_byte_len",
            format!(
                "packed.tet len {packed_len} != MANIFEST packed_byte_len {}",
                manifest.packed_byte_len
            ),
        ));
        return Err(findings);
    }
    findings.push(ok_finding(
        manifest_check(is_current),
        Some(format!("{prefix} byte_len ok")),
    ));
    Ok(findings)
}

fn check_packed_sha(
    packed: &Path,
    manifest: &GenerationManifest,
    is_current: bool,
    mut findings: Vec<VerifyFinding>,
) -> Result<Vec<VerifyFinding>, Vec<VerifyFinding>> {
    match &manifest.packed_sha256 {
        None => {
            if is_current {
                findings.push(warn_finding(
                    "manifest_sha256",
                    "MANIFEST packed_sha256 absent (legacy seal)".to_owned(),
                ));
            }
            Ok(findings)
        }
        Some(expected) => match sha256_file_hex(packed) {
            Ok(actual) if actual == *expected => {
                findings.push(ok_finding("manifest_sha256", Some("match".to_owned())));
                Ok(findings)
            }
            Ok(actual) => {
                findings.push(err_finding(
                    "manifest_sha256",
                    format!("packed.tet sha256 {actual} != MANIFEST {expected}"),
                ));
                Err(findings)
            }
            Err(e) => {
                findings.push(err_finding("manifest_sha256", e.to_string()));
                Err(findings)
            }
        },
    }
}

fn check_packed_layout(
    packed: &Path,
    opts: VerifyOptions,
    prefix: &str,
    mut findings: Vec<VerifyFinding>,
) -> Result<(Vec<VerifyFinding>, Option<VerifySummary>), Vec<VerifyFinding>> {
    match verify_tet_file_with_options(packed, opts) {
        Ok(report) if report.ok => {
            findings.push(ok_finding(
                "packed_layout",
                Some(format!("{prefix} layout-v1 ok")),
            ));
            for f in &report.findings {
                if !f.ok {
                    findings.push(f.clone());
                }
            }
            Ok((findings, report.summary))
        }
        Ok(report) => {
            findings.push(err_finding(
                "packed_layout",
                format!("{prefix} packed.tet failed layout-v1 verify"),
            ));
            for f in report.findings {
                if !f.ok {
                    findings.push(f);
                }
            }
            Err(findings)
        }
        Err(e) => {
            findings.push(err_finding("packed_layout", e.to_string()));
            Err(findings)
        }
    }
}

fn check_publishing_locks(package: &TetPackage) -> Vec<VerifyFinding> {
    let mut findings = Vec::new();
    let dir = package.publishing_dir();
    let Ok(entries) = fs::read_dir(&dir) else {
        return findings;
    };
    for entry in entries.flatten() {
        let name = entry.file_name();
        let Some(name_s) = name.to_str() else {
            continue;
        };
        if name_s.contains("quarantine") {
            findings.push(warn_finding(
                "publishing",
                format!("quarantined publish lock present: {name_s}"),
            ));
            continue;
        }
        if !entry.path().is_dir() {
            continue;
        }
        let Ok(g) = name_s.parse::<u64>() else {
            findings.push(warn_finding(
                "publishing",
                format!("unexpected publishing/ entry: {name_s}"),
            ));
            continue;
        };
        match package.publish_lock_is_stale(g, DEFAULT_PUBLISH_LOCK_STALE) {
            Ok(true) => findings.push(warn_finding(
                "publishing_stale",
                format!(
                    "publishing/{name_s} is stale (>{DEFAULT_PUBLISH_LOCK_STALE:?}); quarantine with tet repair --apply quarantine_publish"
                ),
            )),
            Ok(false) => findings.push(warn_finding(
                "publishing_incomplete",
                format!("publishing/{name_s} present (incomplete publish; CURRENT unchanged)"),
            )),
            Err(e) => findings.push(warn_finding("publishing", e.to_string())),
        }
    }
    if findings
        .iter()
        .all(|f| f.check != "publishing_stale" && f.check != "publishing_incomplete")
    {
        findings.push(ok_finding("publishing", Some("no active locks".to_owned())));
    }
    findings
}

fn check_staging_orphans(package: &TetPackage) -> Vec<VerifyFinding> {
    let staging = package.staging_dir();
    let Ok(entries) = fs::read_dir(&staging) else {
        return vec![ok_finding("staging", Some("absent".to_owned()))];
    };
    let mut sessions = 0usize;
    for entry in entries.flatten() {
        if entry.path().is_dir() {
            sessions += 1;
        }
    }
    if sessions == 0 {
        vec![ok_finding("staging", Some("empty".to_owned()))]
    } else {
        vec![warn_finding(
            "staging",
            format!(
                "{sessions} staging session(s) present (OK if packed.tet verifies; GC optional)"
            ),
        )]
    }
}
