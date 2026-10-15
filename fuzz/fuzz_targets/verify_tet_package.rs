//! Fuzz target for `verify_tet_package` (THI-481).
//!
//! Input is a **length-prefixed** blob (not tar), so libFuzzer can mutate package
//! fields independently:
//!
//! ```text
//! [u32 LE proto_len][PROTOCOL bytes]
//! [u32 LE current_len][CURRENT bytes]     # empty → omit CURRENT
//! [u32 LE manifest_len][MANIFEST.json]
//! [u32 LE packed_len][packed.tet bytes]
//! ```
//!
//! Truncated / oversize lengths are clamped to remaining input. Materialize a
//! minimal `.tet.d` under a tempfile and call verify (must not panic).
//!
//! ```bash
//! mise fuzz-package
//! # or:
//! cargo +nightly fuzz run verify_tet_package
//! ```

#![no_main]

use std::fs;
use std::path::Path;

use libfuzzer_sys::fuzz_target;
use tetration::verify::verify_tet_package;

const MAX_PROTO: usize = 4 * 1024;
const MAX_CURRENT: usize = 4 * 1024;
const MAX_MANIFEST: usize = 64 * 1024;
const MAX_PACKED: usize = 256 * 1024;

fuzz_target!(|data: &[u8]| {
    let _ = fuzz_once(data);
});

fn fuzz_once(data: &[u8]) -> Option<()> {
    let mut cur = data;
    let protocol = take_len_prefixed(&mut cur, MAX_PROTO);
    let current = take_len_prefixed(&mut cur, MAX_CURRENT);
    let manifest = take_len_prefixed(&mut cur, MAX_MANIFEST);
    let packed = take_len_prefixed(&mut cur, MAX_PACKED);

    let dir = tempfile::tempdir().ok()?;
    let root = dir.path().join("fuzz.tet.d");
    materialize_package(&root, protocol, current, manifest, packed)?;
    let _ = verify_tet_package(&root);
    Some(())
}

fn take_len_prefixed<'a>(data: &mut &'a [u8], max: usize) -> &'a [u8] {
    if data.len() < 4 {
        *data = &[];
        return &[];
    }
    let n = u32::from_le_bytes([data[0], data[1], data[2], data[3]]) as usize;
    *data = &data[4..];
    let n = n.min(max).min(data.len());
    let (head, rest) = data.split_at(n);
    *data = rest;
    head
}

fn materialize_package(
    root: &Path,
    protocol: &[u8],
    current: &[u8],
    manifest: &[u8],
    packed: &[u8],
) -> Option<()> {
    fs::create_dir_all(root.join("generations").join("00000001")).ok()?;
    fs::create_dir_all(root.join("staging")).ok()?;
    fs::create_dir_all(root.join("publishing")).ok()?;

    // Empty PROTOCOL → leave absent (open fails as fatal finding, must not panic).
    if !protocol.is_empty() {
        fs::write(root.join("PROTOCOL"), protocol).ok()?;
    }
    if !current.is_empty() {
        fs::write(root.join("CURRENT"), current).ok()?;
    }
    if !manifest.is_empty() {
        fs::write(
            root.join("generations").join("00000001").join("MANIFEST.json"),
            manifest,
        )
        .ok()?;
    }
    if !packed.is_empty() {
        fs::write(
            root.join("generations").join("00000001").join("packed.tet"),
            packed,
        )
        .ok()?;
    }
    Some(())
}
