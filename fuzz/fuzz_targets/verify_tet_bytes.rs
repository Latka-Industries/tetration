//! Fuzz target for `verify_tet_bytes` quick scan (THI-376).
//!
//! ```bash
//! cargo +nightly fuzz run verify_tet_bytes
//! ```

#![no_main]

use std::path::Path;

use libfuzzer_sys::fuzz_target;
use tetration::verify::{VerifyOptions, verify_tet_bytes};

fuzz_target!(|data: &[u8]| {
    let _ = verify_tet_bytes(data, Some(Path::new("fuzz.tet")), VerifyOptions::default());
});
