//! Fuzz target for `verify_tet_bytes` with full chunk decode (THI-376).
//!
//! Slower than [`verify_tet_bytes`]; use for deeper coverage, not the hot loop.
//!
//! ```bash
//! cargo +nightly fuzz run verify_tet_bytes_deep
//! ```

#![no_main]

use std::path::Path;

use libfuzzer_sys::fuzz_target;
use tetration::verify::{VerifyOptions, verify_tet_bytes};

fuzz_target!(|data: &[u8]| {
    let _ = verify_tet_bytes(
        data,
        Some(Path::new("fuzz.tet")),
        VerifyOptions { deep_decode: true },
    );
});
