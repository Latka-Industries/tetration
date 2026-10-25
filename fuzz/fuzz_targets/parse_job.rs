//! Fuzz target for job-file parse + static validate (`parse_job`) — THI-651 / THI-650.
//!
//! Mutated inputs must never panic; invalid TOML / JSON / unknown ops are expected
//! [`JobError::Parse`] / [`JobError::Invalid`] failures. Does **not** run the executor.
//!
//! ```bash
//! cargo +nightly fuzz run parse_job
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use tetration::job::{JobFormat, parse_job};

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    // Cap input size so the harness stays responsive under libFuzzer.
    if text.len() > 64 * 1024 {
        return;
    }
    let _ = parse_job(text, JobFormat::Auto);
});
