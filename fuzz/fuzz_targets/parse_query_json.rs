//! Fuzz target for closed query wire parse (`parse_query_json`) — THI-604 / THI-607.
//!
//! Mutated inputs must never panic; invalid JSON / unknown keys are expected
//! [`TetError::Validation`] failures.
//!
//! ```bash
//! cargo +nightly fuzz run parse_query_json
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use tetration::query::parse_query_json;

fuzz_target!(|data: &[u8]| {
    let Ok(text) = std::str::from_utf8(data) else {
        return;
    };
    // Cap input size so the harness stays responsive under libFuzzer.
    if text.len() > 64 * 1024 {
        return;
    }
    let _ = parse_query_json(text);
});
