//! Fuzz target for `INDEX.json` / [`GenerationSecondaryIndex`] wire (THI-613).
//!
//! Mutated inputs must never panic. Invalid JSON and unknown `kind` values are
//! expected failures or [`SecondaryIndexEntry::Unknown`].
//!
//! ```bash
//! cargo +nightly fuzz run secondary_index_json
//! ```

#![no_main]

use libfuzzer_sys::fuzz_target;
use tetration::catalog::{
    GenerationSecondaryIndex, encode_secondary_index, known_index_entries, sorted_axis_ranges,
    value_postings,
};

fuzz_target!(|data: &[u8]| {
    if data.len() > 64 * 1024 {
        return;
    }
    let Ok(doc) = serde_json::from_slice::<GenerationSecondaryIndex>(data) else {
        return;
    };
    let _ = known_index_entries(&doc);
    let _ = sorted_axis_ranges(&doc);
    let _ = value_postings(&doc);
    let _ = encode_secondary_index(&doc);
});
