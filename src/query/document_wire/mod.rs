//! Flat JSON query document wire format (v1).
//!
//! Top-level reduction keys (`mean`, `sum`, …) replace nested `operation`. Export spill uses
//! top-level `"spill": "path"` (not nested `output`). Axis specs accept `0`, `[0, 1]`, or `[]`
//! (scalar). Parametric ops use `{ "q": …, "axis": … }`.

mod fields;
mod filter;
mod keys;
mod operation;
mod parse;
mod pipeline;
mod serialize;

pub use parse::parse_query_value;

use serde::de::{self, Deserialize, Deserializer};
use serde::ser::{Serialize, Serializer};
use serde_json::Value;

use super::types::QueryDocument;
use serialize::QueryDocumentWire;

impl Serialize for QueryDocument {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        QueryDocumentWire::from(self).serialize(serializer)
    }
}

impl<'de> Deserialize<'de> for QueryDocument {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: Deserializer<'de>,
    {
        let value = Value::deserialize(deserializer)?;
        parse_query_value(&value).map_err(de::Error::custom)
    }
}
