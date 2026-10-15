//! Per-file execution settings stored in the chunk index header (layout v1).

use argus::{LeReader, LeWriter};
use serde::Serialize;

/// Default share of host RAM used when no fixed byte budget is set (25.00% = 2500 bps).
pub const DEFAULT_MEMORY_BUDGET_PERCENT_BPS: u16 = 2500;

/// On-disk execution preferences for a `.tet` file (TIDX header bytes 16–31).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize)]
#[serde(rename_all = "snake_case")]
pub struct FileExecutionSettingsV1 {
    /// Basis points of host available RAM (10000 = 100%). `0` = engine default ([`DEFAULT_MEMORY_BUDGET_PERCENT_BPS`]).
    pub memory_budget_percent_bps: u16,
    /// Fixed RAM cap for dense in-memory decode; `0` = derive from [`Self::memory_budget_percent_bps`].
    pub memory_budget_bytes: u32,
}

impl FileExecutionSettingsV1 {
    pub const WIRE_LEN: usize = 16;

    /// All-zero settings: use engine default percent of detected host RAM.
    #[must_use]
    pub const fn default_engine() -> Self {
        Self {
            memory_budget_percent_bps: 0,
            memory_budget_bytes: 0,
        }
    }

    /// Parse from the 16-byte **reserved** tail of the chunk index header (offset 16 from `TIDX`).
    #[must_use]
    pub fn from_index_header_tail(tail: &[u8]) -> Self {
        debug_assert!(tail.len() >= Self::WIRE_LEN);
        let mut r = LeReader::new(&tail[..Self::WIRE_LEN]);
        let memory_budget_percent_bps = r.take_u16();
        r.skip(2);
        let memory_budget_bytes = r.take_u32();
        Self {
            memory_budget_percent_bps,
            memory_budget_bytes,
        }
    }

    /// Write into the 16-byte reserved tail (bytes 16–31 of the chunk index header).
    pub fn write_index_header_tail(&self, tail: &mut [u8]) {
        debug_assert!(tail.len() >= Self::WIRE_LEN);
        let mut w = LeWriter::new(&mut tail[..Self::WIRE_LEN]);
        w.put_u16(self.memory_budget_percent_bps);
        w.put_u16(0);
        w.put_u32(self.memory_budget_bytes);
        w.put_u64(0);
        debug_assert_eq!(w.position(), Self::WIRE_LEN);
    }

    /// Effective percent basis points: file override, else engine default.
    #[must_use]
    pub const fn effective_percent_bps(self) -> u16 {
        if self.memory_budget_percent_bps == 0 {
            DEFAULT_MEMORY_BUDGET_PERCENT_BPS
        } else {
            self.memory_budget_percent_bps
        }
    }
}
