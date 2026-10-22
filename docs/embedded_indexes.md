# Embedded zone maps + secondary indexes (0.7.0)

Contract for carrying the same logical content as generation `ZONE.json` / `INDEX.json` **inside** a sealed `.tet` image. Parent: [THI-627](https://linear.app/thicclatka/issue/THI-627) / [THI-626](https://linear.app/thicclatka/issue/THI-626) under [THI-609](https://linear.app/thicclatka/issue/THI-609).

Through **0.6.x**, zone maps and secondary indexes are **package sidecars** beside `packed.tet` ([`secondary_indexes.md`](secondary_indexes.md)). A standalone `.tet` has no pushdown. **0.7.0** embeds those UTF-8 JSON blobs in the sealed image so `tet query -t foo.tet` can skip chunks without a `.tet.d` package. `.tet.d` remains the multi-writer path only.

This document locks **where the bytes live**, **versioning**, **discovery**, and **verify**. Writer emit, query load, package publish mirrors, and conformance fixtures shipped with **0.7.0** (THI-627).

## Decision

Reuse today’s builders and JSON document shapes (`GenerationZoneMaps` / secondary-index `version: 1`). Append the blobs as **footer spills** (same family as `metadata_spill`), discovered by refs in `history_json`.

Not a mid-file region (would remap every `payload_offset`). Not a new superblock flag alone (old readers that only honor bit 0 would mis-bound payloads). Not a `layout_version` bump: superblock / dataset directory / TIDX framing stay **1**.

### Why `history_version = 2`

`payload_file_len` subtracts the THST JSON + tail + every known spill length. A reader that only knows `metadata_ref` and sees extra zone/index bytes before `history_json` would treat those bytes as chunk payloads → silent wrong answers.

Bump **`history_version` to 2** whenever the footer may carry `zone_maps_ref` and/or `secondary_index_ref`. Readers that only implement version **1** already refuse unsupported `history_version` ([`COMPATIBILITY.md`](../SPEC/COMPATIBILITY.md) §3.1). That refuse is the compatibility story — not “ignore unknown footer keys.”

`layout_version` stays **1**. Pack / SPEC milestone for the embed cut is **0.7.0** (conformance PR).

## Where the bytes live

EOF suffix when superblock `flags & 1` and `history_version = 2`:

```text
… chunk payloads …
[zone_maps_spill?]         # raw UTF-8 ZONE.json bytes (pretty + trailing newline)
[secondary_index_spill?]   # raw UTF-8 INDEX.json bytes (same spelling as sidecar)
[metadata_spill?]          # unchanged 0.x metadata spill
history_json               # UTF-8; may include zone_maps_ref / secondary_index_ref
history_json_len           # u64 LE
history_version            # u32 LE = 2
THST                       # ASCII magic
```

Spill order toward EOF is fixed: **zone → index → metadata → history_json → tail**. Absent spills omit their span. Writers **MUST NOT** interleave or reorder.

`history_json` gains two optional keys (same shape as `metadata_ref`):

| Key | Meaning |
| --- | ------- |
| `zone_maps_ref` | `{ "offset": u64, "len": u64 }` — byte span of `zone_maps_spill` in the file. Absent → no embedded zone maps. |
| `secondary_index_ref` | `{ "offset": u64, "len": u64 }` — byte span of `secondary_index_spill`. Absent → no embedded secondary index. |

Rules:

- Each ref is mutually exclusive with “no spill”: if the key is present, `len > 0` and the span **MUST** lie after payloads and before `history_json`, with no overlap between spills.
- Inline zone/index JSON inside `history_json` is **not** allowed (size + hash stability). Spills only.
- `metadata` / `metadata_ref` rules are unchanged.
- Unknown keys under `history_json` remain ignorable for catalog bind **only when** `history_version` is understood. Version **2** readers that do not yet implement embed **MUST** still subtract both new spill lengths when computing payload bounds (or refuse the file).

### `packed_sha256` (chicken-egg)

Zone and index documents carry `packed_sha256`. That field binds **separate** `ZONE.json` / `INDEX.json` sidecars (and MANIFEST) to a sealed `packed.tet`. Same-file footer spills do not need a content hash for query pushdown.

| Path | `packed_sha256` |
|------|-----------------|
| **Standalone** embed (convert / writer / `tet index`, generation **`0`**) | Fixed all-zero placeholder. **Do not** full-file SHA-256. Query ignores the field for embedded spills. |
| **Package** seal / store publish (generation `G ≥ 1`) | SHA-256 of the sealed image **before** zone/index spills (same as today’s sidecar seal). |

Package seal sequence:

1. Write payloads (+ optional history footer **without** zone/index refs; `history_version` may be 1 at this step, or 2 with both refs absent).
2. Hash **the entire file bytes at that moment** → `packed_sha256`.
3. Build `ZONE.json` / optional `INDEX.json` with existing APIs (`zone_maps_json_for_packed` / `prepare_secondary_index`), using that hash and generation `G`.
4. Rewrite the footer: insert spills, set refs, set `history_version = 2`, rewrite `history_json` + tail. File grows; `packed_sha256` still names the pre-spill image.

Do **not** re-hash after append. Do **not** include spill bytes in `packed_sha256`.

## Discovery order (query)

For a planned dataset, resolve zone maps and secondary indexes in this order:

1. **Embedded** — if `flags & 1`, `history_version == 2`, and the matching `*_ref` is present: read the spill bytes, parse JSON, check `version` / known kinds (no full-file `packed_sha256` re-hash; standalone placeholder is fine). Failures → treat as missing (fall back); do not refuse the open for pushdown alone.
2. **Generation sidecar** — else if the path resolves to a package generation (or store pin) with `ZONE.json` / `INDEX.json` (+ MANIFEST rules for INDEX, including `packed_sha256` string match): unchanged 0.6.x behavior.
3. **Missing** — no pushdown; answers unchanged.

Embedded wins over a sibling sidecar when both exist (publish **MAY** keep sidecars as transitional mirrors; see below). A stray `ZONE.json` beside a standalone `.tet` that also embeds **MUST** be ignored for planning once the embed refs parse.

Standalone files have no `MANIFEST.json`. Embedded INDEX usability therefore uses the in-document `version` / kind checks only (no manifest `secondary_indexes` token). Packages that embed **SHOULD** still set MANIFEST fields for sidecar mirrors during the transition.

## Package publish (transitional)

At seal / store publish for **0.7.0**:

- Embed zone (+ optional index) into `packed.tet` as above.
- **MAY** still write `ZONE.json` / `INDEX.json` beside `packed.tet` as **mirrors** of the same UTF-8 bytes (same hashes as today).
- MANIFEST `zone_maps` / `secondary_indexes` tokens remain valid for mirror discovery and verify.

A later cut **MAY** demote or drop required sidecars once embed verify is the primary path. This spec does not require dropping them in 0.7.0.

`tet pack` that copies only `packed.tet` then preserves pushdown for the packed standalone file (the point of embed).

## Writer surfaces

| Surface | Expectation |
| ------- | ----------- |
| Package `seal_generation` / store seal | Embed into `packed.tet`; optional sidecar mirrors |
| `TetWriterSession::commit` / convert | **Auto-embed** when sealing a standalone `.tet` |
| `tet index` / `tet index --clear` | Re-embed an older bare file, or drop embeds for a no-pushdown baseline |

Staging / shard writers still do **not** update indexes; seal (or convert/commit) remains the moment of truth.

## Verify

| Check | Rule |
| ----- | ---- |
| `history_version = 2` with unknown/malformed refs | Verify **error**; query open **refuse** (framing) |
| Ref present, spill unreadable / bad JSON | Verify **error** for that blob; query pushdown falls back |
| `packed_sha256` inside ZONE/INDEX ≠ pre-spill file hash | Verify **error** (`zone_maps_stale` / `secondary_indexes_stale` analogs); query falls back |
| Deep recompute | Same as sidecar deep verify: rebuild from payloads and compare documents |
| Sidecar mirror present | Bytes **SHOULD** match the embedded spill; mismatch is verify **error** during the mirror era |
| `history_version = 1` file | No embed refs allowed; ignore unknown keys; unchanged |

Absence of embed on a `history_version = 1` standalone file remains advisory (no pushdown), same as today’s missing sidecars.

## Forward compatibility

- `layout_version` stays **1**.
- `history_version` **1** = pre-embed footer framing (optional `metadata_ref` only among spills).
- `history_version` **2** = may include `zone_maps_ref` / `secondary_index_ref`; readers **MUST** account for those spill lengths in payload bounds or refuse.
- `history_version` other than 1 or 2: refuse.
- INDEX document `version` and kind strings stay on the sidecar contract; embed does not invent a parallel schema.
- Old 0.6 readers: refuse `history_version = 2` files; still open `history_version = 1` packages and use sidecars.

## Out of scope (this issue’s later cuts own them)

Mandatory indexes. Live rewrite of sealed package embeds without a new generation. Changing filter answer semantics.
