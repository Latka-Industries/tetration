# Compatibility and capability-flag policy

**Status:** normative since the **0.5.2** cut (THI-587); pack coverage grew in **0.5.3**.  
**Parent:** [`SPEC/README.md`](README.md) §5.  
**Goal:** say what a reader written today must tolerate in **2046**, and how unknown features are negotiated.

Language: **MUST** / **MUST NOT** / **SHOULD** / **MAY** (RFC 2119).

---

## 1. Version axes (independent)

| Axis | Where it lives | Bumps when |
| ---- | -------------- | ---------- |
| `layout_version` | Superblock `u32` | On-disk container framing cannot be expressed with reserved bits / additive sidecar JSON |
| `index_version` | TIDX header | Chunk-index entry wire length or field meaning changes incompatibly |
| `history_version` | `THST` tail | Footer framing / mutual exclusion rules break |
| `protocol_version` | Package `PROTOCOL` | Package roles / `CURRENT` semantics change incompatibly |
| Query wire | Document keys / response JSON | Closed-key set or answer semantics change incompatibly |
| `TET_ABI_VERSION` | C header + `tet_abi_version()` | C symbols, calling convention, or handle ownership break |

These axes **MUST NOT** be conflated with crate semver. A crate patch **MAY** ship docs and fixtures without bumping any of them. A crate minor on 0.x **MAY** bump query wire or ABI while leaving `layout_version = 1`.

**Relative policy**

- Layout stays **1** while package / query / ABI evolve, as long as sealed `packed.tet` bytes remain readable under this policy.
- Query wire bumps do **not** require a layout bump.
- ABI bumps do **not** require a layout or query bump (and vice versa).
- Package `protocol_version` bumps do **not** change `layout_version` inside `packed.tet`.

---

## 2. Frozen vs evolvable

### 2.1 Frozen for layout v1 (must keep meaning)

A conforming reader of `layout_version = 1` **MUST** treat these as fixed:

| Item | Rule |
| ---- | ---- |
| Magic `TETR` / `TIDX` / `THST` | Exact ASCII at specified offsets |
| Superblock size | 32 bytes; field offsets in [`docs/layout_v1.md`](../docs/layout_v1.md) |
| Chunk index entry size | 104 bytes; field order and types |
| Dtype tags `1…10` | Closed set; meaning frozen |
| Codecs `0` (raw) and `1` (zstd) | Closed set; meaning frozen |
| LE integers; `align8` for index base | Frozen |
| `dataset_id` = 0-based directory index | Frozen |

Writers **MUST** write reserved integer fields as **0**. Readers of v1 **MUST NOT** interpret non-zero reserved fields as defined behavior; see §3.

### 2.2 Evolvable without a layout bump

| Mechanism | How it ages |
| --------- | ----------- |
| Superblock `flags` bits | Bit `0` (`flags & 1`) = history footer (defined). Bits `1…31` are **reserved**. Writers **MUST** write `0`. A future minor **MAY** define a new bit; old readers that only test bit `0` keep working if payload bounds stay correct. |
| Reserved `u32` / `u16` / `u64` slots | Stay zero until a documented layout amendment reassigns them (usually implies `layout_version` or `index_version` bump). |
| Footer JSON object keys | Unknown keys under `history` / `metadata` **MAY** be ignored by readers that only need catalog + chunks. |
| Dataset `attrs` / semantics profiles | Additive strings; **MUST NOT** change fold answers by themselves. |
| Generation sidecars (`ZONE.json`, `INDEX.json`) | Optional; new kinds **MAY** appear beside `packed.tet` without touching the byte image. |
| `MANIFEST.json` keys | Unknown keys **MAY** be ignored (`GenerationManifest` style). Known required keys keep verify semantics. |
| Query top-level keys | **Closed** — additive only by SPEC amendment; unknown keys are hard errors (§3). |
| C ABI symbols | Additive symbols **MAY** appear without bumping `TET_ABI_VERSION` only if old symbols keep identical behavior; any break bumps the ABI. |

### 2.3 What requires a hard version bump

| Change | Bump |
| ------ | ---- |
| Reorder / resize superblock or TIDX entry | `layout_version` and/or `index_version` |
| New mandatory region before payloads that old readers cannot skip | `layout_version` |
| New chunk codec that old readers cannot decode | Prefer refuse-unknown-codec (§3); a layout bump only if the index framing itself changes |
| Rename / remove a query reduction key or change its answer semantics | Query wire major (document in CHANGELOG + SPEC); keep refuse-unknown |
| Remove or change calling convention of an existing C symbol | `TET_ABI_VERSION` |
| Change `CURRENT` / pin semantics incompatibly | `protocol_version` |

---

## 3. Negotiate unknown features

### 3.1 Refuse (hard error)

A compatible reader **MUST refuse** (clear error, no silent wrong answers):

| Unknown / unsupported | Why |
| --------------------- | --- |
| `layout_version ≠ 1` | Container framing unknown |
| `index_version ≠ 1` | Entry wire unknown |
| `history_version ≠ 1` when footer present | Footer framing unknown |
| `protocol_version ≠ 2` for packages | Package roles unknown |
| Chunk `codec` not in `{0, 1}` | Cannot decode payload safely |
| Dtype tag outside `1…10` (if claiming full dtype support) | Element size / LE layout unknown |
| Unknown **top-level** query key or reduction op | Closed control plane; ignoring would drop intent |
| Nested `operation` / `output` objects | Rejected by wire |
| Query over size/depth/rank limits | DoS / undefined planning |

A **minimal** smoke reader (catalog + one raw chunk only) **MAY** refuse reductions and zstd entirely, but **MUST** still refuse unknown layout/index versions and unknown codecs it does not implement — it **MUST NOT** treat compressed bytes as raw.

### 3.2 May ignore (safe)

A compatible reader **MAY ignore** without refusing open:

| Feature | Condition |
| ------- | --------- |
| Unknown footer / metadata / attrs keys | Still bind catalog and decode chunks |
| Unknown `MANIFEST` keys | Still pin `packed.tet` when required keys validate |
| Missing `ZONE.json` / `INDEX.json` | Answers unchanged; no chunk-skip optimization |
| `MANIFEST.secondary_indexes = "omitted"` or pre-0.4.5 manifests without the field | Open OK |
| Declared semantics / caller policy files | Open/query OK; interpretation is caller-side |
| Superblock `flags` bits other than bit `0` | **SHOULD** refuse if a set unknown bit is documented as “must understand”; until such bits exist, writers keep them 0 and readers only honor bit `0` |
| Non-zero reserved integer fields on v1 | **SHOULD** warn in verify tooling; readers **MUST NOT** invent meaning. A future SPEC **MAY** require refuse-nonzero when a bit is claimed |

### 3.3 Capability advertisement (optional)

Implementations **MAY** expose a capability set (CLI flag, JSON field, or compile-time feature) such as:

```text
layout=1 index=1 codecs=raw,zstd dtypes=1-10 query=closed-v1 footer=thst package=2
```

Hosts **SHOULD** fail fast when a required capability is missing rather than partially executing a query. The reference Rust crate is the oracle for answer equality; capability strings are informative.

---

## 4. Examples — must read vs may ignore

### Must read (2046 reader of a 2026 file)

1. Superblock `TETR` + `layout_version = 1`, dataset directory, TIDX `index_version = 1`, codecs `0`/`1`, dtypes `1…10`.
2. Payload bounds using footer suffix when `flags & 1`.
3. Package pin: read `CURRENT` once (or explicit `G`); never re-resolve mid-query.
4. Closed query document: reject unknown keys; for supported ops, match reference aggregates on the same pinned bytes.

### May ignore (still “compatible open”)

1. Footer `metadata.datasets.*.attrs.frame_rate` / `crs` / profile ids — open and mean/sum unchanged.
2. `ZONE.json` / `INDEX.json` — open without them; full scan still correct.
3. Extra `MANIFEST` fields added in a later crate — ignore unknowns.
4. History rows with extra `params` keys — ignore unknowns when not verifying provenance.

### Must refuse

1. `layout_version = 2` file with no migration story.
2. `codec = 2` (undefined) payload.
3. Query `{ "dataset": "a", "mean": [], "fancy": true }` — unknown `fancy`.
4. Package with `protocol_version: 3` when the reader only implements 2.

---

## 5. Writers and forward compatibility

- Writers **MUST** set reserved fields and undefined flag bits to **0**.
- Writers **MUST NOT** reuse generation ids.
- Writers **MUST NOT** rewrite sealed `generations/<G>/packed.tet` in place; publish `G+1`.
- New optional sidecars **SHOULD** be content-addressed or hashed in `MANIFEST` when verify is expected to check them.
- Additive query keys require a SPEC + CHANGELOG amendment and reference-engine support before any “compatible” claim.

---

## 6. Relationship to conformance

The conformance pack encodes this policy as fixtures ([`conformance/`](../conformance/README.md)):

- **Accept** — files a 0.5.x-compatible reader must open; queries whose answers must match.
- **Reject verify** — malformed framing / codec faults.
- **Reject query** — unknown top-level keys and nested `operation` (refuse-unknown).

Reference gate: `cargo test --lib conformance`.
