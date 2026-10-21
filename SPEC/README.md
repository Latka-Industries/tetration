# Tetration normative SPEC (0.6.x)

**Status:** draft for the **0.6.0** cut (windowed stats; builds on 0.5.2 freeze).  
**Audience:** someone implementing a **compatible `.tet` reader** (or package open + pin) without reading the Rust crate.  
**Language:** **MUST** / **MUST NOT** / **SHOULD** / **MAY** follow [RFC 2119](https://datatracker.ietf.org/doc/html/rfc2119).

This tree is the **canonical contract index**. Byte layouts, package roles, query wire tables, and C symbols live in the linked docs; where this SPEC and a linked doc disagree, **file a bug** — do not silently pick one.

| Surface | Normative detail | This SPEC |
| ------- | ---------------- | --------- |
| Layout v1 binary | [`docs/layout_v1.md`](../docs/layout_v1.md) | §1 |
| Writer protocol v2 | [`docs/writer_protocol_v2.md`](../docs/writer_protocol_v2.md) | §2 |
| Query JSON/TOML wire | [`docs/query_engine.md`](../docs/query_engine.md) | §3 |
| C ABI | [`docs/ffi.md`](../docs/ffi.md), [`include/tetration.h`](../include/tetration.h) | §4 |
| Compatibility / capability flags | [`COMPATIBILITY.md`](COMPATIBILITY.md) | §5 |
| Conformance pack | [`conformance/`](../conformance/README.md) | §6 |

**Informative (not required to claim “compatible reader”):** convert/export, GPU routing, CLI presentation, distributed fold harness, declared semantics interpretation ([`docs/semantics.md`](../docs/semantics.md)).

---

## Compatible reader (one paragraph)

A **compatible reader** is any program that, given a sealed layout-v1 `.tet` (or a package pin of `generations/<G>/packed.tet`), (1) validates the superblock magic/`layout_version`, dataset directory, and chunk index against this SPEC and [`layout_v1.md`](../docs/layout_v1.md); (2) decodes raw (`codec = 0`) and zstd (`codec = 1`) chunk payloads into the catalog dtype’s little-endian element stream; and (3) for any query document that uses only the **closed** top-level keys and reduction ops listed in §3, either rejects the document for a documented validation reason or produces the same catalog binding, chunk-touch set, and scalar/partial aggregate answers as the reference Rust engine on the same pinned bytes — without treating unknown top-level query keys as ignorable, without re-resolving package `CURRENT` mid-query, and without requiring attrs/semantics, zone maps, or secondary indexes to change those answers when they are absent.

---

## 0. Scope and non-goals

### In scope (0.5.2)

- Standalone `.tet` layout v1 and optional `THST` footer.
- Package protocol v2: shards → publish → sealed generation; reader pin of `packed.tet`.
- Flat query JSON/TOML control plane (closed keys, limits, operation tiers).
- C ABI versioning policy (`TET_ABI_VERSION`).

### Out of scope (implementers MAY ignore)

- Becoming a plugin host, climate engine, or SQL engine.
- Parallel writers mutating one live TIDX / mmap catalog.
- Requiring secondary indexes (`ZONE.json` / `INDEX.json`) for correct answers when absent.
- Full second-language query engines (optional smoke only — THI-589).

Attrs and declared semantics **MUST NOT** change fold/query answers by themselves ([`docs/semantics.md`](../docs/semantics.md)).

---

## 1. Layout v1 (binary)

**Normative:** [`docs/layout_v1.md`](../docs/layout_v1.md). Summary below is binding; tables there are authoritative for field offsets.

### 1.1 File map

All multi-byte integers are **little-endian**. `align8(n) = (n + 7) & !7`.

| # | Region | Rule |
| - | ------ | ---- |
| 1 | Superblock | Always 32 bytes at offset 0. |
| 2 | Dataset directory | Present iff `dataset_count > 0`; starts at 32. |
| 3 | Padding | 0…7 zero bytes so the chunk index is 8-byte aligned. |
| 4 | Chunk index | At `chunk_index_offset`; length `chunk_index_length`. |
| 5 | Chunk payloads | Each index row’s `payload_offset` … `+ stored_byte_len`. |

**Empty file** (`dataset_count = 0`): regions 2–5 absent; `chunk_index_offset = 32`, `chunk_index_length = 0`; file MAY end at byte 32.

Readers **MUST** ensure `chunk_index_offset + chunk_index_length` fits in the file without overflow. For every index row, `payload_offset + stored_byte_len ≤ file_len` (payload bounds exclude the optional history footer suffix when `flags & 1`).

### 1.2 Superblock (32 bytes)

| Offset | Size | Field | Constraint |
| ------ | ---- | ----- | ---------- |
| 0 | 4 | `magic` | ASCII `TETR` |
| 4 | 4 | `layout_version` | **MUST** be `1` |
| 8 | 4 | `dataset_count` | Count of dataset records |
| 12 | 4 | `flags` | Bit `1` = optional history footer; otherwise write `0` |
| 16 | 8 | `chunk_index_offset` | Offset of chunk index |
| 24 | 8 | `chunk_index_length` | Length of chunk index |

Unknown `layout_version` values **MUST** be refused.

### 1.3 Dataset directory

When `dataset_count > 0`: `u64 dataset_blob_len` at offset 32, then exactly `dataset_count` concatenated records totaling that length.

Each record: `name_len` (`u32`), `dtype` (`u32`), `ndim` (`u32`, range `1…8`), `reserved` (`u32` = `0`), UTF-8 `name`, pad to 8-byte alignment from record start, then `ndim × u64` `shape`, then `ndim × u64` `chunk_shape`.

**Dtype tags (closed):** `1=f32`, `2=f64`, `3=i32`, `4=i64`, `5=u8`, `6=u16`, `7=i16`, `8=u32`, `9=f16`, `10=u64`. Booleans are not a separate tag (import as `u8` 0/1). Unknown tags **MUST** be refused by a conforming reader that claims full dtype support; a minimal reader **MAY** refuse unsupported tags with a clear error.

`dataset_id` in the chunk index is the 0-based index into this list.

### 1.4 Chunk index

- Magic ASCII `TIDX`; `index_version` **MUST** be `1`.
- Header 32 bytes; each entry **104** bytes (`dataset_id`, eight `chunk_index` `u64`s, `payload_offset`, `raw_byte_len`, `stored_byte_len`, `codec`, `reserved`).
- Total length **MUST** equal `32 + entry_count × 104`.
- When `dataset_count > 0`, `chunk_index_offset` **MUST** equal `align8(40 + dataset_blob_len)`.

**Codecs (closed):**

| `codec` | Meaning |
| ------- | ------- |
| `0` | Raw: `stored_byte_len` **MUST** equal `raw_byte_len`; payload is LE element bytes |
| `1` | Zstd frame: `stored_byte_len` = compressed size; decode to exactly `raw_byte_len` bytes |

Other codec values **MUST** be refused (or treated as unsupported capability — see §5).

### 1.5 Optional history footer

When `flags & 1`, the file ends with UTF-8 `history_json`, optional `metadata_spill`, then `history_json_len` (`u64`), `history_version` (`u32` = `1`), magic ASCII `THST`. Readers without footer support **MAY** ignore the flag only for payload-bound purposes when the flag is **0**; when the flag is **1**, payload validation **MUST** use `file_len − footer_suffix`.

Dimension names, coordinate labels, and attrs live in footer `metadata` (inline or spill). They enable name→axis and label→index resolution at plan time; they are **not** a query index by default.

### 1.6 Generation sidecars (not layout regions)

`ZONE.json` and `INDEX.json` beside `packed.tet` are **not** part of the v1 byte image and do **not** bump `layout_version`. A reader that only opens `packed.tet` remains valid. Missing or unusable indexes **MUST NOT** change answers; they only affect chunk-skip optimization. Detail: [`docs/secondary_indexes.md`](../docs/secondary_indexes.md).

---

## 2. Writer protocol v2 (packages)

**Normative:** [`docs/writer_protocol_v2.md`](../docs/writer_protocol_v2.md) (decisions **A–L** locked).

### 2.1 Thesis

```text
N writers → immutable parts/shards → single publish → sealed generation G
Readers only open sealed G (via packed.tet)
```

Parallel writers updating one live mmap’d catalog / TIDX **MUST NOT** be implemented; that behavior is undefined.

### 2.2 Package roles (normative)

```text
example.tet.d/
  PROTOCOL                     # protocol_version: 2
  CURRENT                      # sealed generation id
  generations/<G>/
    MANIFEST.json
    packed.tet                 # layout v1 (required for sealed G)
  staging/<session>/           # writers; not reader-visible
  publishing/<G>/              # exclusive publish lock; not reader-visible
```

- `layout_version` inside `packed.tet` remains **1**; package identity uses **`protocol_version: 2`**.
- Generation ids are monotonic `u64` and **MUST NEVER** be reused.
- Publish builds `packed.tet` + `MANIFEST`, then atomically sets `CURRENT → G` (POSIX rename locally; CAS/conditional write on object stores).
- Staging after pack is **not** required for sealed verify (decision **D**).

### 2.3 Reader pin (normative)

1. Resolve target: explicit `G` **or** read `CURRENT` **once**.
2. **Pin** `generations/<G>/packed.tet` for the lifetime of the handle / query.
3. **MUST NOT** read `staging/` or `publishing/`.
4. **MUST NOT** re-resolve `CURRENT` mid-query. If `CURRENT` advances to `G+1`, pinned readers keep seeing `G` until reopen.
5. Replacing bytes under the pinned `packed.tet` key is **not** a new generation — it is a verify failure on `G`.

Standalone `.tet` uses the layout-v1 reader contract unchanged.

### 2.4 Overlap / schema (publish)

- Overlapping chunk claims with **differing** bytes → **reject**.
- Identical overlap → **allow** (idempotent retry).
- New datasets in `G+1` → **OK**.
- Dtype/shape change of an existing name → **reject** by default (opt-in replace flag; never mutate sealed `G`).

---

## 3. Query wire (JSON / TOML)

**Normative:** [`docs/query_engine.md`](../docs/query_engine.md) — especially *Query document*, *Operations*, *JSON security*, *Pin / consistency*.

### 3.1 Shape

- Flat object: required `dataset` (non-empty string).
- At most **one** top-level reduction key per document.
- Nested `"operation": {…}` and `"output": {…}` **MUST** be rejected.
- TOML is a front-end: convert to the same JSON value tree, then apply the same parser.
- Unknown top-level fields **MUST** be rejected (`deny_unknown_fields`).

### 3.2 Closed top-level keys

**Reserved:** `layout_version`, `dataset`, `selection`, `filter`, `group_by`, `window`, `pipeline`, `spill`, `write`, `execution`.

**Reduction / transform keys (at most one):**  
`sum`, `mean`, `min`, `max`, `count`, `var`, `std`, `product`, `norm_l1`, `norm_l2`, `all_finite`, `any_nan`, `any_inf`, `arg_min`, `arg_max`, `median`, `quantile`, `histogram`, `nan_count`, `inf_count`, `null_count`, `covariance`, `correlation`, `nan_mean`, `nan_std`, `transform`.

**`filter` keys (closed, AND):** `eq`, `ne`, `lt`, `le`, `gt`, `ge`, `is_null`, `is_not_null`.

**`window`:** object with closed fields `axis` (dimension name or index), `size` (positive integer), `mode` (`tumbling` | `sliding`). Requires a reduction. **MUST NOT** combine with `group_by`, `transform`, covariance/correlation, or materialize-required ops. Empty reduction axes mean reduce only within each window along the windowed axis (other axes kept). Tumbling last window **MAY** be shorter; sliding advances by one index. Unknown `window.*` fields **MUST** be rejected.

**`pipeline`:** ordered steps lowered onto the flat document (`select`, `filter`, one reduction with optional `group_by` or `window`, then `spill` or `write`). Unknown steps, repeats, and out-of-order stages **MUST** fail. Maximum **eight** steps. **MUST NOT** combine with those analytic keys at the top level.

### 3.3 Limits (defaults)

Conforming parsers **MUST** enforce at least these caps (reference: `QueryLimits::DEFAULT`):

| Limit | Default |
| ----- | ------- |
| Payload size | 1 MiB |
| JSON composite depth | 64 |
| `dataset` name length | 1024 bytes |
| Operation axis label length | 32 |
| `filter` predicates | 8 |
| `group_by` keys | 8 |
| `pipeline` steps | 8 |
| `window.size` | 1 048 576 (extent checks at plan time) |
| Rank / axes | ≤ `MAX_NDIM` (8) |

Axis tokens are non-negative ASCII decimals or dimension names from footer `dim_names` — not arbitrary expressions.

### 3.4 Implementation tiers (informative for answers, normative for claims)

| Tier | Pattern | Examples |
| ---- | ------- | -------- |
| **A** | Scalar fold (`axes: []`) | `sum`, `mean`, … |
| **B** | Partial-axis fold | same ops with non-empty axes |
| **C** | Materialize-required | `median`, `quantile`, `histogram` |

A reader that claims “compatible aggregates” for an op **MUST** match the reference answers for that op’s tier. A smoke reader **MAY** support only catalog + raw decode and refuse reductions.

### 3.5 Host path rule

The `.tet` / package path is **host-controlled** (CLI flag, API argument). Query JSON **MUST NOT** name the input file path. Spill/write output paths in the document are caller-chosen and **SHOULD** be allowlisted by hosts.

### 3.6 Attrs / indexes vs answers

- Declared attrs / semantics profiles do **not** change answers by themselves.
- Zone maps and value postings may skip chunks; when missing or unusable, planners fall back; answers **MUST** match a full scan of kept cells under the same `filter`.

---

## 4. C ABI versioning

**Normative symbols:** [`include/tetration.h`](../include/tetration.h), [`docs/ffi.md`](../docs/ffi.md).

| Rule | Requirement |
| ---- | ----------- |
| Version macro | `#define TET_ABI_VERSION` and `tet_abi_version()` **MUST** agree |
| When to bump | Breaking C symbol set, calling convention, or handle ownership rules — **independent** of crate semver patch |
| v1 surface | `tet_open` / `tet_close`, `tet_last_error` / `tet_clear_error`, `tet_summary_json`, `tet_query_json`, `tet_verify_json`, `tet_string_free` |
| JSON | Query in / structured result out match the §3 wire and `QueryResponse` JSON |
| Out of v1 | convert/import, writer session, GPU device selection, query history |

Callers **MUST** compare `tet_abi_version()` to the header they compiled against at startup. On 0.x, the C ABI **MAY** change between minors; release notes and `TET_ABI_VERSION` are the signal.

---

## 5. Compatibility / capability flags

**Normative policy:** [`COMPATIBILITY.md`](COMPATIBILITY.md) (THI-587).

Summary:

| Class | Reader behavior |
| ----- | --------------- |
| Unknown `layout_version` / `index_version` / `history_version` / `protocol_version` | **Refuse** |
| Unknown chunk `codec` | **Refuse** |
| Unknown query top-level key / reduction op | **Refuse** document |
| Unknown footer metadata keys / attrs | **MAY ignore**; **MUST NOT** invent answers from them |
| Unknown `MANIFEST` keys | **MAY ignore** |
| Missing `ZONE.json` / `INDEX.json` | **MUST** still open `packed.tet`; **MUST NOT** change answers |

Layout, query wire, package protocol, and `TET_ABI_VERSION` bump **independently**. Reserved fields stay zero until amended.

---

## 6. Conformance

**Pack:** [`conformance/`](../conformance/README.md) · machine index [`conformance/manifest.v1.json`](../conformance/manifest.v1.json).

- **Accept** — healthy `.tet` files + closed query documents with expected aggregate fields.
- **Reject verify** — malformed layout / codec fixtures (stable `check` names).
- **Reject query** — unknown keys / nested `operation` or `output`, empty dataset, dual reduction keys, and related closed-wire policy.

Reference gate: `cargo test --lib conformance` ([`src/tests/conformance.rs`](../src/tests/conformance.rs)). Fixtures remain under [`fixtures/small/tet/`](../fixtures/small/tet/) and [`fixtures/queries/`](../fixtures/queries/); the pack references them by path. Pack milestone **0.5.3** grows accept + reject coverage without changing layout or query ABI.

Optional second-language smoke (catalog + one chunk, not a query engine): [`conformance/smoke/`](../conformance/smoke/README.md).

---

## 7. Versioning of this SPEC

| Field | Meaning |
| ----- | ------- |
| Crate **0.5.2** | First release that freezes this SPEC enough to claim “compatible” via conformance |
| Pack **0.5.3** | Larger accept / reject surface (same layout + query wire; see [`conformance/`](../conformance/README.md)) |
| Crate **0.5.4** | Hygiene + `parse_query_json` fuzz (layout / query wire / pack milestone unchanged) |
| `layout_version` | On-disk container (still **1**) |
| `protocol_version` | Package roles (still **2**) |
| `TET_ABI_VERSION` | C shared library (still **1** at 0.5.x) |
| Query wire | Evolves by additive closed keys or a documented major bump; unknown keys stay hard errors |

Amendments require an explicit PR that updates this file and the linked normative docs together.

---

## Related reading

- [`docs/roadmap.md`](../docs/roadmap.md) — Phase 18 (this SPEC + conformance)
- [`docs/semantics.md`](../docs/semantics.md) — declared attrs vs caller policy
- [`README.md`](../README.md) — install, library sketch, concurrency overview
