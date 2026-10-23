# Semantics profile registry

**Profiles are contracts, not code.** A `semantics_profile` attr is a documentation id that tells *tools* which declared keys to expect and how to interpret them in caller policy. The crate does **not** whitelist ids, load profile bytecode, or change fold/query answers when a profile is set.

Anyone may mint a new profile id. Shipping a profile here only means: we document the contract and (later) optional lint/snippets that target it.

Published guide index: [docs.tetration.sh/guides/semantics](https://docs.tetration.sh/guides/semantics/). Crate contract: [`semantics.md`](semantics.md).

## Registry (0.7.3)

| Profile id | Purpose | Status |
| ---------- | ------- | ------ |
| [`media_v1`](#media_v1) | Video / imaging frames along a time axis | Documented; fixture `fixtures/small/tet/recording.tet` |
| [`timeseries_v1`](#timeseries_v1) | Continuous / sampled signals | Documented |
| [`cf_lite_v1`](#cf_lite_v1) | CF-inspired climate/grid attrs (declare-only) | Documented (prefer over legacy `climate_v1`) |
| [`prov_log_v1`](#prov_log_v1) | PROV-inspired history / params expectations | Documented |
| `climate_v1` | Legacy alias for early 0.5.1 examples | **Alias** → treat as [`cf_lite_v1`](#cf_lite_v1) |
| *(any other string)* | Caller-minted | Valid; tools that do not know it ignore it |

### Shared rules (all profiles)

- Declared facts live in footer `metadata.datasets[<name>].attrs` (string map) and append-only `history`.
- Interpretation lives in `tet-semantics.toml` / `.json` (or later package `SEMANTICS.json`) — **outside** chunk payloads.
- Missing recommended keys are valid. Unknown attr keys are ignored.
- Setting `semantics_profile` **MUST NOT** change chunk planning or fold answers by itself ([`SPEC/README.md`](../SPEC/README.md) §3.6).

---

## `media_v1`

**Intent:** sealed frame stacks / video-like tensors where wall-clock ↔ index mapping is caller-side.

### Recommended attrs

| Key | Role |
| --- | ---- |
| `semantics_profile` | `"media_v1"` |
| `media_kind` | e.g. `"video"` |
| `frame_rate` | decimal string samples per `frame_rate_unit` |
| `frame_rate_unit` | usually `"Hz"` |
| `time_axis` | axis index that advances in time |
| `color_space` | optional soft label (`"sRGB"`, …) |
| `pixel_unit` | optional (`"px"`, …) |
| `start_time` | optional ISO-8601 / epoch origin |

### Tool expectations

- Policy may define `index_to_seconds` / `seconds_to_index` from `frame_rate`.
- Query documents stay index-based; map wall-clock **before** building the query.
- Do **not** decode/scale pixels because this profile is set.

### Example

`fixtures/small/tet/recording.tet` — dataset `frames`, `frame_rate=30`, `time_axis=0`, `semantics_profile=media_v1`.

---

## `timeseries_v1`

**Intent:** 1-D (or leading-time) continuous recordings (e.g. electrophys, sensors).

### Recommended attrs

| Key | Role |
| --- | ---- |
| `semantics_profile` | `"timeseries_v1"` |
| `media_kind` | e.g. `"timeseries"` |
| `sample_rate` / `sample_rate_unit` | decimal + unit (usually `"Hz"`) |
| `time_axis` | time axis index |
| `units` | physical unit of the sample values |
| `start_time` | optional origin |

### Tool expectations

- Same index↔seconds pattern as media, using `sample_rate`.
- Display unit convert (e.g. `uV` → `mV`) is caller policy only.

---

## `cf_lite_v1`

**Intent:** CF-*inspired* declare-only climate / grid labels. **Not** a CF compliance suite and **not** a climate runtime.

Legacy doc/scaffold id **`climate_v1`** means the same contract; prefer writing `cf_lite_v1` on new files.

### Recommended attrs

| Key | Role |
| --- | ---- |
| `semantics_profile` | `"cf_lite_v1"` (or legacy `"climate_v1"`) |
| `units` | e.g. `"K"` |
| `standard_name` | optional CF-style name string |
| `crs` | e.g. `"EPSG:4326"` |
| `uncertainty_kind` / `uncertainty_sibling` | optional error attachment |

### Tool expectations

- Export / notebook helpers **may** look up `standard_name` / `units` for labels.
- Kelvin→°C display convert belongs in policy (`scale` / `offset`), never as an in-crate fold rewrite.
- Convert-time CF decode (`scale_factor` / `add_offset` / `_FillValue` at **import**) is separate from this profile — it changes stored bytes at convert, not at query via profile.

---

## `prov_log_v1`

**Intent:** richer append-only provenance expectations on `history` rows (PROV-*inspired*). **Not** a W3C PROV graph engine.

### Recommended attrs

| Key | Role |
| --- | ---- |
| `semantics_profile` | `"prov_log_v1"` on datasets that participate in the log (optional) |

### History / params expectations

Prefer documenting lineage with first-class history fields and/or `history[].params` rather than inventing parallel attr keys:

| Field / param | PROV-ish role | Example |
| ------------- | ------------- | ------- |
| `agent` | who/what ran the step | `"tet-convert"` |
| `activity` | kind of activity | `"import"` |
| `entity` | output entity label | `"dataset:tas"` |
| `used` (params) | input entity id | `"s3://bucket/in.nc"` |
| `generated` (params) | output entity id | `"generations/00000003/packed.tet"` |
| `note` (params) | free text | `"QC pass"` |

`HistoryEvent::with_prov` sets first-class `agent` / `activity` / `entity` and mirrors them into `params` for older tools. `op` / `source` / `at` / `parents` remain the required row spine ([`semantics.md`](semantics.md)). Unknown history object keys are ignored on read.

### Tool expectations

- Linters may warn when `prov_log_v1` is claimed but history rows lack `agent`/`activity`.
- Tools **MUST NOT** reinterpret chunk values from provenance rows.

---

## Minting a new profile

1. Pick a stable, TOML-safe id (`snake_case` + `_vN`).
2. Document recommended attrs + tool expectations (fold-inert reminder).
3. Optionally add lint rules / policy snippets in your repo — the crate will not reject unknown ids.
