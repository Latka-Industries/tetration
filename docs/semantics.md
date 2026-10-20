# Declared semantics + caller policy (0.5.1)

Optional **declared** science and media facts live in the `.tet` footer. **Interpretation** (unit convert, seconds↔index, geo crops) lives outside the file in a caller policy. The crate is not a plugin host and has no climate- or video-special mode.

Open, verify, and query stay valid when none of these keys are present.

## Two layers

| Layer | Where | What |
| ----- | ----- | ---- |
| **Declared** | Footer `metadata.datasets[<name>].attrs` + append-only `history` | Facts about this sealed file |
| **Interpreted** | `tet-semantics.toml` or `tet-semantics.json` beside the app | What the app does with those facts |

```text
.tet attrs/history  →  facts (units, frame_rate, crs, …)
tet-semantics.toml  →  policy (display °C, index↔seconds, …)
docs examples       →  vocabulary + worked policies
```

Lab-wide conversion graphs do **not** belong inside `packed.tet`. A package `SEMANTICS.json` sidecar is deferred.

## Layout

- Same optional **history footer** as today (`THST`). `layout_version` stays **1**. No new superblock flag.
- Dataset attrs remain `BTreeMap<String, String>` (scalar strings). Nested objects are not native attr values.
- Unknown attr keys are ignored (forward compatible). Readers that do not know this document still open the file.
- Absence of every key below is valid.

## Declared keys

Recommended attr names on a dataset. Values are strings; numeric rates are decimal text (e.g. `"30"`).

| Key | Meaning | Example |
| --- | ------- | ------- |
| `units` | Physical or dimensionless unit token | `"K"`, `"uV"`, `"1"` |
| `frame_rate` | Samples along `time_axis` per `frame_rate_unit` (video / imaging) | `"30"` |
| `frame_rate_unit` | Unit for `frame_rate` | `"Hz"` |
| `sample_rate` | Same idea for continuous signals | `"30000"` |
| `sample_rate_unit` | Unit for `sample_rate` | `"Hz"` |
| `time_axis` | Axis index that advances in time (decimal) | `"0"` |
| `crs` | Coordinate reference id for geo axes | `"EPSG:4326"` |
| `media_kind` | Soft label for docs/tools | `"video"`, `"timeseries"` |
| `color_space` | Soft label when pixels are color | `"sRGB"` |
| `pixel_unit` | Soft label for spatial pixel axes | `"px"` |
| `start_time` | ISO-8601 or epoch string for time origin | `"2026-10-20T15:00:00Z"` |
| `semantics_profile` | Doc profile id (not a Rust enum) | `"tetration.media.v1"` |
| `uncertainty_kind` | How uncertainty is attached | `"1sigma"` |
| `uncertainty_sibling` | Catalog name of an error / σ dataset | `"tas_err"` |
| `standard_name` | Optional CF-style name (string only) | `"air_temperature"` |

Uncertainty is two flat string keys because attrs are strings. Do not require a JSON object in one attr.

**Provenance** uses existing `history` rows (`op`, `source`, `at`, optional `parents` / `params`). Prefer richer `params` (tool, version) over inventing parallel provenance attrs.

Convert may already copy source attrs (`units`, …). That is import, not a domain engine.

## Recommended vocabulary (docs only)

Not enforced on decode. Prefer these spellings when declaring:

- Units: `K`, `degC`, `uV`, `mV`, `Hz`, `1`, `px`
- CRS: `EPSG:####` when an EPSG code exists; otherwise a short WKT or authority string
- Profile ids: `tetration.media.v1`, `tetration.climate.v1`, `tetration.timeseries.v1` (documentation anchors)

## Caller policy

File name convention: **`tet-semantics.toml`** (JSON equivalent allowed). The crate does not load this file unless a future helper is given an explicit path. Sample shape:

```toml
# tet-semantics.toml — outside the .tet

[[profiles]]
id = "tetration.media.v1"
index_to_seconds = "index / frame_rate"
seconds_to_index = "floor(seconds * frame_rate)"

[units.K]
display = "°C"
scale = 1.0
offset = -273.15

[units.Hz]
display = "Hz"
```

Policy miss or missing file → behave as today (indices and raw dtype).

## CLI (this cut)

- **`tet semantics scaffold`**: read declared attrs. If present, emit policy stubs keyed off them. If absent, emit catalog summary plus a link to the semantics guide — **no invented commented config**.
- **`tet semantics set`**: declare or clear attrs on a dataset (rewrites the history footer metadata; does not reinterpret values).

Doc link target for scaffold (site): `https://docs.tetration.sh/guides/semantics/`

## Cold start

A bare `.tet` with no footer metadata is valid.

1. Prefer convert / writer attrs when the source has them.
2. Else `tet semantics set` (or session helpers) to declare facts.
3. Else app-only policy without writing the file (fine for one pipeline; weak for sharing).
4. Re-run scaffold after declarations to get real stubs.

## Out of this cut

- Plugin / dynamic code load inside the crate
- Mandatory semantics on every file
- Full CF / PROV compliance suite
- Built-in climate or media modes / UCUM engine
- Conversion tables stored inside `packed.tet`
- Package `SEMANTICS.json` sidecar
- Required in-tree policy interpreter (docs sample is enough)

## Relation to query

Declared semantics do not change chunk planning or fold answers by themselves. Cell `filter`, zone maps, and value postings are unchanged. Policy may later map “2.5s–4s” to indices before a query document is built; that mapping is caller-side.

## Examples (docs only)

### Video / recording

Tracked fixture: `fixtures/small/tet/recording.tet` — dataset `frames`, shape `[6,2,2]`, attrs `frame_rate=30`, `frame_rate_unit=Hz`, `time_axis=0`, `media_kind=video`, `semantics_profile=tetration.media.v1`, plus a `semantics` history row.

```bash
tet info fixtures/small/tet/recording.tet
tet semantics scaffold fixtures/small/tet/recording.tet -o tet-semantics.toml
```

### Timeseries

Declare continuous-signal rates (no special crate mode):

```bash
tet semantics set signal.tet --dataset voltage \
  --set sample_rate=30000 --set sample_rate_unit=Hz \
  --set time_axis=0 --set media_kind=timeseries \
  --set units=uV --set semantics_profile=tetration.timeseries.v1
```

### Climate as attrs only

Use CF-style string attrs and a doc profile id. There is **no** climate engine in the crate:

```bash
tet semantics set grid.tet --dataset tas \
  --set units=K --set standard_name=air_temperature \
  --set crs=EPSG:4326 --set semantics_profile=tetration.climate.v1
```

Site guide (scaffold link target): [semantics](https://docs.tetration.sh/guides/semantics/).
