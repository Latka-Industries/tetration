# Declared semantics + caller policy

Optional **declared** science and media facts live in the `.tet` footer. **Interpretation** (unit convert, seconds↔index, geo crops) lives outside the file in a caller policy. The crate is not a plugin host and has no climate- or video-special mode.

**Published guide (keys, vocabulary, policy samples, examples):** [docs.tetration.sh/guides/semantics](https://docs.tetration.sh/guides/semantics/). Scaffold link target: `https://docs.tetration.sh/guides/semantics/`.

**Profile contracts (registry):** [`semantics_profiles.md`](semantics_profiles.md) — `media.v1`, `timeseries.v1`, `cf_lite.v1`, `prov_log.v1`, open minting. Aliases `media_v1` / `v1.media` (and the same pattern for the others, plus `climate_v1`) normalize to those ids.

Open, verify, and query stay valid when none of these keys are present. Declared attrs do **not** change chunk planning or fold answers by themselves.

## Contract (crate-local)

- Same optional **history footer** (`THST`). `layout_version` stays **1**. No new superblock flag.
- Dataset attrs remain `BTreeMap<String, String>` (scalar strings). Nested objects are not native attr values.
- Unknown attr keys are ignored. Absence of every recommended key is valid.
- Uncertainty is two flat string keys (`uncertainty_kind`, `uncertainty_sibling`) — not a JSON object in one attr.
- Provenance uses `history` rows (`op`, `source`, `at`, optional `parents` / `params`, optional PROV-shaped `agent` / `activity` / `entity`). See [`semantics_profiles.md`](semantics_profiles.md#prov_logv1).
- Caller policy file convention: `tet-semantics.toml` / `.json`. The crate does not load it unless a helper is given an explicit path.
- Lab-wide conversion graphs do **not** belong inside `packed.tet`. Optional package **`SEMANTICS.json`** (declare-only, beside `packed.tet` in `generations/<G>/`) may mirror declared attrs for tools; ignore-unknown; **MUST NOT** change fold answers. CLI: `tet semantics package-write` / `package-show`.

## Profiles = contracts

`semantics_profile` is a **doc id**, not a Rust enum. The crate does not whitelist ids. Documented contracts live in [`semantics_profiles.md`](semantics_profiles.md). Canonical spelling is `name.vN` (`media.v1`). Legacy `climate_v1` / `cf_lite_v1` / `v1.cf_lite` normalize to `cf_lite.v1`.

## CLI

- **`tet semantics scaffold`**: declared attrs → policy stubs; none → catalog + site doc URL (no invented commented config).
- **`tet semantics set`**: declare or clear attrs on a dataset (rewrites footer metadata; does not reinterpret values).
- **`tet semantics lint`**: advisory profile-contract checks (e.g. claims `cf_lite.v1` but missing `units`). Reference policy snippets: [`semantics_snippets/`](semantics_snippets/).

## Out of scope

Plugin / dynamic code load; mandatory semantics; full CF / PROV suite; built-in climate or media modes / UCUM engine; conversion tables inside `packed.tet`; required in-tree policy interpreter that auto-applies CF scale because a profile is set.

Tracked fixture for the media example: `fixtures/small/tet/recording.tet`.
