# Declared semantics + caller policy

Optional **declared** science and media facts live in the `.tet` footer. **Interpretation** (unit convert, seconds↔index, geo crops) lives outside the file in a caller policy. The crate is not a plugin host and has no climate- or video-special mode.

**Published guide (keys, vocabulary, policy samples, examples):** [docs.tetration.sh/guides/semantics](https://docs.tetration.sh/guides/semantics/). Scaffold link target: `https://docs.tetration.sh/guides/semantics/`.

Open, verify, and query stay valid when none of these keys are present. Declared attrs do **not** change chunk planning or fold answers by themselves.

## Contract (crate-local)

- Same optional **history footer** (`THST`). `layout_version` stays **1**. No new superblock flag.
- Dataset attrs remain `BTreeMap<String, String>` (scalar strings). Nested objects are not native attr values.
- Unknown attr keys are ignored. Absence of every recommended key is valid.
- Uncertainty is two flat string keys (`uncertainty_kind`, `uncertainty_sibling`) — not a JSON object in one attr.
- Provenance uses existing `history` rows (`op`, `source`, `at`, optional `parents` / `params`).
- Caller policy file convention: `tet-semantics.toml` / `.json`. The crate does not load it unless a helper is given an explicit path.
- Lab-wide conversion graphs do **not** belong inside `packed.tet`. Package `SEMANTICS.json` is deferred.

## CLI

- **`tet semantics scaffold`**: declared attrs → policy stubs; none → catalog + site doc URL (no invented commented config).
- **`tet semantics set`**: declare or clear attrs on a dataset (rewrites footer metadata; does not reinterpret values).

## Out of scope

Plugin / dynamic code load; mandatory semantics; full CF / PROV suite; built-in climate or media modes / UCUM engine; conversion tables inside `packed.tet`; required in-tree policy interpreter.

Tracked fixture for the media example: `fixtures/small/tet/recording.tet`.
