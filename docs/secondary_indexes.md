# Optional secondary indexes (0.4.5)

Decision for [THI-568](https://linear.app/thicclatka/issue/THI-568). [THI-571](https://linear.app/thicclatka/issue/THI-571) builds and verifies this blob. [THI-570](https://linear.app/thicclatka/issue/THI-570) teaches the planner one predicate. [THI-572](https://linear.app/thicclatka/issue/THI-572) is the user-facing writeup in `layout_v1` / `query_engine` / `roadmap` / `CHANGELOG`. Those children follow this note.

Zone maps answer “can this chunk possibly match?” A secondary index answers a narrower question: “which chunks contain this exact value?” The index is optional. A missing or unusable index falls back to zone maps and a full chunk read. It must not change answers.

## Decision

One generation-scoped sidecar, `generations/<G>/INDEX.json`, named from the sealed generation’s `MANIFEST.json`.

Not inside `packed.tet`. Not a layout-v1 capability flag. Not a pointer at the package root.

### Why not a capability flag

`packed.tet` is layout v1. Readers from 0.4.0 through 0.4.4 open that image and ignore sibling files they do not know. A new superblock bit or a new region would be a byte-layout change. The history footer already uses the one defined flag. This cut does not take another bit, even if today’s reader happens to ignore unknown bits.

### Why not a catalog pointer outside the generation

`CURRENT` names a generation id. The index for `G` has to disappear with `G` (prune deletes the generation prefix). A root-level pointer would dangle or need a rewrite on every publish. Zone maps already use the pattern that fits: the blob sits next to `packed.tet`, and the manifest names it.

## Where the bytes live

```text
generations/<G>/
  packed.tet      # layout v1, unchanged
  ZONE.json       # 0.4 zone maps
  INDEX.json      # this cut, only when the manifest says "present"
  MANIFEST.json   # discovery + hashes
```

Store key: `generations/{G:08}/INDEX.json`, same id padding as `zone_maps_key`.

`tet pack` still copies only `packed.tet`. A standalone `.tet` has no manifest and no index. This cut does not honor a stray `INDEX.json` beside a standalone file: there is no manifest hash to check it against, and pack does not copy the sidecar. Sibling `ZONE.json` behavior is unchanged.

Staging writers do not create or update `INDEX.json`. Compact/publish recomputes it from the sealed `packed.tet` (payloads and the footer already in that image) and writes it once. The next generation gets a new file. Sealed bytes are not edited in place. Same rule as zone maps.

## How the planner knows an index exists

`GenerationManifest` does not use `deny_unknown_fields`, so 0.4.0–0.4.4 readers ignore new keys. Add two optional fields:

| Field | Meaning |
| ----- | ------- |
| `secondary_indexes` | Absent, `"omitted"`, or `"present"`. Any other string is unsupported. |
| `secondary_indexes_sha256` | Lowercase hex SHA-256 of the `INDEX.json` bytes. Required when the value is `"present"`. |

Constants for the impl children: file name `INDEX.json`; tokens `present` and `omitted`; document `version` **1**; kind string `value_postings`; cardinality cap **64** (`VALUE_POSTINGS_MAX_CARDINALITY`).

Discovery order for a package or store pin:

1. Read the pinned generation’s manifest. Do not re-read `CURRENT`.
2. Absent or `"omitted"` → no index. Fall back.
3. `"present"` → read the sibling file or the store key above.
4. Anything else, or a failed read → do not use a posting list. Fall back. Verify reports the manifest problem separately.
5. Use the blob only when all of these hold: JSON parses, `version` is 1, `generation` is the pin, `packed_sha256` equals the sealed image, and the bytes match `secondary_indexes_sha256`. Any miss → fall back. A bad posting list must not drop chunks.
6. Apply entries with `kind == "value_postings"` for the dataset being planned. No entry for that dataset → fall back for that dataset, even if other datasets are indexed. Unknown `kind` values are skipped.

The manifest token is the existence check. “A file happens to be there” is not.

## Forward compatibility

- `layout_version` stays **1**. No new superblock bits. No new region in `packed.tet`.
- Old readers open `packed.tet` and ignore unknown manifest keys and unknown siblings.
- `INDEX.json` with `version` other than 1: the query ignores the document and falls back. The package still opens.
- A later `kind` in the same file does not break a reader that only knows `value_postings`.

## What this cut indexes

One kind: **value posting lists** for `filter` equality (`eq`) on a low-cardinality cell value.

That is the predicate class that zone maps answer poorly. Overlapping chunk ranges all survive a min/max test; the posting list keeps only chunks that contain the value.

Coordinate-label equality (label → chunk list) and sorted-axis range lookup are the same file shape, but not this cut. Footer labels already map to positions, so a label walk skips the same payload I/O without a sidecar. A later `kind` can land in this document without a layout change.

### When a list is built

At seal, per dataset:

- Walk non-null cells. Nulls are the zone-map rule: fill value and NaN.
- When the number of distinct raw values is **≤ 64**, emit one `value_postings` entry that lists every distinct value.
- Above 64, omit that dataset. Do not write a partial list. A value left out would look like “no chunk has it” and drop a chunk that does.
- When no dataset qualifies, set `secondary_indexes` to `"omitted"` and do not write `INDEX.json`.
- When at least one dataset qualifies, write `INDEX.json` and set `"present"` plus the hash.

The cap is fixed for the cut. It is not a query-document field.

### `INDEX.json`

Pretty JSON and a trailing newline, same spelling as `ZONE.json`, so the hash is stable.

```json
{
  "version": 1,
  "generation": 1,
  "packed_sha256": "<hex of packed.tet>",
  "indexes": [
    {
      "kind": "value_postings",
      "dataset_id": 0,
      "dtype": 3,
      "postings": [
        { "value_le": "03000000", "chunks": [[0], [2]] }
      ]
    }
  ]
}
```

- `value_le` is lowercase hex of the raw little-endian element, the same encoding as zone `min_le` / `max_le`.
- `chunks` are rank-length chunk-grid coordinates, the same shape as zone `chunk_index`.
- A chunk is listed under a value only when at least one non-null cell has that bit pattern.
- Sort `indexes` by `dataset_id`, postings by `value_le`, and chunk coordinates lexicographically, so two seals of the same image hash the same.

### Planner (THI-570)

1. Zone-map pushdown stays as it is in 0.4.1–0.4.4. Only those drops increment `zone_filter_skipped_chunks`.
2. When the filter has an `eq` literal, a usable `value_postings` entry exists for that dataset, and the literal encodes as exactly one raw value of `dtype` (no rounding), drop surviving chunks that are not in that value’s posting. Count those on a new `ReadPlan` field `index_filter_skipped_chunks` (JSON, default 0).
3. Otherwise skip step 2. Cell filtering inside a kept chunk is unchanged.
4. `ne`, ranges, and null tests do not consult the posting list. An `eq` that is not an exact dtype value does not either.
5. Missing, omitted, unknown version, unknown kind, hash mismatch, or `packed_sha256` mismatch: step 2 does not run. Answers stay on the zone-map path.

Geometric touch count is `chunk_count + zone_filter_skipped_chunks + index_filter_skipped_chunks`.

## Verify (THI-571)

Absence is advisory. A blob the manifest claims is required.

| `secondary_indexes` | `INDEX.json` | Hash | Deep recompute differs | Query |
| ------------------- | ------------ | ---- | ---------------------- | ----- |
| absent (pre-0.4.5) | warning `secondary_indexes` | — | — | fallback |
| `"omitted"` | file must be absent (ok). A stray file is error `secondary_indexes` | — | — | fallback |
| `"present"` | missing or unreadable → error `secondary_indexes` | mismatch → error `secondary_indexes_sha256`. Missing hash field → error | error `secondary_indexes_stale` | fallback (do not apply postings) |
| any other string | error `secondary_indexes` | — | — | fallback |

`"present"` whose JSON `version` is not 1, but whose bytes match the manifest hash: query falls back; verify warns `secondary_indexes_version` and does not recompute. The package still opens. A hash mismatch is still an error.

Staging is not checked. Seal is the only writer.

## Out of this cut

Mandatory indexes. Updating an index without a new generation. Sorted-axis indexes and coordinate-label postings. A public index-builder API. Bytes inside `packed.tet`. Changing what `zone_filter_skipped_chunks` means.
