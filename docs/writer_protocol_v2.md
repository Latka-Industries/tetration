# Writer protocol v2 — shards, generations, publish

**Status:** shipped in tetration **0.2.0** — **decisions A–L locked** (2026-10-15); revise only by explicit amendment.  
**Tracks:** [THI-462](https://linear.app/thicclatka/issue/THI-462) · series [THI-458](https://linear.app/thicclatka/issue/THI-458) · umbrella [THI-19](https://linear.app/thicclatka/issue/THI-19)  
**Depends on:** sealed layout v1 ([`layout_v1.md`](layout_v1.md)) as the **reader-visible** byte image.

## Thesis

v1 = one coordinated writer seals an immutable `.tet`; unlimited mmap readers.

**Beyond v1** = concurrent production under an explicit protocol:

```text
N writers → immutable parts/shards → single publish → sealed generation G
Readers only open sealed G (via packed.tet)
```

Same *logical* model locally and on object stores: cloud is transport for parts + the generation pointer, not a different concurrency story. **0.2 atomicity is POSIX rename; 0.3.1 object stores use CAS / conditional write** (`S3Store`, decision **K**).

### Explicit non-goal

**Parallel writers freely updating one live mmap’d catalog / TIDX.** That remains undefined and forbidden. Multi-writer means multi-*part*, then one committer.

---

## Locked decisions (A–L)

| ID | Decision | Lock |
| -- | -------- | ---- |
| **A** | What 0.2 readers open | Only `generations/<G>/packed.tet` (layout v1). Multi-file generation mmap = later. |
| **B** | What a shard is | Sealed layout-v1-compatible fragment: datasets + chunk payloads for **claimed chunk ranges** (mini-`.tet` or same framing). |
| **C** | Publish set | **Explicit part-id list** in `MANIFEST`. Staging scan is sugar that must freeze that list before build. |
| **D** | Staging after pack | Not required. Sealed verify checks **`packed.tet`**; `MANIFEST` keeps part ids + checksums for provenance only. |
| **E** | Overlap / schema | **Reject** overlapping chunk claims in one `G`. **New datasets in G+1 OK.** Dtype/shape change of an existing name = **reject** (rename or new package). |
| **F** | Package identity | Root `PROTOCOL` with `protocol_version: 2`. Generation ids = monotonic `u64`, **never reuse**. |
| **G** | Publish lock | Exclusive create of `publishing/<G>/`. Stale lock: age > N (default **1h**) + no heartbeat → verify/repair may **quarantine**; no auto-steal without `--force`. |
| **H** | Append | Standalone `.tet` + `TetWriterSession::open_append` = **v1**. Package “append” = ingest → **publish G+1**. Never mutate `packed.tet` in place. |
| **I** | Compact / publish / pack | **`publish`** = build `packed.tet` + flip `CURRENT`. **Compact** = optional pre-merge inside the same commit. **`pack` CLI** = copy sealed `G` → standalone `.tet` (no `CURRENT` flip). |
| **J** | GC / pins | Retain at least **`current` and `current-1`**. Older GC is opt-in. Do not unlink under an active mmap. |
| **K** | Atomic `CURRENT` | 0.2: POSIX write-temp + `rename` on same filesystem. 0.3: CAS/conditional write; same logical pointer. |
| **L** | Impl order | Package skeleton roles (`CURRENT`, `generations/`, `staging/`, `publishing/`) are **normative for 0.2** before/with shard API ([THI-468](https://linear.app/thicclatka/issue/THI-468) + [THI-463](https://linear.app/thicclatka/issue/THI-463)). |

### Soft defaults (locked in code unless amended)

- **Part files:** `part-<id>.tet` (layout-v1 mini-`.tet`), not a separate `.shard` codec
- **Heartbeat:** `publishing/<G>/HEARTBEAT` with unix timestamp text; stale = mtime age > N
- Retain window beyond `current` / `current-1` — still open
- Cross-host committer election — **0.3.2** (`begin_publish_store` / `PublishOptions::force` on `PackageStore`)

---

## Vocabulary

| Term | Meaning |
| ---- | ------- |
| **Package** | Directory store **`.tet.d`** with `PROTOCOL`, `CURRENT`, `staging/`, `generations/`, `publishing/`. |
| **Shard / part** | Immutable layout-v1-compatible fragment (decision **B**). Sealed after part finalize; never mutated. |
| **Part id** | Stable id recorded in the publish set and `MANIFEST` (decision **C**). |
| **Generation `G`** | Published image identified by monotonic `u64` (never reused). Reader-visible form in 0.2: **`packed.tet`**. |
| **`CURRENT`** | Atomic pointer naming the latest sealed generation id. |
| **Publish** | Build `packed.tet` for `G`, write `MANIFEST`, flip `CURRENT → G` (decision **I**). |
| **Committer** | Exactly one process role allowed to publish a given `G`. |
| **Pin** | Reader opens `packed.tet` for a specific `G` and keeps that handle for the query (no mid-query re-resolve of `CURRENT`). |
| **Standalone `.tet`** | Layout-v1 single file outside a package (v1 writers/readers; `pack` export target). |

---

## Lifecycle (roles, not one object)

Three **roles** objects pass through — do not implement as “mv shard dir into publishing.”

| Role | Who writes | Reader-visible? | Rules |
| ---- | ---------- | --------------- | ----- |
| **staging** | Shard writers | **No** | Incomplete temps OK. Crash = garbage; committer ignores unsealed parts. |
| **publishing** | Committer only | **No** | Exclusive `publishing/<G>/`. Incomplete `G` must never become `CURRENT`. |
| **sealed** | Nobody | **Yes** | `generations/<G>/packed.tet` (+ `MANIFEST`) immutable. |

```text
writers:   create staging/<session>/ → seal parts (PART_META + payloads)
committer: snapshot part-id list → publishing/<G>/ → packed.tet + MANIFEST
           → rename CURRENT → G → sealed
readers:   read CURRENT once (or explicit G) → pin generations/<G>/packed.tet
```

---

## Package layout (normative roles for 0.2)

Filenames below are the 0.2 default; roles are mandatory (decision **L**).

```text
example.tet.d/
  PROTOCOL                     # protocol_version: 2 (+ optional created_at)
  CURRENT                      # generation id (text or tiny JSON)
  generations/
    00000001/
      MANIFEST.json            # generation id, part ids, checksums, catalog summary
      packed.tet               # REQUIRED for sealed G in 0.2 (decision A)
    00000002/
      …
  staging/
    <writer_session_id>/
      PART_META.json           # sealed flag, part id, claims, content hash
      part-<id>.tet            # or .shard — layout-v1-compatible fragment (decision B)
  publishing/
    <G>/                       # exclusive create = publish lock (decision G)
      …
```

**Atomicity of `CURRENT` (0.2):** write-temp + `rename` on the same filesystem (decision **K**). Cross-volume packages: pack to standalone `.tet`, then copy — do not rely on rename across volumes.

**N=1 path:** one writer seals one part → publish `G=1` → readers use `packed.tet`. Equivalent UX to today’s single-file write when the tool also runs `pack` or writes a standalone `.tet` directly (v1 path remains valid — decision **H**).

---

## Writer protocol

### Shard writer (many)

1. Create unique `staging/<session_id>/` (fail on collision).
2. Write part payload to a temp name; fsync; rename to final part name.
3. Bind **part id + content hash** in `PART_META`; set `sealed: true` **last** (meta-last). Committer ignores sessions without sealed meta or with hash mismatch.
4. Exit. Do **not** update `CURRENT`.

**Claims (decision E):** `PART_META` declares dataset names and chunk-grid ranges covered. For a given publish set:

- Overlapping chunk claims → **publish error**
- Filling disjoint chunks of an existing dataset → OK
- Introducing a **new** dataset name → OK (typically in `G+1` relative to prior packed catalog)
- Changing dtype or global shape of an existing dataset name → **reject**

### Committer (exactly one per publish)

1. **Freeze** an explicit **part-id list** (decision **C**). If using a staging scan, snapshot the list into memory/`MANIFEST` draft *before* building; parts sealed after the snapshot are not in this `G`.
2. Exclusive-create `publishing/<G>/` (decision **G**). On collision: abort (not corruption). Heartbeat file while working; default stale threshold **1h**.
3. Build **`packed.tet`** (layout v1) from the frozen parts (optional compact/merge first — decision **I**).
4. Fsync `packed.tet`; write `MANIFEST.json` with: `G`, frozen part ids, provenance checksums, catalog summary, checksum of `packed.tet`.
5. Move/rename completed tree into `generations/<G>/` (must contain `packed.tet` + `MANIFEST`).
6. Atomically set `CURRENT` → `G`.
7. Staging GC optional afterward (decision **D**): sealed verify must **not** require staging files.

**Stale lock:** if `publishing/<G>/` is older than N and heartbeat is dead, `tet verify` / `tet repair` may **quarantine** the dir. Taking over requires explicit **`--force`** (decision **G**). On `PackageStore`, the same rule is [`PublishOptions::force`](../src/catalog/store/publish_lock.rs) via `begin_publish_store` / `publish_staged_parts_with_options` (0.3.2).

### Crash behavior

| Failure | Result |
| ------- | ------ |
| Writer dies mid-part | Unsealed staging; ignored by committer |
| Committer dies mid-publish | `CURRENT` unchanged; incomplete `publishing/<G>/` ignored or quarantined; retry or `--force` |
| Crash after `CURRENT` rename | `G` is live; GC staging only |

---

## Reader protocol

1. Resolve target: explicit `G` **or** read `CURRENT` **once**.
2. **Pin** `generations/<G>/packed.tet` for the lifetime of the handle / query (decision **A**).
3. Never read `staging/` or `publishing/`. Never re-resolve `CURRENT` mid-query.
4. If `CURRENT` later advances to `G+1`, pinned readers keep seeing `G` until reopen.

**Standalone `.tet`:** unchanged v1 reader contract. `tet pack` copies a sealed generation’s `packed.tet` to a portable path without flipping `CURRENT` (decision **I**).

---

## Relationship to layout v1

| Concern | Rule |
| ------- | ---- |
| Reader-visible bytes in a package (0.2) | Always layout-v1 **`packed.tet`**. |
| `layout_version` | Remains **1** inside `packed.tet`. Package uses **`protocol_version: 2`**. |
| Multi-file generation mmap | **Out of scope for 0.2** (decision **A**). |
| `TetWriterSession::open_append` on standalone `.tet` | Still v1 (decision **H**). |
| Package append | New parts + publish **G+1** only. |

---

## Verify / repair expectations

| Condition | Severity | Finding `check` |
| --------- | -------- | --------------- |
| Missing / invalid `PROTOCOL` or `protocol_version ≠ 2` | **Error** | `protocol` |
| `CURRENT` absent while sealed `generations/` exist | **Error** | `current` |
| `CURRENT` unparseable | **Error** | `current` |
| `CURRENT` → missing `generations/<G>/packed.tet` | **Error** | `current_packed` |
| Bad / missing `MANIFEST` for CURRENT | **Error** | `current_manifest` / `manifest_*` |
| `packed.tet` fails layout-v1 verify | **Error** | `packed_layout` (+ nested e.g. `chunk_coords`) |
| Checksum / length mismatch vs `MANIFEST` | **Error** | `manifest_sha256` / `manifest_byte_len` |
| Overlapping claims at publish time | **Error** (publish reject) | — (never seals) |
| Duplicate chunk coords inside sealed `packed.tet` | **Error** | `chunk_coords` |
| `MANIFEST` part ids missing from staging | **OK** if `packed.tet` verifies (decision **D**) | — |
| Incomplete `publishing/<G>/` (live lock) | **Warn** | `publishing_incomplete` |
| Stale / quarantinable `publishing/` | **Warn** | `publishing_stale` / `publishing` |
| Orphan `staging/` | **Warn** | `staging` |
| Generation reuse of id | **Error** | (publish / seal path) |
| Standalone `.tet` | Existing v1 verify unchanged | — |

Crash mid-publish must leave prior `CURRENT` readable; incomplete `G` is never reader-visible until `set_current`.

**GC (decision J):** default retain `current` and `current-1`. Deleting older `generations/<G>/` is opt-in; never unlink a `packed.tet` that may still be mmap’d.

Library: [`verify_tet_package`](../src/verify/package.rs). CLI: `tet verify path.tet.d`. Repair: `tet repair path.tet.d --apply quarantine_publish` for stale locks. Finding-code table: [`docs/cli.md`](cli.md#package-finding-check-names). Reject / mid-publish coverage: `src/tests/package_verify.rs`, `src/tests/publish.rs` ([THI-466](https://linear.app/thicclatka/issue/THI-466), [THI-480](https://linear.app/thicclatka/issue/THI-480)).

---

## CLI surface (aligned with decision I)

| Command | Role |
| ------- | ---- |
| `tet ingest` | Shard writer → `staging/` |
| `tet publish` | Freeze part list → build `packed.tet` → flip `CURRENT` |
| `tet compact` | Optional merge of parts **during** publish (not a separate visibility step) |
| `tet pack` | Copy sealed `G`’s `packed.tet` → standalone `.tet` (optional `--remove-package` after verified copy) |

[THI-467](https://linear.app/thicclatka/issue/THI-467) · [THI-464](https://linear.app/thicclatka/issue/THI-464).

---

## Implementation order

1. This RFC (THI-462) — **locked A–L** ✅
2. Package skeleton + `PROTOCOL` / `CURRENT` roles — THI-468 (partial: `TetPackage`) ✅
3. Shard writer API — THI-463 (`TetShardWriter` / `PartMeta`) ✅
4. Compact-inside-publish + atomic publish — THI-464 (`publish_parts` / `merge_tet_files`)
5. Verify / repair / fuzz — THI-466 (`verify_tet_package`, `tet verify` on `.tet.d`)
6. CLI ingest / publish / pack — THI-467 (`tet ingest` / `publish` / `compact` / `pack`)
7. ~~tet-py **0.2.0** + zero-copy mmap → NumPy~~ — [THI-21](https://linear.app/thicclatka/issue/THI-21) **Done**

---

## Related reading

- Iceberg / Delta — snapshot + atomic pointer (cousin)
- Zarr — many writers → chunk blobs (write-shape cousin)
- [`layout_v1.md`](layout_v1.md) — sealed single-file contract
- [`roadmap.md`](roadmap.md) — Phase 17; cloud (12) builds on sealed generations + **K**
