# Second-language smoke reader (THI-589)

Optional proof that [`SPEC/README.md`](../../SPEC/README.md) §1 is implementable **without** the Rust crate.

| File | Role |
| ---- | ---- |
| [`read_catalog_chunk.py`](read_catalog_chunk.py) | Python **stdlib only** — open a `.tet`, print catalog, decode one raw `f32` chunk |

Not a query engine. Not tet-py. Failure means SPEC ambiguity (or a broken fixture), not a product commitment to forever maintain a second full reader.

```bash
# from repo root
python3 conformance/smoke/read_catalog_chunk.py fixtures/small/tet/sample.tet
mise run smoke:py
```

CI runs the smoke on Ubuntu after the lib suite (Python 3 is on the runner image).
