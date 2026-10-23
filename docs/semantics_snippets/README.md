# Semantics one-file examples

Sample `tet-semantics.toml` files for documented profile ids ([`semantics_profiles.md`](../semantics_profiles.md)).

| File | Profile |
| ---- | ------- |
| [`media.v1.toml`](media.v1.toml) | `media.v1` (canonical apply example) |
| [`timeseries.v1.toml`](timeseries.v1.toml) | `timeseries.v1` |
| [`cf_lite.v1.toml`](cf_lite.v1.toml) | `cf_lite.v1` |
| [`prov_log.v1.toml`](prov_log.v1.toml) | `prov_log.v1` |

Each file is declare + policy. `tet semantics apply` stamps `dim_names` and attrs only. Profile formulas and `[units.*]` stay in the TOML. The crate does not load these unless you pass `--from`.
