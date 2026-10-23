# Semantics policy reference snippets

Sample `tet-semantics.toml` fragments for documented profile ids ([`semantics_profiles.md`](../semantics_profiles.md)).

| File | Profile |
| ---- | ------- |
| [`media_v1.toml`](media_v1.toml) | `media_v1` |
| [`cf_lite_v1.toml`](cf_lite_v1.toml) | `cf_lite_v1` |
| [`prov_log_v1.toml`](prov_log_v1.toml) | `prov_log_v1` |

These are **caller policy** examples. The crate does not load them unless a helper is given an explicit path. Lint declared attrs with `tet semantics lint`.
