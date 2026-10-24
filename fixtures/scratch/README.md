# Wizard scratch (gitignored)

Mutable clip for the metadata wizard dogfood loop ([THI-638](https://linear.app/thicclatka/issue/THI-638)).

| Path | Role |
| ---- | ---- |
| `wizard.tet` | Working `.tet` (reset with mise) |
| `tet-semantics.toml` | Optional TOML from scaffold / wizard |

## Reset

```bash
# Bare frames [6,2,2] — no dim_names / attrs (cold start: name axes interactively)
mise semantics:scratch

# Or start from the tracked worked example (dims + media attrs already stamped)
mise semantics:scratch:prefill
```

## Loop

```bash
mise semantics:info
mise semantics:scaffold   # after you have attrs, or to see empty scaffold
mise semantics:wizard     # interactive (THI-638)
mise semantics:apply      # if you edited tet-semantics.toml by hand
```

Do **not** mutate `fixtures/small/tet/recording.tet` for wizard experiments — always reset scratch.
