# `bin/` — `tet` CLI

The default binary (`cargo run`, `default-run = "tet"`). Entry: [`tet.rs`](tet.rs). Subcommand bodies live under [`tet/`](tet/) as `#[path = …]` modules (Cargo only compiles `tet.rs` as the bin target).

Full flag reference: [`docs/cli.md`](../../docs/cli.md).

## Commands

| Module           | Command                  | Library API                       |
| ---------------- | ------------------------ | --------------------------------- |
| `tet/info.rs`    | `tet info`               | `catalog`, `query::cli::info`     |
| `tet/verify.rs`  | `tet verify`             | `verify`, `repair`                |
| `tet/repair.rs`  | `tet repair`             | `repair`                          |
| `tet/query.rs`   | `tet query` (`q`)        | `query::execute_*`, `cli::output` |
| `tet/qhist.rs`   | `tet qhist` (`hist`)     | `query::cli::history`             |
| `tet/convert.rs` | `tet convert`            | `convert::convert_to_tet_*`       |
| `tet/export.rs`  | `tet export`             | `export::export_tet_*`            |
| `tet/args.rs`    | clap `Cli` / `Commands`  | —                                 |
| `tet/util.rs`    | stdin query read, errors | `parse_query_text`                |

## `tet query` flags (high level)

- `-t` / `--tet` — attach catalog
- `-x` / `--execute` — run operation
- `--format` — `full`, `json`, `stats`, `plan`, `quiet`, `table`
- `--preview N`, `--device DEVICE`, `--spill-allow DIR`

## Adding a subcommand

1. Add variant to `args.rs` `Commands`
2. Create `tet/<name>.rs` with `run_<name>`
3. `#[path = "tet/<name>.rs"] mod <name>;` in `tet.rs`
4. Wire in `run()` match arm

Keep heavy logic in the library crates (`query`, `catalog`, …); CLI should parse args and call library functions.
