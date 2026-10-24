//! Reset the gitignored metadata-wizard scratch clip (THI-638).
//!
//! ```bash
//! mise semantics:scratch
//! # or:
//! cargo run --example gen_wizard_scratch
//! ```

use std::fs;
use std::path::Path;

use tetration::catalog::write_wizard_scratch_fixture;

fn main() {
    let dir = Path::new("fixtures/scratch");
    fs::create_dir_all(dir).expect("create fixtures/scratch");
    let path = dir.join("wizard.tet");
    write_wizard_scratch_fixture(&path).expect("write wizard scratch");
    // Drop leftover TOML from a prior loop so confirm/redo starts clean.
    let _ = fs::remove_file(dir.join("tet-semantics.toml"));
    println!("wrote {}", path.display());
    println!("bare frames [6,2,2] — no dim_names / attrs (cold-start wizard)");
}
