#!/usr/bin/env bash
# Remove mutated fuzz corpus + crash artifacts; keep *.tet / *.pkgfuzz / *.json / *.bin seeds.
set -euo pipefail
root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$root"
removed=0
if [[ -d fuzz/corpus ]]; then
  while IFS= read -r -d '' f; do
    rm -f "$f"
    removed=$((removed + 1))
  done < <(find fuzz/corpus -type f ! -name '*.tet' ! -name '*.pkgfuzz' ! -name '*.json' ! -name '*.bin' -print0 2>/dev/null)
fi
rm -rf fuzz/artifacts
echo "fuzz-clean: removed $removed mutated corpus file(s); artifacts wiped; *.tet / *.pkgfuzz / *.json / *.bin seeds kept"
