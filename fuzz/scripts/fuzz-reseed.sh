#!/usr/bin/env bash
# Refresh fuzz corpus *.tet / *.pkgfuzz / *.json seeds from fixtures (+ tiny reject stubs).
set -euo pipefail
root="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"
cd "$root"
mkdir -p \
  fuzz/corpus/verify_tet_bytes \
  fuzz/corpus/verify_tet_bytes_deep \
  fuzz/corpus/verify_tet_package \
  fuzz/corpus/parse_query_json \
  fuzz/corpus/parse_job \
  fuzz/corpus/secondary_index_json \
  fuzz/corpus/secondary_index_seal

cp fixtures/small/tet/sample.tet \
   fixtures/small/tet/plan.tet \
   fixtures/small/tet/multichunk_u8.tet \
   fixtures/small/tet/multichunk_u32.tet \
   fixtures/small/tet/multichunk_f16.tet \
   fuzz/corpus/verify_tet_bytes/
cp fixtures/small/tet/sample.tet \
   fixtures/small/tet/plan.tet \
   fixtures/small/tet/multichunk_u8.tet \
   fuzz/corpus/verify_tet_bytes_deep/
if [[ -d fixtures/small/tet/reject ]]; then
  cp fixtures/small/tet/reject/*.tet fuzz/corpus/verify_tet_bytes/ 2>/dev/null || true
  cp fixtures/small/tet/reject/*.tet fuzz/corpus/verify_tet_bytes_deep/ 2>/dev/null || true
fi
: > fuzz/corpus/verify_tet_bytes/empty.tet
printf 'TET\x00' > fuzz/corpus/verify_tet_bytes/too_short.tet
printf 'XXXX\x00\x00\x00\x00' > fuzz/corpus/verify_tet_bytes/bad_magic.tet
cp fuzz/corpus/verify_tet_bytes/empty.tet \
   fuzz/corpus/verify_tet_bytes/too_short.tet \
   fuzz/corpus/verify_tet_bytes/bad_magic.tet \
   fuzz/corpus/verify_tet_bytes_deep/

# --- package seeds (THI-481): length-prefixed PROTOCOL|CURRENT|MANIFEST|packed ---
enc=fuzz/scripts/encode_pkgfuzz.py
proto_ok='{"protocol_version":2}'
packed=fixtures/small/tet/sample.tet
packed_len=$(wc -c < "$packed" | tr -d ' ')
packed_sha=$(sha256sum "$packed" | awk '{print $1}')
manifest_ok=$(printf '{"generation":1,"protocol_version":2,"part_ids":["p0"],"packed_byte_len":%s,"packed_sha256":"%s"}' "$packed_len" "$packed_sha")

python3 "$enc" fuzz/corpus/verify_tet_package/healthy.pkgfuzz \
  --protocol-text "$proto_ok" --current-text $'1\n' --manifest-text "$manifest_ok" --packed "$packed"
python3 "$enc" fuzz/corpus/verify_tet_package/bad_protocol.pkgfuzz \
  --protocol-text '{"protocol_version":99}' --current-text $'1\n' --manifest-text "$manifest_ok" --packed "$packed"
python3 "$enc" fuzz/corpus/verify_tet_package/current_missing_gen.pkgfuzz \
  --protocol-text "$proto_ok" --current-text $'9\n' --manifest-text "$manifest_ok" --packed "$packed"
python3 "$enc" fuzz/corpus/verify_tet_package/missing_current.pkgfuzz \
  --protocol-text "$proto_ok" --manifest-text "$manifest_ok" --packed "$packed"
python3 "$enc" fuzz/corpus/verify_tet_package/manifest_sha_mismatch.pkgfuzz \
  --protocol-text "$proto_ok" --current-text $'1\n' \
  --manifest-text "$(printf '{"generation":1,"protocol_version":2,"part_ids":["p0"],"packed_byte_len":%s,"packed_sha256":"%s"}' "$packed_len" "$(printf '0%.0s' {1..64})")" \
  --packed "$packed"
python3 "$enc" fuzz/corpus/verify_tet_package/manifest_len_mismatch.pkgfuzz \
  --protocol-text "$proto_ok" --current-text $'1\n' \
  --manifest-text '{"generation":1,"protocol_version":2,"part_ids":["p0"],"packed_byte_len":1}' \
  --packed "$packed"
python3 "$enc" fuzz/corpus/verify_tet_package/empty_proto.pkgfuzz --protocol-text ''
# Reject packed.tet inside an otherwise coherent package (layout-v1 fail path).
reject=fixtures/small/tet/reject/truncated_tidx.tet
reject_len=$(wc -c < "$reject" | tr -d ' ')
python3 "$enc" fuzz/corpus/verify_tet_package/reject_packed.pkgfuzz \
  --protocol-text "$proto_ok" --current-text $'1\n' \
  --manifest-text "$(printf '{"generation":1,"protocol_version":2,"part_ids":["p0"],"packed_byte_len":%s}' "$reject_len")" \
  --packed "$reject"

# --- query wire seeds (THI-604 / THI-613 label range; post-0.6 shapes via conformance) ---
# Covered shapes: fixtures (mean/sum/var/slice/quantile) + conformance accept/reject
# (window rolling/bins/pad/align/step, group_by labels, align, filter, write) +
# label-range / label_filter stubs below. No separate pipeline/transform seed set —
# those keys are exercised by mutations from the existing JSON tree.
cp fixtures/queries/*.json fuzz/corpus/parse_query_json/ 2>/dev/null || true
cp conformance/reject/queries/*.json fuzz/corpus/parse_query_json/ 2>/dev/null || true
cp conformance/accept/queries/*.json fuzz/corpus/parse_query_json/ 2>/dev/null || true

# --- job file seeds (THI-651): fixtures + reject stubs (parse/validate only) ---
cp fixtures/jobs/*.json fixtures/jobs/*.toml fuzz/corpus/parse_job/ 2>/dev/null || true
cat > fuzz/corpus/parse_job/ok_verify.json <<'EOF'
{"schema_version":1,"steps":[{"op":"verify","path":"a.tet"}]}
EOF
cat > fuzz/corpus/parse_job/ok_verify.toml <<'EOF'
schema_version = 1
[[step]]
op = "verify"
path = "a.tet"
EOF
cat > fuzz/corpus/parse_job/ok_query_inline.json <<'EOF'
{"schema_version":1,"steps":[{"op":"query","query":{"dataset":"a","mean":[]}}]}
EOF
cat > fuzz/corpus/parse_job/ok_convert_verify.toml <<'EOF'
schema_version = 1
[[step]]
op = "convert"
from = "raw/x.csv"
to = "work/x.tet"
[[step]]
op = "verify"
path = "work/x.tet"
EOF
cat > fuzz/corpus/parse_job/reject_unknown_op.json <<'EOF'
{"schema_version":1,"steps":[{"op":"worker","path":"a.tet"}]}
EOF
cat > fuzz/corpus/parse_job/reject_unknown_op.toml <<'EOF'
schema_version = 1
[[step]]
op = "qhist"
path = "a.tet"
EOF
cat > fuzz/corpus/parse_job/reject_bad_schema_version.json <<'EOF'
{"schema_version":2,"steps":[{"op":"verify","path":"a.tet"}]}
EOF
cat > fuzz/corpus/parse_job/reject_bad_schema_version.toml <<'EOF'
schema_version = 99
[[step]]
op = "verify"
path = "a.tet"
EOF
cat > fuzz/corpus/parse_job/reject_missing_schema.json <<'EOF'
{"steps":[{"op":"verify","path":"a.tet"}]}
EOF
cat > fuzz/corpus/parse_job/reject_empty_steps.json <<'EOF'
{"schema_version":1,"steps":[]}
EOF
cat > fuzz/corpus/parse_job/reject_unknown_top_level.json <<'EOF'
{"schema_version":1,"retries":3,"steps":[{"op":"verify","path":"a.tet"}]}
EOF
# Deep nesting: serde_json recursion limit must fail closed (no panic).
python3 - <<'PY'
from pathlib import Path
nested = "{" + "".join('"a":{' for _ in range(80)) + '"x":1' + "}" * 80 + "}\n"
Path("fuzz/corpus/parse_job/reject_huge_nesting.json").write_text(nested)
PY

# --- secondary INDEX.json seeds (THI-613 sorted_axis_range) ---
cat > fuzz/corpus/secondary_index_json/empty_indexes.json <<'EOF'
{"version":1,"generation":1,"packed_sha256":"aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa","indexes":[]}
EOF
cat > fuzz/corpus/secondary_index_json/sorted_axis_one_segment.json <<'EOF'
{
  "version": 1,
  "generation": 1,
  "packed_sha256": "bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb",
  "indexes": [
    {
      "kind": "sorted_axis_range",
      "dataset_id": 0,
      "axis": 0,
      "ascending": true,
      "segments": [
        {
          "lo_le": "0000000000000000",
          "hi_le": "000000000000f03f",
          "chunks": [[0]]
        }
      ]
    }
  ]
}
EOF
cat > fuzz/corpus/secondary_index_json/value_postings_one.json <<'EOF'
{
  "version": 1,
  "generation": 1,
  "packed_sha256": "cccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccccc",
  "indexes": [
    {
      "kind": "value_postings",
      "dataset_id": 0,
      "dtype": 1,
      "postings": [
        {"value_le": "0000803f", "chunks": [[0, 0]]}
      ]
    }
  ]
}
EOF
cat > fuzz/corpus/secondary_index_json/unknown_kind.json <<'EOF'
{
  "version": 1,
  "generation": 1,
  "packed_sha256": "dddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddddd",
  "indexes": [
    {"kind": "future_kind_v99", "payload": {"x": 1}}
  ]
}
EOF
cat > fuzz/corpus/secondary_index_json/coord_label_postings_one.json <<'EOF'
{
  "version": 1,
  "generation": 1,
  "packed_sha256": "eeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeeee",
  "indexes": [
    {
      "kind": "coord_label_postings",
      "dataset_id": 0,
      "axis": 0,
      "postings": [
        {"label": "nyc", "chunks": [[0], [1]]}
      ]
    }
  ]
}
EOF
# Label-range query seeds for parse_query_json (THI-613 selection wire).
cat > fuzz/corpus/parse_query_json/sum_label_range.json <<'EOF'
{"dataset":"data","layout_version":1,"selection":[{"start_label":"10","stop_label":"14"},{}],"sum":[]}
EOF
cat > fuzz/corpus/parse_query_json/mean_label_range.json <<'EOF'
{"dataset":"a","selection":[{"start_label":"20.0","stop_label":"40.0"}],"mean":[]}
EOF
# Label-eq query seeds for parse_query_json (THI-614 label_filter wire).
cat > fuzz/corpus/parse_query_json/sum_label_filter.json <<'EOF'
{"dataset":"a","label_filter":{"axis":"station","eq":"nyc"},"sum":[]}
EOF
cat > fuzz/corpus/parse_query_json/sum_label_filter_and_gt.json <<'EOF'
{"dataset":"temperature","label_filter":{"axis":0,"eq":"bos"},"filter":{"gt":0},"sum":[]}
EOF

# Seal-time value-postings seeds (dtype_sel, n_elems, chunk_elems, payload).
python3 - <<'PY'
from pathlib import Path
import struct
root = Path("fuzz/corpus/secondary_index_seal")
root.mkdir(parents=True, exist_ok=True)
root.joinpath("u8_low_card.bin").write_bytes(
    bytes([0, 16, 8]) + bytes([1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1])
)
root.joinpath("u8_high_card.bin").write_bytes(bytes([0, 65, 65]) + bytes(range(65)))
root.joinpath("f32_high_card.bin").write_bytes(
    bytes([1, 128, 64])
    + b"".join(struct.pack("<f", float(i) * 0.1) for i in range(128))
)
root.joinpath("f32_low_card.bin").write_bytes(
    bytes([1, 64, 16])
    + b"".join(struct.pack("<f", v) for v in [1.0, 2.0, 1.0, 2.0] * 16)
)
print(f"secondary_index_seal: {len(list(root.glob('*.bin')))} seeds")
PY

echo "fuzz-reseed: $(ls fuzz/corpus/verify_tet_bytes/*.tet | wc -l) quick; $(ls fuzz/corpus/verify_tet_bytes_deep/*.tet | wc -l) deep; $(ls fuzz/corpus/verify_tet_package/*.pkgfuzz | wc -l) package; $(ls fuzz/corpus/parse_query_json/*.json | wc -l) query; $(ls fuzz/corpus/parse_job/*.{json,toml} 2>/dev/null | wc -l) job; $(ls fuzz/corpus/secondary_index_json/*.json | wc -l) index; $(ls fuzz/corpus/secondary_index_seal/*.bin | wc -l) seal"
