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
  fuzz/corpus/secondary_index_json

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

# --- query wire seeds (THI-604 / THI-613 label range) ---
cp fixtures/queries/*.json fuzz/corpus/parse_query_json/ 2>/dev/null || true
cp conformance/reject/queries/*.json fuzz/corpus/parse_query_json/ 2>/dev/null || true
cp conformance/accept/queries/*.json fuzz/corpus/parse_query_json/ 2>/dev/null || true

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
# Label-range query seeds for parse_query_json (THI-613 selection wire).
cat > fuzz/corpus/parse_query_json/sum_label_range.json <<'EOF'
{"dataset":"data","layout_version":1,"selection":[{"start_label":"10","stop_label":"14"},{}],"sum":[]}
EOF
cat > fuzz/corpus/parse_query_json/mean_label_range.json <<'EOF'
{"dataset":"a","selection":[{"start_label":"20.0","stop_label":"40.0"}],"mean":[]}
EOF

echo "fuzz-reseed: $(ls fuzz/corpus/verify_tet_bytes/*.tet | wc -l) quick; $(ls fuzz/corpus/verify_tet_bytes_deep/*.tet | wc -l) deep; $(ls fuzz/corpus/verify_tet_package/*.pkgfuzz | wc -l) package; $(ls fuzz/corpus/parse_query_json/*.json | wc -l) query; $(ls fuzz/corpus/secondary_index_json/*.json | wc -l) index"
