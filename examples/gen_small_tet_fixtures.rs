//! Write tracked `fixtures/small/tet/*.tet` (same layout as `src/tests/fixture.rs` helpers).
//!
//! ```bash
//! cargo run --example gen_small_tet_fixtures
//! ```

use std::path::PathBuf;

use tetration::catalog::{
    CHUNK_PAYLOAD_CODEC_V1, DATASET_DTYPE_TAG_V1, FooterBlobV1, RawArrayWrite, TetMetadataV1,
    embed_indexes_in_tet, write_footer_blob, write_raw_array_file, write_recording_fixture,
};

fn write_multichunk_2x3(path: &PathBuf, dtype: u32, data: &[u8]) {
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: "a",
            dtype,
            shape: &[2, 3],
            chunk_shape: &[2, 2],
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data,
            file_execution: None,
        },
    )
    .unwrap();
}

fn le_f32_1_to_6() -> Vec<u8> {
    (1_u8..=6)
        .flat_map(|n| f32::from(n).to_le_bytes())
        .collect()
}

fn le_u8_1_to_6() -> Vec<u8> {
    (1_u8..=6).collect()
}

fn le_u32_1_to_6() -> Vec<u8> {
    (1_u32..=6).flat_map(u32::to_le_bytes).collect()
}

fn le_f16_1_to_6() -> Vec<u8> {
    (1_u32..=6)
        .flat_map(|n| half::f16::from_f32(n as f32).to_bits().to_le_bytes())
        .collect()
}

fn write_large(path: &PathBuf) {
    const SHAPE: [u64; 2] = [34, 64];
    const CHUNK: [u64; 2] = [4, 4];
    let ne = usize::try_from(SHAPE[0] * SHAPE[1]).unwrap();
    let mut data = vec![0u8; ne * 4];
    for (slot, i) in data.as_chunks_mut::<4>().0.iter_mut().zip(0_u32..) {
        slot.copy_from_slice(&(i as f32).to_le_bytes());
    }
    write_raw_array_file(
        path,
        &RawArrayWrite {
            name: "a",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &SHAPE,
            chunk_shape: &CHUNK,
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )
    .unwrap();
}

fn write_plan(path: &PathBuf) {
    write_multichunk_2x3(path, DATASET_DTYPE_TAG_V1.f32, &le_f32_1_to_6());
    write_footer_blob(
        path,
        &FooterBlobV1 {
            history: Vec::new(),
            metadata: Some(TetMetadataV1::default()),
            metadata_ref: None,
        },
    )
    .unwrap();
    let mut data = std::fs::read(path).unwrap();
    const TAIL: usize = 16;
    let json_len = u64::from_le_bytes(
        data[data.len() - TAIL..data.len() - TAIL + 8]
            .try_into()
            .unwrap(),
    );
    let json_end = data.len() - TAIL;
    let json_start = json_end - usize::try_from(json_len).unwrap();
    for b in &mut data[json_start..json_end] {
        *b = b'X';
    }
    std::fs::write(path, &data).unwrap();
}

fn main() {
    let dir = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("fixtures/small/tet");
    std::fs::create_dir_all(&dir).unwrap();

    write_raw_array_file(
        &dir.join("sample.tet"),
        &RawArrayWrite {
            name: "temperature",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &[2, 3],
            chunk_shape: &[2, 2],
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &le_f32_1_to_6(),
            file_execution: None,
        },
    )
    .unwrap();
    write_large(&dir.join("large.tet"));
    write_plan(&dir.join("plan.tet"));
    write_multichunk_2x3(
        &dir.join("multichunk_u8.tet"),
        DATASET_DTYPE_TAG_V1.u8,
        &le_u8_1_to_6(),
    );
    write_multichunk_2x3(
        &dir.join("multichunk_u32.tet"),
        DATASET_DTYPE_TAG_V1.u32,
        &le_u32_1_to_6(),
    );
    write_multichunk_2x3(
        &dir.join("multichunk_f16.tet"),
        DATASET_DTYPE_TAG_V1.f16,
        &le_f16_1_to_6(),
    );
    write_recording_fixture(&dir.join("recording.tet")).unwrap();

    // 0.7.0: two chunks [1,2] and [10,11] with ZONE/INDEX embedded in the image.
    let embedded = dir.join("embedded_zone.tet");
    let vals = [1.0f32, 2.0, 10.0, 11.0];
    let data: Vec<u8> = vals.iter().flat_map(|v| v.to_le_bytes()).collect();
    write_raw_array_file(
        &embedded,
        &RawArrayWrite {
            name: "a",
            dtype: DATASET_DTYPE_TAG_V1.f32,
            shape: &[4],
            chunk_shape: &[2],
            chunk_codec: CHUNK_PAYLOAD_CODEC_V1.raw,
            data: &data,
            file_execution: None,
        },
    )
    .unwrap();
    embed_indexes_in_tet(&embedded, 0).unwrap();

    println!("wrote {}", dir.display());
}
