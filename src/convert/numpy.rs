//! `NumPy` `.npy` / `.npz` → `.tet`.
//!
//! `.npz` imports every `.npy` member as its own dataset. Duplicate catalog names,
//! Fortran order, and object or pickle dtypes are hard errors.
//!
//! `.npy` payloads are read from the data offset in tiles. `.npz` is a zip of `.npy`
//! members: stored members are sliced in place; compressed members are inflated to a
//! temporary file and then read the same way. Neither path allocates the full array.

use std::fs::{self, File};
use std::io::{self, Read, Seek, SeekFrom};
use std::path::{Path, PathBuf};
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};
use std::time::Instant;

use zip::CompressionMethod;
use zip::ZipArchive;

use crate::catalog::MAX_NDIM;
use crate::utils::dtype::ElementDtype;

use super::shared::{
    ImportPlan, ensure_non_empty, ensure_strict, note_plan, write_plans_streaming,
};
use super::tile_io::tile_axis_ranges;
use super::{ConvertError, ConvertInputFormat, ConvertProgress, ConvertReport, report};

/// Default `.npy` tile target when the source has no chunk grid (64 MiB).
pub(crate) const NUMPY_DEFAULT_CHUNK_BYTES: u64 = 64 * 1024 * 1024;

/// Import one `.npy` array into a `.tet` file.
///
/// # Errors
///
/// Returns [`ConvertError`] when the header cannot be read, the array is Fortran-order
/// or pickled, or no supported numeric array remains.
pub fn convert_npy_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_npy_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_npy_to_tet`], invoking `progress` after each chunk payload is written.
///
/// # Errors
///
/// Returns [`ConvertError`] when import fails.
pub fn convert_npy_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_numpy_inner(
        input,
        output,
        parallel_jobs,
        false,
        ConvertInputFormat::Npy,
        None,
        progress,
    )
}

/// Import every `.npy` member of an `.npz` into one `.tet` file.
///
/// Each member becomes one dataset (`weights.npy` → `weights`). Stored members are read
/// from their zip offset. Compressed members are inflated one at a time to a temp file,
/// then tiled. Two members that would share a catalog name are an error.
///
/// # Errors
///
/// Returns [`ConvertError`] when the zip cannot be read, catalog names collide, or an
/// array is Fortran-order or pickled.
pub fn convert_npz_to_tet(input: &Path, output: &Path) -> Result<ConvertReport, ConvertError> {
    convert_npz_to_tet_with_progress(input, output, 0, None::<fn(ConvertProgress)>)
}

/// Like [`convert_npz_to_tet`], invoking `progress` after each chunk payload is written.
///
/// # Errors
///
/// Returns [`ConvertError`] when import fails.
pub fn convert_npz_to_tet_with_progress(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_numpy_inner(
        input,
        output,
        parallel_jobs,
        false,
        ConvertInputFormat::Npz,
        None,
        progress,
    )
}

pub(crate) fn convert_numpy_inner(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    format: ConvertInputFormat,
    chunk_bytes: Option<u64>,
    progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    convert_numpy_with_chunk_bytes(
        input,
        output,
        parallel_jobs,
        strict,
        format,
        chunk_bytes.unwrap_or(NUMPY_DEFAULT_CHUNK_BYTES),
        progress,
    )
}

pub(crate) fn convert_numpy_with_chunk_bytes(
    input: &Path,
    output: &Path,
    parallel_jobs: usize,
    strict: bool,
    format: ConvertInputFormat,
    chunk_bytes: u64,
    mut progress: Option<impl FnMut(ConvertProgress)>,
) -> Result<ConvertReport, ConvertError> {
    let started = Instant::now();
    let loaded = load_arrays(input, format)?;
    let _cleanup = loaded.cleanup;

    let mut plans = Vec::new();
    let mut skipped = Vec::new();
    let mut locs = Vec::new();
    for array in loaded.arrays {
        let name = array.name.clone();
        let decision = plan_array(&array, chunk_bytes)?;
        match decision {
            ArrayDecision::Import(plan) => {
                locs.push(NumpyLoc {
                    path: array.payload_path,
                    data_offset: array.data_offset,
                });
                note_plan(Ok(*plan), &mut plans, &mut skipped)?;
            }
            ArrayDecision::Skip { reason } => {
                super::shared::record_skip(&mut skipped, name, &reason);
            }
        }
    }

    ensure_strict(input, &skipped, strict)?;
    ensure_non_empty(
        input,
        &plans
            .iter()
            .map(|plan| plan.name.clone())
            .collect::<Vec<_>>(),
    )?;

    let mut progress_bridge = |done: u64, total: u64, dataset: &str| {
        if let Some(ref mut cb) = progress {
            cb(ConvertProgress {
                chunks_done: done,
                chunks_total: total,
                dataset: dataset.to_owned(),
            });
        }
    };

    let parallel_jobs = super::parallel::resolve_parallel_jobs(parallel_jobs);
    let source = NumpySource {
        locs: locs.into(),
        plans: plans.clone().into(),
    };
    let write_acc = write_plans_streaming(
        output,
        &plans,
        parallel_jobs,
        |job, buf| source.fill_tile(job, buf),
        Some(&mut progress_bridge as &mut dyn FnMut(u64, u64, &str)),
    )?;
    let history = super::import_metadata::finish_convert_footer_with_write_acc(
        output,
        format.history_source(),
        &plans,
        Some(write_acc),
    )?;

    Ok(report(
        input,
        output,
        &plans,
        skipped,
        history,
        started.elapsed().as_secs_f64(),
    ))
}

/// Chunk grid for an unchunked C-order array.
///
/// Tiles stay at or under `target_bytes` by splitting the leading axis and keeping every
/// later axis whole, so each tile is one contiguous read. Arrays smaller than the target
/// stay one tile.
#[must_use]
pub(crate) fn numpy_chunk_shape(shape: &[u64], elem_size: usize, target_bytes: u64) -> Vec<u64> {
    if shape.is_empty() {
        return Vec::new();
    }
    let elem = u64::try_from(elem_size).unwrap_or(1).max(1);
    let target_elems = (target_bytes / elem).max(1);
    if product(shape) <= target_elems {
        return shape.to_vec();
    }
    let mut chunk = shape.to_vec();
    for axis in 0..shape.len() {
        let rest = product(&shape[axis + 1..]);
        if rest == 0 || rest > target_elems {
            chunk[axis] = 1;
            continue;
        }
        let fit = (target_elems / rest).max(1).min(shape[axis].max(1));
        chunk[axis] = fit.max(1);
        for later in (axis + 1)..shape.len() {
            chunk[later] = shape[later].max(1);
        }
        return chunk;
    }
    chunk
}

struct LoadedArrays {
    arrays: Vec<NpyArray>,
    cleanup: TempCleanup,
}

struct NpyArray {
    name: String,
    payload_path: PathBuf,
    data_offset: u64,
    header: NpyHeader,
}

struct NpyHeader {
    dtype: Option<ElementDtype>,
    descr: String,
    shape: Vec<u64>,
    fortran_order: bool,
    object_dtype: bool,
    reject_reason: Option<String>,
}

struct NumpyLoc {
    path: PathBuf,
    data_offset: u64,
}

struct NumpySource {
    locs: Arc<[NumpyLoc]>,
    plans: Arc<[ImportPlan]>,
}

impl NumpySource {
    fn fill_tile(
        &self,
        job: &crate::catalog::StreamTileJob<'_>,
        buf: &mut [u8],
    ) -> Result<(), ConvertError> {
        let plan = &self.plans[job.dataset_id];
        let loc = &self.locs[job.dataset_id];
        let spec = plan.tile_read(job);
        read_contiguous_tile(&loc.path, loc.data_offset, &spec, buf)
    }
}

enum ArrayDecision {
    Import(Box<ImportPlan>),
    Skip { reason: String },
}

struct TempCleanup(Vec<PathBuf>);

impl Drop for TempCleanup {
    fn drop(&mut self) {
        for path in self.0.drain(..) {
            let _ = fs::remove_file(path);
        }
    }
}

fn load_arrays(input: &Path, format: ConvertInputFormat) -> Result<LoadedArrays, ConvertError> {
    match format {
        ConvertInputFormat::Npy => {
            let mut file = File::open(input).map_err(|err| numpy_io(input, &err))?;
            let header = parse_npy_header(&mut file, input)?;
            let data_offset = file
                .stream_position()
                .map_err(|err| numpy_io(input, &err))?;
            Ok(LoadedArrays {
                arrays: vec![NpyArray {
                    name: "data".to_owned(),
                    payload_path: input.to_path_buf(),
                    data_offset,
                    header,
                }],
                cleanup: TempCleanup(Vec::new()),
            })
        }
        ConvertInputFormat::Npz => load_npz(input),
        other => Err(ConvertError::Numpy(format!(
            "internal convert dispatched {other:?} to the NumPy importer"
        ))),
    }
}

fn load_npz(input: &Path) -> Result<LoadedArrays, ConvertError> {
    let members = npz_npy_members(input)?;
    let mut loaded = LoadedArrays {
        arrays: Vec::with_capacity(members.len()),
        cleanup: TempCleanup(Vec::new()),
    };
    for member in &members {
        let name = dataset_name_from_member(&member.name);
        let array = if member.compression == CompressionMethod::Stored {
            read_stored_member(input, member, name)?
        } else {
            let temp = inflate_member(input, &member.name)?;
            loaded.cleanup.0.push(temp.clone());
            read_npy_array(&temp, input, name)?
        };
        loaded.arrays.push(array);
    }
    ensure_unique_dataset_names(input, &loaded.arrays)?;
    Ok(loaded)
}

fn read_stored_member(
    input: &Path,
    member: &ZipMember,
    name: String,
) -> Result<NpyArray, ConvertError> {
    let mut file = File::open(input).map_err(|err| numpy_io(input, &err))?;
    let payload_at = stored_data_offset(&mut file, member)?;
    file.seek(SeekFrom::Start(payload_at))
        .map_err(|err| numpy_io(input, &err))?;
    let header = parse_npy_header(&mut file, input)?;
    let data_offset = file
        .stream_position()
        .map_err(|err| numpy_io(input, &err))?;
    Ok(NpyArray {
        name,
        payload_path: input.to_path_buf(),
        data_offset,
        header,
    })
}

fn read_npy_array(payload: &Path, source: &Path, name: String) -> Result<NpyArray, ConvertError> {
    let mut file = File::open(payload).map_err(|err| numpy_io(payload, &err))?;
    let header = parse_npy_header(&mut file, source)?;
    let data_offset = file
        .stream_position()
        .map_err(|err| numpy_io(payload, &err))?;
    Ok(NpyArray {
        name,
        payload_path: payload.to_path_buf(),
        data_offset,
        header,
    })
}

fn ensure_unique_dataset_names(input: &Path, arrays: &[NpyArray]) -> Result<(), ConvertError> {
    let mut seen = std::collections::BTreeSet::new();
    for array in arrays {
        if !seen.insert(array.name.clone()) {
            return Err(ConvertError::Numpy(format!(
                "npz {} has more than one array named `{}`",
                input.display(),
                array.name
            )));
        }
    }
    Ok(())
}

struct ZipMember {
    name: String,
    compression: CompressionMethod,
    header_start: u64,
    data_start: Option<u64>,
}

fn npz_npy_members(input: &Path) -> Result<Vec<ZipMember>, ConvertError> {
    let file = File::open(input).map_err(|err| numpy_io(input, &err))?;
    let mut archive = ZipArchive::new(file).map_err(|err| {
        ConvertError::Numpy(format!("failed to open npz {}: {err}", input.display()))
    })?;
    let mut members = Vec::new();
    for index in 0..archive.len() {
        let member = archive.by_index(index).map_err(|err| {
            ConvertError::Numpy(format!("failed to read npz member {index}: {err}"))
        })?;
        if member.encrypted() {
            return Err(ConvertError::Numpy(format!(
                "npz {} contains an encrypted member",
                input.display()
            )));
        }
        let name = member.name().map_err(|err| {
            ConvertError::Numpy(format!("npz member {index} has a non-utf8 name: {err}"))
        })?;
        if name.ends_with('/') || !name.to_ascii_lowercase().ends_with(".npy") {
            continue;
        }
        members.push(ZipMember {
            name: name.into_owned(),
            compression: member.compression(),
            header_start: member.header_start(),
            data_start: member.data_start(),
        });
    }
    members.sort_by(|a, b| a.name.cmp(&b.name));
    Ok(members)
}

fn stored_data_offset(file: &mut File, member: &ZipMember) -> Result<u64, ConvertError> {
    if let Some(offset) = member.data_start {
        return Ok(offset);
    }
    file.seek(SeekFrom::Start(member.header_start))
        .map_err(|err| numpy_io(Path::new("npz"), &err))?;
    let mut header = [0u8; 30];
    file.read_exact(&mut header)
        .map_err(|err| numpy_io(Path::new("npz"), &err))?;
    if &header[0..4] != b"PK\x03\x04" {
        return Err(ConvertError::Numpy(
            "npz local header is not a stored zip entry".to_owned(),
        ));
    }
    let name_len = u64::from(u16::from_le_bytes([header[26], header[27]]));
    let extra_len = u64::from(u16::from_le_bytes([header[28], header[29]]));
    member
        .header_start
        .checked_add(30)
        .and_then(|value| value.checked_add(name_len))
        .and_then(|value| value.checked_add(extra_len))
        .ok_or_else(|| ConvertError::Numpy("npz data offset overflow".to_owned()))
}

fn inflate_member(archive_path: &Path, member_name: &str) -> Result<PathBuf, ConvertError> {
    let file = File::open(archive_path).map_err(|err| numpy_io(archive_path, &err))?;
    let mut archive = ZipArchive::new(file).map_err(|err| {
        ConvertError::Numpy(format!(
            "failed to open npz {}: {err}",
            archive_path.display()
        ))
    })?;
    let mut member = archive
        .by_name(member_name)
        .map_err(|err| ConvertError::Numpy(format!("npz member `{member_name}` missing: {err}")))?;
    let temp = temp_npy_path();
    let mut out = File::create(&temp).map_err(|err| numpy_io(&temp, &err))?;
    io::copy(&mut member, &mut out).map_err(|err| {
        let _ = fs::remove_file(&temp);
        ConvertError::Numpy(format!(
            "failed to inflate npz member `{member_name}`: {err}"
        ))
    })?;
    Ok(temp)
}

fn temp_npy_path() -> PathBuf {
    static COUNTER: AtomicU64 = AtomicU64::new(0);
    let n = COUNTER.fetch_add(1, Ordering::Relaxed);
    std::env::temp_dir().join(format!("tet-npz-{}-{n}.npy", std::process::id()))
}

fn dataset_name_from_member(member: &str) -> String {
    let trimmed = member.trim_end_matches('/');
    let stem = trimmed.strip_suffix(".npy").unwrap_or(trimmed);
    if stem.is_empty() {
        "data".to_owned()
    } else {
        stem.replace('\\', "/")
    }
}

fn plan_array(array: &NpyArray, chunk_bytes: u64) -> Result<ArrayDecision, ConvertError> {
    let header = &array.header;
    if header.fortran_order {
        return Err(ConvertError::Numpy(format!(
            "array `{}` is Fortran-order; only C-order NumPy arrays are supported",
            array.name
        )));
    }
    if header.object_dtype {
        return Err(ConvertError::Numpy(format!(
            "array `{}` is an object or pickled dtype ({}); pickling is not supported",
            array.name, header.descr
        )));
    }
    if let Some(reason) = &header.reject_reason {
        return Err(ConvertError::Numpy(format!(
            "array `{}`: {reason}",
            array.name
        )));
    }
    if header.shape.is_empty() {
        return Ok(ArrayDecision::Skip {
            reason: "scalar".to_owned(),
        });
    }
    if header.shape.contains(&0) {
        return Ok(ArrayDecision::Skip {
            reason: "empty".to_owned(),
        });
    }
    if header.shape.len() > MAX_NDIM {
        return Err(ConvertError::Numpy(format!(
            "array `{}` has ndim {} (max {MAX_NDIM})",
            array.name,
            header.shape.len()
        )));
    }
    let Some(dtype) = header.dtype else {
        return Ok(ArrayDecision::Skip {
            reason: format!("unsupported dtype {}", header.descr),
        });
    };
    let chunk_shape = numpy_chunk_shape(&header.shape, dtype.elem_size(), chunk_bytes);
    Ok(ArrayDecision::Import(Box::new(ImportPlan::new(
        array.name.clone(),
        dtype,
        header.shape.clone(),
        chunk_shape,
        None,
    ))))
}

fn parse_npy_header(file: &mut File, path: &Path) -> Result<NpyHeader, ConvertError> {
    let mut magic = [0u8; 6];
    file.read_exact(&mut magic)
        .map_err(|err| numpy_io(path, &err))?;
    if magic != *b"\x93NUMPY" {
        return Err(ConvertError::Numpy(format!(
            "{} is not a NumPy .npy array (bad magic)",
            path.display()
        )));
    }
    let mut version = [0u8; 2];
    file.read_exact(&mut version)
        .map_err(|err| numpy_io(path, &err))?;
    let (major, _minor) = (version[0], version[1]);
    let header_len = match major {
        1 => {
            let mut len_buf = [0u8; 2];
            file.read_exact(&mut len_buf)
                .map_err(|err| numpy_io(path, &err))?;
            u64::from(u16::from_le_bytes(len_buf))
        }
        2 | 3 => {
            let mut len_buf = [0u8; 4];
            file.read_exact(&mut len_buf)
                .map_err(|err| numpy_io(path, &err))?;
            u64::from(u32::from_le_bytes(len_buf))
        }
        other => {
            return Err(ConvertError::Numpy(format!(
                "unsupported .npy version {other}.x in {}",
                path.display()
            )));
        }
    };
    let header_len = usize::try_from(header_len).map_err(|_| {
        ConvertError::Numpy(format!(".npy header length overflow in {}", path.display()))
    })?;
    let mut header_bytes = vec![0u8; header_len];
    file.read_exact(&mut header_bytes)
        .map_err(|err| numpy_io(path, &err))?;
    let header = String::from_utf8(header_bytes).map_err(|_| {
        ConvertError::Numpy(format!(".npy header is not UTF-8 in {}", path.display()))
    })?;
    parse_header_dict(&header).map_err(|detail| {
        ConvertError::Numpy(format!(
            "failed to parse .npy header in {}: {detail}",
            path.display()
        ))
    })
}

fn parse_header_dict(header: &str) -> Result<NpyHeader, String> {
    let fortran_order = header_bool(header, "fortran_order")?;
    let descr = header_string_or_structured(header, "descr")?;
    let shape = header_shape(header)?;
    let (dtype, object_dtype, reject_reason) = classify_descr(&descr);
    Ok(NpyHeader {
        dtype,
        descr,
        shape,
        fortran_order,
        object_dtype,
        reject_reason,
    })
}

fn header_bool(header: &str, key: &str) -> Result<bool, String> {
    let after = field_after_colon(header, key)?;
    if after.starts_with("True") {
        Ok(true)
    } else if after.starts_with("False") {
        Ok(false)
    } else {
        Err(format!("expected True or False for {key}"))
    }
}

fn header_string_or_structured(header: &str, key: &str) -> Result<String, String> {
    let after = field_after_colon(header, key)?;
    if after.starts_with('[') {
        return Ok(after.split(']').next().unwrap_or(after).to_owned());
    }
    let quote = after
        .chars()
        .next()
        .filter(|ch| *ch == '\'' || *ch == '"')
        .ok_or_else(|| format!("expected a quoted descr, got {after}"))?;
    let body = &after[quote.len_utf8()..];
    let end = body
        .find(quote)
        .ok_or_else(|| "unterminated descr string".to_owned())?;
    Ok(body[..end].to_owned())
}

fn header_shape(header: &str) -> Result<Vec<u64>, String> {
    let after = field_after_colon(header, "shape")?;
    let start = after
        .find('(')
        .ok_or_else(|| "shape is missing '('".to_owned())?;
    let end = after
        .find(')')
        .ok_or_else(|| "shape is missing ')'".to_owned())?;
    let inner = after[start + 1..end].trim();
    if inner.is_empty() {
        return Ok(Vec::new());
    }
    inner
        .split(',')
        .filter(|part| !part.trim().is_empty())
        .map(|part| {
            part.trim()
                .parse::<u64>()
                .map_err(|_| format!("bad shape component {part}"))
        })
        .collect()
}

fn field_after_colon<'a>(header: &'a str, key: &str) -> Result<&'a str, String> {
    let needle = format!("'{key}'");
    let needle_dq = format!("\"{key}\"");
    let idx = header
        .find(&needle)
        .or_else(|| header.find(&needle_dq))
        .ok_or_else(|| format!("header missing {key}"))?;
    let after_key = if header[idx..].starts_with(&needle) {
        &header[idx + needle.len()..]
    } else {
        &header[idx + needle_dq.len()..]
    };
    let colon = after_key
        .find(':')
        .ok_or_else(|| format!("header missing ':' after {key}"))?;
    Ok(after_key[colon + 1..].trim_start())
}

fn classify_descr(descr: &str) -> (Option<ElementDtype>, bool, Option<String>) {
    if descr.starts_with('[') {
        return (None, false, None);
    }
    let (endian, rest) = match descr.as_bytes().first().copied() {
        Some(b'<' | b'|') => (descr.as_bytes()[0], &descr[1..]),
        Some(b'>') => {
            return (None, false, None);
        }
        _ => {
            return (None, descr.contains('O'), None);
        }
    };
    let _ = endian;
    let Some(type_char) = rest.chars().next() else {
        return (None, false, None);
    };
    if type_char == 'O' {
        return (None, true, None);
    }
    let size: u32 = rest[type_char.len_utf8()..].parse().unwrap_or(0);
    let dtype = match (type_char, size) {
        ('f', 2) => Some(ElementDtype::F16),
        ('f', 4) => Some(ElementDtype::F32),
        ('f', 8) => Some(ElementDtype::F64),
        ('i', 2) => Some(ElementDtype::I16),
        ('i', 4) => Some(ElementDtype::I32),
        ('i', 8) => Some(ElementDtype::I64),
        ('u', 1) => Some(ElementDtype::U8),
        ('u', 2) => Some(ElementDtype::U16),
        ('u', 4) => Some(ElementDtype::U32),
        ('u', 8) => Some(ElementDtype::U64),
        _ => None,
    };
    (dtype, false, None)
}

fn read_contiguous_tile(
    path: &Path,
    data_offset: u64,
    spec: &super::shared::ImportTileRead<'_>,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let ranges = tile_axis_ranges(*spec);
    let starts: Vec<u64> = ranges.iter().map(|(start, _)| *start).collect();
    let extents: Vec<u64> = ranges.iter().map(|(start, end)| end - start).collect();
    let elem_size = spec.dtype.elem_size();
    let count = product(&extents);
    let nbytes = count
        .checked_mul(u64::try_from(elem_size).unwrap_or(1))
        .ok_or_else(|| ConvertError::Numpy("tile byte length overflow".to_owned()))?;
    if usize::try_from(nbytes).ok() != Some(buf.len()) {
        return Err(ConvertError::Numpy(format!(
            "tile byte length mismatch (expected {nbytes}, got {})",
            buf.len()
        )));
    }
    if tile_is_contiguous(spec.shape, &starts, &extents) {
        let start = linear_index(spec.shape, &starts)?;
        let offset = data_offset
            .checked_add(
                start
                    .checked_mul(u64::try_from(elem_size).unwrap_or(1))
                    .ok_or_else(|| ConvertError::Numpy("tile offset overflow".to_owned()))?,
            )
            .ok_or_else(|| ConvertError::Numpy("tile offset overflow".to_owned()))?;
        return read_exact_at(path, offset, buf);
    }
    read_rows(
        path,
        data_offset,
        spec.shape,
        &starts,
        &extents,
        elem_size,
        buf,
    )
}

fn tile_is_contiguous(shape: &[u64], starts: &[u64], extents: &[u64]) -> bool {
    let count = product(extents);
    if count <= 1 {
        return true;
    }
    let mut last = starts.to_vec();
    for (coord, extent) in last.iter_mut().zip(extents) {
        *coord = coord.saturating_add(extent.saturating_sub(1));
    }
    let Ok(start) = linear_index(shape, starts) else {
        return false;
    };
    let Ok(end) = linear_index(shape, &last) else {
        return false;
    };
    end.saturating_sub(start).saturating_add(1) == count
}

fn read_rows(
    path: &Path,
    data_offset: u64,
    shape: &[u64],
    starts: &[u64],
    extents: &[u64],
    elem_size: usize,
    buf: &mut [u8],
) -> Result<(), ConvertError> {
    let ndim = shape.len();
    if ndim == 0 {
        return Ok(());
    }
    let row_elems = extents[ndim - 1];
    let row_bytes = usize::try_from(
        row_elems
            .checked_mul(u64::try_from(elem_size).unwrap_or(1))
            .ok_or_else(|| ConvertError::Numpy("row byte length overflow".to_owned()))?,
    )
    .map_err(|_| ConvertError::Numpy("row byte length overflow".to_owned()))?;
    let rows = product(&extents[..ndim - 1]);
    let mut coord = starts.to_vec();
    let mut out_off = 0usize;
    for _ in 0..rows {
        let index = linear_index(shape, &coord)?;
        let offset = data_offset
            .checked_add(
                index
                    .checked_mul(u64::try_from(elem_size).unwrap_or(1))
                    .ok_or_else(|| ConvertError::Numpy("row offset overflow".to_owned()))?,
            )
            .ok_or_else(|| ConvertError::Numpy("row offset overflow".to_owned()))?;
        let end = out_off + row_bytes;
        read_exact_at(path, offset, &mut buf[out_off..end])?;
        out_off = end;
        bump_coord(&mut coord, starts, extents);
    }
    Ok(())
}

fn bump_coord(coord: &mut [u64], starts: &[u64], extents: &[u64]) {
    if coord.len() < 2 {
        return;
    }
    for axis in (0..coord.len() - 1).rev() {
        let limit = starts[axis] + extents[axis];
        coord[axis] += 1;
        if coord[axis] < limit {
            return;
        }
        coord[axis] = starts[axis];
    }
}

fn linear_index(shape: &[u64], coord: &[u64]) -> Result<u64, ConvertError> {
    let mut index = 0u64;
    for dim in 0..shape.len() {
        index = index
            .checked_mul(shape[dim])
            .and_then(|value| value.checked_add(coord[dim]))
            .ok_or_else(|| ConvertError::Numpy("linear index overflow".to_owned()))?;
    }
    Ok(index)
}

fn read_exact_at(path: &Path, offset: u64, buf: &mut [u8]) -> Result<(), ConvertError> {
    let mut file = File::open(path).map_err(|err| numpy_io(path, &err))?;
    file.seek(SeekFrom::Start(offset))
        .map_err(|err| numpy_io(path, &err))?;
    file.read_exact(buf).map_err(|err| numpy_io(path, &err))
}

fn product(shape: &[u64]) -> u64 {
    shape
        .iter()
        .try_fold(1u64, |acc, &dim| acc.checked_mul(dim))
        .unwrap_or(u64::MAX)
}

fn numpy_io(path: &Path, err: &io::Error) -> ConvertError {
    ConvertError::Numpy(format!("{}: {err}", path.display()))
}

#[cfg(test)]
mod tests {
    use super::numpy_chunk_shape;

    #[test]
    fn small_array_is_one_tile() {
        assert_eq!(
            numpy_chunk_shape(&[32, 32, 32], 4, 64 * 1024 * 1024),
            vec![32, 32, 32]
        );
    }

    #[test]
    fn leading_axis_splits_to_target() {
        assert_eq!(numpy_chunk_shape(&[8, 4], 4, 32), vec![2, 4]);
    }
}
