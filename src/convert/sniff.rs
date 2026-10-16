//! Input format detection: file extension (with compression suffix peel) then magic bytes.

use std::fs::File;
use std::io::{self, Read};
use std::path::{Path, PathBuf};

use super::zarr::{is_zarr_v2_directory, is_zarr_v3_directory};
use super::{ConvertError, ConvertInputFormat};

/// HDF5 convert input recognition (extensions + file signature).
pub struct Hdf5ConvertInput;

impl Hdf5ConvertInput {
    pub const MAGIC: [u8; 8] = *b"\x89HDF\r\n\x1a\n";
    pub const EXTENSIONS: &[&str] = &["h5", "hdf5", "hdf", "he2", "he5"];
    pub const SUPPORTED_EXTENSIONS: &str = ".h5, .hdf5, .hdf, .he2, .he5";
}

/// Classic `NetCDF` convert input recognition (extensions + CDF magic).
pub struct NetcdfConvertInput;

impl NetcdfConvertInput {
    pub const NETCDF3_V1: [u8; 4] = *b"CDF\x01";
    pub const NETCDF3_V2: [u8; 4] = *b"CDF\x02";
    pub const NETCDF3_MAGICS: [[u8; 4]; 2] = [Self::NETCDF3_V1, Self::NETCDF3_V2];
    pub const EXTENSIONS: &[&str] = &["nc", "netcdf", "nc4", "nc3", "cdf", "ncdf"];
    pub const SUPPORTED_EXTENSIONS: &str = ".nc, .netcdf, .nc4, .nc3, .cdf, .ncdf";
}

/// CSV / TSV convert input recognition. Delimiter follows the extension.
pub struct CsvConvertInput;

impl CsvConvertInput {
    pub const CSV_EXTENSIONS: &[&str] = &["csv"];
    pub const TSV_EXTENSIONS: &[&str] = &["tsv"];
    pub const SUPPORTED_EXTENSIONS: &str = ".csv (comma), .tsv (tab)";
}

/// `NumPy` `.npy` / `.npz` convert input recognition.
pub struct NumpyConvertInput;

impl NumpyConvertInput {
    pub const MAGIC: [u8; 6] = *b"\x93NUMPY";
    pub const NPY_EXTENSIONS: &[&str] = &["npy"];
    pub const NPZ_EXTENSIONS: &[&str] = &["npz"];
    pub const SUPPORTED_EXTENSIONS: &str = ".npy, .npz";
}

/// Parquet convert input recognition.
pub struct ParquetConvertInput;

impl ParquetConvertInput {
    pub const MAGIC: [u8; 4] = *b"PAR1";
    pub const EXTENSIONS: &[&str] = &["parquet", "pq"];
    pub const SUPPORTED_EXTENSIONS: &str = ".parquet, .pq";
}

/// Zarr v3 directory store recognition (`.zarr` extension or root `zarr.json`).
pub struct ZarrConvertInput;

impl ZarrConvertInput {
    pub const EXTENSIONS: &[&str] = &["zarr"];
    pub const SUPPORTED_EXTENSIONS: &str =
        ".zarr directory store (v3 zarr.json, or v2 .zgroup / .zarray)";
}

/// Compression suffixes stripped before extension matching during convert sniff.
pub struct ConvertCompressionSuffixes;

impl ConvertCompressionSuffixes {
    pub const SUFFIXES: &[&str] = &["gz", "bz2", "xz", "zst", "zip"];
}

/// Detect importer from extension, then (when needed) the first bytes of the file.
///
/// Extension lookup is ASCII case-insensitive. Known compression suffixes (`.gz`, `.bz2`, …)
/// are stripped once before matching (e.g. `model.nc.gz` → `NetCDF`). When the extension is
/// missing or unrecognized, classic `NetCDF` (`CDF\\x01` / `CDF\\x02`), HDF5
/// (`\\x89HDF\\r\\n\\x1a\\n`), and `NumPy` (`\\x93NUMPY`) signatures are checked. NetCDF-4
/// (HDF5 container) should use a `NetCDF` extension so the `NetCDF` importer is selected.
/// `.npz` is extension-only (a bare zip signature is not treated as `NumPy`). Zarr v3
/// directory stores are detected when the path is a directory containing root `zarr.json`
/// (`zarr_format: 3`).
///
/// # Errors
///
/// Returns [`ConvertError::UnsupportedInputExtension`] when neither extension nor signature
/// match, or [`ConvertError::Catalog`] on I/O failure while sniffing.
pub fn detect_convert_format(path: &Path) -> Result<ConvertInputFormat, ConvertError> {
    if path.is_dir() && (is_zarr_v3_directory(path) || is_zarr_v2_directory(path)) {
        return Ok(ConvertInputFormat::Zarr);
    }

    let logical = logical_path_for_detection(path);
    let ext = logical
        .extension()
        .and_then(|e| e.to_str())
        .map(str::to_ascii_lowercase)
        .unwrap_or_default();

    if let Some(fmt) = format_from_extension(&ext) {
        return Ok(fmt);
    }

    if path.is_dir() && (is_zarr_v3_directory(path) || is_zarr_v2_directory(path)) {
        return Ok(ConvertInputFormat::Zarr);
    }

    match sniff_format_from_file(path) {
        Ok(fmt) => Ok(fmt),
        Err(e) if e.kind() == io::ErrorKind::InvalidData => Err(unsupported_input(path, &ext)),
        Err(e) => Err(ConvertError::Catalog(crate::catalog::CatalogError::Io(e))),
    }
}

fn format_from_extension(ext: &str) -> Option<ConvertInputFormat> {
    if Hdf5ConvertInput::EXTENSIONS.contains(&ext) {
        Some(ConvertInputFormat::H5)
    } else if NetcdfConvertInput::EXTENSIONS.contains(&ext) {
        Some(ConvertInputFormat::Netcdf)
    } else if ZarrConvertInput::EXTENSIONS.contains(&ext) {
        Some(ConvertInputFormat::Zarr)
    } else if NumpyConvertInput::NPY_EXTENSIONS.contains(&ext) {
        Some(ConvertInputFormat::Npy)
    } else if NumpyConvertInput::NPZ_EXTENSIONS.contains(&ext) {
        Some(ConvertInputFormat::Npz)
    } else if CsvConvertInput::CSV_EXTENSIONS.contains(&ext)
        || CsvConvertInput::TSV_EXTENSIONS.contains(&ext)
    {
        Some(ConvertInputFormat::Csv)
    } else if ParquetConvertInput::EXTENSIONS.contains(&ext) {
        Some(ConvertInputFormat::Parquet)
    } else {
        None
    }
}

fn logical_path_for_detection(path: &Path) -> PathBuf {
    let mut current = path.to_path_buf();
    while let Some(ext) = current.extension().and_then(|e| e.to_str()) {
        let ext_lower = ext.to_ascii_lowercase();
        if ConvertCompressionSuffixes::SUFFIXES.contains(&ext_lower.as_str()) {
            let stem = current.file_stem().unwrap_or_default();
            current = current.with_file_name(stem);
        } else {
            break;
        }
    }
    current
}

fn sniff_format_from_file(path: &Path) -> io::Result<ConvertInputFormat> {
    let mut f = File::open(path)?;
    let mut header = [0u8; 8];
    let n = f.read(&mut header)?;
    if n == Hdf5ConvertInput::MAGIC.len() && header == Hdf5ConvertInput::MAGIC {
        return Ok(ConvertInputFormat::H5);
    }
    if n >= NetcdfConvertInput::NETCDF3_V1.len()
        && NetcdfConvertInput::NETCDF3_MAGICS
            .iter()
            .any(|magic| header[..magic.len()] == magic[..])
    {
        return Ok(ConvertInputFormat::Netcdf);
    }
    if n >= NumpyConvertInput::MAGIC.len() && header[..6] == NumpyConvertInput::MAGIC {
        return Ok(ConvertInputFormat::Npy);
    }
    if n >= ParquetConvertInput::MAGIC.len() && header[..4] == ParquetConvertInput::MAGIC {
        return Ok(ConvertInputFormat::Parquet);
    }
    Err(io::Error::new(
        io::ErrorKind::InvalidData,
        "unrecognized convert input signature",
    ))
}

fn unsupported_input(path: &Path, ext: &str) -> ConvertError {
    ConvertError::UnsupportedInputExtension {
        path: path.display().to_string(),
        ext: ext.to_owned(),
        h5_ext: Hdf5ConvertInput::SUPPORTED_EXTENSIONS,
        nc_ext: NetcdfConvertInput::SUPPORTED_EXTENSIONS,
        zarr_ext: ZarrConvertInput::SUPPORTED_EXTENSIONS,
        npy_ext: NumpyConvertInput::SUPPORTED_EXTENSIONS,
    }
}
