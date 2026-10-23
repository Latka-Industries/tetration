# Documentation: https://docs.brew.sh/Formula-Cookbook
class Tetration < Formula
  desc "Mmap-oriented chunked tensor format and tet CLI (query, convert, export, info)"
  homepage "https://docs.tetration.sh"
  url "https://github.com/Latka-Industries/tetration/archive/refs/tags/v0.7.2.tar.gz"
  sha256 "fd96bd5d77b5331158ed1d9d2aeb1d35772e15b0fe873f60733bb0e2b00d2b93"
  license any_of: ["MIT", "Apache-2.0"]

  depends_on "pkgconf" => :build
  depends_on "rust" => :build

  depends_on "hdf5"
  depends_on "netcdf"

  def install
    hdf5 = Formula["hdf5"].opt_prefix
    netcdf = Formula["netcdf"].opt_prefix
    ENV["HDF5_DIR"] = hdf5
    ENV["HDF5_ROOT"] = hdf5
    ENV["HDF5_INCLUDE_DIR"] = "#{hdf5}/include"
    ENV["HDF5_LIB_DIR"] = "#{hdf5}/lib"
    ENV["NETCDF_DIR"] = netcdf
    ENV.prepend_path "PKG_CONFIG_PATH", "#{hdf5}/lib/pkgconfig"
    ENV.prepend_path "PKG_CONFIG_PATH", "#{netcdf}/lib/pkgconfig"
    system "cargo", "install", *std_cargo_args
  end

  test do
    assert_match "Tetration CLI", shell_output("#{bin}/tet --help")
  end
end
