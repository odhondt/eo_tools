# %%
"""One-burst interpolation evaluation based on test-s1-core.py.

Run without arguments from the repository root. Settings and paths are below.
Geometry, calibration, phase warping and primary amplitude are computed once.
Only secondary SLC alignment varies. Results are written to summary.md, with
GeoTIFFs for visual comparison through the usual TiTiler/show_cog setup.
"""

import logging
import sys
import warnings
from datetime import datetime
from math import pi
from pathlib import Path
from time import perf_counter

import folium
import numba
import numpy as np
import rasterio as rio

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eo_tools.S1.core import lla_to_ecef, load_dem_coords
from eo_tools.S1.core import S1IWSwath, coregister
from eo_tools.S1.process import amplitude, coherence, sar2geo
from eo_tools.S1.util import remap
from eo_tools_dev.util import palette_phi, serve_map, show_cog
from scripts.benchmark_remap import remap_reference

logging.basicConfig(level=logging.INFO)
logging.getLogger("httpx").setLevel(logging.WARNING)

# Same test products as test-s1-core.py, restricted to one burst.
data_dir = "/data/S1"
primary_path = f"{data_dir}/S1A_IW_SLC__1SDV_20230904T063730_20230904T063757_050174_0609E3_DAA1.zip"
secondary_path = f"{data_dir}/S1A_IW_SLC__1SDV_20230916T063730_20230916T063757_050349_060FCD_6814.zip"
out_dir = Path("/data/res") / datetime.now().strftime("remap-burst-%Y%m%d-%H%M%S-%f")
ref_dir = Path(
    "/data/reference/S1_InSAR_VV_2023-09-04-063730__2023-09-16-063730_Morocco"
)
iw = 1
pol = "vv"
burst = 3
up = 1.8
threads = 4

variants = {
    "bicubic-original": (remap_reference, "bicubic"),
    "bicubic-optimized": (remap, "bicubic"),
    "sinc-original": (remap_reference, "sinc"),
    "sinc-lut": (remap, "sinc"),
    "lanczos-original": (remap_reference, "lanczos"),
    "lanczos-lut": (remap, "lanczos"),
}
baselines = {
    "bicubic": "bicubic-original",
    "sinc": "sinc-original",
    "lanczos": "lanczos-original",
}


def main() -> None:
    numba.set_num_threads(threads)
    out_dir.mkdir(parents=True, exist_ok=False)
    print("Preparing shared burst inputs and geometry", flush=True)
    start = perf_counter()
    prm = S1IWSwath(primary_path, iw=iw, pol=pol)
    sec = S1IWSwath(secondary_path, iw=iw, pol=pol)
    dem_file = prm.fetch_dem_burst(burst, dem_dir="/data/tmp", force_download=False)
    lat, lon, alt, _, composite_crs = load_dem_coords(dem_file, up)
    dem_x, dem_y, dem_z = lla_to_ecef(lat, lon, alt, composite_crs)
    az_p2g, rg_p2g = prm.geocode_burst(dem_x, dem_y, dem_z, burst_idx=burst)
    az_s2g, rg_s2g = sec.geocode_burst(dem_x, dem_y, dem_z, burst_idx=burst)
    arr_p = prm.read_burst(burst, True)
    arr_s = sec.read_burst(burst, True)
    arr_p /= prm.calibration_factor(burst, cal_type="beta")
    arr_s /= sec.calibration_factor(burst, cal_type="beta")
    pdb_s = sec.deramp_burst(burst)
    arr_s_de = arr_s * np.exp(1j * pdb_s).astype(np.complex64)
    az_s2p, rg_s2p = coregister(arr_p, az_p2g, rg_p2g, az_s2g, rg_s2g)

    # These operations are identical for every variant.
    pdb_s2p = remap_reference(pdb_s, az_s2p, rg_s2p, kernel="bicubic")
    reramp = np.exp(-1j * pdb_s2p).astype(np.complex64)
    rg_p = np.broadcast_to(np.arange(arr_p.shape[1]), arr_p.shape)
    pht_p = prm.phi_topo(rg_p)
    pht_s = sec.phi_topo(rg_s2p.ravel()).reshape(arr_p.shape)
    pha_topo = np.exp(-1j * (pht_p - pht_s)).astype(np.complex64)
    prm_file = str(out_dir / "primary.tif")
    lut_file = str(out_dir / "lut.tif")
    _write_slc(prm_file, arr_p)
    with rio.open(dem_file) as ds:
        profile = ds.profile.copy()
        # geocode_burst upsamples the DEM in memory; preserve that grid.
        height, width = az_p2g.shape
        profile.update(
            height=height,
            width=width,
            transform=ds.transform
            * ds.transform.scale(ds.width / width, ds.height / height),
        )
    profile.update(
        driver="GTiff",
        count=2,
        dtype="float64",
        nodata=np.nan,
        tiled=True,
        blockxsize=512,
        blockysize=512,
    )
    with rio.open(lut_file, "w", **profile) as ds:
        ds.write(np.stack((az_p2g, rg_p2g)))
    amp_file = str(out_dir / "amp.tif")
    amplitude(prm_file, amp_file, multilook=[2, 8])
    sar2geo(amp_file, lut_file, str(out_dir / "amp_geo.tif"), kernel="bicubic")
    preparation_seconds = perf_counter() - start
    nonfinite = np.count_nonzero(~np.isfinite(arr_s_de))
    print(f"Shared preparation complete: {preparation_seconds:.3f}s", flush=True)

    # Compile on small arrays: no extra full-burst warm-up runs. Exercise both
    # finite inputs and the normal propagation of border NaNs.
    rr, cc = np.indices((16, 16), dtype=np.float64)
    image = np.ones((16, 16), dtype=arr_s_de.dtype)
    for function, kernel in variants.values():
        print(f"Warming {function.__name__}, kernel={kernel}", flush=True)
        function(image, rr, cc, kernel)
        bad = image.copy()
        bad[0, 0] = complex(np.nan, np.nan)
        function(bad, rr, cc, kernel)

    header = (
        f"# One-burst remap evaluation\n\n"
        f"Primary: `{primary_path}`\n\nSecondary: `{secondary_path}`\n\n"
        f"IW{iw}, {pol.upper()}, burst {burst}; {threads} threads. "
        f"Shared preparation: {preparation_seconds:.3f} s (includes cold geometry JIT).\n\n"
        f"Nonfinite secondary source samples: {nonfinite}/{arr_s_de.size}. "
        "Candidates run directly without a whole-image validity scan or reference fallback. "
        "NaNs propagate through each stencil; smaller support can increase finite border coverage.\n\n"
        "Each variant is timed once after small-array JIT warm-up. "
        "Variant wall time includes alignment, reramping, "
        "writing the secondary, coherence and phase/coherence geocoding; shared preparation, "
        "error comparisons and map display are excluded. Geocoding uses production "
        "bicubic; phase warping uses original bicubic. No ESD or spectral filtering.\n\n"
        "Errors and speedups use each kernel's original implementation as the baseline; "
        "there are no cross-kernel error comparisons. Sinc is normalized in both its "
        "original reference and LUT implementations. Errors are RMS / max absolute "
        "on common valid pixels; phase is wrapped and measured in radians. "
        "Coverage columns list common finite pixels / gained finite pixels / lost finite "
        "pixels relative to that kernel's baseline. Pixels nonfinite in either output "
        "are excluded from error statistics; retained interior zeros are unchanged. "
        "LUTs are approximations (benchmark acceptance: atol=1e-6, rtol=1e-6 for "
        "remapped samples, not thresholds for these downstream metrics). "
        "SNAP is used only for visual inspection, not as numerical ground truth. "
        "Primary amplitude is computed once "
        "and reused unchanged, so its difference is zero by construction.\n\n"
        "| Variant | Baseline | Remap s | Remap speedup | Variant wall s | Phase RMS / max rad | Coherence RMS / max | Primary amplitude difference | Phase pixels (common / gained / lost) | Coherence pixels (common / gained / lost) |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    rows = []
    summary = out_dir / "summary.md"
    summary.write_text(header)
    baseline_seconds = {}
    for name, (function, kernel) in variants.items():
        directory = out_dir / name
        directory.mkdir()
        print(f"Running {name}", flush=True)
        start = perf_counter()
        arr_s2p = function(arr_s_de, az_s2p, rg_s2p, kernel)
        remap_seconds = perf_counter() - start
        print(f"  Interpolation complete: {remap_seconds:.3f}s", flush=True)
        arr_s2p *= reramp
        arr_s2p *= pha_topo
        sec_file = str(directory / "secondary.tif")
        coh_file = str(directory / "coh.tif")
        _write_slc(sec_file, arr_s2p)
        print(
            "  Reramping and secondary SLC write complete; computing coherence",
            flush=True,
        )
        coherence(
            prm_file,
            sec_file,
            coh_file,
            box_size=[3, 3],
            multilook=[1, 4],
            magnitude=False,
        )
        print("  Coherence complete; geocoding coherence", flush=True)
        sar2geo(
            coh_file,
            lut_file,
            str(directory / "coh_geo.tif"),
            kernel="bicubic",
            magnitude_only=True,
        )
        print("  Coherence geocoding complete; geocoding phase", flush=True)
        sar2geo(
            coh_file,
            lut_file,
            str(directory / "phi_geo.tif"),
            kernel="bicubic",
            write_phase=True,
        )
        total = perf_counter() - start
        print(
            f"  Phase geocoding complete; variant wall time: {total:.3f}s", flush=True
        )
        baseline = baselines[kernel]
        print(f"  Comparing phase and coherence with {baseline}", flush=True)
        if name == baseline:
            baseline_seconds[kernel] = remap_seconds
        phase_error, phase_coverage = _errors(
            out_dir / baseline / "phi_geo.tif",
            directory / "phi_geo.tif",
            phase=True,
        )
        coherence_error, coherence_coverage = _errors(
            out_dir / baseline / "coh_geo.tif", directory / "coh_geo.tif"
        )
        row = f"| {name} | {baseline} | {remap_seconds:.3f} | {baseline_seconds[kernel] / remap_seconds:.2f}x | {total:.3f} | {phase_error} | {coherence_error} | 0 (shared) | {phase_coverage} | {coherence_coverage} |"
        rows.append(row)
        summary.write_text(header + "\n".join(rows) + "\n")
        print(row, flush=True)
        print(f"  Summary updated: {summary}", flush=True)

    print("Preparing visual comparison layers", flush=True)
    m = folium.Map()
    show_cog(str(ref_dir / "phi.tif"), m, rescale=f"{-pi},{pi}", colormap=palette_phi())
    show_cog(str(ref_dir / "coh.tif"), m, rescale="0,1")
    show_cog(str(out_dir / "amp_geo.tif"), m, rescale="0,1")
    for name in variants:
        show_cog(
            str(out_dir / name / "phi_geo.tif"),
            m,
            rescale=f"{-pi},{pi}",
            colormap=palette_phi(),
        )
        show_cog(str(out_dir / name / "coh_geo.tif"), m, rescale="0,1")
    folium.LayerControl().add_to(m)
    m.save(str(out_dir / "comparison.html"))
    print(f"Results: {summary}", flush=True)
    serve_map(m)


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


def _write_slc(filename: str, image: np.ndarray) -> None:
    with rio.open(
        filename,
        "w",
        driver="GTiff",
        width=image.shape[1],
        height=image.shape[0],
        count=1,
        dtype=image.dtype,
        nodata=np.nan,
        tiled=True,
    ) as ds:
        ds.write(image, 1)


def _errors(reference: Path, candidate: Path, phase: bool = False) -> tuple[str, str]:
    with rio.open(reference) as ds:
        expected = ds.read(1, masked=True).filled(np.nan)
        grid = (ds.shape, ds.transform, ds.crs)
    with rio.open(candidate) as ds:
        if (ds.shape, ds.transform, ds.crs) != grid:
            raise ValueError("Comparison requires identical output grids")
        actual = ds.read(1, masked=True).filled(np.nan)
    left, right = np.isfinite(expected), np.isfinite(actual)
    valid = left & right
    gained = np.count_nonzero(right & ~left)
    lost = np.count_nonzero(left & ~right)
    coverage = f"{np.count_nonzero(valid)} / {gained} / {lost}"
    if not valid.any():
        return "no valid pixels", coverage
    difference = actual[valid].astype(np.float64) - expected[valid]
    if phase:
        difference = np.angle(np.exp(1j * difference))
    rms = np.sqrt(np.mean(difference**2))
    return f"{rms:.3e} / {np.max(np.abs(difference)):.3e}", coverage


if __name__ == "__main__":
    with warnings.catch_warnings():
        # SAR intermediates intentionally use pixel coordinates. Keep other
        # georeferencing warnings and unrelated Numba diagnostics visible.
        warnings.filterwarnings(
            "ignore",
            message=r"Dataset has no geotransform, gcps, or rpcs\..*",
            category=rio.errors.NotGeoreferencedWarning,
            module=r"rasterio(\..*)?$",
        )
        warnings.filterwarnings(
            "ignore",
            message=r".*First-class function type feature is experimental",
            category=numba.core.errors.NumbaExperimentalFeatureWarning,
            module=r"(.*[./])?benchmark_remap$",
        )
        main()
