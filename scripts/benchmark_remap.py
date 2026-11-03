"""Validate and benchmark remappers against the original production algorithm.

Run from the repository root, for example::

    python scripts/benchmark_remap.py --size 512 --repeats 5 --threads 4
    python scripts/benchmark_remap.py --check-only
    python scripts/benchmark_remap.py --lut --kernels sinc lanczos6

The reference retains the original production algorithm with sinc normalized.
The candidate precomputes axis weights, uses separable accumulation, tightens
radius-six support, and handles complex data in one pass. Optional LUTs use
linearly interpolated, normalized float64 phase
vectors (4096 intervals by default). No fastmath is used. NaN/Inf values
propagate through the candidate stencil without a whole-image fallback.
Tighter support can change boundary coverage; nonfinite-source validation
compares common finite outputs. Coordinate arrays are float64.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Callable
from pathlib import Path
from time import perf_counter

import numba
import numpy as np
from numba import njit, prange

# Allow direct execution from a source checkout without an editable install.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eo_tools.S1 import util  # noqa: E402

KERNELS = ("nearest", "bilinear", "bicubic", "bicubic6", "sinc", "lanczos6")
DTYPES = ("float32", "float64", "complex64", "complex128")


def remap_reference(
    img: np.ndarray, rr: np.ndarray, cc: np.ndarray, kernel: str = "bicubic"
) -> np.ndarray:
    """Run the official algorithm with normalized sinc."""
    if np.iscomplexobj(img):
        return _reference(img.real, rr, cc, kernel) + 1j * _reference(
            img.imag, rr, cc, kernel
        )
    return _reference(img, rr, cc, kernel)


def remap_optimized(
    img: np.ndarray, rr: np.ndarray, cc: np.ndarray, kernel: str = "bicubic"
) -> np.ndarray:
    """Run the separable candidate with real or complex dtype specialization."""
    zero = np.complex128(0) if np.iscomplexobj(img) else np.float64(0)
    missing = complex(np.nan, np.nan) if np.iscomplexobj(img) else np.nan
    return _optimized(img, rr, cc, kernel, zero, missing)


def remap_lut(
    img: np.ndarray,
    rr: np.ndarray,
    cc: np.ndarray,
    kernel: str = "sinc",
    phases: int = 4096,
) -> np.ndarray:
    """Interpolate with a cached normalized fractional-phase coefficient LUT."""
    table = _phase_table(kernel, phases)
    zero = np.complex128(0) if np.iscomplexobj(img) else np.float64(0)
    missing = complex(np.nan, np.nan) if np.iscomplexobj(img) else np.nan
    return _lut(img, rr, cc, table, zero, missing)


def validate_lut(args: argparse.Namespace) -> None:
    """Compare LUT output with direct evaluation, including phase-bin edges."""
    rng = np.random.default_rng(783)
    size = 40
    rr, cc = _coordinates(size, (32, 32), "random", rng)
    # All phase nodes, bin midpoints, and positions around the stencil rollover.
    fractions = np.concatenate(
        (
            np.linspace(0, 1, 2 * args.lut_phases + 1),
            [np.nextafter(0.0, 1.0), 1e-12, 1 - 1e-12, np.nextafter(1.0, 0.0)],
        )
    )
    edge = np.array([-np.inf, np.nan, -0.1, 0, 0.37, 39, 39.9, 40, np.inf])
    er, ec = np.meshgrid(edge, edge, indexing="ij")
    rr = np.concatenate((rr.ravel(), 15 + fractions, er.ravel()))
    cc = np.concatenate((cc.ravel(), 17 + fractions[::-1], ec.ravel()))
    rows, cols = np.indices((size, size), dtype=float)
    for dtype in args.dtypes:
        image = _image(size, dtype, rng)
        is_complex = np.iscomplexobj(image)
        for kernel in args.kernels:
            if kernel not in ("sinc", "lanczos6"):
                continue
            max_error, max_rms = 0.0, 0.0
            sources = [
                image,
                image[::2, ::2],
                np.full_like(image, 2 + 3j if is_complex else 2),
            ]
            for frequency in (0.03, 0.2, 0.45):
                wave = np.exp(2j * np.pi * frequency * (rows + 0.7 * cols))
                sources.append((wave if is_complex else wave.real).astype(dtype))
            for source in sources:
                expected = remap_optimized(source, rr, cc, kernel)
                actual = remap_lut(source, rr, cc, kernel, args.lut_phases)
                error, rms = _compare(
                    expected, actual, dtype, (args.lut_rtol, args.lut_atol)
                )
                max_error, max_rms = max(max_error, error), max(max_rms, rms)
            for source in (image, np.full_like(image, 2 + 3j if is_complex else 2)):
                actual = remap_lut(source, rows, cols, kernel, args.lut_phases)
                rtol, atol = _tolerance(dtype)
                np.testing.assert_allclose(actual, source, rtol=rtol, atol=atol)
            constant = np.full_like(image, 2 + 3j if is_complex else 2)
            actual = remap_lut(constant, rr, cc, kernel, args.lut_phases)
            valid = np.isfinite(actual)
            rtol, atol = _tolerance(dtype)
            np.testing.assert_allclose(
                actual[valid], constant[0, 0], rtol=rtol, atol=atol
            )
            for bad in (np.nan, np.inf):
                source = image.copy()
                source[12, 12] = bad
                _compare(
                    remap_reference(source, rr, cc, kernel),
                    remap_lut(source, rr, cc, kernel, args.lut_phases),
                    dtype,
                    (args.lut_rtol, args.lut_atol),
                    finite_only=True,
                )
            error, rms = _check_border(
                image,
                kernel,
                lambda img, rr, cc, kernel: remap_lut(
                    img, rr, cc, kernel, args.lut_phases
                ),
                (args.lut_rtol, args.lut_atol),
            )
            max_error, max_rms = max(max_error, error), max(max_rms, rms)
            try:
                remap_lut(image, rr, cc[:-1], kernel, args.lut_phases)
            except ValueError:
                pass
            else:
                raise AssertionError("Mismatched coordinate shapes were accepted")
            print(
                f"LUT validation {dtype:10s} {kernel:8s}: "
                f"max_abs={max_error:.3e} rel_rms={max_rms:.3e}",
                flush=True,
            )


def validate(kernels: list[str], dtypes: list[str]) -> None:
    """Check parity, constants, identity and analytic sinusoidal response."""
    rng = np.random.default_rng(781)
    size = 40
    rr, cc = _coordinates(size, (32, 32), "random", rng)
    # Include borders, invalid coordinates, integers and adjacent floats.
    edge = np.array(
        [
            -np.inf,
            np.nan,
            -0.1,
            0,
            0.37,
            1,
            6,
            6 - 1e-12,
            6 + 1e-12,
            size - 1,
            size - 0.1,
            size,
            np.inf,
        ]
    )
    er, ec = np.meshgrid(edge, edge, indexing="ij")
    rr = np.concatenate((rr.ravel(), er.ravel()))
    cc = np.concatenate((cc.ravel(), ec.ravel()))
    rows, cols = np.indices((size, size), dtype=float)
    worst = dict.fromkeys(dtypes, 0.0)
    cases = 0
    for dtype in dtypes:
        image = _image(size, dtype, rng)
        is_complex = np.iscomplexobj(image)
        value = 2 + 3j if is_complex else 2.0
        rtol, atol = _tolerance(dtype)
        for kernel in kernels:
            for source in (image, image[::2, ::2], np.full_like(image, value)):
                expected = remap_reference(source, rr, cc, kernel)
                actual = remap_optimized(source, rr, cc, kernel)
                error, _ = _compare(expected, actual, dtype)
                worst[dtype] = max(worst[dtype], error)
                cases += 1

            constant = remap_optimized(np.full_like(image, value), rr, cc, kernel)
            valid = (rr >= 0) & (rr < size) & (cc >= 0) & (cc < size)
            np.testing.assert_allclose(constant[valid], value, rtol=rtol, atol=atol)
            identity = remap_optimized(image, rows, cols, kernel)
            np.testing.assert_allclose(identity, image, rtol=rtol, atol=atol)

            # Nonfinite propagation may differ with the tighter stencil.
            # Compare only outputs finite in both implementations.
            for bad in (
                (np.nan, np.inf, complex(1, np.nan)) if is_complex else (np.nan, np.inf)
            ):
                source = image.copy()
                source[12, 12] = bad
                _compare(
                    remap_reference(source, rr, cc, kernel),
                    remap_optimized(source, rr, cc, kernel),
                    dtype,
                    finite_only=True,
                )
                cases += 1

            if kernel in ("sinc", "lanczos6"):
                error, _ = _check_border(image, kernel, remap_optimized)
                worst[dtype] = max(worst[dtype], error)
                cases += 1

            # Independent finite-stencil frequency response (including near Nyquist).
            for frequency in (0.03, 0.2, 0.45):
                wave = np.exp(2j * np.pi * frequency * (rows + 0.7 * cols))
                source = (wave if is_complex else wave.real).astype(dtype)
                wr = np.array([15.0, 15.01, 15.37, 15.5, 15.99])
                wc = np.array([17.7, 17.5, 17.11, 17.0, 17.99])
                expected = _wave_response(wr, wc, frequency, kernel)
                if not is_complex:
                    expected = expected.real
                actual = remap_optimized(source, wr, wc, kernel)
                np.testing.assert_allclose(actual, expected, rtol=rtol, atol=atol)
                cases += 1
            # Shape errors must remain explicit.
            for remapper in (remap_reference, remap_optimized):
                try:
                    remapper(image, rr, cc[:-1], kernel)
                except ValueError:
                    pass
                else:
                    raise AssertionError("Mismatched coordinate shapes were accepted")
    print(f"Validation passed: {cases} comparisons plus constant/identity/shape checks")
    for dtype, error in worst.items():
        print(f"  {dtype:10s} maximum absolute parity error: {error:.3e}")


def benchmark(args: argparse.Namespace) -> None:
    """Report warmed-up end-to-end timings, alternating execution order."""
    rng = np.random.default_rng(782)
    print(f"NumPy {np.__version__}; Numba {numba.__version__}")
    print(
        f"threads={numba.get_num_threads()}, source/target={args.size}x{args.size}, "
        f"repeats={args.repeats}; compilation excluded, no source scan or fallback"
    )
    print("Times are median [min, max]; throughput is candidate megapixels/second.")
    for dtype in args.dtypes:
        image = _image(args.size, dtype, rng)
        for layout in args.layouts:
            rr, cc = _coordinates(args.size, image.shape, layout, rng)
            for kernel in args.kernels:
                expected = remap_reference(image, rr, cc, kernel)
                actual = remap_optimized(image, rr, cc, kernel)
                error, relative_rms = _compare(expected, actual, dtype)
                timings = [[], []]
                functions = [remap_reference, remap_optimized]
                lut_error = lut_rms = 0.0
                if args.lut and kernel in ("sinc", "lanczos6"):
                    lut_actual = remap_lut(image, rr, cc, kernel, args.lut_phases)
                    lut_error, lut_rms = _compare(
                        actual, lut_actual, dtype, (args.lut_rtol, args.lut_atol)
                    )
                    functions.append(
                        lambda img, rr, cc, kernel: remap_lut(
                            img, rr, cc, kernel, args.lut_phases
                        )
                    )
                    timings.append([])
                for repeat in range(args.repeats):
                    order = np.roll(np.arange(len(functions)), repeat % len(functions))
                    for index in order:
                        start = perf_counter()
                        functions[index](image, rr, cc, kernel)
                        timings[index].append(perf_counter() - start)
                medians = [np.median(values) for values in timings]
                ranges = [
                    f"{median:.4f} [{min(values):.4f}, {max(values):.4f}]"
                    for median, values in zip(medians, timings)
                ]
                lut_report = ""
                if len(functions) == 3:
                    lut_report = (
                        f" lut={ranges[2]} s lut_gain={medians[1] / medians[2]:.2f}x"
                        f" total_gain={medians[0] / medians[2]:.2f}x"
                        f" lut_max_abs={lut_error:.2e} lut_rel_rms={lut_rms:.2e}"
                    )
                print(
                    f"{dtype:10s} {layout:6s} {kernel:8s} "
                    f"ref={ranges[0]} s opt={ranges[1]} s "
                    f"speedup={medians[0] / medians[1]:.2f}x "
                    f"MPix/s={rr.size / medians[1] / 1e6:.2f} "
                    f"max_abs={error:.2e} rel_rms={relative_rms:.2e}{lut_report}",
                    flush=True,
                )


def main() -> None:
    """Parse benchmark options and always validate before timing."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--repeats", type=int, default=5)
    parser.add_argument("--threads", type=int, default=min(4, numba.get_num_threads()))
    parser.add_argument("--kernels", nargs="+", choices=KERNELS, default=list(KERNELS))
    parser.add_argument("--dtypes", nargs="+", choices=DTYPES, default=list(DTYPES))
    parser.add_argument(
        "--layouts",
        nargs="+",
        choices=("smooth", "random"),
        default=["smooth", "random"],
    )
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument(
        "--lut", action="store_true", help="Also validate/time sinc and Lanczos LUTs"
    )
    parser.add_argument("--lut-phases", type=int, default=4096)
    parser.add_argument("--lut-atol", type=float, default=1e-6)
    parser.add_argument("--lut-rtol", type=float, default=1e-6)
    args = parser.parse_args()
    if args.size < 16 or args.repeats < 1 or args.threads < 1:
        parser.error("size must be >=16; repeats and threads must be positive")
    numba.set_num_threads(args.threads)
    if args.lut_phases < 2 or args.lut_atol < 0 or args.lut_rtol < 0:
        parser.error("LUT phases must be >=2 and tolerances nonnegative")
    if args.lut:
        for kernel in args.kernels:
            if kernel in ("sinc", "lanczos6"):
                start = perf_counter()
                table = _phase_table(kernel, args.lut_phases)
                print(
                    f"{kernel} LUT: {table.shape}, float64, {table.nbytes / 1024:.1f} KiB, "
                    f"build={(perf_counter() - start) * 1000:.2f} ms (excluded from timings)",
                    flush=True,
                )
    validate(args.kernels, args.dtypes)
    if args.lut:
        print(f"LUT acceptance: atol={args.lut_atol:g}, rtol={args.lut_rtol:g}")
        validate_lut(args)
    if not args.check_only:
        benchmark(args)


# -----------------------------------------------------------------------------
# Private helpers
# -----------------------------------------------------------------------------


# Original production algorithm, retained as an independent normalized baseline.
@njit(parallel=True, nogil=True)
def _reference(
    img: np.ndarray, rr: np.ndarray, cc: np.ndarray, kernel: str = "bicubic"
) -> np.ndarray:

    if rr.shape != cc.shape:
        raise ValueError("Coordinate arrays must have the same shape.")

    arr_out = np.full_like(rr, np.nan, dtype=img.dtype)
    normalize = False
    if kernel == "nearest":
        ker = util._ker_near
        H = 0
    elif kernel == "bilinear":
        ker = util._ker_lin
        H = 0
    elif kernel == "bicubic":
        ker = util._ker_cub
        H = 1
    elif kernel == "bicubic6":
        ker = util._ker_cub6
        H = 2
    elif kernel == "sinc":
        ker = util._ker_sinc
        H = 6
        normalize = True
    elif kernel == "lanczos6":
        ker = util._ker_lanczos6
        H = 6
        normalize = True
    else:
        raise ValueError("Unknown interpolation type.")

    for idx in prange(len(rr.flat)):
        r = rr.flat[idx]
        c = cc.flat[idx]

        # change boundaries if using other kernels
        rmin = np.floor(r) - H
        rmax = np.ceil(r) + H
        cmin = np.floor(c) - H
        cmax = np.ceil(c) + H

        if np.isnan(r) | np.isnan(c):
            continue
        is_in_image = (r >= 0) & (r < img.shape[0]) & (c >= 0) & (c < img.shape[1])
        if not is_in_image:
            continue
        val = 0.0
        weight_sum = 0.0
        for i in range(int(rmin), int(rmax) + 1):
            for j in range(int(cmin), int(cmax) + 1):
                # using nearest neighbor on image border
                i2 = min(max(0, i), img.shape[0] - 1)
                j2 = min(max(0, j), img.shape[1] - 1)
                weight = ker(r - i) * ker(c - j)
                val += weight * img[i2, j2]
                weight_sum += weight
        if normalize:
            val /= weight_sum
        arr_out.flat[idx] = val
    return arr_out


def _kernel(kernel: str) -> tuple[Callable[[float], float], int, bool]:
    if kernel == "nearest":
        return util._ker_near, 0, False
    if kernel == "bilinear":
        return util._ker_lin, 0, False
    if kernel == "bicubic":
        return util._ker_cub, 1, False
    if kernel == "bicubic6":
        return util._ker_cub6, 2, False
    if kernel == "sinc":
        return util._ker_sinc, 6, True
    if kernel == "lanczos6":
        return util._ker_lanczos6, 6, True
    raise ValueError("Unknown interpolation type.")


_optimized = util._remap
_phase_table = util._phase_table
_lut = util._remap_lut


def _tolerance(dtype: str) -> tuple[float, float]:
    return (1e-5, 1e-6) if dtype in ("float32", "complex64") else (1e-12, 1e-12)


def _check_border(
    image: np.ndarray,
    kernel: str,
    remapper: Callable,
    tolerance: tuple[float, float] | None = None,
) -> tuple[float, float]:
    """Check that border NaNs stay local and do not disable the fast path."""
    source = image.copy()
    source[:2] = complex(np.nan, np.nan) if np.iscomplexobj(source) else np.nan
    rr = np.array([7.25, 8.25, 15.25])
    cc = np.full(rr.shape, 15.25)
    expected = remap_reference(source, rr, cc, kernel)
    actual = remapper(source, rr, cc, kernel)
    # The original visits a zero-weight NaN in row 1 for the first target;
    # the tighter candidate stencil starts at row 2 and stays finite.
    assert not np.isfinite(expected[0])
    assert np.isfinite(actual).all()
    assert np.isfinite(expected[1:]).all()
    return _compare(expected, actual, str(image.dtype), tolerance, finite_only=True)


def _compare(
    expected: np.ndarray,
    actual: np.ndarray,
    dtype: str,
    tolerance: tuple[float, float] | None = None,
    finite_only: bool = False,
) -> tuple[float, float]:
    assert actual.shape == expected.shape
    assert actual.dtype == expected.dtype
    if not finite_only:
        # A complex sample can have only one nonfinite component.
        for left, right in ((expected.real, actual.real), (expected.imag, actual.imag)):
            np.testing.assert_array_equal(np.isnan(left), np.isnan(right))
            np.testing.assert_array_equal(np.isposinf(left), np.isposinf(right))
            np.testing.assert_array_equal(np.isneginf(left), np.isneginf(right))
    valid = np.isfinite(expected) & np.isfinite(actual)
    rtol, atol = _tolerance(dtype) if tolerance is None else tolerance
    np.testing.assert_allclose(actual[valid], expected[valid], rtol=rtol, atol=atol)
    if not valid.any():
        return 0.0, 0.0
    difference = np.abs(actual[valid] - expected[valid]).astype(np.float64)
    scale = np.sqrt(np.mean(np.abs(expected[valid]).astype(np.float64) ** 2))
    relative_rms = np.sqrt(np.mean(difference**2)) / max(scale, np.finfo(float).tiny)
    return float(difference.max()), float(relative_rms)


def _image(size: int, dtype: str, rng: np.random.Generator) -> np.ndarray:
    image = rng.standard_normal((size, size))
    if np.issubdtype(np.dtype(dtype), np.complexfloating):
        image = image + 1j * rng.standard_normal(image.shape)
    return image.astype(dtype)


def _coordinates(
    size: int, shape: tuple[int, int], layout: str, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    if layout == "random":
        return rng.uniform(0, size - 1, shape), rng.uniform(0, size - 1, shape)
    rr, cc = np.meshgrid(
        np.linspace(0, size - 1, shape[0]),
        np.linspace(0, size - 1, shape[1]),
        indexing="ij",
    )
    return rr + 0.37 + 0.05 * np.sin(cc / 31), cc + 0.23 + 0.05 * np.sin(rr / 43)


def _wave_response(
    rr: np.ndarray, cc: np.ndarray, frequency: float, kernel: str
) -> np.ndarray:
    # Evaluate the analytical plane wave on the full reference stencil in Python.
    ker, half_width, normalize = _kernel(kernel)
    values = []
    for r, c in zip(rr, cc):
        ri = np.arange(np.floor(r) - half_width, np.ceil(r) + half_width + 1)
        ci = np.arange(np.floor(c) - half_width, np.ceil(c) + half_width + 1)
        rw = np.array([ker(float(r - i)) for i in ri])
        cw = np.array([ker(float(c - i)) for i in ci])
        if normalize:
            rw /= rw.sum()
            cw /= cw.sum()
        values.append(
            np.sum(rw * np.exp(2j * np.pi * frequency * ri))
            * np.sum(cw * np.exp(2j * np.pi * frequency * 0.7 * ci))
        )
    return np.array(values)


if __name__ == "__main__":
    main()
