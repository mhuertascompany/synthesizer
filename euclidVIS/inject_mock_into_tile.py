"""Inject a Synthesizer Euclid VIS mock into a calibrated Euclid tile."""

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits
from scipy.signal import fftconvolve
from scipy.spatial import cKDTree

matplotlib.use("Agg")
import matplotlib.pyplot as plt


AB_ZEROPOINT_MICROJY = 23.9


def parse_args():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--tile",
        type=Path,
        default=here / "images" / (
            "EUC_MER_BGSUB-MOSAIC-VIS_TILE101834044-F20EA5_"
            "20250828T043943.523084Z_00.00.fits"
        ),
    )
    parser.add_argument(
        "--mock",
        type=Path,
        default=here / "images" / "mocks" / "euclid_vis_98_raw.fits",
    )
    parser.add_argument(
        "--psf",
        type=Path,
        default=here / "images" / (
            "EUC_MER_CATALOG-PSF-VIS_TILE101834044-D89068_"
            "20250828T052759.955095Z_00.00.fits"
        ),
        help="MER catalog-PSF grid corresponding to the input tile",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=here / "images" / "tile101834044_with_mock98.fits",
    )
    parser.add_argument(
        "--diagnostic",
        type=Path,
        default=here / "images" / "tile101834044_mock98_diagnostic.png",
    )
    parser.add_argument("--x", type=int, help="Zero-based tile x pixel")
    parser.add_argument("--y", type=int, help="Zero-based tile y pixel")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--avoid-radius", type=int, default=30,
        help="Radius in pixels checked for existing sources",
    )
    parser.add_argument(
        "--detection-sigma", type=float, default=5.0,
        help="Reject positions containing pixels this far above the tile sky",
    )
    parser.add_argument("--max-position-attempts", type=int, default=10000)
    parser.add_argument(
        "--allow-source-overlap", action="store_true",
        help="Allow an explicit --x/--y position even if a source is detected",
    )
    parser.add_argument("--cutout-size", type=int, default=121)
    return parser.parse_args()


def convolve_with_psf(stamp, psf):
    """Convolve a stamp with a normalized PSF, preserving integrated flux."""
    kernel = np.asarray(psf, dtype=np.float64)
    kernel[~np.isfinite(kernel)] = 0.0
    kernel_sum = np.sum(kernel, dtype=np.float64)
    if not np.isfinite(kernel_sum) or kernel_sum <= 0:
        raise ValueError("Selected PSF stamp has non-positive integrated flux")
    kernel /= kernel_sum
    convolved = fftconvolve(stamp, kernel, mode="same")
    original_flux = np.sum(stamp, dtype=np.float64)
    convolved_flux = np.sum(convolved, dtype=np.float64)
    if convolved_flux != 0:
        convolved *= original_flux / convolved_flux
    return convolved


def load_nearest_psf(psf_path, x, y):
    """Extract the nearest 1-based MER PSF-grid stamp for a tile position."""
    with fits.open(psf_path, memmap=True) as hdul:
        mosaic = hdul[1].data
        header = hdul[1].header
        table = hdul[2].data
        stamp_size = int(header["STMPSIZE"])
        coordinates = np.column_stack((table["x"], table["y"]))
        valid = np.isfinite(coordinates).all(axis=1)
        valid &= np.isfinite(table["FWHM"])
        if not np.any(valid):
            raise ValueError("The PSF catalog contains no valid stamps")
        valid_indices = np.flatnonzero(valid)
        # MER table coordinates are FITS-style (one based); CLI pixels are zero based.
        nearest_local = cKDTree(coordinates[valid]).query((x + 1, y + 1))[1]
        index = int(valid_indices[nearest_local])
        x_center = float(table["x_center"][index]) - 1.0
        y_center = float(table["y_center"][index]) - 1.0
        half = stamp_size // 2
        x0 = int(round(x_center)) - half
        y0 = int(round(y_center)) - half
        psf = np.asarray(
            mosaic[y0:y0 + stamp_size, x0:x0 + stamp_size], dtype=np.float64
        ).copy()
        if psf.shape != (stamp_size, stamp_size):
            raise ValueError(f"PSF stamp {index} falls outside its mosaic")
        metadata = {
            "index": index,
            "x": float(table["x"][index]),
            "y": float(table["y"][index]),
            "ra": float(table["RA"][index]),
            "dec": float(table["Dec"][index]),
            "fwhm": float(table["FWHM"][index]),
            "pixel_scale": np.sqrt(abs(
                header["CD1_1"] * header["CD2_2"]
                - header["CD1_2"] * header["CD2_1"]
            )) * 3600.0,
        }
    return psf, metadata


def estimate_tile_background(data):
    """Robustly estimate tile background and pixel noise from a sparse sample."""
    stride = max(1, int(np.sqrt(data.size / 250000)))
    sample = np.asarray(data[::stride, ::stride], dtype=np.float64)
    sample = sample[np.isfinite(sample)]
    if sample.size == 0:
        raise ValueError("Tile contains no finite pixels")
    median = float(np.median(sample))
    sigma = float(1.4826 * np.median(np.abs(sample - median)))
    if not np.isfinite(sigma) or sigma <= 0:
        raise ValueError("Could not estimate a positive tile noise level")
    return median, sigma


def position_has_source(data, x, y, radius, threshold):
    patch = data[y - radius:y + radius + 1, x - radius:x + radius + 1]
    return patch.size == 0 or not np.all(np.isfinite(patch)) or np.any(patch > threshold)


def choose_empty_position(data, stamp_shape, args, background, noise):
    """Choose or validate a position without a significant observed source."""
    margin_x = max(stamp_shape[1] // 2 + 1, args.avoid_radius + 1)
    margin_y = max(stamp_shape[0] // 2 + 1, args.avoid_radius + 1)
    threshold = background + args.detection_sigma * noise

    if (args.x is None) != (args.y is None):
        raise ValueError("Provide both --x and --y, or neither")
    if args.x is not None:
        x, y = int(args.x), int(args.y)
        if not (margin_x <= x < data.shape[1] - margin_x and
                margin_y <= y < data.shape[0] - margin_y):
            raise ValueError("Injection position is too close to the tile boundary")
        if (not args.allow_source_overlap and
                position_has_source(data, x, y, args.avoid_radius, threshold)):
            raise ValueError(
                "An observed source or invalid pixel is present at the requested "
                "position; choose another position or pass --allow-source-overlap"
            )
        return x, y, threshold

    rng = np.random.default_rng(args.seed)
    for _ in range(args.max_position_attempts):
        x = int(rng.integers(margin_x, data.shape[1] - margin_x))
        y = int(rng.integers(margin_y, data.shape[0] - margin_y))
        if not position_has_source(data, x, y, args.avoid_radius, threshold):
            return x, y, threshold
    raise RuntimeError(
        f"No empty position found in {args.max_position_attempts} attempts; "
        "reduce --avoid-radius or --detection-sigma only after inspecting the tile"
    )


def overlap_slices(tile_shape, stamp_shape, x, y):
    """Return matching tile and stamp slices for a centered injection."""
    stamp_y0 = y - stamp_shape[0] // 2
    stamp_x0 = x - stamp_shape[1] // 2
    tile_y0 = max(stamp_y0, 0)
    tile_x0 = max(stamp_x0, 0)
    tile_y1 = min(stamp_y0 + stamp_shape[0], tile_shape[0])
    tile_x1 = min(stamp_x0 + stamp_shape[1], tile_shape[1])
    if tile_y0 >= tile_y1 or tile_x0 >= tile_x1:
        raise ValueError("The requested injection position is outside the tile")
    stamp_y_slice = slice(tile_y0 - stamp_y0, tile_y1 - stamp_y0)
    stamp_x_slice = slice(tile_x0 - stamp_x0, tile_x1 - stamp_x0)
    return (
        (slice(tile_y0, tile_y1), slice(tile_x0, tile_x1)),
        (stamp_y_slice, stamp_x_slice),
    )


def diagnostic_plot(before, after, injected, output):
    finite = before[np.isfinite(before)]
    vmin, vmax = np.percentile(finite, [1, 99])
    diff_limit = np.nanpercentile(np.abs(injected), 99.5)
    fig, axes = plt.subplots(1, 3, figsize=(13, 4), constrained_layout=True)
    panels = [
        (before, "Original tile", "gray", vmin, vmax),
        (after, "Tile + mock", "gray", vmin, vmax),
        (injected, "Injected signal", "magma", 0, diff_limit),
    ]
    for axis, (image, title, cmap, low, high) in zip(axes, panels):
        artist = axis.imshow(
            image, origin="lower", cmap=cmap, vmin=low, vmax=high
        )
        axis.set_title(title)
        axis.set_xlabel("x pixel")
        axis.set_ylabel("y pixel")
        fig.colorbar(artist, ax=axis, fraction=0.046, pad=0.04)
    fig.suptitle("Euclid VIS mock injection [ADU/s]")
    fig.savefig(output, dpi=180)
    plt.close(fig)


def main():
    args = parse_args()
    with fits.open(args.tile, memmap=True) as tile_hdul:
        tile_header = tile_hdul[0].header.copy()
        tile_shape = tile_hdul[0].data.shape
        background, tile_noise = estimate_tile_background(tile_hdul[0].data)
    tile_unit = tile_header.get("BUNIT", "").replace(" ", "").lower()
    if tile_unit not in {"adu/s", "adu/s."}:
        raise ValueError(
            f"Expected to apply an ADU/s calibration to tile BUNIT={tile_unit!r}"
        )
    if "MAGZERO" not in tile_header:
        raise KeyError("Tile has no MAGZERO AB zeropoint")
    zeropoint = float(tile_header["MAGZERO"])

    mock, mock_header = fits.getdata(args.mock, header=True)
    if mock_header.get("BUNIT", "").strip().lower() != "ujy":
        raise ValueError("Mock BUNIT must be uJy")
    mock_pixel_scale = float(mock_header["PIXSCALE"])
    tile_pixel_scale = np.sqrt(
        abs(
            tile_header["CD1_1"] * tile_header["CD2_2"]
            - tile_header["CD1_2"] * tile_header["CD2_1"]
        )
    ) * 3600.0
    if not np.isclose(mock_pixel_scale, tile_pixel_scale, rtol=0, atol=1e-4):
        raise ValueError(
            "Mock and tile pixel scales differ; explicit resampling is required "
            f"({mock_pixel_scale} vs {tile_pixel_scale} arcsec/pixel)"
        )

    # For an image calibrated in ADU/s, m_AB = ZP - 2.5 log10(ADU/s).
    # Since 1 microJy has m_AB approximately 23.9, no EXPTIME factor belongs
    # in this conversion.
    adu_per_second_per_microjy = 10 ** (
        -0.4 * (AB_ZEROPOINT_MICROJY - zeropoint)
    )
    with fits.open(args.tile, memmap=True) as tile_hdul:
        x, y, source_threshold = choose_empty_position(
            tile_hdul[0].data, mock.shape, args, background, tile_noise
        )
    psf_kernel, psf_metadata = load_nearest_psf(args.psf, x, y)
    if not np.isclose(
        psf_metadata["pixel_scale"], tile_pixel_scale, rtol=0, atol=1e-4
    ):
        raise ValueError(
            "PSF and tile pixel scales differ; explicit PSF resampling is required "
            f"({psf_metadata['pixel_scale']} vs {tile_pixel_scale} arcsec/pixel)"
        )
    psf_mock_microjy = convolve_with_psf(
        np.asarray(mock, dtype=np.float64), psf_kernel
    )
    injected_stamp = psf_mock_microjy * adu_per_second_per_microjy
    tile_slice, stamp_slice = overlap_slices(
        tile_shape, injected_stamp.shape, x, y
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.tile, args.output)
    with fits.open(args.output, mode="update", memmap=True) as output_hdul:
        target = output_hdul[0].data[tile_slice]
        signal = injected_stamp[stamp_slice].astype(target.dtype)
        target += signal
        output_hdul[0].header["HISTORY"] = (
            "Injected pre-PSF Synthesizer mock with inject_mock_into_tile.py"
        )
        output_hdul[0].header["MOCKFILE"] = args.mock.name
        output_hdul[0].header["MOCKX"] = x
        output_hdul[0].header["MOCKY"] = y
        output_hdul[0].header["MOCKZP"] = zeropoint
        output_hdul[0].header["MOCKPSF"] = args.psf.name
        output_hdul[0].header["PSFINDX"] = psf_metadata["index"]
        output_hdul[0].header["PSFFWHM"] = psf_metadata["fwhm"]
        output_hdul[0].header["PSFX"] = psf_metadata["x"]
        output_hdul[0].header["PSFY"] = psf_metadata["y"]
        output_hdul.flush()

    half = args.cutout_size // 2
    cutout_y = slice(max(y - half, 0), min(y + half + 1, tile_shape[0]))
    cutout_x = slice(max(x - half, 0), min(x + half + 1, tile_shape[1]))
    with fits.open(args.tile, memmap=True) as original_hdul:
        before = original_hdul[0].data[cutout_y, cutout_x].copy()
    with fits.open(args.output, memmap=True) as injected_hdul:
        after = injected_hdul[0].data[cutout_y, cutout_x].copy()
    diagnostic_plot(before, after, after - before, args.diagnostic)

    total_microjy = float(np.sum(mock, dtype=np.float64))
    total_adu_per_second = float(np.sum(injected_stamp, dtype=np.float64))
    ab_magnitude = AB_ZEROPOINT_MICROJY - 2.5 * np.log10(total_microjy)
    print(f"Tile pixel scale: {tile_pixel_scale:.6f} arcsec/pixel")
    print(f"Tile zeropoint: {zeropoint:.4f} AB for 1 ADU/s")
    print(f"Conversion: {adu_per_second_per_microjy:.6f} ADU/s per uJy")
    print(
        f"PSF: grid index {psf_metadata['index']}, "
        f"FWHM={psf_metadata['fwhm']:.4f} arcsec, "
        f"nearest grid pixel=({psf_metadata['x']:.1f}, {psf_metadata['y']:.1f})"
    )
    print(
        f"Empty-position selection: background={background:.6g}, "
        f"noise={tile_noise:.6g}, threshold={source_threshold:.6g} ADU/s"
    )
    print(f"Mock total: {total_microjy:.6f} uJy = AB {ab_magnitude:.4f}")
    print(f"Injected total: {total_adu_per_second:.6f} ADU/s")
    print(f"Position: zero-based pixel (x={x}, y={y})")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.diagnostic}")


if __name__ == "__main__":
    main()
