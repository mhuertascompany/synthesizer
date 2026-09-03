"""Inject a Synthesizer Euclid VIS mock into a calibrated Euclid tile."""

import argparse
import shutil
from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits
from scipy.ndimage import gaussian_filter

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
    parser.add_argument(
        "--psf-fwhm",
        type=float,
        default=0.18,
        help="Placeholder Gaussian PSF FWHM in arcsec",
    )
    parser.add_argument("--cutout-size", type=int, default=121)
    return parser.parse_args()


def normalized_gaussian_psf(stamp, fwhm_arcsec, pixel_scale_arcsec):
    """Convolve a stamp while preserving its integrated flux exactly."""
    sigma_pixels = fwhm_arcsec / (2.355 * pixel_scale_arcsec)
    convolved = gaussian_filter(stamp, sigma=sigma_pixels, mode="constant")
    original_flux = np.sum(stamp, dtype=np.float64)
    convolved_flux = np.sum(convolved, dtype=np.float64)
    if convolved_flux != 0:
        convolved *= original_flux / convolved_flux
    return convolved, sigma_pixels


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
    psf_mock_microjy, sigma_pixels = normalized_gaussian_psf(
        np.asarray(mock, dtype=np.float64),
        args.psf_fwhm,
        tile_pixel_scale,
    )
    injected_stamp = psf_mock_microjy * adu_per_second_per_microjy

    x = args.x
    y = args.y
    if x is None:
        x = int(round(float(tile_header.get("CRPIX1", tile_shape[1] / 2)) - 1))
    if y is None:
        y = int(round(float(tile_header.get("CRPIX2", tile_shape[0] / 2)) - 1))
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
        output_hdul[0].header["MOCKFWHM"] = args.psf_fwhm
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
    print(f"Placeholder PSF: sigma={sigma_pixels:.4f} pixels")
    print(f"Mock total: {total_microjy:.6f} uJy = AB {ab_magnitude:.4f}")
    print(f"Injected total: {total_adu_per_second:.6f} ADU/s")
    print(f"Position: zero-based pixel (x={x}, y={y})")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.diagnostic}")


if __name__ == "__main__":
    main()
