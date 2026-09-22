#!/usr/bin/env python3
"""Compare matched observed DESI spectra with simulated spectra."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


HERE = Path(__file__).resolve().parent
DEFAULT_DATA_ROOT = Path("/Users/marchuertascompany/Documents/data/euclid_desi")
DEFAULT_LINES = {
    "[O II]": 3727.0,
    "Hβ": 4861.0,
    "[O III]": 5007.0,
    "Hα": 6563.0,
}
TIER_COLORS = {"low": "#377eb8", "median": "#ff7f00", "high": "#4daf4a"}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--manifest", type=Path, default=HERE / "desi_observed_examples.csv"
    )
    parser.add_argument(
        "--observed-dir", type=Path, default=DEFAULT_DATA_ROOT / "spectra"
    )
    parser.add_argument("--mock-dir", type=Path, default=HERE / "images" / "mocks")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_DATA_ROOT / "desi_observed_vs_mocks.png",
    )
    parser.add_argument(
        "--smooth-pixels",
        type=int,
        default=5,
        help="Boxcar width used only in the normalized rest-frame panels",
    )
    parser.add_argument(
        "--feasibgs-only",
        action="store_true",
        help="Only plot mocks whose primary FITS header has FEASIBGS=True",
    )
    return parser.parse_args()


def named_column(data, name):
    names = {column.upper(): column for column in data.names}
    if name.upper() not in names:
        raise KeyError(f"Missing {name}; available columns: {data.names}")
    return np.asarray(data[names[name.upper()]]).squeeze()


def read_observed(path):
    with fits.open(path, memmap=False) as hdul:
        table_hdu = next(
            hdu
            for hdu in hdul[1:]
            if getattr(hdu.data, "names", None)
            and "WAVELENGTH" in {name.upper() for name in hdu.data.names}
        )
        wave = named_column(table_hdu.data, "WAVELENGTH").astype(float)
        flux = named_column(table_hdu.data, "FLUX").astype(float)
        ivar = named_column(table_hdu.data, "IVAR").astype(float)
        mask = named_column(table_hdu.data, "MASK")
    good = (
        np.isfinite(wave)
        & np.isfinite(flux)
        & np.isfinite(ivar)
        & (ivar > 0)
        & (mask == 0)
    )
    return wave[good], flux[good], ivar[good]


def read_mock(path):
    # DESI MASK image HDUs use unsigned-integer FITS scaling, which Astropy
    # cannot expose through a memory map.
    with fits.open(path, memmap=False) as hdul:
        if "B_FLUX" in hdul:
            pieces = []
            for arm in "BRZ":
                wave = np.asarray(hdul[f"{arm}_WAVELENGTH"].data).squeeze().astype(float)
                flux = np.asarray(hdul[f"{arm}_FLUX"].data).squeeze().astype(float)
                ivar = np.asarray(hdul[f"{arm}_IVAR"].data).squeeze().astype(float)
                mask = np.asarray(hdul[f"{arm}_MASK"].data).squeeze()
                good = (
                    np.isfinite(wave)
                    & np.isfinite(flux)
                    & np.isfinite(ivar)
                    & (ivar > 0)
                    & (mask == 0)
                )
                pieces.append((wave[good], flux[good], ivar[good]))
            wave = np.concatenate([piece[0] for piece in pieces])
            flux = np.concatenate([piece[1] for piece in pieces])
            ivar = np.concatenate([piece[2] for piece in pieces])
        else:
            table = hdul[1].data
            wave = named_column(table, "WAVELENGTH").astype(float)
            flux = named_column(table, "FLUX").astype(float)
            ivar = np.ones_like(flux)
            # Early mocks stored f_nu in cgs without a FITS unit keyword.
            if np.nanmedian(np.abs(flux)) < 1e-20:
                speed_of_light_angstrom_s = 2.99792458e18
                flux = flux * speed_of_light_angstrom_s / wave**2 / 1e-17

    order = np.argsort(wave)
    return wave[order], flux[order], ivar[order]


def smooth(values, width):
    if width <= 1:
        return values
    kernel = np.ones(width, dtype=float) / width
    return np.convolve(values, kernel, mode="same")


def robust_normalize(flux):
    finite = flux[np.isfinite(flux)]
    center = np.median(finite)
    p16, p84 = np.percentile(finite, [16, 84])
    scale = p84 - p16
    if not np.isfinite(scale) or scale <= 0:
        scale = np.std(finite)
    return (flux - center) / scale


def set_robust_limits(axis, arrays, lower=0.5, upper=99.5, padding=0.12):
    values = np.concatenate([array[np.isfinite(array)] for array in arrays])
    low, high = np.percentile(values, [lower, upper])
    span = max(high - low, 1e-6)
    axis.set_ylim(low - padding * span, high + padding * span)


def load_manifest(path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    grouped = {}
    for row in rows:
        grouped.setdefault(row["mock_id"], []).append(row)
    return grouped


def is_feasibgs_mock(path):
    """Return whether a mock was generated by the feasiBGS forward model."""
    return bool(fits.getheader(path, 0).get("FEASIBGS", False))


def main():
    args = parse_args()
    grouped = load_manifest(args.manifest)
    if args.feasibgs_only:
        grouped = {
            mock_id: rows
            for mock_id, rows in grouped.items()
            if is_feasibgs_mock(args.mock_dir / rows[0]["mock_file"])
        }
        if not grouped:
            raise ValueError("No FEASIBGS=True mocks were found in the manifest")
    missing = []
    for rows in grouped.values():
        mock_path = args.mock_dir / rows[0]["mock_file"]
        if not mock_path.exists():
            missing.append(mock_path)
        for row in rows:
            observed_path = args.observed_dir / row["observed_file"]
            if not observed_path.exists():
                missing.append(observed_path)
    if missing:
        formatted = "\n".join(f"  {path}" for path in missing)
        raise FileNotFoundError(f"Missing input spectra:\n{formatted}")

    figure, axes = plt.subplots(
        len(grouped), 2, figsize=(16, 4.2 * len(grouped)), squeeze=False
    )

    for row_index, (mock_id, rows) in enumerate(grouped.items()):
        absolute_axis, feature_axis = axes[row_index]
        mock_z = float(rows[0]["mock_z"])
        mock_logm = float(rows[0]["mock_logm"])
        mock_wave, mock_flux, _ = read_mock(args.mock_dir / rows[0]["mock_file"])

        absolute_axis.plot(
            mock_wave, mock_flux, color="black", linewidth=0.8, alpha=0.9, label="mock"
        )
        mock_feature_flux = robust_normalize(smooth(mock_flux, args.smooth_pixels))
        feature_axis.plot(
            mock_wave / (1 + mock_z),
            mock_feature_flux,
            color="black",
            linewidth=1.0,
            alpha=0.9,
            label="mock",
        )

        absolute_fluxes = [mock_flux]
        feature_fluxes = [mock_feature_flux]
        rest_waves = [mock_wave / (1 + mock_z)]
        for row in rows:
            tier = row["snr_tier"]
            color = TIER_COLORS.get(tier)
            observed_z = float(row["observed_z"])
            wave, flux, _ = read_observed(args.observed_dir / row["observed_file"])
            label = (
                f"observed {tier}: z={observed_z:.3f}, "
                f"S/N={float(row['snr_combined']):.1f}"
            )
            absolute_axis.plot(
                wave, flux, color=color, linewidth=0.55, alpha=0.65, label=label
            )
            feature_flux = robust_normalize(smooth(flux, args.smooth_pixels))
            rest_wave = wave / (1 + observed_z)
            feature_axis.plot(
                rest_wave,
                feature_flux,
                color=color,
                linewidth=0.65,
                alpha=0.7,
                label=label,
            )
            absolute_fluxes.append(flux)
            feature_fluxes.append(feature_flux)
            rest_waves.append(rest_wave)

        set_robust_limits(absolute_axis, absolute_fluxes)
        set_robust_limits(feature_axis, feature_fluxes)
        absolute_axis.set_xlim(3600, 9824)
        feature_axis.set_xlim(
            max(np.nanmin(wave) for wave in rest_waves),
            min(np.nanmax(wave) for wave in rest_waves),
        )
        for label, wavelength in DEFAULT_LINES.items():
            if feature_axis.get_xlim()[0] <= wavelength <= feature_axis.get_xlim()[1]:
                feature_axis.axvline(wavelength, color="0.75", linestyle=":", linewidth=0.7)
                feature_axis.text(
                    wavelength,
                    0.97,
                    label,
                    rotation=90,
                    color="0.4",
                    fontsize=7,
                    ha="right",
                    va="top",
                    transform=feature_axis.get_xaxis_transform(),
                )

        title = f"mock {mock_id}: z={mock_z:.3f}, log(M*/M☉)={mock_logm:.2f}"
        absolute_axis.set_title(title + " — observed frame")
        feature_axis.set_title(title + " — rest-frame features")
        absolute_axis.set_ylabel(
            r"flux [$10^{-17}$ erg s$^{-1}$ cm$^{-2}$ Å$^{-1}$]"
        )
        feature_axis.set_ylabel("robust-normalized flux")
        absolute_axis.grid(alpha=0.15)
        feature_axis.grid(alpha=0.15)
        absolute_axis.legend(fontsize=7, loc="upper right")

    axes[-1, 0].set_xlabel("observed wavelength [Å]")
    axes[-1, 1].set_xlabel("rest wavelength [Å]")
    figure.suptitle("Matched observed DESI spectra versus TNG50 mocks", fontsize=15)
    figure.tight_layout(rect=(0, 0, 1, 0.98))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
