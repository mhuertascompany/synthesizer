#!/usr/bin/env python3
"""Compare raw DESI fiber spectra generated with different dust opacities."""

import argparse
import csv
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from astropy.io import fits


LINE_WINDOWS = {
    "[O II]": (3727.0, 30.0),
    "Hbeta": (4861.3, 30.0),
    "[O III]": (5006.8, 70.0),
    "Halpha+[N II]": (6562.8, 70.0),
}
HEADER_RATIOS = {
    "[O II]": "O2PKCONT",
    "Hbeta": "HBPKCONT",
    "[O III]": "O3PKCONT",
    "Halpha+[N II]": "HAPKCONT",
}
MICROJY_CGS = 1.0e-29


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "spectra",
        type=Path,
        nargs="+",
        help="Raw FITS spectra, including the existing kappa=20 reference.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--metrics",
        type=Path,
        help="CSV destination; defaults to OUTPUT with a .csv suffix.",
    )
    return parser.parse_args()


def load_raw(path):
    with fits.open(path, memmap=True) as hdul:
        data = hdul[1].data
        header = hdul[1].header.copy()
        wave = np.asarray(data["wavelength"], dtype=float)
        fnu = np.asarray(data["flux"], dtype=float)
    order = np.argsort(wave)
    return wave[order], fnu[order], header


def local_line_metrics(wave, fnu, redshift, center, half_width):
    rest = wave / (1.0 + redshift)
    line = np.abs(rest - center) <= half_width
    continuum = (
        ((rest >= center - half_width - 100.0) & (rest <= center - half_width - 20.0))
        | ((rest >= center + half_width + 20.0) & (rest <= center + half_width + 100.0))
    )
    if not np.any(line) or not np.any(continuum):
        return np.nan, np.nan
    continuum_level = float(np.nanmedian(fnu[continuum]))
    if not np.isfinite(continuum_level) or continuum_level <= 0:
        return np.nan, np.nan
    peak_to_continuum = float(np.nanmax(fnu[line]) / continuum_level)
    line_excess = fnu[line] / continuum_level - 1.0
    line_wave = rest[line]
    ew_rest = float(
        np.sum(
            0.5
            * (line_excess[1:] + line_excess[:-1])
            * np.diff(line_wave)
        )
    )
    return peak_to_continuum, ew_rest


def main():
    args = parse_args()
    if len(args.spectra) < 2:
        raise ValueError("Provide at least two raw spectra")

    records = []
    for path in args.spectra:
        wave, fnu, header = load_raw(path)
        records.append(
            {
                "path": path,
                "wave": wave,
                "fnu": fnu,
                "header": header,
                "kappa": float(header["KAPPA"]),
                "redshift": float(header["REDSHIFT"]),
            }
        )
    records.sort(key=lambda record: record["kappa"])

    subhalos = {int(record["header"]["SUBHALO"]) for record in records}
    projections = {
        (round(float(record["header"]["PHI"]), 10), round(float(record["header"]["THETA"]), 10))
        for record in records
    }
    if len(subhalos) != 1:
        raise ValueError(f"Spectra contain different subhalos: {sorted(subhalos)}")
    if len(projections) != 1:
        raise ValueError(
            "Spectra use different PHI/THETA values; this is not a controlled kappa test"
        )

    colors = plt.cm.viridis(np.linspace(0.1, 0.9, len(records)))
    figure = plt.figure(figsize=(14, 11), constrained_layout=True)
    grid = figure.add_gridspec(3, 2, height_ratios=[1.15, 1, 1])
    full_axis = figure.add_subplot(grid[0, :])
    line_axes = [
        figure.add_subplot(grid[1, 0]),
        figure.add_subplot(grid[1, 1]),
        figure.add_subplot(grid[2, 0]),
        figure.add_subplot(grid[2, 1]),
    ]

    for color, record in zip(colors, records):
        label = rf"$\kappa={record['kappa']:g}$"
        full_axis.plot(
            record["wave"], record["fnu"] / MICROJY_CGS,
            color=color, lw=0.8, label=label,
        )
        for axis, (name, (center, half_width)) in zip(line_axes, LINE_WINDOWS.items()):
            observed_center = center * (1.0 + record["redshift"])
            shown = np.abs(record["wave"] - observed_center) <= (half_width + 120.0) * (1.0 + record["redshift"])
            axis.plot(
                record["wave"][shown], record["fnu"][shown] / MICROJY_CGS,
                color=color, lw=1.0, label=label,
            )

    full_axis.set(
        xlabel=r"Observed wavelength [$\AA$]",
        ylabel=r"Raw fiber $f_\nu$ [$\mu$Jy]",
        title=f"Subhalo {next(iter(subhalos))}: dust-opacity comparison",
    )
    full_axis.legend(ncol=len(records), frameon=False)
    full_axis.grid(alpha=0.2)

    for axis, (name, (center, _)) in zip(line_axes, LINE_WINDOWS.items()):
        redshift = records[0]["redshift"]
        axis.axvline(center * (1.0 + redshift), color="0.5", ls="--", lw=0.8)
        axis.set(
            xlabel=r"Observed wavelength [$\AA$]",
            ylabel=r"Raw fiber $f_\nu$ [$\mu$Jy]",
            title=name,
        )
        axis.grid(alpha=0.2)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(args.output, dpi=180)
    plt.close(figure)

    rows = []
    for record in records:
        r_band = (record["wave"] >= 5600.0) & (record["wave"] <= 7000.0)
        median_fnu = float(np.nanmedian(record["fnu"][r_band]))
        r_ab = -2.5 * np.log10(median_fnu) - 48.6 if median_fnu > 0 else np.nan
        base = {
            "file": str(record["path"]),
            "subhalo": int(record["header"]["SUBHALO"]),
            "redshift": record["redshift"],
            "kappa": record["kappa"],
            "phi": float(record["header"]["PHI"]),
            "theta": float(record["header"]["THETA"]),
            "median_fnu_5600_7000_microjy": median_fnu / MICROJY_CGS,
            "fiber_r_ab_proxy": r_ab,
        }
        for name, (center, half_width) in LINE_WINDOWS.items():
            peak, ew = local_line_metrics(
                record["wave"], record["fnu"], record["redshift"], center, half_width
            )
            key = name.lower().replace(" ", "_").replace("[", "").replace("]", "").replace("+", "_plus_")
            base[f"{key}_peak_to_continuum"] = peak
            base[f"{key}_ew_rest_angstrom"] = ew
            header_key = HEADER_RATIOS[name]
            base[f"{key}_header_peak_to_continuum"] = record["header"].get(header_key, np.nan)
        rows.append(base)

    metrics = args.metrics or args.output.with_suffix(".csv")
    metrics.parent.mkdir(parents=True, exist_ok=True)
    with metrics.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {args.output}")
    print(f"Wrote {metrics}")


if __name__ == "__main__":
    main()
