#!/usr/bin/env python3
"""Select observed DESI spectra matched to the local mock examples."""

from __future__ import annotations

import argparse
import csv
import math
import warnings
from pathlib import Path

import numpy as np
from astropy.table import Table
from astropy.units import UnitsWarning


HERE = Path(__file__).resolve().parent
DEFAULT_CATALOG = Path(
    "/Users/marchuertascompany/Documents/data/euclid_desi/"
    "catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.0.fits"
)
DEFAULT_OUTPUT = HERE / "desi_observed_examples.csv"

# The first entry is read from its FITS header. The other two masses came from
# the matched-mock catalog used when those spectra were generated.
MOCKS = (
    {
        "mock_id": 119472,
        "mock_z": 0.13792321543679534,
        "mock_logm": math.log10(12011743057.408295),
    },
    {"mock_id": 534154, "mock_z": 0.24886828397654662, "mock_logm": 10.5},
    {
        "mock_id": 477652,
        "mock_z": 0.948777568795041,
        "mock_logm": math.log10(5.77e9),
    },
)


def text_column(column):
    """Return a FITS byte/string column as stripped Unicode."""
    return np.char.strip(np.asarray(column).astype("U"))


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--catalog", type=Path, default=DEFAULT_CATALOG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--per-mock", type=int, default=3)
    parser.add_argument("--z-tolerance", type=float, default=0.025)
    parser.add_argument("--mass-tolerance", type=float, default=0.25)
    return parser.parse_args()


def main():
    args = parse_args()
    if args.per_mock < 1:
        raise ValueError("--per-mock must be positive")

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        catalog = Table.read(args.catalog)

    redshift = np.asarray(catalog["Z"], dtype=float)
    logm = np.asarray(catalog["LOGM"], dtype=float)
    arm_snr = np.column_stack(
        [np.asarray(catalog[f"SNR_SPEC_{arm}"], dtype=float) for arm in "BRZ"]
    )
    combined_snr = np.sqrt(np.nansum(np.clip(arm_snr, 0, None) ** 2, axis=1))

    valid = (
        (text_column(catalog["SPECTYPE"]) == "GALAXY")
        & (text_column(catalog["chosen_survey"]) == "DEEP")
        & np.isfinite(redshift)
        & np.isfinite(logm)
        & np.isfinite(combined_snr)
        & (redshift > 0)
        & (redshift < 1.5)
        & (logm > 9.5)
        & (logm < 13)
        & (combined_snr > 0)
    )

    if args.per_mock == 1:
        quantiles = np.array([0.5])
        tier_names = ["median"]
    else:
        quantiles = np.linspace(0.2, 0.8, args.per_mock)
        tier_names = [f"q{quantile:.2f}" for quantile in quantiles]
        if args.per_mock == 3:
            tier_names = ["low", "median", "high"]

    rows = []
    used_targetids = set()
    all_valid_indices = np.flatnonzero(valid)

    for mock in MOCKS:
        z0 = mock["mock_z"]
        mass0 = mock["mock_logm"]
        local = (
            valid
            & (np.abs(redshift - z0) < args.z_tolerance)
            & (np.abs(logm - mass0) < args.mass_tolerance)
        )
        candidates = np.flatnonzero(local)

        if candidates.size < args.per_mock:
            distance2 = (
                ((redshift[all_valid_indices] - z0) / args.z_tolerance) ** 2
                + ((logm[all_valid_indices] - mass0) / args.mass_tolerance) ** 2
            )
            candidates = all_valid_indices[np.argsort(distance2)[:200]]

        snr_targets = np.quantile(combined_snr[candidates], quantiles)
        for tier, snr_target in zip(tier_names, snr_targets):
            score = np.abs(combined_snr[candidates] - snr_target) / max(
                snr_target, 1e-6
            )
            score += 0.05 * (
                np.abs(redshift[candidates] - z0) / args.z_tolerance
                + np.abs(logm[candidates] - mass0) / args.mass_tolerance
            )
            for index in candidates[np.argsort(score)]:
                targetid = int(catalog["TARGETID"][index])
                if targetid not in used_targetids:
                    break
            used_targetids.add(targetid)
            rows.append(
                {
                    "mock_id": mock["mock_id"],
                    "mock_file": f"desi_spectrum_{mock['mock_id']}.fits",
                    "mock_z": z0,
                    "mock_logm": mass0,
                    "snr_tier": tier,
                    "targetid": targetid,
                    "observed_file": f"TARGETID_{targetid}.fits",
                    "object_id": int(catalog["object_id"][index]),
                    "observed_z": redshift[index],
                    "observed_logm": logm[index],
                    "delta_z": redshift[index] - z0,
                    "delta_logm": logm[index] - mass0,
                    "snr_b": arm_snr[index, 0],
                    "snr_r": arm_snr[index, 1],
                    "snr_z": arm_snr[index, 2],
                    "snr_combined": combined_snr[index],
                    "survey": text_column(catalog["SURVEY"])[index],
                    "program": text_column(catalog["PROGRAM"])[index],
                    "chosen_survey": text_column(catalog["chosen_survey"])[index],
                }
            )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"Selected {len(rows)} observed spectra -> {args.output}")
    for row in rows:
        print(
            f"mock {row['mock_id']} {row['snr_tier']:>6}: "
            f"TARGETID {row['targetid']}  z={row['observed_z']:.4f}  "
            f"logM={row['observed_logm']:.3f}  "
            f"S/N={row['snr_combined']:.2f}"
        )


if __name__ == "__main__":
    main()
