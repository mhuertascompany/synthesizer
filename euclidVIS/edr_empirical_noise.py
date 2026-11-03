"""DESI donor selection and empirical release-matched noise utilities.

The module name is retained for compatibility with existing manifests and
batch scripts, but the current donor archive is DESI DR1 rather than EDR.
"""

from __future__ import annotations

import argparse
import os
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from astropy.io import fits
from astropy.units import UnitsWarning
from scipy.spatial import cKDTree


TARGETID_PATTERN = re.compile(r"^TARGETID_(\d+)\.fits$")


def _text(values):
    return np.char.strip(np.asarray(values).astype("U"))


def discover_spectrum_targetids(spectra_dir):
    """Discover available TARGETIDs with one efficient directory scan."""
    targetids = set()
    with os.scandir(spectra_dir) as entries:
        for entry in entries:
            if not entry.is_file():
                continue
            match = TARGETID_PATTERN.match(entry.name)
            if match:
                targetids.add(int(match.group(1)))
    if not targetids:
        raise FileNotFoundError(
            f"No TARGETID_<id>.fits spectra found under {spectra_dir}"
        )
    return targetids


def load_donor_candidates(catalog_path, spectra_dir, sample):
    """Load valid DESI noise donors and their matching coordinates."""
    required = {
        "TARGETID",
        "SPECTYPE",
        "PROGRAM",
        "SURVEY",
        "chosen_survey",
        "Z",
        "LOGM",
        "LOGSFR",
        "LOGSFR_ERR",
        "FLUX_R",
        "FLUX_IVAR_R",
        "SNR_SPEC_B",
        "SNR_SPEC_R",
        "SNR_SPEC_Z",
    }
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UnitsWarning)
        with fits.open(catalog_path, memmap=True) as hdul:
            table = hdul[1].data
            missing = sorted(required - set(table.names))
            if missing:
                raise KeyError(f"DESI donor catalog is missing columns: {missing}")
            columns = {name: np.asarray(table[name]) for name in required}

    targetid = columns["TARGETID"].astype(np.int64)
    redshift = columns["Z"].astype(float)
    log_mass = columns["LOGM"].astype(float)
    log_sfr = columns["LOGSFR"].astype(float)
    log_sfr_error = columns["LOGSFR_ERR"].astype(float)
    flux_r = columns["FLUX_R"].astype(float)
    flux_ivar_r = columns["FLUX_IVAR_R"].astype(float)
    program = np.char.lower(_text(columns["PROGRAM"]))
    combined_snr = np.sqrt(
        sum(
            np.clip(columns[f"SNR_SPEC_{arm}"].astype(float), 0, None) ** 2
            for arm in "BRZ"
        )
    )

    valid = (
        (_text(columns["SPECTYPE"]) == "GALAXY")
        & (_text(columns["chosen_survey"]) == "DEEP")
        & np.isin(program, ("bright", "dark"))
        & np.isfinite(redshift)
        & np.isfinite(log_mass)
        & np.isfinite(log_sfr)
        & np.isfinite(log_sfr_error)
        & np.isfinite(flux_r)
        & np.isfinite(flux_ivar_r)
        & np.isfinite(combined_snr)
        & (redshift >= float(sample["z_min"]))
        & (redshift < float(sample["z_max"]))
        & (log_mass > float(sample["log_mass_min"]))
        & (log_mass <= float(sample["log_mass_max"]))
        & (flux_r > 0)
        & (flux_ivar_r > 0)
        & (combined_snr > 0)
    )

    available = discover_spectrum_targetids(spectra_dir)
    valid &= np.fromiter(
        (int(value) in available for value in targetid),
        dtype=bool,
        count=len(targetid),
    )
    indices = np.flatnonzero(valid)
    if indices.size == 0:
        raise ValueError("No DESI donor galaxies pass the selection and file checks")

    return pd.DataFrame(
        {
            "targetid": targetid[indices],
            "spectrum_path": [
                str(Path(spectra_dir) / f"TARGETID_{value}.fits")
                for value in targetid[indices]
            ],
            "program": program[indices],
            "survey": _text(columns["SURVEY"])[indices],
            "redshift": redshift[indices],
            "log_mass": log_mass[indices],
            "log_sfr": log_sfr[indices],
            "log_sfr_error": log_sfr_error[indices],
            "log_ssfr": (log_sfr - log_mass)[indices],
            "flux_r": flux_r[indices],
            "log_flux_r": np.log10(flux_r[indices]),
            "snr_b": columns["SNR_SPEC_B"].astype(float)[indices],
            "snr_r": columns["SNR_SPEC_R"].astype(float)[indices],
            "snr_z": columns["SNR_SPEC_Z"].astype(float)[indices],
            "snr_combined": combined_snr[indices],
        }
    )


def assign_noise_donors(targets, catalog_path, spectra_dir, sample, config, seed):
    """Assign reproducible, usually unique DESI donors to mock targets."""
    donors = load_donor_candidates(catalog_path, spectra_dir, sample)
    z_scale = float(config.get("donor_redshift_scale", 0.05))
    mass_scale = float(config.get("donor_log_mass_scale", 0.2))
    ssfr_scale = float(config.get("donor_log_ssfr_scale", 0.6))
    flux_scale = float(config.get("donor_log_flux_scale", 0.4))
    ssfr_floor = float(config.get("donor_log_ssfr_floor", -13.0))
    ssfr_ceiling = float(config.get("donor_log_ssfr_ceiling", -8.0))
    match_ssfr = bool(config.get("donor_match_ssfr", False))
    match_flux = bool(config.get("donor_match_flux", False))
    k_neighbors = int(config.get("donor_k_neighbors", 64))
    if (
        z_scale <= 0
        or mass_scale <= 0
        or ssfr_scale <= 0
        or flux_scale <= 0
        or k_neighbors <= 0
    ):
        raise ValueError("DESI donor scales and donor_k_neighbors must be positive")
    if ssfr_floor >= ssfr_ceiling:
        raise ValueError("donor_log_ssfr_floor must be below its ceiling")

    target_mass_column = (
        "tng_log_mass" if "tng_log_mass" in targets.columns else "target_log_mass"
    )
    donor_coordinate_columns = [
        donors["redshift"].to_numpy() / z_scale,
        donors["log_mass"].to_numpy() / mass_scale,
    ]
    target_coordinate_columns = [
        targets["target_redshift"].to_numpy(dtype=float) / z_scale,
        targets[target_mass_column].to_numpy(dtype=float) / mass_scale,
    ]
    match_dimensions = ["redshift", "stellar_mass"]
    if match_ssfr:
        if "tng_log_ssfr" not in targets.columns:
            raise KeyError(
                "SFR matching requires a tng_log_ssfr target column; run "
                "measure_mock_matching_properties.py first"
            )
        donor_coordinate_columns.append(
            donors["log_ssfr"].clip(ssfr_floor, ssfr_ceiling).to_numpy()
            / ssfr_scale
        )
        target_coordinate_columns.append(
            targets["tng_log_ssfr"]
            .clip(ssfr_floor, ssfr_ceiling)
            .to_numpy(dtype=float)
            / ssfr_scale
        )
        match_dimensions.append("specific_sfr")
    if match_flux:
        if "mock_log_fiber_flux_r" not in targets.columns:
            raise KeyError(
                "Flux matching requires a mock_log_fiber_flux_r target column; "
                "run measure_mock_matching_properties.py first"
            )
        donor_coordinate_columns.append(
            donors["log_flux_r"].to_numpy() / flux_scale
        )
        target_coordinate_columns.append(
            targets["mock_log_fiber_flux_r"].to_numpy(dtype=float) / flux_scale
        )
        match_dimensions.append("r_flux")
    donor_coordinates = np.column_stack(donor_coordinate_columns)
    target_coordinates = np.column_stack(target_coordinate_columns)
    if np.any(~np.isfinite(target_coordinates)):
        raise ValueError("Mock donor-matching coordinates contain non-finite values")
    tree = cKDTree(donor_coordinates)
    k = min(k_neighbors, len(donors))
    distances, neighbor_indices = tree.query(target_coordinates, k=k)
    if k == 1:
        distances = distances[:, None]
        neighbor_indices = neighbor_indices[:, None]

    rng = np.random.default_rng(int(seed))
    used = set()
    assignments = []
    assignment_distances = []
    for distance, neighbors in zip(distances, neighbor_indices):
        unused = np.asarray(
            [index for index in neighbors if int(index) not in used], dtype=int
        )
        choices = unused if unused.size else np.asarray(neighbors, dtype=int)
        choice_distances = np.asarray(
            [distance[np.flatnonzero(neighbors == choice)[0]] for choice in choices]
        )
        weights = np.exp(-0.5 * np.minimum(choice_distances, 8.0) ** 2)
        weights = weights / weights.sum() if weights.sum() > 0 else None
        selected = int(rng.choice(choices, p=weights))
        used.add(selected)
        assignments.append(donors.iloc[selected])
        assignment_distances.append(
            float(distance[np.flatnonzero(neighbors == selected)[0]])
        )

    assigned = pd.DataFrame(assignments).reset_index(drop=True)
    result = targets.copy().reset_index(drop=True)
    result["noise_donor_targetid"] = assigned["targetid"].astype(np.int64)
    result["noise_donor_path"] = assigned["spectrum_path"]
    result["noise_donor_program"] = assigned["program"]
    result["noise_donor_survey"] = assigned["survey"]
    result["noise_donor_redshift"] = assigned["redshift"].astype(float)
    result["noise_donor_log_mass"] = assigned["log_mass"].astype(float)
    result["noise_donor_log_sfr"] = assigned["log_sfr"].astype(float)
    result["noise_donor_log_sfr_error"] = assigned["log_sfr_error"].astype(float)
    result["noise_donor_log_ssfr"] = assigned["log_ssfr"].astype(float)
    result["noise_donor_flux_r"] = assigned["flux_r"].astype(float)
    result["noise_donor_log_flux_r"] = assigned["log_flux_r"].astype(float)
    for arm in "brz":
        result[f"noise_donor_snr_{arm}"] = assigned[f"snr_{arm}"].astype(float)
    result["noise_donor_snr_combined"] = assigned["snr_combined"].astype(float)
    result["noise_donor_delta_z"] = (
        result["noise_donor_redshift"] - result["target_redshift"]
    )
    result["noise_donor_delta_log_mass"] = (
        result["noise_donor_log_mass"] - result["target_log_mass"]
    )
    result["noise_donor_delta_tng_log_mass"] = (
        result["noise_donor_log_mass"] - result[target_mass_column]
    )
    if match_ssfr:
        result["noise_donor_delta_log_ssfr"] = (
            result["noise_donor_log_ssfr"] - result["tng_log_ssfr"]
        )
    if match_flux:
        result["noise_donor_log_flux_ratio"] = (
            result["noise_donor_log_flux_r"]
            - result["mock_log_fiber_flux_r"]
        )
    result["noise_donor_match_distance"] = assignment_distances
    result["noise_donor_match_dimensions"] = "+".join(match_dimensions)
    return result


def load_noise_template(path):
    """Read a simplified DESI WAVELENGTH/IVAR/MASK spectrum."""
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore", message="File may have been truncated")
        with fits.open(path, memmap=False) as hdul:
            data = hdul[1].data
            names = {name.upper(): name for name in data.names}
            wave = np.asarray(data[names["WAVELENGTH"]], dtype=float).squeeze()
            ivar = np.asarray(data[names["IVAR"]], dtype=float).squeeze()
            mask = np.asarray(data[names["MASK"]]).squeeze()
    order = np.argsort(wave)
    wave, ivar, mask = wave[order], ivar[order], mask[order]
    if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0):
        raise ValueError(f"Invalid DESI wavelength grid in {path}")
    return wave, ivar, mask


def apply_noise_template(output_path, donor_path, seed, release="DR1"):
    """Replace noise using an empirical DESI release IVAR/mask template."""
    release = re.sub(r"[^A-Z0-9]+", "_", str(release).upper()).strip("_")
    if not release:
        raise ValueError("DESI empirical-noise release label cannot be empty")
    donor_wave, donor_ivar, donor_mask = load_noise_template(donor_path)
    donor_good = (
        np.isfinite(donor_wave)
        & np.isfinite(donor_ivar)
        & (donor_ivar > 0)
        & (donor_mask == 0)
    )
    if donor_good.sum() < 2:
        raise ValueError(f"DESI donor {donor_path} has insufficient valid IVAR")
    donor_variance = np.full(donor_ivar.shape, np.nan, dtype=float)
    donor_variance[donor_good] = 1.0 / donor_ivar[donor_good]
    rng = np.random.default_rng(int(seed))

    with fits.open(output_path, mode="update", memmap=False) as hdul:
        for arm in "BRZ":
            wave = np.asarray(hdul[f"{arm}_WAVELENGTH"].data, dtype=float).squeeze()
            flux_hdu = hdul[f"{arm}_FLUX"]
            ivar_hdu = hdul[f"{arm}_IVAR"]
            mask_hdu = hdul[f"{arm}_MASK"]
            signal = np.asarray(flux_hdu.data, dtype=float).squeeze()
            instrument_mask = np.asarray(mask_hdu.data).squeeze()

            variance = np.interp(
                wave,
                donor_wave[donor_good],
                donor_variance[donor_good],
                left=np.nan,
                right=np.nan,
            )
            right = np.searchsorted(donor_wave, wave, side="left")
            right = np.clip(right, 0, donor_wave.size - 1)
            left = np.clip(right - 1, 0, donor_wave.size - 1)
            nearest = np.where(
                np.abs(wave - donor_wave[left]) <= np.abs(donor_wave[right] - wave),
                left,
                right,
            )
            empirical_mask = np.bitwise_or(
                instrument_mask.astype(mask_hdu.data.dtype),
                donor_mask[nearest].astype(mask_hdu.data.dtype),
            )
            valid = np.isfinite(variance) & (variance > 0) & (empirical_mask == 0)
            noisy_flux = signal.copy()
            noisy_flux[valid] += rng.normal(0.0, np.sqrt(variance[valid]))
            empirical_ivar = np.zeros_like(variance)
            empirical_ivar[valid] = 1.0 / variance[valid]

            flux_hdu.data[...] = noisy_flux.reshape(flux_hdu.data.shape)
            ivar_hdu.data[...] = empirical_ivar.reshape(ivar_hdu.data.shape)
            mask_hdu.data[...] = empirical_mask.reshape(mask_hdu.data.shape)

        target_match = TARGETID_PATTERN.match(Path(donor_path).name)
        header = hdul[0].header
        header["NOISEMOD"] = (
            f"{release}_IVAR",
            f"Empirical DESI {release} inverse-variance noise",
        )
        header["DONREL"] = (release, "DESI noise donor data release")
        header["DONOR"] = (
            int(target_match.group(1)) if target_match else -1,
            "DESI noise donor TARGETID",
        )
        hdul.flush()


def parse_args():
    parser = argparse.ArgumentParser(
        description="Assign empirical DESI noise donors to a matched mock manifest."
    )
    parser.add_argument("--targets", type=Path, required=True)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--spectra-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--copy-list", type=Path)
    parser.add_argument(
        "--assigned-spectra-dir",
        type=Path,
        help="Path to record in the manifest after donors are transferred",
    )
    parser.add_argument("--seed", type=int, default=190734905)
    parser.add_argument("--z-min", type=float, default=0.0)
    parser.add_argument("--z-max", type=float, default=1.5)
    parser.add_argument("--log-mass-min", type=float, default=9.5)
    parser.add_argument("--log-mass-max", type=float, default=14.0)
    parser.add_argument("--neighbors", type=int, default=64)
    parser.add_argument("--redshift-scale", type=float, default=0.05)
    parser.add_argument("--log-mass-scale", type=float, default=0.2)
    parser.add_argument("--match-ssfr", action="store_true")
    parser.add_argument("--match-flux", action="store_true")
    parser.add_argument("--log-ssfr-scale", type=float, default=0.6)
    parser.add_argument("--log-flux-scale", type=float, default=0.4)
    parser.add_argument("--log-ssfr-floor", type=float, default=-13.0)
    parser.add_argument("--log-ssfr-ceiling", type=float, default=-8.0)
    return parser.parse_args()


def main():
    args = parse_args()
    targets = pd.read_csv(args.targets)
    required = {"target_redshift", "target_log_mass"}
    missing = sorted(required - set(targets.columns))
    if missing:
        raise KeyError(f"Matched mock manifest is missing columns: {missing}")
    sample = {
        "z_min": args.z_min,
        "z_max": args.z_max,
        "log_mass_min": args.log_mass_min,
        "log_mass_max": args.log_mass_max,
    }
    config = {
        "donor_k_neighbors": args.neighbors,
        "donor_redshift_scale": args.redshift_scale,
        "donor_log_mass_scale": args.log_mass_scale,
        "donor_match_ssfr": args.match_ssfr,
        "donor_match_flux": args.match_flux,
        "donor_log_ssfr_scale": args.log_ssfr_scale,
        "donor_log_flux_scale": args.log_flux_scale,
        "donor_log_ssfr_floor": args.log_ssfr_floor,
        "donor_log_ssfr_ceiling": args.log_ssfr_ceiling,
    }
    assigned = assign_noise_donors(
        targets,
        args.catalog,
        args.spectra_dir,
        sample,
        config,
        args.seed,
    )
    source_paths = assigned["noise_donor_path"].copy()
    if args.assigned_spectra_dir is not None:
        assigned["noise_donor_path"] = [
            str(args.assigned_spectra_dir / Path(path).name)
            for path in source_paths
        ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    assigned.to_csv(args.output, index=False)
    copy_list = args.copy_list or args.output.with_suffix(".files.txt")
    copy_list.parent.mkdir(parents=True, exist_ok=True)
    unique_paths = source_paths.drop_duplicates()
    copy_list.write_text("\n".join(unique_paths) + "\n")
    print(f"Assigned {len(assigned)} mocks using {len(unique_paths)} DESI donors")
    print(assigned.groupby("noise_donor_program").size().to_string())
    print(f"Manifest: {args.output}")
    print(f"Files to transfer: {copy_list}")


if __name__ == "__main__":
    main()
