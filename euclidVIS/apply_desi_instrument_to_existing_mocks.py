#!/usr/bin/env python3
"""Apply the feasiBGS DESI instrument and noise model to raw mock spectra.

This is deliberately a post-processing stage: the expensive intrinsic
``*_raw.fits`` spectra are read but never modified.  The output is a standard
DESI B/R/Z ``Spectra`` file named ``desi_spectrum_<subhalo>.fits``.
"""

import argparse
import copy
import shutil
import sys
import traceback
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from astropy.io import fits

from apply_edr_noise_to_existing_mocks import (
    header_metadata,
    load_raw_spectrum,
    raw_and_output_paths,
)
from desi_feasibgs import simulate_feasibgs_exposure


REQUIRED_MANIFEST_COLUMNS = {"snapshot", "subhalo_id"}
REQUIRED_DESI_EXTENSIONS = {
    "FIBERMAP",
    "B_WAVELENGTH",
    "B_FLUX",
    "B_IVAR",
    "B_MASK",
    "B_RESOLUTION",
    "R_WAVELENGTH",
    "R_FLUX",
    "R_IVAR",
    "R_MASK",
    "R_RESOLUTION",
    "Z_WAVELENGTH",
    "Z_FLUX",
    "Z_IVAR",
    "Z_MASK",
    "Z_RESOLUTION",
}


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_popeye.yaml"),
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        help=(
            "Override paths.output_path in the config. The raw spectra are "
            "read from this root and final DESI files are written beside them."
        ),
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--limit", type=int,
        help="Process only the first N manifest rows for a smoke test.",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Replace an existing final desi_spectrum_<id>.fits product.",
    )
    parser.add_argument(
        "--backup-existing", action="store_true",
        help="Save an existing final spectrum as *.pre_feasibgs.fits once.",
    )
    parser.add_argument(
        "--response-only", action="store_true",
        help="Apply the DESI response/resolution without stochastic noise.",
    )
    return parser.parse_args()


def validate_desi_output(path, response_only):
    """Validate the standard camera products and noise provenance."""
    with fits.open(path, memmap=False) as hdul:
        extension_names = {hdu.name.upper() for hdu in hdul}
        missing = sorted(REQUIRED_DESI_EXTENSIONS - extension_names)
        if missing:
            raise RuntimeError(f"{path} is missing DESI extensions: {missing}")

        header = hdul[0].header
        expected_noise = "NONE" if response_only else "FEASIBGS"
        if not bool(header.get("FEASIBGS", False)):
            raise RuntimeError(f"{path} is missing FEASIBGS provenance")
        if str(header.get("NOISEMOD", "")).strip().upper() != expected_noise:
            raise RuntimeError(
                f"{path} has NOISEMOD={header.get('NOISEMOD')!r}; "
                f"expected {expected_noise!r}"
            )

        for band in "BRZ":
            wave = np.asarray(hdul[f"{band}_WAVELENGTH"].data, dtype=float)
            flux = np.asarray(hdul[f"{band}_FLUX"].data, dtype=float)
            ivar = np.asarray(hdul[f"{band}_IVAR"].data, dtype=float)
            mask = np.asarray(hdul[f"{band}_MASK"].data)
            if wave.ndim != 1 or wave.size < 2 or np.any(np.diff(wave) <= 0):
                raise RuntimeError(f"Invalid {band}-camera wavelength grid in {path}")
            if flux.shape[-1] != wave.size or ivar.shape != flux.shape:
                raise RuntimeError(f"Invalid {band}-camera flux/IVAR shapes in {path}")
            if mask.shape != flux.shape:
                raise RuntimeError(f"Invalid {band}-camera mask shape in {path}")
            if np.any(~np.isfinite(flux)) or np.any(~np.isfinite(ivar)):
                raise RuntimeError(f"Non-finite {band}-camera flux or IVAR in {path}")
            if np.any(ivar < 0) or not np.any(ivar > 0):
                raise RuntimeError(f"Invalid {band}-camera inverse variance in {path}")


def process_row(row, config, overwrite, backup_existing, response_only):
    raw_path, output_path = raw_and_output_paths(row, config)
    subhalo_id = int(row["subhalo_id"])
    snapshot = int(row["snapshot"])
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "subhalo_id": subhalo_id, "path": output_path}

    wave, fnu, raw_header = load_raw_spectrum(raw_path)
    desi_config = copy.deepcopy(config["desi"])
    # This stage represents the physical feasiBGS realization. Empirical EDR
    # donor noise, when desired, remains a later and separately auditable step.
    for key in list(desi_config):
        if key.startswith("empirical_noise_donor"):
            desi_config.pop(key)
    desi_config["no_noise"] = bool(response_only)

    if output_path.exists() and backup_existing:
        backup_path = output_path.with_name(
            f"{output_path.stem}.pre_feasibgs.fits"
        )
        if not backup_path.exists():
            shutil.copy2(output_path, backup_path)

    base_seed = int(desi_config.get("noise_seed", 42))
    noise_seed = (base_seed + snapshot * 1000003 + subhalo_id) % 4294967295
    simulate_feasibgs_exposure(
        wave,
        fnu,
        output_path,
        desi_config,
        noise_seed,
        metadata=header_metadata(raw_header),
    )
    validate_desi_output(output_path, response_only)
    return {"status": "written", "subhalo_id": subhalo_id, "path": output_path}


def process_row_safely(row, config, overwrite, backup_existing, response_only):
    try:
        return process_row(
            row, config, overwrite, backup_existing, response_only
        )
    except Exception as error:
        return {
            "status": "failed",
            "subhalo_id": int(row["subhalo_id"]),
            "error": str(error),
            "traceback": traceback.format_exc(),
        }


def main():
    args = parse_args()
    if args.n_jobs is not None and args.n_jobs <= 0:
        raise ValueError("--n-jobs must be positive")
    if args.limit is not None and args.limit <= 0:
        raise ValueError("--limit must be positive")
    if args.backup_existing and not args.overwrite:
        raise ValueError("--backup-existing requires --overwrite")

    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    if args.output_root is not None:
        config["paths"]["output_path"] = str(args.output_root)

    manifest = pd.read_csv(args.manifest)
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise KeyError(f"Manifest is missing columns: {missing_columns}")
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()
    if manifest.empty:
        raise ValueError("Manifest contains no rows to process")

    missing_raw = []
    for row in manifest.to_dict("records"):
        raw_path, _ = raw_and_output_paths(row, config)
        if not raw_path.is_file():
            missing_raw.append(raw_path)
    if missing_raw:
        preview = "\n".join(f"  {path}" for path in missing_raw[:10])
        raise FileNotFoundError(
            f"{len(missing_raw)} raw spectra are missing:\n{preview}"
        )

    n_jobs = args.n_jobs or int(config.get("optimization", {}).get("n_jobs", 1))
    records = manifest.to_dict("records")
    mode = "response only" if args.response_only else "response plus physical noise"
    print(
        f"Applying feasiBGS {mode} to {len(records)} existing raw spectra "
        f"with n_jobs={n_jobs}.",
        flush=True,
    )
    with ProcessPoolExecutor(max_workers=n_jobs) as executor:
        results = list(
            executor.map(
                process_row_safely,
                records,
                [config] * len(records),
                [args.overwrite] * len(records),
                [args.backup_existing] * len(records),
                [args.response_only] * len(records),
            )
        )

    counts = pd.Series([result["status"] for result in results]).value_counts()
    print(counts.to_string(), flush=True)
    for result in results:
        if result["status"] == "written":
            print(f"WROTE: subhalo {result['subhalo_id']}: {result['path']}")

    failures = [result for result in results if result["status"] == "failed"]
    for failure in failures:
        print(
            f"ERROR: subhalo {failure['subhalo_id']}: {failure['error']}\n"
            f"{failure['traceback']}",
            file=sys.stderr,
            flush=True,
        )
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
