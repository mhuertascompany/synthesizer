"""Apply empirical EDR noise to existing noiseless DESI mock spectra."""

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

from desi_feasibgs import simulate_feasibgs_exposure


REQUIRED_MANIFEST_COLUMNS = {
    "snapshot",
    "subhalo_id",
    "noise_donor_path",
    "noise_donor_targetid",
    "noise_donor_program",
}


def parse_args():
    parser = argparse.ArgumentParser(
        description=(
            "Run existing noiseless *_raw.fits spectra through feasiBGS and "
            "apply the EDR donor IVAR/mask recorded in a matched manifest."
        )
    )
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_matched.yaml"),
    )
    parser.add_argument("--n-jobs", type=int, default=None)
    parser.add_argument(
        "--limit",
        type=int,
        help="Process only the first N manifest rows for a smoke test.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace an existing final desi_spectrum_<id>.fits product.",
    )
    parser.add_argument(
        "--backup-existing",
        action="store_true",
        help="Save the previous final spectrum as *.pre_edr.fits once.",
    )
    return parser.parse_args()


def raw_and_output_paths(row, config):
    output_root = Path(config["paths"]["output_path"])
    desi_subdir = config["paths"].get("desi_subdir", "DESI")
    snapshot = int(row["snapshot"])
    subhalo_id = int(row["subhalo_id"])
    directory = output_root / f"sn{snapshot}" / desi_subdir
    raw_path = directory / f"desi_spectrum_{subhalo_id}_raw.fits"
    output_path = directory / f"desi_spectrum_{subhalo_id}.fits"
    return raw_path, output_path


def load_raw_spectrum(path):
    """Load the pre-instrument observed-frame F_nu spectrum."""
    with fits.open(path, memmap=False) as hdul:
        spectrum_hdu = next(
            (
                hdu
                for hdu in hdul
                if getattr(hdu, "data", None) is not None
                and getattr(hdu.data, "names", None) is not None
            ),
            None,
        )
        if spectrum_hdu is None:
            raise ValueError(f"No binary-table spectrum found in {path}")
        names = {name.upper(): name for name in spectrum_hdu.data.names}
        missing = {"WAVELENGTH", "FLUX"} - set(names)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        wave = np.asarray(
            spectrum_hdu.data[names["WAVELENGTH"]], dtype=np.float64
        ).squeeze()
        fnu = np.asarray(
            spectrum_hdu.data[names["FLUX"]], dtype=np.float64
        ).squeeze()
        header = spectrum_hdu.header.copy()

    if wave.ndim != 1 or fnu.ndim != 1 or wave.shape != fnu.shape:
        raise ValueError(f"Invalid wavelength/flux arrays in {path}")
    if wave.size < 2 or np.any(~np.isfinite(wave)) or np.any(np.diff(wave) <= 0):
        raise ValueError(f"Invalid wavelength grid in {path}")
    if np.any(~np.isfinite(fnu)):
        raise ValueError(f"Non-finite flux values in {path}")
    if str(header.get("SPECTYPE", "FNU")).strip().upper() != "FNU":
        raise ValueError(f"Expected an F_nu raw spectrum in {path}")
    if str(header.get("WAVEUNIT", "Angstrom")).strip().lower() not in {
        "angstrom",
        "angstroms",
        "aa",
    }:
        raise ValueError(f"Expected Angstrom wavelengths in {path}")
    return wave, fnu, header


def header_metadata(header):
    metadata = {}
    for key in (
        "REDSHIFT",
        "SUBHALO",
        "MASS",
        "PHI",
        "THETA",
        "SNAP",
        "KAPPA",
        "DTM",
        "CURVE",
        "VELSHIFT",
    ):
        if key in header:
            value = header[key]
            if isinstance(value, np.generic):
                value = value.item()
            metadata[key] = value
    return metadata


def process_row(row, config, overwrite, backup_existing):
    raw_path, output_path = raw_and_output_paths(row, config)
    subhalo_id = int(row["subhalo_id"])
    snapshot = int(row["snapshot"])
    if output_path.exists() and not overwrite:
        return {"status": "skipped", "subhalo_id": subhalo_id, "path": output_path}

    wave, fnu, raw_header = load_raw_spectrum(raw_path)
    desi_config = copy.deepcopy(config["desi"])
    desi_config["empirical_noise_donor"] = str(row["noise_donor_path"])
    desi_config["empirical_noise_donor_targetid"] = int(
        row["noise_donor_targetid"]
    )
    desi_config["empirical_noise_donor_program"] = str(
        row["noise_donor_program"]
    )
    base_seed = int(desi_config.get("noise_seed", 42))
    noise_seed = (base_seed + snapshot * 1000003 + subhalo_id) % 4294967295

    if output_path.exists() and backup_existing:
        backup_path = output_path.with_name(f"{output_path.stem}.pre_edr.fits")
        if not backup_path.exists():
            shutil.copy2(output_path, backup_path)

    simulate_feasibgs_exposure(
        wave,
        fnu,
        output_path,
        desi_config,
        noise_seed,
        metadata=header_metadata(raw_header),
    )
    with fits.open(output_path, memmap=False) as hdul:
        noise_model = hdul[0].header.get("NOISEMOD")
        donor = hdul[0].header.get("DONOR")
    if noise_model != "EDR_IVAR" or donor != int(row["noise_donor_targetid"]):
        raise RuntimeError(
            f"Output validation failed for subhalo {subhalo_id}: "
            f"NOISEMOD={noise_model}, DONOR={donor}"
        )
    return {"status": "written", "subhalo_id": subhalo_id, "path": output_path}


def process_row_safely(row, config, overwrite, backup_existing):
    try:
        return process_row(row, config, overwrite, backup_existing)
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
    if str(config.get("desi", {}).get("noise_model", "")).lower() != "feasibgs_edr":
        raise ValueError("config desi.noise_model must be 'feasibgs_edr'")

    manifest = pd.read_csv(args.manifest)
    missing_columns = sorted(REQUIRED_MANIFEST_COLUMNS - set(manifest.columns))
    if missing_columns:
        raise KeyError(f"Manifest is missing columns: {missing_columns}")
    if args.limit is not None:
        manifest = manifest.iloc[: args.limit].copy()
    if manifest.empty:
        raise ValueError("Manifest contains no rows to process")

    missing_raw = []
    missing_donors = []
    for row in manifest.to_dict("records"):
        raw_path, _ = raw_and_output_paths(row, config)
        if not raw_path.is_file():
            missing_raw.append(raw_path)
        donor_path = Path(str(row["noise_donor_path"]))
        if not donor_path.is_file():
            missing_donors.append(donor_path)
    if missing_raw or missing_donors:
        messages = []
        if missing_raw:
            preview = "\n".join(f"  {path}" for path in missing_raw[:10])
            messages.append(f"{len(missing_raw)} raw spectra are missing:\n{preview}")
        if missing_donors:
            unique = list(dict.fromkeys(missing_donors))
            preview = "\n".join(f"  {path}" for path in unique[:10])
            messages.append(f"{len(unique)} donor spectra are missing:\n{preview}")
        raise FileNotFoundError("\n".join(messages))

    n_jobs = args.n_jobs or int(config.get("optimization", {}).get("n_jobs", 1))
    records = manifest.to_dict("records")
    print(
        f"Applying empirical EDR noise to {len(records)} existing raw spectra "
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
            )
        )
    counts = pd.Series([result["status"] for result in results]).value_counts()
    print(counts.to_string(), flush=True)

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
