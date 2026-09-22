"""Convert noisy DESI B/R/Z FITS products into AstroPT DESI tokens."""

import argparse
import csv
from pathlib import Path

import numpy as np
from astropy.io import fits


WAVE_MIN = 3600.0
WAVE_STEP = 0.8
N_WAVE = 7781
PATCH_SIZE = 10


def astropt_wavelength_grid():
    """Return the 7781-bin grid encoded in the supplied checkpoint."""
    return WAVE_MIN + WAVE_STEP * np.arange(N_WAVE, dtype=np.float64)


def _camera_array(hdul, band, kind, required=True):
    name = f"{band.upper()}_{kind.upper()}"
    try:
        array = np.asarray(hdul[name].data)
    except (KeyError, TypeError):
        if required:
            raise KeyError(f"Missing required DESI extension {name}")
        return None
    if array.ndim == 2:
        if array.shape[0] != 1:
            raise ValueError(f"{name} contains {array.shape[0]} spectra; expected one")
        array = array[0]
    return np.asarray(array).reshape(-1)


def read_and_coadd_desi(path, target_wave):
    """Interpolate and inverse-variance-combine the DESI cameras."""
    numerator = np.zeros_like(target_wave, dtype=np.float64)
    total_ivar = np.zeros_like(target_wave, dtype=np.float64)
    with fits.open(path, memmap=True) as hdul:
        metadata = {
            "subhalo": hdul[0].header.get("SUBHALO", -1),
            "snapshot": hdul[0].header.get("SNAP", -1),
            "redshift": hdul[0].header.get("REDSHIFT", np.nan),
            "mass": hdul[0].header.get("MASS", np.nan),
        }
        for band in ("b", "r", "z"):
            wave = _camera_array(hdul, band, "wavelength").astype(np.float64)
            flux = _camera_array(hdul, band, "flux").astype(np.float64)
            ivar = _camera_array(hdul, band, "ivar").astype(np.float64)
            mask = _camera_array(hdul, band, "mask", required=False)
            valid = np.isfinite(wave) & np.isfinite(flux) & np.isfinite(ivar)
            valid &= ivar > 0
            if mask is not None:
                valid &= mask == 0
            if np.count_nonzero(valid) < 2:
                continue
            wave_valid = wave[valid]
            order = np.argsort(wave_valid)
            wave_valid = wave_valid[order]
            flux_valid = flux[valid][order]
            ivar_valid = ivar[valid][order]
            covered = (target_wave >= wave_valid[0]) & (target_wave <= wave_valid[-1])
            interp_flux = np.interp(target_wave[covered], wave_valid, flux_valid)
            interp_ivar = np.interp(target_wave[covered], wave_valid, ivar_valid)
            numerator[covered] += interp_flux * interp_ivar
            total_ivar[covered] += interp_ivar

    combined_flux = np.zeros_like(target_wave, dtype=np.float64)
    valid = total_ivar > 0
    combined_flux[valid] = numerator[valid] / total_ivar[valid]
    if np.count_nonzero(valid) == 0:
        raise ValueError(f"No valid B/R/Z samples in {path}")
    return combined_flux.astype(np.float32), total_ivar.astype(np.float32), metadata


def make_tokens(flux, normalization="asinh"):
    """Apply checkpoint normalization and split into 10-value tokens."""
    if normalization == "asinh":
        model_flux = np.arcsinh(flux)
    elif normalization == "none":
        model_flux = flux.copy()
    else:
        raise ValueError("normalization must be 'asinh' or 'none'")
    pad = (-len(model_flux)) % PATCH_SIZE
    padded = np.pad(model_flux, (0, pad), constant_values=0.0)
    tokens = padded.reshape(-1, PATCH_SIZE).astype(np.float32)
    positions = np.arange(len(tokens), dtype=np.int64)
    sample_mask = np.pad(
        np.ones(len(model_flux), dtype=np.uint8), (0, pad), constant_values=0
    ).reshape(-1, PATCH_SIZE)
    return tokens, positions, sample_mask


def output_name(path):
    snapshot_dir = next((p for p in path.parents if p.name.startswith("sn")), None)
    prefix = f"{snapshot_dir.name}_" if snapshot_dir is not None else ""
    return f"{prefix}{path.stem}_astropt.npz"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument(
        "--pattern", default="sn*/DESI/desi_spectrum_*.fits",
        help="Recursive glob relative to input-root",
    )
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--normalization", choices=("asinh", "none"), default="asinh")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    paths = sorted(
        p for p in args.input_root.glob(args.pattern)
        if not p.name.endswith("_raw.fits")
    )
    if args.limit is not None:
        paths = paths[:args.limit]
    if not paths:
        raise FileNotFoundError(
            f"No processed DESI spectra matched {args.input_root / args.pattern}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    target_wave = astropt_wavelength_grid()
    rows = []
    for index, path in enumerate(paths, start=1):
        destination = args.output_dir / output_name(path)
        if destination.exists() and not args.overwrite:
            with np.load(destination) as saved:
                metadata = {key: saved[key].item() for key in (
                    "subhalo", "snapshot", "redshift", "mass"
                )}
        else:
            flux, ivar, metadata = read_and_coadd_desi(path, target_wave)
            tokens, positions, sample_mask = make_tokens(flux, args.normalization)
            np.savez_compressed(
                destination,
                wavelength=target_wave.astype(np.float32),
                flux=flux,
                ivar=ivar,
                tokens=tokens,
                positions=positions,
                sample_mask=sample_mask,
                normalization=args.normalization,
                source_path=str(path.resolve()),
                **metadata,
            )
        rows.append({
            "input": str(path.resolve()),
            "prepared": str(destination.resolve()),
            **metadata,
        })
        print(f"[{index}/{len(paths)}] {path.name} -> {destination.name}", flush=True)

    manifest = args.output_dir / "manifest.csv"
    with manifest.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("input", "prepared", "subhalo", "snapshot", "redshift", "mass"),
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Prepared {len(rows)} spectra on {target_wave[0]:.1f}-{target_wave[-1]:.1f} A")
    print(f"Wrote {manifest}")


if __name__ == "__main__":
    main()
