"""Add TNG specific-SFR and synthetic fiber r flux to a mock manifest."""

import argparse
from pathlib import Path

import illustris_python as il
import numpy as np
import pandas as pd
import yaml
from astropy.io import fits


NANOMAGGY_FNU_CGS = 3.631e-29


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).with_name("config_matched.yaml"),
    )
    parser.add_argument("--r-wave-min", type=float, default=5600.0)
    parser.add_argument("--r-wave-max", type=float, default=7000.0)
    parser.add_argument("--log-ssfr-floor", type=float, default=-13.0)
    return parser.parse_args()


def raw_spectrum_path(row, config):
    root = Path(config["paths"]["output_path"])
    subdir = config["paths"].get("desi_subdir", "DESI")
    return (
        root
        / f"sn{int(row['snapshot'])}"
        / subdir
        / f"desi_spectrum_{int(row['subhalo_id'])}_raw.fits"
    )


def load_raw_fnu(path):
    with fits.open(path, memmap=False) as hdul:
        table_hdu = next(
            (
                hdu
                for hdu in hdul
                if getattr(hdu, "data", None) is not None
                and getattr(hdu.data, "names", None) is not None
            ),
            None,
        )
        if table_hdu is None:
            raise ValueError(f"No binary-table spectrum found in {path}")
        names = {name.upper(): name for name in table_hdu.data.names}
        missing = {"WAVELENGTH", "FLUX"} - set(names)
        if missing:
            raise KeyError(f"{path} is missing columns: {sorted(missing)}")
        wave = np.asarray(
            table_hdu.data[names["WAVELENGTH"]], dtype=float
        ).squeeze()
        fnu = np.asarray(table_hdu.data[names["FLUX"]], dtype=float).squeeze()
    return wave, fnu


def synthetic_top_hat_flux_nanomaggy(path, wave_min, wave_max):
    """Return mean observed F_nu in a simple r-like top-hat passband."""
    wave, fnu = load_raw_fnu(path)
    selected = (
        np.isfinite(wave)
        & np.isfinite(fnu)
        & (wave >= float(wave_min))
        & (wave <= float(wave_max))
    )
    if selected.sum() < 2:
        raise ValueError(
            f"{path} does not cover {wave_min:.0f}-{wave_max:.0f} Angstrom"
        )
    selected_wave = wave[selected]
    selected_fnu = fnu[selected]
    if hasattr(np, "trapezoid"):
        integral = np.trapezoid(selected_fnu, selected_wave)
    else:
        integral = np.trapz(selected_fnu, selected_wave)
    mean_fnu = integral / (selected_wave[-1] - selected_wave[0])
    return float(mean_fnu / NANOMAGGY_FNU_CGS)


def add_tng_sfr(table, config, log_ssfr_floor):
    """Read only the SubhaloSFR group-catalog field once per snapshot."""
    tng_path = config["paths"]["tng_path"]
    table["tng_sfr_msun_per_year"] = np.nan
    table["tng_log_sfr"] = np.nan
    table["tng_log_ssfr"] = np.nan
    for snapshot, indices in table.groupby("snapshot").groups.items():
        values = il.groupcat.loadSubhalos(
            tng_path,
            int(snapshot),
            fields=["SubhaloSFR"],
        )
        if isinstance(values, dict):
            values = values["SubhaloSFR"]
        values = np.asarray(values, dtype=float)
        for index in indices:
            subhalo_id = int(table.at[index, "subhalo_id"])
            sfr = max(float(values[subhalo_id]), 0.0)
            log_mass = float(table.at[index, "tng_log_mass"])
            log_ssfr = max(
                np.log10(sfr) - log_mass if sfr > 0 else -np.inf,
                float(log_ssfr_floor),
            )
            table.at[index, "tng_sfr_msun_per_year"] = sfr
            table.at[index, "tng_log_sfr"] = log_ssfr + log_mass
            table.at[index, "tng_log_ssfr"] = log_ssfr
    return table


def main():
    args = parse_args()
    if args.r_wave_min >= args.r_wave_max:
        raise ValueError("--r-wave-min must be less than --r-wave-max")
    with args.config.open() as stream:
        config = yaml.safe_load(stream)
    table = pd.read_csv(args.manifest)
    required = {"snapshot", "subhalo_id", "tng_log_mass"}
    missing = sorted(required - set(table.columns))
    if missing:
        raise KeyError(f"Manifest is missing columns: {missing}")

    table = add_tng_sfr(table, config, args.log_ssfr_floor)
    fluxes = []
    for row in table.to_dict("records"):
        path = raw_spectrum_path(row, config)
        if not path.is_file():
            raise FileNotFoundError(path)
        fluxes.append(
            synthetic_top_hat_flux_nanomaggy(
                path,
                args.r_wave_min,
                args.r_wave_max,
            )
        )
    fluxes = np.asarray(fluxes)
    if np.any(~np.isfinite(fluxes)) or np.any(fluxes <= 0):
        raise ValueError("One or more synthetic mock r fluxes are non-positive")
    table["mock_fiber_flux_r_nanomaggy"] = fluxes
    table["mock_log_fiber_flux_r"] = np.log10(fluxes)
    table["mock_fiber_r_abmag"] = 22.5 - 2.5 * np.log10(fluxes)
    table["mock_r_wave_min_angstrom"] = float(args.r_wave_min)
    table["mock_r_wave_max_angstrom"] = float(args.r_wave_max)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output, index=False)
    print(f"Wrote {len(table)} enriched targets to {args.output}")
    print(
        table[
            [
                "tng_log_ssfr",
                "mock_fiber_flux_r_nanomaggy",
                "mock_fiber_r_abmag",
            ]
        ].describe().to_string()
    )


if __name__ == "__main__":
    main()
