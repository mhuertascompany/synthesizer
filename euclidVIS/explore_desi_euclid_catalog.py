"""Explore the matched DESI DR1 and Euclid DR1 physical properties."""

from pathlib import Path

import matplotlib
import numpy as np
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
INPUT = (
    HERE
    / "catalogs"
    / "catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits"
)
OUTPUT_DIRECTORY = HERE / "catalogs"
LOG_MASS_MIN = 9.5


def plot_distributions(
    redshift,
    log_mass,
    selection,
    label,
    output,
    redshift_max=None,
):
    """Plot redshift and stellar mass for one Euclid survey component."""
    selected = (
        selection
        & np.isfinite(redshift)
        & np.isfinite(log_mass)
        & (log_mass > LOG_MASS_MIN)
    )
    if redshift_max is not None:
        selected &= redshift < redshift_max
    valid_redshift = selected
    valid_mass = selected
    valid_both = valid_redshift & valid_mass
    selected_redshifts = redshift[valid_redshift]
    redshift_min = np.floor(selected_redshifts.min() * 10) / 10
    redshift_max = np.ceil(selected_redshifts.max() * 10) / 10
    redshift_edges = np.linspace(
        redshift_min,
        redshift_max,
        int(round((redshift_max - redshift_min) / 0.05)) + 1,
    )
    mass_max = np.ceil(log_mass[valid_mass].max() * 10) / 10
    mass_edges = np.arange(LOG_MASS_MIN, mass_max + 0.05, 0.05)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].hist(
        redshift[valid_redshift],
        bins=redshift_edges,
        histtype="stepfilled",
        alpha=0.75,
    )
    axes[0].set(xlabel="Photometric redshift", ylabel="Number of galaxies")
    axes[0].set_title(f"Redshift ({valid_redshift.sum():,} galaxies)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.2)

    axes[1].hist(
        log_mass[valid_mass],
        bins=mass_edges,
        histtype="stepfilled",
        alpha=0.75,
    )
    axes[1].set(
        xlabel=r"Median $\log_{10}(M_\star/M_\odot)$",
        ylabel="Number of galaxies",
    )
    axes[1].set_title(f"Stellar mass ({valid_mass.sum():,} galaxies)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    cuts = f"log(Mstar/Msun) > {LOG_MASS_MIN}"
    if redshift_max is not None:
        cuts += f", z < {redshift_max}"
    fig.suptitle(
        f"DESI DR1–Euclid DR1 matched catalog: Euclid {label}, {cuts}"
    )
    fig.savefig(output, dpi=180)
    plt.close(fig)
    return int(selection.sum()), int(valid_both.sum())


def main():
    with fits.open(INPUT, memmap=True) as hdul:
        catalog = hdul[1].data
        redshift = np.asarray(catalog["phz_pp_median_redshift"])
        log_mass = np.asarray(catalog["phz_pp_median_stellarmass"])
        chosen_survey = np.char.strip(
            np.asarray(catalog["chosen_survey"]).astype(str)
        )

    print(f"Rows: {len(redshift):,}")
    for survey in ("DEEP", "WIDE"):
        output = OUTPUT_DIRECTORY / (
            "catalog_MER_DR1_DESI_DR1_"
            f"{survey.lower()}_v1.1_histograms.png"
        )
        rows, finite = plot_distributions(
            redshift,
            log_mass,
            chosen_survey == survey,
            survey,
            output,
            redshift_max=1.5 if survey == "DEEP" else None,
        )
        print(f"Euclid {survey}: {rows:,} rows; {finite:,} finite pairs")
        print(f"Wrote {output}")


if __name__ == "__main__":
    main()
