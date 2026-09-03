"""Convert the Euclid deep-field physical-property catalog and plot it.

The input has tens of millions of rows, so this script streams CSV chunks into
a memory-mapped FITS binary table and accumulates histograms incrementally.
"""

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
from astropy.io import fits

matplotlib.use("Agg")
import matplotlib.pyplot as plt


HERE = Path(__file__).resolve().parent
INPUT = HERE / "catalogs" / "physical properties deep-result.csv"
OUTPUT_FITS = HERE / "catalogs" / "physical_properties_deep.fits"
OUTPUT_PLOT = HERE / "catalogs" / "physical_properties_deep_histograms.png"
CHUNK_SIZE = 1_000_000

COLUMNS = {
    "object_id": "OBJECT_ID",
    "phz_pp_median_redshift": "REDSHIFT",
    "phz_pp_median_stellarmass": "LOGMSTAR",
    "phz_pp_median_sfr": "LOGSFR",
}

FITS_DTYPE = np.dtype(
    [
        ("OBJECT_ID", ">i8"),
        ("REDSHIFT", ">f4"),
        ("LOGMSTAR", ">f4"),
        ("LOGSFR", ">f4"),
    ]
)


def count_rows(path):
    """Count data rows without materializing the CSV."""
    with path.open("rb") as stream:
        return sum(block.count(b"\n") for block in iter(
            lambda: stream.read(16 * 1024 * 1024), b""
        )) - 1


def create_empty_fits(path, nrows):
    """Create a standards-compliant, fixed-size FITS binary table."""
    primary = fits.Header()
    primary["SIMPLE"] = True
    primary["BITPIX"] = 8
    primary["NAXIS"] = 0
    primary["EXTEND"] = True

    table = fits.Header()
    table["XTENSION"] = "BINTABLE"
    table["BITPIX"] = 8
    table["NAXIS"] = 2
    table["NAXIS1"] = FITS_DTYPE.itemsize
    table["NAXIS2"] = nrows
    table["PCOUNT"] = 0
    table["GCOUNT"] = 1
    table["TFIELDS"] = 4
    table["EXTNAME"] = "PHYSICAL_PROPERTIES"
    definitions = [
        ("OBJECT_ID", "K", ""),
        ("REDSHIFT", "E", ""),
        ("LOGMSTAR", "E", "log10(Msun)"),
        ("LOGSFR", "E", "log10(Msun/yr)"),
    ]
    for index, (name, form, unit) in enumerate(definitions, start=1):
        table[f"TTYPE{index}"] = name
        table[f"TFORM{index}"] = form
        if unit:
            table[f"TUNIT{index}"] = unit

    primary_bytes = primary.tostring(endcard=True, padding=True).encode("ascii")
    table_bytes = table.tostring(endcard=True, padding=True).encode("ascii")
    data_bytes = nrows * FITS_DTYPE.itemsize
    padded_data_bytes = ((data_bytes + 2879) // 2880) * 2880
    with path.open("wb") as stream:
        stream.write(primary_bytes)
        stream.write(table_bytes)
        stream.truncate(len(primary_bytes) + len(table_bytes) + padded_data_bytes)
    return len(primary_bytes) + len(table_bytes)


def main():
    nrows = count_rows(INPUT)
    print(f"Converting {nrows:,} rows from {INPUT}", flush=True)
    data_offset = create_empty_fits(OUTPUT_FITS, nrows)
    output = np.memmap(
        OUTPUT_FITS,
        dtype=FITS_DTYPE,
        mode="r+",
        offset=data_offset,
        shape=(nrows,),
    )

    redshift_edges = np.linspace(0, 6, 121)
    mass_edges = np.linspace(6, 13, 141)
    redshift_counts = np.zeros(redshift_edges.size - 1, dtype=np.int64)
    mass_counts = np.zeros(mass_edges.size - 1, dtype=np.int64)
    finite_redshift = finite_mass = 0
    row_start = 0

    reader = pd.read_csv(
        INPUT,
        usecols=list(COLUMNS),
        dtype={
            "object_id": np.int64,
            "phz_pp_median_redshift": np.float32,
            "phz_pp_median_stellarmass": np.float32,
            "phz_pp_median_sfr": np.float32,
        },
        chunksize=CHUNK_SIZE,
    )
    for chunk_number, chunk in enumerate(reader, start=1):
        size = len(chunk)
        row_end = row_start + size
        for csv_name, fits_name in COLUMNS.items():
            output[fits_name][row_start:row_end] = chunk[csv_name].to_numpy()

        redshift = chunk["phz_pp_median_redshift"].to_numpy()
        mass = chunk["phz_pp_median_stellarmass"].to_numpy()
        good_redshift = np.isfinite(redshift)
        good_mass = np.isfinite(mass)
        finite_redshift += int(good_redshift.sum())
        finite_mass += int(good_mass.sum())
        redshift_counts += np.histogram(redshift[good_redshift], redshift_edges)[0]
        mass_counts += np.histogram(mass[good_mass], mass_edges)[0]
        row_start = row_end
        print(f"  chunk {chunk_number}: {row_end:,}/{nrows:,}", flush=True)

    output.flush()
    del output
    if row_start != nrows:
        raise RuntimeError(f"Expected {nrows} rows but wrote {row_start}")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axes[0].stairs(redshift_counts, redshift_edges, fill=True, alpha=0.75)
    axes[0].set(xlabel="Photometric redshift", ylabel="Number of galaxies")
    axes[0].set_title(f"Redshift ({finite_redshift:,} finite values)")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.2)

    axes[1].stairs(mass_counts, mass_edges, fill=True, alpha=0.75)
    axes[1].set(
        xlabel=r"Median $\log_{10}(M_\star/M_\odot)$",
        ylabel="Number of galaxies",
    )
    axes[1].set_title(f"Stellar mass ({finite_mass:,} finite values)")
    axes[1].set_yscale("log")
    axes[1].grid(alpha=0.2)
    fig.suptitle("Euclid deep-field physical-property catalog")
    fig.savefig(OUTPUT_PLOT, dpi=180)
    plt.close(fig)

    print(f"Wrote {OUTPUT_FITS}", flush=True)
    print(f"Wrote {OUTPUT_PLOT}", flush=True)


if __name__ == "__main__":
    main()
