#!/usr/bin/env python3
"""Create a wavelength-trimmed Synthesizer HDF5 grid.

The high-resolution production grids are deliberately broad in wavelength and
can be several GB.  Euclid VIS plus DESI only need a small rest-frame interval,
so this utility copies the complete grid schema while slicing every spectral
dataset along its wavelength axis.  Line tables and all other metadata are
preserved; Synthesizer filters line tables to the retained spectral range when
the resulting grid is loaded.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import h5py
import numpy as np


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Source Synthesizer HDF5 grid")
    parser.add_argument("output", type=Path, help="Trimmed output HDF5 grid")
    parser.add_argument("--lam-min", type=float, required=True, help="Minimum rest wavelength [Angstrom]")
    parser.add_argument("--lam-max", type=float, required=True, help="Maximum rest wavelength [Angstrom]")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def copy_attrs(source, destination):
    """Copy every HDF5 attribute without changing its type."""
    for key, value in source.attrs.items():
        destination.attrs[key] = value


def copy_trimmed_grid(input_path, output_path, lam_min, lam_max, overwrite=False):
    """Copy a grid, slicing wavelength-dependent spectra to a contiguous range."""
    input_path = input_path.resolve()
    output_path = output_path.resolve()
    if input_path == output_path:
        raise ValueError("Input and output grid paths must differ")
    if not input_path.is_file():
        raise FileNotFoundError(input_path)
    if not np.isfinite(lam_min) or not np.isfinite(lam_max) or lam_min >= lam_max:
        raise ValueError("Require finite --lam-min < --lam-max")
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Output exists: {output_path}; pass --overwrite to replace it")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = output_path.with_name(f".{output_path.name}.tmp")
    if temporary_path.exists():
        temporary_path.unlink()

    try:
        with h5py.File(input_path, "r") as source:
            if "spectra/wavelength" not in source:
                raise KeyError("Input grid has no spectra/wavelength dataset")
            wavelength = np.asarray(source["spectra/wavelength"][...], dtype=float)
            if wavelength.ndim != 1 or len(wavelength) < 2:
                raise ValueError("spectra/wavelength must be a non-empty 1D array")
            if np.any(~np.isfinite(wavelength)) or np.any(np.diff(wavelength) <= 0):
                raise ValueError("spectra/wavelength must be finite and strictly increasing")

            selected = np.flatnonzero((wavelength >= lam_min) & (wavelength <= lam_max))
            if len(selected) < 2:
                raise ValueError("Requested wavelength interval contains fewer than two samples")
            start = int(selected[0])
            stop = int(selected[-1]) + 1
            retained_wavelength = wavelength[start:stop]

            with h5py.File(temporary_path, "w") as destination:
                copy_attrs(source, destination)
                destination.attrs["trimmed_from"] = input_path.name
                destination.attrs["trim_lam_min_angstrom"] = float(retained_wavelength[0])
                destination.attrs["trim_lam_max_angstrom"] = float(retained_wavelength[-1])

                for name in source:
                    if name != "spectra":
                        source.copy(name, destination)

                source_spectra = source["spectra"]
                destination_spectra = destination.create_group("spectra")
                copy_attrs(source_spectra, destination_spectra)

                for name, dataset in source_spectra.items():
                    destination_name = f"spectra/{name}"
                    if name == "wavelength":
                        copied = destination.create_dataset(
                            destination_name,
                            data=dataset[start:stop],
                            dtype=dataset.dtype,
                        )
                        copy_attrs(dataset, copied)
                        continue

                    if dataset.ndim > 0 and dataset.shape[-1] == len(wavelength):
                        trimmed_shape = dataset.shape[:-1] + (stop - start,)
                        copied = destination.create_dataset(
                            destination_name,
                            shape=trimmed_shape,
                            dtype=dataset.dtype,
                        )
                        copy_attrs(dataset, copied)
                        if dataset.ndim == 1:
                            copied[...] = dataset[start:stop]
                        else:
                            for index in range(dataset.shape[0]):
                                copied[index, ...] = dataset[index, ..., start:stop]
                    else:
                        source.copy(dataset, destination_spectra, name=name)

                destination.flush()

        os.replace(temporary_path, output_path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()

    return retained_wavelength


def main():
    args = parse_args()
    wavelength = copy_trimmed_grid(
        args.input,
        args.output,
        args.lam_min,
        args.lam_max,
        overwrite=args.overwrite,
    )
    spacing = np.diff(wavelength)
    resolving_power = wavelength[:-1] / spacing
    print(f"Wrote {args.output}")
    print(
        f"Retained {len(wavelength)} wavelength samples: "
        f"{wavelength[0]:.3f}-{wavelength[-1]:.3f} Angstrom"
    )
    print(
        f"Median delta-lambda={np.median(spacing):.4g} Angstrom; "
        f"median R={np.median(resolving_power):.1f}"
    )


if __name__ == "__main__":
    main()
