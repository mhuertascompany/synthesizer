#!/usr/bin/env python3
"""Vera smoke test for the optional feasiBGS DESI backend."""

import argparse
import tempfile
from pathlib import Path

import numpy as np

from desi_feasibgs import (
    C_ANGSTROM_PER_SECOND,
    fnu_cgs_to_feasibgs_flambda,
    simulate_feasibgs_exposure,
)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--imports-only", action="store_true",
        help="Only import feasiBGS and the DESI stack; do not simulate an exposure.",
    )
    args = parser.parse_args()

    from feasibgs import forwardmodel  # noqa: F401
    from feasibgs import skymodel  # noqa: F401
    import desimodel  # noqa: F401
    import desisim  # noqa: F401
    import desispec  # noqa: F401
    import specsim  # noqa: F401
    print("PASS: feasiBGS, desimodel, desisim, desispec, and specsim import.")
    if args.imports_only:
        return

    # A logarithmic grid reproduces Synthesizer's nonuniform wavelength input.
    # The adapter must resample this before passing it to desisim.
    wave = np.geomspace(3500.0, 10000.0, 12000)
    # Construct a flat F_lambda=1e-17 spectrum through the inverse relation.
    fnu = 1e-17 * wave**2 / C_ANGSTROM_PER_SECOND
    converted = fnu_cgs_to_feasibgs_flambda(wave, fnu)
    np.testing.assert_allclose(converted, 1.0, rtol=2e-15)

    config = {
        "exposure_time_seconds": 10.0,
        "airmass": 1.1,
        "seeing_arcsec": 1.1,
        # Exercise the same feasiBGS compatibility path used in production.
        "sky_model": "bright",
        "moon_illumination": 0.7,
        "moon_altitude_deg": 60.0,
        "moon_separation_deg": 80.0,
        "sun_altitude_deg": -30.0,
        "sun_separation_deg": 180.0,
        "input_dlambda_angstrom": 1.0,
        "output_dlambda_angstrom": 1.0,
    }
    with tempfile.TemporaryDirectory(prefix="feasibgs-smoke-") as tmpdir:
        output = Path(tmpdir) / "desi-smoke.fits"
        spectra = simulate_feasibgs_exposure(
            wave, fnu, output, config, seed=42,
            metadata={"OBJECT": "FEASIBGS SMOKE TEST"},
        )
        bands = set(spectra.bands)
        if bands != {"b", "r", "z"}:
            raise RuntimeError(f"Expected DESI b/r/z cameras, got {sorted(bands)}")
        for band in bands:
            if not np.all(np.isfinite(spectra.flux[band])):
                raise RuntimeError(f"Non-finite {band}-camera flux")
            if not np.all(np.isfinite(spectra.ivar[band])):
                raise RuntimeError(f"Non-finite {band}-camera inverse variance")
        if not output.exists():
            raise RuntimeError("feasiBGS did not write the requested FITS file")
    print("PASS: unit conversion and noisy DESI b/r/z exposure simulation.")


if __name__ == "__main__":
    main()
