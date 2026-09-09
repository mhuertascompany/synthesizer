"""Small, optional adapter between Synthesizer spectra and feasiBGS."""

import os
import tempfile
from pathlib import Path

import numpy as np
from astropy.io import fits


C_ANGSTROM_PER_SECOND = 2.99792458e18
FEASIBGS_FLUX_SCALE = 1e-17


def fnu_cgs_to_feasibgs_flambda(wavelength_angstrom, fnu_cgs):
    """Convert F_nu cgs to feasiBGS's 1e-17 F_lambda-per-Angstrom units."""
    wave = np.asarray(wavelength_angstrom, dtype=np.float64)
    fnu = np.asarray(fnu_cgs, dtype=np.float64)
    if wave.ndim != 1 or fnu.ndim != 1 or wave.shape != fnu.shape:
        raise ValueError("wavelength and F_nu must be same-length 1D arrays")
    if np.any(~np.isfinite(wave)) or np.any(wave <= 0):
        raise ValueError("wavelengths must be finite and positive")
    if np.any(~np.isfinite(fnu)):
        raise ValueError("F_nu contains non-finite values")

    # F_lambda[per Angstrom] = F_nu * c[Angstrom/s] / lambda[Angstrom]^2.
    return fnu * C_ANGSTROM_PER_SECOND / wave**2 / FEASIBGS_FLUX_SCALE


def _bright_sky(feasibgs_sky, config):
    """Return the feasiBGS refit-KS moon/twilight sky model."""
    # feasiBGS calls the private compatibility helper
    # ``specsim.config.is_string``. It was removed from newer specsim releases,
    # although the equivalent check is simply isinstance(value, str).
    import specsim.config

    if not hasattr(specsim.config, "is_string"):
        specsim.config.is_string = lambda value: isinstance(value, str)
    sky_wave, sky_brightness = feasibgs_sky.Isky_newKS_twi(
        float(config.get("airmass", 1.1)),
        float(config.get("moon_illumination", 0.7)),
        float(config.get("moon_altitude_deg", 60.0)),
        float(config.get("moon_separation_deg", 80.0)),
        float(config.get("sun_altitude_deg", -30.0)),
        float(config.get("sun_separation_deg", 180.0)),
    )
    # feasiBGS passes these directly to np.interp alongside wave.to_value().
    # Newer Astropy versions reject mixing that unitless array with a Quantity.
    if hasattr(sky_wave, "to_value"):
        sky_wave = sky_wave.to_value("Angstrom")
    if hasattr(sky_brightness, "value"):
        sky_brightness = sky_brightness.value
    return (
        np.asarray(sky_wave, dtype=np.float64),
        np.asarray(sky_brightness, dtype=np.float64),
    )


def _resample_uniform_for_desi(wave, flux, config, desimodel_io):
    """Resample one F_lambda spectrum to the uniform grid desisim requires."""
    step = float(config.get("input_dlambda_angstrom", 0.1))
    if not np.isfinite(step) or step <= 0:
        raise ValueError("desi.input_dlambda_angstrom must be positive")

    params = desimodel_io.load_desiparams()
    desi_min = float(params["ccd"]["b"]["wavemin"])
    desi_max = float(params["ccd"]["z"]["wavemax"])
    if wave[0] > desi_min or wave[-1] < desi_max:
        raise ValueError(
            "Synthesizer spectrum does not cover the full DESI wavelength "
            f"range {desi_min:.1f}-{desi_max:.1f} Angstrom; input covers "
            f"{wave[0]:.1f}-{wave[-1]:.1f} Angstrom"
        )

    # Extend by one grid point because feasiBGS explicitly requires its input
    # to bracket (not merely equal) the DESI camera limits.
    grid_min = np.floor(desi_min / step) * step - step
    grid_max = np.ceil(desi_max / step) * step + step
    uniform_wave = grid_min + step * np.arange(
        int(np.ceil((grid_max - grid_min) / step)) + 1, dtype=np.float64
    )
    uniform_flux = np.interp(uniform_wave, wave, flux)
    return uniform_wave, uniform_flux


def simulate_feasibgs_exposure(
    wavelength_angstrom,
    fnu_cgs,
    output_path,
    config,
    seed,
    metadata=None,
):
    """Simulate and write one noisy DESI B/R/Z exposure with feasiBGS."""
    try:
        from feasibgs import forwardmodel as forwardmodel
        from feasibgs import skymodel as skymodel
        import desimodel.io as desimodel_io
    except ImportError as error:
        raise ImportError(
            "The DESI noise_model is 'feasibgs', but feasiBGS or one of its "
            "DESI dependencies is unavailable. Install feasiBGS and ensure "
            "its repository is on PYTHONPATH before running this job."
        ) from error

    wave = np.asarray(wavelength_angstrom, dtype=np.float64)
    fnu = np.asarray(fnu_cgs, dtype=np.float64)
    order = np.argsort(wave)
    wave = wave[order]
    flux = fnu_cgs_to_feasibgs_flambda(wave, fnu[order])
    wave, unique_indices = np.unique(wave, return_index=True)
    flux = flux[unique_indices]
    wave, flux = _resample_uniform_for_desi(
        wave, flux, config, desimodel_io
    )
    flux = np.clip(flux, 0.0, None)[None, :]

    sky_name = str(config.get("sky_model", "bright")).lower()
    if sky_name == "dark":
        isky = None
    elif sky_name in {"bright", "newks", "new_ks"}:
        isky = _bright_sky(skymodel, config)
    else:
        raise ValueError("desi.sky_model must be 'dark' or 'bright'")

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    simulator = forwardmodel.fakeDESIspec()
    temp_handle, temp_name = tempfile.mkstemp(
        prefix=f".{output_path.stem}.", suffix=".fits", dir=output_path.parent
    )
    os.close(temp_handle)
    os.unlink(temp_name)
    try:
        spectra = simulator.simExposure(
            wave,
            flux,
            airmass=float(config.get("airmass", 1.1)),
            exptime=float(config.get("exposure_time_seconds", 180.0)),
            seeing=float(config.get("seeing_arcsec", 1.1)),
            seed=int(seed),
            skyerr=float(config.get("sky_subtraction_error", 0.0)),
            Isky=isky,
            nonoise=bool(config.get("no_noise", False)),
            dwave_out=float(config.get("output_dlambda_angstrom", 1.0)),
            filename=temp_name,
        )

        # feasiBGS writes standard DESI extensions. Add mock provenance only
        # to the primary header so the DESI data model remains intact.
        with fits.open(temp_name, mode="update") as hdul:
            header = hdul[0].header
            header["FEASIBGS"] = (True, "Exposure simulated with feasiBGS")
            header["SKYMODEL"] = (sky_name.upper(), "feasiBGS sky model")
            header["RNGSEED"] = (int(seed), "Noise random seed")
            header["EXPTIME"] = float(config.get("exposure_time_seconds", 180.0))
            header["AIRMASS"] = float(config.get("airmass", 1.1))
            header["SEEING"] = float(config.get("seeing_arcsec", 1.1))
            for key, value in (metadata or {}).items():
                header[key] = value
            hdul.flush()
        os.replace(temp_name, output_path)
    finally:
        if os.path.exists(temp_name):
            os.unlink(temp_name)
    return spectra
