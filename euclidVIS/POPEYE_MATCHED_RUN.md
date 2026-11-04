# Popeye Euclid--DESI matched production run

This workflow draws a fixed sample from the Euclid-DEEP objects in the matched
DESI--Euclid catalog, with `z < 1.5` and `log10(Mstar/Msun) > 9.5`. TNG50
subhalos are matched in redshift and stellar mass. Products use `kappa=0.0795`,
the Calzetti attenuation curve, velocity-shifted particle spectra, a 1.5-arcsec
DESI fiber, and Euclid VIS plus NISP Y/J/H imaging.

The expensive production stage writes noiseless, pre-PSF Euclid images and
high-resolution intrinsic DESI-fiber spectra. PSFs, detector/background noise,
and empirical DESI DR1 noise are intentionally deferred to post-processing.

## Required inputs

The TNG50 public release is expected at:

```text
/mnt/sdceph/users/sgenel/Illustris_IllustrisTNG_public_data_release/L35n2160TNG/output
```

Two untracked data assets must also be staged on Popeye (the grid is public;
the reference catalog is private):

```text
/mnt/sdceph/users/$USER/synthesizer_grids/
  bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res_1200-22000A.hdf5

/mnt/sdceph/users/$USER/euclid_desi_mocks/reference/
  catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits
```

Create those directories with:

```bash
mkdir -p /mnt/sdceph/users/$USER/synthesizer_grids
mkdir -p /mnt/sdceph/users/$USER/euclid_desi_mocks/reference
```

The VIS and NISP Y/J/H transmission files are tracked in this repository and
do not need to be copied separately.

### Preparing the high-resolution grid

The DESI production workflow uses official Syndex release 241, whose native
optical wavelength grid has approximately constant `R=5450`. The complete grid
is 8.3 GB and spans wavelengths irrelevant to this project. Download it once,
verify the published SHA-256, then make a physical 1200--22000 Angstrom
rest-frame subset. The long-wavelength limit is required by NISP-H:

```bash
cd /mnt/sdceph/users/$USER/synthesizer_grids
curl -L --fail --retry 20 --retry-all-errors --continue-at - \
  -o bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res.hdf5.part \
  https://synthesizer-data-api.universe-engine.workers.dev/v1/releases/241/download
echo "fca1244e6dfb1d25f196b18778159a01620769782d53f1d622ccaec38218ca6b  bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res.hdf5.part" \
  | sha256sum --check
mv bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res.hdf5.part \
  bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res.hdf5

python3 /mnt/home/$USER/python/synthesizer/euclidVIS/trim_synthesizer_grid.py \
  bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res.hdf5 \
  bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res_1200-22000A.hdf5 \
  --lam-min 1200 --lam-max 22000
```

Do not replace this with an interpolation of the older `R=300` grid: a finer
output array cannot recover stellar or nebular spectral structure that was not
present in the source grid.

## Python environment

Activate the Popeye environment containing the installed Synthesizer package
and its compiled extensions. It must import:

```bash
python3 -c 'import astropy, h5py, illustris_python, joblib, numpy, pandas, scipy, unyt, yaml; import synthesizer; print("environment OK")'
```

If the correct executable is not the active `python3`, set it explicitly:

```bash
export POPEYE_PYTHON=/path/to/environment/bin/python
```

If `illustris_python` is a source checkout rather than installed, set:

```bash
export POPEYE_ILLUSTRIS_PYTHON_ROOT=/path/to/illustris_python
```

## Smoke test

Use a separate output directory for a 16-object test:

```bash
cd /path/to/synthesizer/euclidVIS
export POPEYE_OUTPUT_ROOT=/mnt/sdceph/users/$USER/euclid_desi_mocks/smoke16_kappa0p0795
./popeye_submit_matched.sh 16 42 4
```

This submits one serial selection job followed by a four-task production array.

## Production run

After the smoke test succeeds:

```bash
unset POPEYE_OUTPUT_ROOT
./popeye_submit_matched.sh 1000 42 4
```

The default output is:

```text
/mnt/sdceph/users/$USER/euclid_desi_mocks/tngmatched_kappa0p0795
```

The 1,000 objects are split into 250 four-galaxy array tasks. At most four
tasks run simultaneously, using four full `gen` nodes and staying within the
documented 4-node/256-core general-partition limit.

For every selected galaxy, the intrinsic stage writes:

```text
sn<snapshot>/Euclid_noiseless/euclid_vis_<subhalo>_raw.fits
sn<snapshot>/Euclid_noiseless/euclid_nisp_y_<subhalo>_raw.fits
sn<snapshot>/Euclid_noiseless/euclid_nisp_j_<subhalo>_raw.fits
sn<snapshot>/Euclid_noiseless/euclid_nisp_h_<subhalo>_raw.fits
sn<snapshot>/DESI/desi_spectrum_<subhalo>_raw.fits
```

All four images are in microJy per pixel (`BUNIT=uJy`) and have
`IMTYPE=PRE_PSF`. VIS uses 0.1 arcsec/pixel; NISP uses 0.3 arcsec/pixel. The
spectrum is an observed-frame, 1.5-arcsec fiber `F_nu` spectrum. This workflow
does not generate NISP slitless/grism spectra; that requires a separate
dispersion/PSF/contamination forward model.

Monitor the IDs printed by the submitter with `squeue`, or inspect:

```text
OUTPUT_ROOT/logs/select.JOBID.{out,err}
OUTPUT_ROOT/logs/mocks.ARRAY_JOBID_TASKID.{out,err}
```

## Restarting

The master selection and random orientations are deterministic for a fixed
sample size and seed. Completed four-band Euclid/raw-DESI sets are skipped. Re-running
the same submit command is therefore safe and regenerates only incomplete
objects. Each array task writes its own manifest under `run_manifests/`; no
cross-node process appends to a shared catalog.

## DESI instrument and noise post-processing

The intrinsic spectra are intentionally preserved as
`desi_spectrum_<subhalo>_raw.fits`. A separate environment contains feasiBGS
and the DESI simulation stack:

```text
/mnt/home/$USER/.venvs/desi-noise
/mnt/home/$USER/python/feasiBGS
/mnt/sdceph/users/$USER/desimodel/data
```

Run the synthetic B/R/Z exposure smoke test before submitting real spectra:

```bash
module load python/3.11.11
export DESIMODEL=/mnt/sdceph/users/$USER/desimodel
export PYTHONPATH=/mnt/home/$USER/python/feasiBGS:$PWD
/mnt/home/$USER/.venvs/desi-noise/bin/python check_feasibgs.py
```

Then test one existing high-resolution mock. The fourth argument limits the
post-processing to the first manifest row:

```bash
./popeye_submit_desi_instrument.sh \
  /path/to/OUTPUT_ROOT/matched_sample.csv \
  /path/to/OUTPUT_ROOT \
  1 1
```

The default model is one 180-second bright-time exposure with airmass 1.1 and
1.1-arcsec seeing. It applies DESI throughput, camera-dependent line-spread
functions, sky/source/detector noise, and writes standard B/R/Z `FLUX`, `IVAR`,
`MASK`, and `RESOLUTION` extensions. The final primary header contains
`FEASIBGS=True` and `NOISEMOD=FEASIBGS`. The raw spectrum is never changed.

After inspecting this physical-noise baseline, the DR1-matched alternative
is to enrich `matched_sample.csv` with mock fiber flux and the fiber SFR
diagnostics stored in each raw FITS header. Assign DR1 donors using redshift,
mass, fiber flux, and fiber sSFR before running
`apply_edr_noise_to_existing_mocks.py`. That later stage keeps the DESI response
but replaces the simulated variance with the matched DR1 coadd IVAR and mask.
The older `edr_*` filenames are retained only for compatibility with manifests
already generated on Teide; new products record `NOISEMOD=DR1_IVAR` and
`DONREL=DR1` in their primary headers.
