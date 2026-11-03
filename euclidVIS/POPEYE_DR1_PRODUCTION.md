# Popeye production run with DESI DR1-matched noise

Production is split into an expensive, restart-safe intrinsic stage on Popeye
and an inexpensive DR1 post-processing stage. Do **not** run the 180-second
physical-noise diagnostic over the production sample; DR1 donor IVAR and masks
provide the target coadd depth.

Use this production root throughout:

```bash
export PROD=/mnt/sdceph/users/mhuertas/euclid_desi_mocks/tngmatched_kappa0p0795
```

## 0. Required catalog

Popeye must contain the full private v1.1 reference catalog:

```text
/mnt/sdceph/users/mhuertas/euclid_desi_mocks/reference/
catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits
```

The small `popeye_smoke_*_mapped.fits` files are test fixtures and must not be
used for production selection.

## 1. Generate intrinsic Euclid and DESI products on Popeye

```bash
cd /mnt/home/mhuertas/python/synthesizer/euclidVIS
module purge
module load python/3.11.11
export POPEYE_PYTHON=/mnt/home/mhuertas/.venvs/synth/bin/python
export POPEYE_ILLUSTRIS_PYTHON_ROOT=/mnt/home/mhuertas/python/illustris_python
export POPEYE_OUTPUT_ROOT="$PROD"
./popeye_submit_matched.sh 1000 42 4
```

This writes `matched_sample.csv`, Euclid images, and high-resolution
`desi_spectrum_<subhalo>_raw.fits` products. It does not add DESI noise. The
printed Slurm array is limited to four simultaneous full nodes.

After the array completes, verify the raw count:

```bash
find "$PROD" -path '*/DESI/desi_spectrum_*_raw.fits' | wc -l
```

The result must equal the row count of `matched_sample.csv` before continuing.

## 2. Measure properties used for donor matching

```bash
./popeye_submit_mock_properties.sh "$PROD"
```

This produces:

```text
$PROD/matched_sample_mock_properties.csv
```

It adds mock fiber r flux and TNG sSFR without changing any FITS products.

## 3. Assign DR1 donors on Teide

Transfer `matched_sample_mock_properties.csv` to the Teide working directory,
then run from the Teide checkout of `euclidVIS`:

```bash
sbatch teide_dr1_donors.sb \
  /home/mhuertas/iac18_mhuertas_shared/mhuertas/euclid_desi_mocks/matched_sample_mock_properties.csv \
  /home/mhuertas/iac18_mhuertas_shared/mhuertas/euclid_desi_mocks \
  190734905
```

The job matches donors in redshift, stellar mass, fiber flux, and sSFR. It
creates:

```text
matched_sample_dr1_noise_flux_sfr.csv
selected_dr1_donors.tar.gz
```

Transfer both files to Popeye. The donor archive must be extracted into:

```bash
mkdir -p /mnt/sdceph/users/mhuertas/desi_dr1_noise_donors/spectra
tar -xzf selected_dr1_donors.tar.gz \
  -C /mnt/sdceph/users/mhuertas/desi_dr1_noise_donors/spectra
cp matched_sample_dr1_noise_flux_sfr.csv "$PROD/"
```

If `scp` fails because the remote SSH server has no SFTP subsystem, use
`rsync -av --progress -e ssh SOURCE DESTINATION` for each transfer.

## 4. Apply the DR1 response/noise products on Popeye

First process three rows:

```bash
./popeye_submit_dr1_noise.sh \
  "$PROD/matched_sample_dr1_noise_flux_sfr.csv" \
  "$PROD" 3 3
```

Verify that the three final files contain `FEASIBGS=True`,
`NOISEMOD=DR1_IVAR`, `DONREL=DR1`, and the expected `DONOR` TARGETID. Then run
the complete manifest; the first three completed files are skipped:

```bash
./popeye_submit_dr1_noise.sh \
  "$PROD/matched_sample_dr1_noise_flux_sfr.csv" \
  "$PROD" 8
```

Each final `desi_spectrum_<subhalo>.fits` has standard B/R/Z `FLUX`, `IVAR`,
`MASK`, and `RESOLUTION` extensions. Intrinsic `*_raw.fits` files remain
unchanged, so the DR1 post-processing can be repeated without regenerating
TNG50 galaxies or Euclid images.

Set `POPEYE_DR1_OVERWRITE=1` only when intentionally replacing existing final
spectra; the job saves each old file once as `*.pre_dr1.fits`.
