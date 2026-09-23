# Empirical DESI EDR noise workflow

The full EDR spectrum archive is on Teide, while the TNG50 generation runs on
Vera. Donors are therefore assigned on Teide and only the selected spectra are
transferred to Vera.

## 1. Create the target/subhalo manifest on Vera

Activate the Synthesizer environment and run the inexpensive selection stage:

```bash
cd /u/mhuertas/python/synthesizer/euclidVIS
python3 generate_matched_sample.py \
  --config config_matched.yaml \
  --n_mocks 100 \
  --seed 42 \
  --selection_only \
  --skip_donor_assignment
```

This writes:

```text
/u/mhuertas/data/euclid/tngmatched_euclid_desi/matched_sample.csv
```

Copy that small CSV to Teide.

## 2. Assign donors on Teide

Teide compute nodes do not have internet access, so create the small reusable
environment once on a login node. Store it in the shared project area rather
than the limited home filesystem:

```bash
module purge
module load Miniconda3/23.5.2-0
conda create --yes \
  --prefix /home/mhuertas/iac18_mhuertas_shared/mhuertas/euclid_desi_mocks/envs/edr-noise \
  python=3.10 numpy pandas scipy astropy
```

Then submit the Teide batch job from the `euclidVIS` directory of the same code
revision. The job only reads the prepared environment and writes both output
files alongside the input manifest:

```bash
sbatch teide_edr_donors.sb
```

The defaults assume the input manifest is:

```text
/home/mhuertas/iac18_mhuertas_shared/mhuertas/euclid_desi_mocks/matched_sample.csv
```

An alternative manifest, output directory, and seed can be supplied as the
three positional arguments:

```bash
sbatch teide_edr_donors.sb /path/to/matched_sample.csv /path/to/output 190734905
```

The equivalent direct Python command, for an already prepared compute-node
environment, is:

```bash
python3 edr_empirical_noise.py \
  --targets /path/to/matched_sample.csv \
  --catalog /home/mhuertas/iac18_aasensio_shared/euclid_dr1/catalog/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.0.fits \
  --spectra-dir /home/mhuertas/iac18_aasensio_shared/euclid_dr1/spectra \
  --output /path/to/matched_sample_edr_noise.csv \
  --copy-list /path/to/edr_donor_files.txt \
  --assigned-spectra-dir /u/mhuertas/data/euclid/edr_noise_donors/spectra \
  --seed 190734905
```

The selector keeps DEEP galaxies with valid spectra, `PROGRAM` equal to
`bright` or `dark`, `0 <= z < 1.5`, and `log(M*) > 9.5`. For each mock it
queries the 64 nearest donors in redshift and stellar mass, makes a seeded
random choice weighted by distance, and avoids reusing donors when possible.

The output manifest records the donor TARGETID, program, survey, redshift,
stellar mass, r-band flux, per-arm S/N, and final Vera path. The text file
contains the source Teide paths that need to be transferred.

## 3. Stage the selected spectra on Teide

```bash
mkdir -p /path/to/selected_edr_donors
while IFS= read -r spectrum; do
  cp -p "$spectrum" /path/to/selected_edr_donors/
done < /path/to/edr_donor_files.txt

tar -C /path/to/selected_edr_donors \
  -czf /path/to/selected_edr_donors.tar.gz .
```

Transfer both `matched_sample_edr_noise.csv` and the tar archive to Vera. On
Vera, extract the spectra into:

```text
/u/mhuertas/data/euclid/edr_noise_donors/spectra
```

## 4. Generate the mocks from the exact assigned manifest

If the noiseless `desi_spectrum_<subhalo>_raw.fits` products already exist,
do not regenerate the TNG galaxies or Euclid images. Apply feasiBGS response
and empirical donor noise as a post-processing step instead. First run a
three-spectrum smoke test on Vera:

```bash
sbatch vera_apply_edr_noise.sb \
  /u/mhuertas/data/euclid/edr_noise_donors/matched_sample_edr_noise.csv 3 3
```

The second argument is the number of parallel workers and the third is the
number of manifest rows to process. Inspect the job log and output headers,
then run the complete manifest with nine workers:

```bash
sbatch vera_apply_edr_noise.sb \
  /u/mhuertas/data/euclid/edr_noise_donors/matched_sample_edr_noise.csv 9
```

The raw products are never modified. Before replacing an existing final
`desi_spectrum_<subhalo>.fits`, the job saves it once as
`desi_spectrum_<subhalo>.pre_edr.fits`.

Only use the complete generation job below when a raw spectrum is missing or
the underlying galaxy spectrum itself must be regenerated.

```bash
sbatch vera_matched.sb 100 42 \
  /u/mhuertas/data/euclid/edr_noise_donors/matched_sample_edr_noise.csv
```

feasiBGS is run without stochastic noise so it still supplies the DESI
instrument response and resolution. The selected donor's wavelength-dependent
IVAR and mask are interpolated onto each B/R/Z camera grid, and a fresh seeded
Gaussian realization is added. Output headers contain `NOISEMOD=EDR_IVAR`,
`DONOR=<TARGETID>`, and `DONPROG=<bright|dark>`.
