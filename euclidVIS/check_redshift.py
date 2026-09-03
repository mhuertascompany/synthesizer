
import os
import numpy as np
from astropy.io import fits
import glob
import yaml

def check_redshift():
    # Load config to get paths
    with open("config.yaml", 'r') as f:
        config = yaml.safe_load(f)

    snap = config['simulation']['snap_number']
    base_dir = os.path.join(config['paths']['output_path'], f"sn{snap}")
    desi_subdir = config['paths'].get('desi_subdir', 'DESI')
    desi_dir = os.path.join(base_dir, desi_subdir)

    print(f"Checking directory: {desi_dir}")

    # Find a raw file
    files = glob.glob(os.path.join(desi_dir, "*_raw.fits"))
    if not files:
        print("No raw files found!")
        return

    # Check first 3 files
    for fpath in files[:3]:
        print(f"\n--- Checking {os.path.basename(fpath)} ---")
        with fits.open(fpath) as hdul:
            # Raw header might be in Primary or Table
            # BinTableHDU is usually extension 1
            idx = 1 if len(hdul) > 1 else 0
            hdr = hdul[idx].header
            data = hdul[idx].data

            # Note: Raw fits might not have the full header if I didn't add it?
            # In generate_euclid_vis.py:
            # col_raw = ...
            # fits.BinTableHDU.from_columns(...).writeto(..., overwrite=True)
            # I did NOT add header keywords to _raw.fits!
            # I added them to the processed .fits.

            # Let's check the processed file header for Z
            processed_path = fpath.replace("_raw.fits", ".fits")
            z_header = -1.0
            if os.path.exists(processed_path):
                 with fits.open(processed_path) as h2:
                     z_header = h2[1].header.get('REDSHIFT', -99.0)
                     print(f"Header REDSHIFT (from processed file): {z_header:.4f}")

            # Check wavelength array
            if 'wavelength' in data.names:
                wav = data['wavelength']
                print(f"Wavelength range: {wav.min():.1f} - {wav.max():.1f} A")
                print(f"First 5 wavelengths: {wav[:5]}")

                # Check for shift assumption
                # Known line: H-alpha 6562.8
                # In raw spectrum, is there a feature? hard to tell from numbers.

                # Heuristic:
                # Typical FSPS starts at ~91A or similar.
                # If z=0, min ~ 91.
                # If z=0.5, min ~ 136.

                if z_header > 0:
                    expected_min = 91 * (1 + z_header) # Roughly
                    print(f"If starting at 91A rest, expected min ~ {expected_min:.1f}")
            else:
                print("No 'wavelength' column found.")

if __name__ == "__main__":
    check_redshift()
