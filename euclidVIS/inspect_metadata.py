
from astropy.io import fits
import os
import glob
import pandas as pd

# 1. Check Catalog
cat_path = "/u/mhuertas/data/euclid/tngmocks/sn99/catalog.csv"
if os.path.exists(cat_path):
    print("--- Catalog Snippet ---")
    df = pd.read_csv(cat_path)
    print(df.head())
    print(f"Columns: {df.columns.tolist()}")
else:
    print("Catalog not found.")

# 2. Check FITS Header
euclid_dir = "/u/mhuertas/data/euclid/tngmocks/sn99/Euclid/"
fits_files = glob.glob(os.path.join(euclid_dir, "*.fits"))
if fits_files:
    test_file = fits_files[0]
    print(f"\n--- Header Snippet for {os.path.basename(test_file)} ---")
    with fits.open(test_file) as hdul:
        print(hdul[0].header)
else:
    print("\nNo FITS files found in Euclid directory.")
