
import numpy as np
import yaml
import os
from unyt import Myr, yr, kpc, Msun
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
import illustris_python as il
from astropy.cosmology import Planck15 as cosmo

# This script is to be run on VERA to diagnose the 'extremely old' theory
# and check grid axes.

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths = config['paths']
sim = config['simulation']
subhalo_id = 96768 # The one we are debugging

# Re-implement minimal loader to check ages
directory = paths['tng_path']
snap_number = sim['snap_number']
h = 0.6774 # TNG50 h

header = il.groupcat.loadHeader(directory, snap_number)
scale_factor = header["Time"]
universe_age = cosmo.age(1.0 / scale_factor - 1)

print(f"Header: z={header['Redshift']:.3f}, a={scale_factor:.3f}")
print(f"Universe age at snapshot: {universe_age:.3f}")

star_fields = ["GFM_StellarFormationTime", "Masses"]
out_stars = il.snapshot.loadSubhalo(directory, snap_number, subhalo_id, "stars", fields=star_fields)

if out_stars["count"] > 0:
    form_time = out_stars["GFM_StellarFormationTime"]
    mask = form_time > 0.0
    form_time = form_time[mask]
    
    _ages = cosmo.age(1.0 / form_time - 1)
    ages_yr = (universe_age - _ages).value * 1e9
    
    print(f"Total stars: {len(ages_yr)}")
    print(f"Ages (yr) range: {np.min(ages_yr):.1e} to {np.max(ages_yr):.1e}")
    young_mask = ages_yr < 1e7
    num_young = np.sum(young_mask)
    print(f"Number of stars < 10 Myr: {num_young}")
    
    if num_young > 0:
        print(f"Sample of young ages (yr): {ages_yr[young_mask][:5]}")
else:
    print("No stars found for this subhalo.")

# Check Grid
print("\nChecking Grid...")
grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
print(f"Grid name: {sim['grid_name']}")
print(f"Grid axes: {grid.axes}")
print(f"Grid extract axes: {grid._extract_axes}")
print(f"Reprocessed: {grid.reprocessed}")
