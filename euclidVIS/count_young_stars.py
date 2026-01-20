import numpy as np
import illustris_python as il
from unyt import yr, Msun, kpc
import yaml
import os

# Load config to get paths
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths = config['paths']
snap = config['simulation']['snap_number']
subhalo_id = 96768

print(f"Checking Subhalo {subhalo_id} for young stars...")

# Load star formation times
star_fields = ["GFM_StellarFormationTime"]
out_stars = il.snapshot.loadSubhalo(paths['tng_path'], snap, subhalo_id, "stars", fields=star_fields)

if out_stars["count"] > 0:
    form_time = out_stars["GFM_StellarFormationTime"]
    # Filter out wind particles (negative values)
    mask = form_time > 0.0
    form_time = form_time[mask]
    
    # Header for scale factor
    header = il.groupcat.loadHeader(paths['tng_path'], snap)
    scale_factor = header["Time"]
    redshift = header["Redshift"]
    
    # Calculate ages roughly (scale factor vs formation time)
    # A simple way to check "young" in TNG is to compare with the current scale factor
    # Young stars have form_time close to scale_factor
    # If scale_factor = 1.0 (z=0), young stars ( < 10 Myr) have form_time > 0.999
    
    num_total = len(form_time)
    num_young = np.sum(form_time > (scale_factor * 0.999)) # Very rough 10Myr estimate at z=0
    
    print(f"Total star particles: {num_total}")
    print(f"Rough count of young particles (top 0.1% of cosmic time): {num_young}")
    
    # Find the indices of the first 10 young stars to see where they sit in the array
    young_indices = np.where(form_time > (scale_factor * 0.999))[0]
    if len(young_indices) > 0:
        print(f"First 10 young star indices: {young_indices[:10]}")
    else:
        print("No very young stars found with this rough cut.")
else:
    print("No stars found.")
