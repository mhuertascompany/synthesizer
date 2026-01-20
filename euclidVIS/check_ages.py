
import numpy as np
import yaml
import os
import sys
from unyt import Myr, yr, kpc, Msun, Angstrom
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.emission_models import ReprocessedEmission
from astropy.cosmology import Planck15 as cosmo

try:
    import illustris_python as il
except ImportError:
    print("ERROR: illustris_python not found. Check environment.")
    sys.exit(1)

print("Starting spatial distribution diagnostic...", flush=True)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths, sim, obs = config['paths'], config['simulation'], config['observation']
subhalo_id = 96768 
directory, snap_number = paths['tng_path'], sim['snap_number']
h = 0.6774 

header = il.groupcat.loadHeader(directory, snap_number)
scale_factor, redshift = header["Time"], header["Redshift"]
universe_age = cosmo.age(1.0 / scale_factor - 1)

# Center of subhalo from catalog
subhalo = il.groupcat.loadSingle(directory, snap_number, subhalo_id=subhalo_id)
gal_centre = subhalo['SubhaloPos'] * (scale_factor / h)

star_fields = ["GFM_StellarFormationTime", "Coordinates"]
out_stars = il.snapshot.loadSubhalo(directory, snap_number, subhalo_id, "stars", fields=star_fields)

if out_stars["count"] > 0:
    form_time = out_stars["GFM_StellarFormationTime"]
    coods = out_stars["Coordinates"] * (scale_factor / h)
    
    # Center coordinates
    coods -= gal_centre
    
    mask_form = form_time > 0.0
    _ages = cosmo.age(1.0 / form_time[mask_form] - 1)
    ages_yr = (universe_age - _ages).value * 1e9
    young_mask = ages_yr < 1e7
    
    # Coordinates of young stars
    young_coods = coods[mask_form][young_mask]
    radii = np.sqrt(np.sum(young_coods**2, axis=1))
    
    print(f"\n--- SPATIAL DIAGNOSTICS for Subhalo {subhalo_id} ---")
    print(f"Total young stars (<10 Myr): {len(radii)}")
    print(f"Radii of young stars (kpc): {np.sort(radii)}")
    
    fov_kpc = obs['fov_kpc']
    fov_limit = fov_kpc / 2.0
    print(f"FOV limit (radius): {fov_limit} kpc")
    
    in_fov = radii < fov_limit
    print(f"Young stars in FOV: {np.sum(in_fov)}")
    
    if np.sum(in_fov) == 0 and len(radii) > 0:
        print(f"WARNING: All {len(radii)} young stars are OUTSIDE the {fov_kpc}kpc FOV!")
        print(f"Minimum young star radius: {np.min(radii):.2f} kpc")

else:
    print("No stars found.")
