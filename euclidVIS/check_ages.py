
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

print("Starting definitive H-alpha diagnostic...", flush=True)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths, sim = config['paths'], config['simulation']
subhalo_id = 96768 
directory, snap_number = paths['tng_path'], sim['snap_number']
h = 0.6774 

header = il.groupcat.loadHeader(directory, snap_number)
scale_factor = header["Time"]
universe_age = cosmo.age(1.0 / scale_factor - 1)

star_fields = ["GFM_StellarFormationTime", "Coordinates", "GFM_InitialMass", "GFM_Metallicity"]
out_stars = il.snapshot.loadSubhalo(directory, snap_number, subhalo_id, "stars", fields=star_fields)

if out_stars["count"] > 0:
    form_time = out_stars["GFM_StellarFormationTime"]
    mask_form = form_time > 0.0
    imasses = (out_stars["GFM_InitialMass"][mask_form] * 1e10) / h
    metallicities = out_stars["GFM_Metallicity"][mask_form]
    
    _ages = cosmo.age(1.0 / form_time[mask_form] - 1)
    ages_yr = (universe_age - _ages).value * 1e9
    young_mask = ages_yr < 1e7
    
    print(f"\n--- POPULATION STATISTICS ---")
    print(f"Total stars: {len(ages_yr)}")
    print(f"Young stars (<10 Myr): {np.sum(young_mask)}")
    print(f"Youngest age: {np.min(ages_yr):.2e} yr")
    print(f"Youngest metallicity: {metallicities[young_mask][np.argmin(ages_yr[young_mask])]:.4f}")

    # Load Grid
    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
    print(f"\nGrid: {sim['grid_name']}")
    print(f"Grid reprocessed: {grid.reprocessed}")
    
    # 1. TEST YOUNG STARS ONLY
    print("\n--- YOUNG STARS ONLY SPECTRUM ---")
    galaxy_young = Galaxy()
    galaxy_young.load_stars(
        initial_masses=imasses[young_mask] * Msun, 
        ages=ages_yr[young_mask] * yr, 
        metallicities=metallicities[young_mask]
    )
    
    model = ReprocessedEmission(grid=grid)
    spec_young = galaxy_young.get_spectra(model)
    
    lam = spec_young.lam.to(Angstrom).value
    lnu = spec_young.lnu.value
    
    # Rest-frame H-alpha region
    ha_peak_idx = np.argmin(np.abs(lam - 6563.0))
    ha_cont_idx = np.argmin(np.abs(lam - 6600.0)) # Nearby continuum
    
    peak_flux = lnu[ha_peak_idx]
    cont_flux = lnu[ha_cont_idx]
    ratio = peak_flux / cont_flux
    
    print(f"Young Population H-alpha (6563A) flux: {peak_flux:.2e}")
    print(f"Young Population Continuum (6600A) flux: {cont_flux:.2e}")
    print(f"Young Population Peak/Continuum Ratio: {ratio:.1f}")
    
    if ratio > 2.0:
        print("SUCCESS: H-alpha EMISSION detected in young population!")
    else:
        print("FAILURE: H-alpha ABSORPTION or no feature detected in young population.")

    # 2. TEST FULL GALAXY
    print("\n--- FULL GALAXY SPECTRUM ---")
    galaxy_all = Galaxy()
    galaxy_all.load_stars(
        initial_masses=imasses * Msun, 
        ages=ages_yr * yr, 
        metallicities=metallicities
    )
    spec_all = galaxy_all.get_spectra(model)
    lnu_all = spec_all.lnu.value
    
    peak_all = lnu_all[ha_peak_idx]
    cont_all = lnu_all[ha_cont_idx]
    ratio_all = peak_all / cont_all
    
    print(f"Full Galaxy Peak/Continuum Ratio: {ratio_all:.2f}")

else:
    print("No stars found.")

print("\nDIAGNOSTIC COMPLETE.")
