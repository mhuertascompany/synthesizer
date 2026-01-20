
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

# VERBOSE DIAGNOSTIC SCRIPT
print("Starting check_ages.py diagnostic...", flush=True)

try:
    with open('config.yaml', 'r') as f:
        config = yaml.safe_load(f)
except Exception as e:
    print(f"ERROR loading config: {e}")
    sys.exit(1)

paths = config['paths']
sim = config['simulation']
subhalo_id = 96768 

# Re-implement minimal loader
directory = paths['tng_path']
snap_number = sim['snap_number']
h = 0.6774 

try:
    print(f"Connecting to TNG catalog at {directory}...", flush=True)
    header = il.groupcat.loadHeader(directory, snap_number)
    scale_factor = header["Time"]
    universe_age = cosmo.age(1.0 / scale_factor - 1)
except Exception as e:
    print(f"ERROR reading TNG header: {e}")
    sys.exit(1)

print(f"\n--- HEADER DIAGNOSTICS ---")
print(f"Header: z={header['Redshift']:.3f}, a={scale_factor:.3f}")
print(f"Universe age at snapshot: {universe_age:.3f}")

try:
    print(f"Loading subhalo {subhalo_id} particles...", flush=True)
    star_fields = ["GFM_StellarFormationTime", "GFM_InitialMass", "GFM_Metallicity", "Coordinates"]
    out_stars = il.snapshot.loadSubhalo(directory, snap_number, subhalo_id, "stars", fields=star_fields)
except Exception as e:
    print(f"ERROR loading particles: {e}")
    sys.exit(1)

if out_stars["count"] > 0:
    print(f"\n--- AGE DIAGNOSTICS ---")
    form_time = out_stars["GFM_StellarFormationTime"]
    mask = form_time > 0.0
    form_time = form_time[mask]
    imasses = (out_stars["GFM_InitialMass"][mask] * 1e10) / h
    metallicities = out_stars["GFM_Metallicity"][mask]
    coods = out_stars["Coordinates"][mask] * (scale_factor / h)
    
    _ages = cosmo.age(1.0 / form_time - 1)
    ages_yr = (universe_age - _ages).value * 1e9
    
    print(f"Total stars: {len(ages_yr)}")
    print(f"Ages (yr) range: {np.min(ages_yr):.2e} to {np.max(ages_yr):.2e}")
    young_mask = ages_yr < 1e7
    num_young = np.sum(young_mask)
    print(f"Number of stars < 10 Myr (potential emitters): {num_young}")
    
    # Create a Galaxy object to test integrated spectra
    print("\nInitializing Galaxy object...", flush=True)
    galaxy = Galaxy()
    # Add fesc_ly_alpha to the galaxy directly to see if get_param finds it
    galaxy.fesc_ly_alpha = 1.0
    galaxy.fesc = 0.0
    
    galaxy.load_stars(
        initial_masses=imasses * Msun, 
        ages=ages_yr * yr, 
        metallicities=metallicities, 
        coordinates=coods * kpc
    )

    print("\n--- GRID DIAGNOSTICS ---")
    try:
        grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
        print(f"Grid name: {sim['grid_name']}")
        print(f"Grid axes: {grid.axes}")
        print(f"Grid extract axes: {grid._extract_axes}")
        print(f"Grid reprocessed: {grid.reprocessed}")
    except Exception as e:
        print(f"ERROR loading grid: {e}")
        sys.exit(1)

    print("\n--- INTEGRATED SPECTRUM TEST ---")
    # 1. Test standard ReprocessedEmission
    print("Running Model 1: Standard ReprocessedEmission...", flush=True)
    try:
        model_std = ReprocessedEmission(grid=grid)
        spec_std = galaxy.get_spectra(model_std)
        
        # H-alpha wavelength is ~6563A
        ha_idx = np.argmin(np.abs(spec_std.lam.to(Angstrom).value - 6563.0))
        ha_flux = spec_std.lnu[ha_idx].value
        print(f"Result 1: H-alpha flux = {ha_flux:.3e}")
        
        # Check for absorption (if flux is negative or extremely low)
        # Note: lnu is usually positive. 0.0 means missing emission.
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"ERROR calculating standard spectrum: {e}")

    # 2. Test with explicit U fix
    print("\nRunning Model 2: ReprocessedEmission with fixed U=-2.0...", flush=True)
    try:
        model_fixed = ReprocessedEmission(grid=grid, log10_ionisation_parameter=-2.0, ionisation_parameter=0.01)
        spec_fixed = galaxy.get_spectra(model_fixed)
        ha_flux_fixed = spec_fixed.lnu[ha_idx].value
        print(f"Result 2: H-alpha flux = {ha_flux_fixed:.3e}")
    except Exception as e:
        print(f"ERROR calculating fixed spectrum: {e}")

else:
    print("No stars found for this subhalo.")

print("\nDIAGNOSTIC COMPLETE.", flush=True)
