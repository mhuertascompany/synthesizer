
import numpy as np
import yaml
import os
from unyt import Myr, yr, kpc, Msun, Angstrom
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.emission_models import ReprocessedEmission
import illustris_python as il
from astropy.cosmology import Planck15 as cosmo

# This script is to be run on VERA to diagnose the 'extremely old' theory
# and check grid axes, and test integrated spectra as suggested by devs.

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths = config['paths']
sim = config['simulation']
subhalo_id = 96768 

# Re-implement minimal loader
directory = paths['tng_path']
snap_number = sim['snap_number']
h = 0.6774 

header = il.groupcat.loadHeader(directory, snap_number)
scale_factor = header["Time"]
universe_age = cosmo.age(1.0 / scale_factor - 1)

print(f"--- HEADER DIAGNOSTICS ---")
print(f"Header: z={header['Redshift']:.3f}, a={scale_factor:.3f}")
print(f"Universe age at snapshot: {universe_age:.3f}")

star_fields = ["GFM_StellarFormationTime", "GFM_InitialMass", "GFM_Metallicity", "Coordinates"]
out_stars = il.snapshot.loadSubhalo(directory, snap_number, subhalo_id, "stars", fields=star_fields)

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
    print(f"Ages (yr) range: {np.min(ages_yr):.1e} to {np.max(ages_yr):.1e}")
    young_mask = ages_yr < 1e7
    num_young = np.sum(young_mask)
    print(f"Number of stars < 10 Myr: {num_young}")
    
    # Create a Galaxy object to test integrated spectra
    galaxy = Galaxy()
    galaxy.load_stars(
        initial_masses=imasses * Msun, 
        ages=ages_yr * yr, 
        metallicities=metallicities, 
        coordinates=coods * kpc
    )

    print("\n--- GRID DIAGNOSTICS ---")
    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
    print(f"Grid axes: {grid.axes}")
    print(f"Grid extract axes: {grid._extract_axes}")
    print(f"Reprocessed: {grid.reprocessed}")

    print("\n--- INTEGRATED SPECTRUM TEST ---")
    # 1. Test standard ReprocessedEmission (WITHOUT manual U fix)
    model_std = ReprocessedEmission(grid=grid)
    spec_std = galaxy.get_spectra(model_std)
    
    # H-alpha wavelength is ~6563A
    try:
        ha_idx = np.argmin(np.abs(spec_std.lam.to(Angstrom).value - 6563.0))
        ha_flux = spec_std.lnu[ha_idx].value
        print(f"Standard ReprocessedEmission H-alpha flux: {ha_flux:.3e}")
    except Exception as e:
        print(f"Could not calculate H-alpha for standard model: {e}")

    # 2. Test with explicit U fix (if standard is zero)
    model_fixed = ReprocessedEmission(grid=grid, log10_ionisation_parameter=-2.0, ionisation_parameter=0.01)
    spec_fixed = galaxy.get_spectra(model_fixed)
    ha_flux_fixed = spec_fixed.lnu[ha_idx].value
    print(f"Fixed ReprocessedEmission (-2.0) H-alpha flux: {ha_flux_fixed:.3e}")

else:
    print("No stars found for this subhalo.")
