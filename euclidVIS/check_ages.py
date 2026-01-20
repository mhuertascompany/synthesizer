
import numpy as np
import yaml
import os
import sys
from unyt import Myr, yr, kpc, Msun, Angstrom, erg, s, Hz
from synthesizer.grid import Grid
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.emission_models import ReprocessedEmission, AttenuatedEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from astropy.cosmology import Planck15 as cosmo
from astropy.io import fits

try:
    import illustris_python as il
except ImportError:
    print("ERROR: illustris_python not found. Check environment.")
    sys.exit(1)

print("Starting full spectrum generation (Unmasked/Unrotated to FITS)...", flush=True)

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths, sim = config['paths'], config['simulation']
subhalo_id = 96768 
directory, snap_number = paths['tng_path'], sim['snap_number']

try:
    header = il.groupcat.loadHeader(directory, snap_number)
    scale_factor = header["Time"]
    h = header["HubbleParam"]
    redshift = header["Redshift"]
    universe_age = cosmo.age(1.0 / scale_factor - 1)
except Exception as e:
    print(f"ERROR reading TNG header: {e}")
    sys.exit(1)

try:
    print(f"Loading subhalo {subhalo_id} particles...", flush=True)
    star_fields = ["GFM_StellarFormationTime", "GFM_InitialMass", "GFM_Metallicity", "Coordinates"]
    out_stars = il.snapshot.loadSubhalo(directory, snap_number, subhalo_id, "stars", fields=star_fields)
except Exception as e:
    print(f"ERROR loading data: {e}")
    sys.exit(1)

if out_stars["count"] > 0:
    form_time = out_stars["GFM_StellarFormationTime"]
    mask_form = form_time > 0.0
    
    form_time = form_time[mask_form]
    imasses = (out_stars["GFM_InitialMass"][mask_form] * 1e10) / h
    metallicities = out_stars["GFM_Metallicity"][mask_form]
    coods = out_stars["Coordinates"][mask_form] * (scale_factor / h)
    
    subhalo = il.groupcat.loadSingle(directory, snap_number, subhaloID=subhalo_id)
    gal_centre = subhalo['SubhaloPos'] * (scale_factor / h)
    coods_centered = coods - gal_centre
    
    _ages = cosmo.age(1.0 / form_time - 1)
    ages_yr = (universe_age - _ages).value * 1e9
    
    print("Generating integrated spectra...", flush=True)
    galaxy = Galaxy()
    galaxy.load_stars(
        initial_masses=imasses * Msun, 
        ages=ages_yr * yr, 
        metallicities=metallicities, 
        coordinates=coods_centered * kpc
    )

    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
    nebular_model = ReprocessedEmission(grid=grid)
    attenuated_model = AttenuatedEmission(grid=grid, dust_curve=Calzetti2000(), apply_to=nebular_model, emitter="stellar")
    
    spec_intrinsic = galaxy.get_spectra(nebular_model)
    tau_v_const = np.full(len(ages_yr), 0.5)
    spec_attenuated = galaxy.get_spectra(attenuated_model, tau_v=tau_v_const)
    
    # SAVE TO FITS
    output_name = f"subhalo_{subhalo_id}_full_spectrum.fits"
    
    lam = spec_intrinsic.lam.to(Angstrom).value
    lnu_int = spec_intrinsic.lnu.to(erg/s/Hz).value
    lnu_att = spec_attenuated.lnu.to(erg/s/Hz).value
    
    col1 = fits.Column(name='WAVELENGTH', format='D', array=lam, unit='Angstrom')
    col2 = fits.Column(name='LNU_INTRINSIC', format='D', array=lnu_int, unit='erg/s/Hz')
    col3 = fits.Column(name='LNU_ATTENUATED', format='D', array=lnu_att, unit='erg/s/Hz')
    
    hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
    hdu.header['SUBHALO'] = subhalo_id
    hdu.header['REDSHIFT'] = redshift
    hdu.header['UNIV_AGE'] = universe_age.value
    hdu.header['UNIV_UNT'] = 'Gyr'
    hdu.writeto(output_name, overwrite=True)
    
    print(f"\n--- SUCCESS ---")
    print(f"Full spectrum saved to: {output_name}")
    
    ha_idx = np.argmin(np.abs(lam - 6563.0))
    hb_idx = np.argmin(np.abs(lam - 4861.0))
    print(f"H-alpha (6563A) Intrinsic flux: {lnu_int[ha_idx]:.2e}")
    print(f"Intrinsic Ha/Hb ratio: {lnu_int[ha_idx] / lnu_int[hb_idx]:.2f}")

else:
    print("No stars found.")

print("\nDIAGNOSTIC COMPLETE.", flush=True)
