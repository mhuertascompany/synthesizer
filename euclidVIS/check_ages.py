
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

print("Starting PRODUCTION-RESOURCED full spectrum generation...", flush=True)

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
    
    imasses_all = (out_stars["GFM_InitialMass"][mask_form] * 1e10) / h
    metallicities_all = out_stars["GFM_Metallicity"][mask_form]
    coods_all = out_stars["Coordinates"][mask_form] * (scale_factor / h)
    
    subhalo = il.groupcat.loadSingle(directory, snap_number, subhaloID=subhalo_id)
    gal_centre = subhalo['SubhaloPos'] * (scale_factor / h)
    coods_all -= gal_centre
    
    _ages = cosmo.age(1.0 / form_time[mask_form] - 1)
    ages_yr_all = (universe_age - _ages).value * 1e9
    
    num_stars = len(ages_yr_all)
    print(f"Total stars to process: {num_stars}")

    # Load Grid and TRUNCATE wavelength range (900 - 20000 A) - as suggested by devs
    print("Loading and truncating grid (900 - 20000 A)...", flush=True)
    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
    grid.reduce_rest_frame_range(900 * Angstrom, 20000 * Angstrom)
    
    n_lam = len(grid.lam)
    print(f"Number of wavelength bins: {n_lam}")

    # Define Models
    nebular_model = ReprocessedEmission(grid=grid)
    attenuated_model = AttenuatedEmission(grid=grid, dust_curve=Calzetti2000(), apply_to=nebular_model, emitter="stellar")
    
    # ACCUMULATORS
    lnu_int_total = np.zeros(n_lam)
    lnu_att_total = np.zeros(n_lam)
    
    chunk_size = 50000 # Smaller chunks + 240GB RAM should be bulletproof
    print(f"Processing in chunks of {chunk_size}...", flush=True)
    
    for i in range(0, num_stars, chunk_size):
        end = min(i + chunk_size, num_stars)
        print(f"  Stars {i} to {end}...", flush=True)
        
        chunk_stars = Stars(
            initial_masses=imasses_all[i:end] * Msun,
            ages=ages_yr_all[i:end] * yr,
            metallicities=metallicities_all[i:end],
            coordinates=coods_all[i:end] * kpc
        )
        
        # 1. Intrinsic
        spec_int = chunk_stars.get_spectra(nebular_model)
        lnu_int_total += spec_int.lnu.to(erg/s/Hz).value
        
        # 2. Attenuated
        tau_v_chunk = np.full(end - i, 0.33)
        try:
            # Using get_particle_spectra with 8 threads
            dict_att = chunk_stars.get_particle_spectra(emission_model=attenuated_model, tau_v=tau_v_chunk, nthreads=8)
            spec_att_parts = dict_att['attenuated']
            if hasattr(spec_att_parts, 'nsed') and spec_att_parts.nsed > 1:
                lnu_att_total += np.sum(spec_att_parts.lnu.to(erg/s/Hz).value, axis=0)
            else:
                lnu_att_total += spec_att_parts.lnu.to(erg/s/Hz).value
        except Exception as e:
            print(f"    Chunk fallback: {e}")
            spec_att_fallback = spec_int.apply_attenuation(tau_v=0.33, dust_curve=Calzetti2000())
            lnu_att_total += spec_att_fallback.lnu.to(erg/s/Hz).value

    # SAVE TO FITS
    output_name = f"subhalo_{subhalo_id}_full_spectrum.fits"
    lam = grid.lam.to(Angstrom).value
    
    col1 = fits.Column(name='WAVELENGTH', format='D', array=lam, unit='Angstrom')
    col2 = fits.Column(name='LNU_INTRINSIC', format='D', array=lnu_int_total, unit='erg/s/Hz')
    col3 = fits.Column(name='LNU_ATTENUATED', format='D', array=lnu_att_total, unit='erg/s/Hz')
    
    hdu = fits.BinTableHDU.from_columns([col1, col2, col3])
    hdu.header['SUBHALO'] = subhalo_id
    hdu.header['REDSHIFT'] = redshift
    hdu.header['UNIV_AGE'] = universe_age.value
    hdu.header['UNIV_UNT'] = 'Gyr'
    hdu.writeto(output_name, overwrite=True)
    
    print(f"\n--- SUCCESS ---")
    print(f"Full spectrum saved to: {output_name}")
    
    ha_idx = np.argmin(np.abs(lam - 6563.0))
    print(f"H-alpha (6563A) Intrinsic flux: {lnu_int_total[ha_idx]:.2e}")

else:
    print("No stars found.")

print("\nDIAGNOSTIC COMPLETE.", flush=True)
