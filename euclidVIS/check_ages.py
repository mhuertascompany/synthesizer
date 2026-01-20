
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

print("Starting FINAL CORRECTED diagnostic spectrum generation...", flush=True)

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

z_obs = redshift
d_p = cosmo.luminosity_distance(z_obs).to('cm')

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
    print(f"Total stars: {num_stars}")

    # Load and truncate grid correctly
    print("Loading grid...", flush=True)
    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
    
    print("Truncating grid range (900-20000 A)...", flush=True)
    # CRITICAL: Added inplace=True
    grid.reduce_rest_frame_range(900 * Angstrom, 20000 * Angstrom, inplace=True)
    
    n_lam = len(grid.lam)
    print(f"Number of wavelength bins: {n_lam}")

    # Models
    nebular_model = ReprocessedEmission(grid=grid)
    attenuated_model = AttenuatedEmission(grid=grid, dust_curve=Calzetti2000(), apply_to=nebular_model, emitter="stellar")
    
    # ACCUMULATORS
    lnu_att_total = np.zeros(n_lam)
    
    chunk_size = 50000
    for i in range(0, num_stars, chunk_size):
        end = min(i + chunk_size, num_stars)
        chunk_stars = Stars(
            initial_masses=imasses_all[i:end] * Msun,
            ages=ages_yr_all[i:end] * yr,
            metallicities=metallicities_all[i:end],
            coordinates=coods_all[i:end] * kpc
        )
        
        tau_v_chunk = np.full(end - i, 0.33)
        res_att = chunk_stars.get_particle_spectra(emission_model=attenuated_model, tau_v=tau_v_chunk, nthreads=8)
        spec_att_parts = res_att['attenuated'] if isinstance(res_att, dict) else res_att
        
        if hasattr(spec_att_parts, 'nsed') and spec_att_parts.nsed > 1:
            lnu_att_total += np.sum(spec_att_parts.lnu.to(erg/s/Hz).value, axis=0)
        else:
            lnu_att_total += spec_att_parts.lnu.to(erg/s/Hz).value

    # OBSERVED FRAME CONVERSION
    lam_obs = grid.lam.to(Angstrom).value * (1 + z_obs)
    flux_obs = (1 + z_obs) * lnu_att_total / (4 * np.pi * d_p.value**2)

    # SAVE TO FITS (DESI Standard Format)
    output_name = f"subhalo_{subhalo_id}_full_spectrum.fits"
    
    col1 = fits.Column(name='wavelength', format='D', array=lam_obs, unit='Angstrom')
    col2 = fits.Column(name='flux', format='D', array=flux_obs, unit='erg/s/cm^2/Hz')
    
    hdu = fits.BinTableHDU.from_columns([col1, col2])
    hdu.header['REDSHIFT'] = z_obs
    hdu.header['SUBHALO'] = subhalo_id
    hdu.writeto(output_name, overwrite=True)
    
    print(f"\n--- SUCCESS ---")
    print(f"Full spectrum saved to: {output_name}")
    print(f"Format: wavelength, flux (Observed-frame)")

else:
    print("No stars found.")
