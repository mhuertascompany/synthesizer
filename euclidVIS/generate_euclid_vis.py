import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
import time
from unyt import Myr, yr, kpc, arcsec, Angstrom, Msun, pc, km, s, unyt_quantity, unyt_array, rad
# from synthesizer.load_data.load_illustris import load_IllustrisTNG
from synthesizer.grid import Grid
from synthesizer.imaging import Image
from synthesizer.instruments.filters import FilterCollection, Filter
from synthesizer.emission_models import AttenuatedEmission, ReprocessedEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.kernel_functions import Kernel
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars
from synthesizer.particle.galaxy import Galaxy
from synthesizer.load_data.utils import age_lookup_table, lookup_age
import illustris_python as il
from tqdm import tqdm
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

def load_IllustrisTNG_fixed(
    directory=".",
    snap_number=99,
    stellar_mass_limit=1e10,
    subhalo_ids=None,
    max_galaxies=None,
    verbose=True,
    dtm=0.3,
    physical=True,
    metals=True,
    age_lookup=True,
    age_lookup_delta_a=1e-4,
):
    """Fixed version of load_IllustrisTNG to handle coordinate scaling and debug gas."""
    snap_number = int(snap_number)
    if verbose: print("Loading header information...", flush=True)
    header = il.groupcat.loadHeader(directory, snap_number)
    scale_factor = header["Time"].astype(np.float32)
    redshift = header["Redshift"].astype(np.float32)
    h = header["HubbleParam"]

    if verbose: print("Loading subhalo catalogue...", flush=True)
    fields = ["SubhaloMassType", "SubhaloPos"]
    output = il.groupcat.loadSubhalos(directory, snap_number, fields=fields)
    stellar_mass = output["SubhaloMassType"][:, 4]
    
    if subhalo_ids is not None:
        subhalo_mask = np.zeros(len(stellar_mass), dtype=bool)
        subhalo_mask[subhalo_ids] = True
    else:
        subhalo_mask = (stellar_mass * 1e10) > float(stellar_mass_limit)

    subhalo_pos = output["SubhaloPos"][subhalo_mask]
    if verbose: print(f"Loaded {np.sum(subhalo_mask)} galaxies above cut", flush=True)

    galaxies = []
    processed_count = 0
    all_indices = np.where(subhalo_mask)[0]
    
    for idx, pos in tqdm(zip(all_indices, subhalo_pos), total=len(all_indices), disable=not verbose):
        if max_galaxies is not None and processed_count >= max_galaxies:
            break
            
        galaxy = Galaxy(verbose=False)
        galaxy.redshift = redshift
        if physical: pos *= scale_factor
        galaxy.centre = pos * kpc

        # Load Stars
        star_fields = ["GFM_StellarFormationTime", "Coordinates", "Masses", "GFM_InitialMass", "GFM_Metallicity", "SubfindHsml"]
        if metals: star_fields.append("GFM_Metals")
        out_stars = il.snapshot.loadSubhalo(directory, snap_number, idx, "stars", fields=star_fields)
        if out_stars["count"] > 0:
            mask = out_stars["GFM_StellarFormationTime"] <= 0.0
            imasses = out_stars["GFM_InitialMass"][~mask]
            form_time = out_stars["GFM_StellarFormationTime"][~mask]
            coods = out_stars["Coordinates"][~mask]
            metallicities = out_stars["GFM_Metallicity"][~mask]
            masses = out_stars["Masses"][~mask]
            hsml = out_stars["SubfindHsml"][~mask]
            masses = (masses * 1e10) / h
            imasses = (imasses * 1e10) / h
            if physical:
                coods *= scale_factor
                hsml *= scale_factor
            cosmo_astropy = cosmo
            universe_age = cosmo_astropy.age(1.0 / scale_factor - 1)
            if age_lookup:
                scale_factors, age_grid = age_lookup_table(cosmo_astropy, redshift=redshift, delta_a=age_lookup_delta_a, low_lim=1e-4)
                _ages = lookup_age(form_time, scale_factors, age_grid)
            else:
                _ages = cosmo_astropy.age(1.0 / form_time - 1)
            ages = (universe_age - _ages).value * 1e9  # yr
            galaxy.load_stars(initial_masses=imasses * Msun, ages=ages * yr, metallicities=metallicities, coordinates=coods * kpc, current_masses=masses * Msun, smoothing_lengths=hsml * kpc if hsml is not None else None)

        # Load Gas
        gas_fields = ["StarFormationRate", "Coordinates", "Masses", "GFM_Metallicity", "SubfindHsml"]
        out_gas = il.snapshot.loadSubhalo(directory, snap_number, idx, "gas", fields=gas_fields)
        if out_gas["count"] > 0:
            g_masses = out_gas["Masses"]
            g_sfr = out_gas["StarFormationRate"]
            g_coods = out_gas["Coordinates"]
            g_hsml = out_gas["SubfindHsml"]
            g_metals = out_gas["GFM_Metallicity"]
            g_masses = (g_masses * 1e10) / h
            star_forming = g_sfr > 0.0
            if physical:
                g_coods *= scale_factor
                g_hsml *= scale_factor
            galaxy.load_gas(coordinates=g_coods * kpc, masses=g_masses * Msun, metallicities=g_metals, star_forming=star_forming, smoothing_lengths=g_hsml * kpc, dust_to_metal_ratio=dtm)
            # PROPER DEBUG PRINT
            # print(f"  DEBUG: Subhalo {idx} loaded {out_gas['count']} gas particles.", flush=True)
        else:
            # print(f"  DEBUG: Subhalo {idx} has NO gas particles in snapshot.", flush=True)
            pass

        galaxies.append(galaxy)
        processed_count += 1

    # Update subhalo_mask to match the number of galaxies loaded
    if max_galaxies is not None and len(galaxies) < np.sum(subhalo_mask):
        new_mask = np.zeros_like(subhalo_mask, dtype=bool)
        new_mask[all_indices[:len(galaxies)]] = True
        subhalo_mask = new_mask

    return galaxies, subhalo_mask

def load_config():
    """Load configuration from YAML and override with command-line arguments."""
    default_config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    
    parser = argparse.ArgumentParser(description='Generate Euclid VIS image from TNG data.')
    parser.add_argument('--config', type=str, default=default_config_path, help='Path to config file')
    
    # Paths
    parser.add_argument('--tng_path', type=str, help='Path to TNG data')
    parser.add_argument('--grid_dir', type=str, help='Path to spectral grids')
    parser.add_argument('--output_path', type=str, help='Path for output FITS')
    
    # Simulation
    parser.add_argument('--snap', type=int, help='TNG snapshot number')
    parser.add_argument('--stellar_mass_limit', type=float, help='Stellar mass limit (Msun)')
    parser.add_argument('--batch', type=bool, help='Enable batch processing')
    parser.add_argument('--max_galaxies', type=int, help='Max galaxies to process in batch mode')
    parser.add_argument('--subhalo_ids', type=int, nargs='+', help='Subhalo IDs to load')
    parser.add_argument('--grid_name', type=str, help='Spectral grid name')
    
    # Observation
    parser.add_argument('--z_obs', type=float, help='Observation redshift override')
    parser.add_argument('--fov_kpc', type=float, help='FOV in kpc')
    parser.add_argument('--pixel_scale', type=float, help='Pixel scale in arcsec/pixel')
    parser.add_argument('--fwhm', type=float, help='PSF FWHM in arcsec')
    
    # Optimization
    parser.add_argument('--particle_limit', type=int, help='Max particles to process')
    parser.add_argument('--nthreads', type=int, help='Number of threads for spectral calculation')
    
    # DESI
    parser.add_argument('--desi', action='store_true', help='Enable DESI mock spectra generation')
    parser.add_argument('--no-desi', action='store_false', dest='desi', help='Disable DESI mock spectra generation')
    parser.set_defaults(desi=None)
    
    # Projection
    parser.add_argument('--projection_type', type=str, choices=['random', 'face-on', 'edge-on', 'manual'], help='Type of projection')
    parser.add_argument('--phi', type=float, help='Rotation angle phi (rad)')
    parser.add_argument('--theta', type=float, help='Rotation angle theta (rad)')
    
    args = parser.parse_args()
    
    # Load YAML config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"WARNING: Config file {args.config} not found. Using internal defaults.")
        config = {
            'paths': {}, 'simulation': {}, 'observation': {}, 
            'model': {}, 'optimization': {}, 'projection': {}, 'desi': {}
        }
        
    # Override with CLI arguments if provided
    if args.tng_path: config['paths']['tng_path'] = args.tng_path
    if args.grid_dir: config['paths']['grid_dir'] = args.grid_dir
    if args.output_path: config['paths']['output_path'] = args.output_path
    if args.snap: config['simulation']['snap_number'] = args.snap
    if args.stellar_mass_limit: config['simulation']['stellar_mass_limit'] = args.stellar_mass_limit
    if args.batch is not None: config['simulation']['batch'] = args.batch
    if args.max_galaxies: config['simulation']['max_galaxies'] = args.max_galaxies
    if args.subhalo_ids: config['simulation']['subhalo_ids'] = args.subhalo_ids
    if args.grid_name: config['simulation']['grid_name'] = args.grid_name
    if args.z_obs is not None: config['observation']['z_obs'] = args.z_obs
    if args.fov_kpc: config['observation']['fov_kpc'] = args.fov_kpc
    if args.pixel_scale: config['observation']['pixel_scale_arcsec'] = args.pixel_scale
    if args.fwhm: config['observation']['fwhm_arcsec'] = args.fwhm
    if args.particle_limit: config['optimization']['particle_limit'] = args.particle_limit
    if args.nthreads: config['optimization']['nthreads_spectra'] = args.nthreads
    
    if args.desi is not None: 
        if 'desi' not in config: config['desi'] = {}
        config['desi']['enabled'] = args.desi
    
    if args.projection_type: 
        if 'projection' not in config: config['projection'] = {}
        config['projection']['type'] = args.projection_type
    if args.phi is not None: config['projection']['phi'] = args.phi
    if args.theta is not None: config['projection']['theta'] = args.theta
    
    return config

def safe_rotate(component, phi=None, theta=None, proj_type='manual'):
    """Safely rotate a particle component, handling cases where velocities are None."""
    if component is None:
        return

    if proj_type == 'face-on':
        if component.velocities is None:
            print(f"  WARNING: Cannot rotate {component.name} face-on without velocities. Skipping.")
            return
        component.rotate_face_on()
    elif proj_type == 'edge-on':
        if component.velocities is None:
            print(f"  WARNING: Cannot rotate {component.name} edge-on without velocities. Skipping.")
            return
        component.rotate_edge_on()
    else:
        # manual or random
        # If velocities are None, we temporarily set them to zeros to avoid library bugs
        # then set them back to None after rotation.
        vels_none = component.velocities is None
        if vels_none:
            # Create zeros with correct velocity units (km/s)
            component.velocities = unyt_array(np.zeros_like(component.coordinates.value), units=km/s)
        
        component.rotate_particles(phi=phi, theta=theta)
        
        if vels_none:
            component.velocities = None

def process_galaxy(target_galaxy, subhalo_id, grid, vis_filter, model, config):
    """Core logic to process a single galaxy and save its outputs."""
    obs = config['observation']
    mod = config['model']
    opt = config['optimization']
    paths = config['paths']
    desi_conf = config.get('desi', {})
    proj = config.get('projection', {})
    
    # Initial check for stellar component
    if target_galaxy.stars is None:
        print(f"  ERROR: Subhalo {subhalo_id} has no stars! Skipping.", flush=True)
        return
    
    # Get stellar mass for metadata (sum of initial masses)
    stellar_mass = float(np.sum(target_galaxy.stars.initial_masses).to(Msun).value)

    print(f"\nProcessing Subhalo {subhalo_id}...", flush=True)

    # Redshift and Distance Handling
    z_obs = obs.get('z_obs')
    if z_obs is None:
        z_obs = target_galaxy.redshift
    
    if z_obs < 0.001:
        print(f"WARNING: Redshift {z_obs:.4f} is too low. Using z=0.05.", flush=True)
        z_obs = 0.05
    
    d_lum = unyt_quantity.from_astropy(cosmo.luminosity_distance(z_obs).to(u.cm))
    scale_kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_obs).value / 60.0
    
    # 1. Coordinate Projections and Rotation
    proj_type = proj.get('type', 'manual')
    if proj_type == 'random':
        phi, theta = np.random.uniform(0, 2*np.pi), np.random.uniform(0, np.pi)
        print(f"  Projection: random (phi={phi:.3f}, theta={theta:.3f})")
        safe_rotate(target_galaxy.stars, phi=phi*rad, theta=theta*rad, proj_type='random')
        safe_rotate(target_galaxy.gas, phi=phi*rad, theta=theta*rad, proj_type='random')
    elif proj_type == 'face-on':
        print("  Projection: face-on")
        safe_rotate(target_galaxy.stars, proj_type='face-on')
        safe_rotate(target_galaxy.gas, proj_type='face-on')
    elif proj_type == 'edge-on':
        print("  Projection: edge-on")
        safe_rotate(target_galaxy.stars, proj_type='edge-on')
        safe_rotate(target_galaxy.gas, proj_type='edge-on')
    elif proj_type == 'manual':
        phi, theta = proj.get('phi', 0.0), proj.get('theta', 0.0)
        if phi != 0 or theta != 0:
            print(f"  Projection: manual (phi={phi:.3f}, theta={theta:.3f})")
            safe_rotate(target_galaxy.stars, phi=phi*rad, theta=theta*rad, proj_type='manual')
            safe_rotate(target_galaxy.gas, phi=phi*rad, theta=theta*rad, proj_type='manual')

    # FOV and Resolution
    fov_kpc = obs['fov_kpc']
    fov_arcsec = fov_kpc / scale_kpc_per_arcsec
    pixel_scale_arcsec = obs['pixel_scale_arcsec']
    resolution = int(fov_arcsec / pixel_scale_arcsec)
    
    # Filtering and Downsampling
    particle_limit = opt.get('particle_limit', -1)
    if particle_limit < 0: particle_limit = float('inf')
    fov_limit = fov_kpc / 2 * kpc
    
    # 1. Coordinate Projections and Rotation for Stars
    # clipping to FOV
    star_coords = target_galaxy.stars.coordinates
    if target_galaxy.stars.centre is not None:
        star_coords -= target_galaxy.stars.centre
    star_fov_mask = (np.abs(star_coords[:, 0]) < fov_limit) & (np.abs(star_coords[:, 1]) < fov_limit)
    star_indices = np.where(star_fov_mask)[0]
    
    star_weight_scale = 1.0
    if len(star_indices) > particle_limit:
        sampled_indices = np.random.choice(star_indices, int(particle_limit), replace=False)
        star_weight_scale = len(star_indices) / int(particle_limit)
    else:
        sampled_indices = star_indices

    opt_stars = Stars(
        initial_masses=target_galaxy.stars.initial_masses[sampled_indices],
        ages=target_galaxy.stars.ages[sampled_indices],
        metallicities=target_galaxy.stars.metallicities[sampled_indices],
        coordinates=target_galaxy.stars.coordinates[sampled_indices],
        current_masses=target_galaxy.stars.current_masses[sampled_indices],
        smoothing_lengths=target_galaxy.stars.smoothing_lengths[sampled_indices],
        velocities=target_galaxy.stars.velocities[sampled_indices] if target_galaxy.stars.velocities is not None else None,
        redshift=target_galaxy.stars.redshift,
        centre=target_galaxy.stars.centre
    )

    # --- ROBUST INITIALIZATION ---
    chunk_size = opt.get('chunk_size', 500000)
    num_stars = len(opt_stars.initial_masses)
    tau_v = np.zeros(num_stars) # Default to no dust
    opt_gas = None
    
    # 2. Gas / Dust Processing
    # Check if gas exists and has particles
    target_gas = getattr(target_galaxy, 'gas', None)
    if target_gas is not None and hasattr(target_gas, 'masses') and target_gas.masses is not None and len(target_gas.masses) > 0:
        
        if not hasattr(target_gas, 'dust_masses'):
            target_gas.dust_masses = target_gas.masses * target_gas.metallicities * mod['dust_to_metal']
        
        gas_coords = target_gas.coordinates
        if target_gas.centre is not None:
            gas_coords -= target_gas.centre
            
        gas_fov_mask = (np.abs(gas_coords[:, 0]) < fov_limit + 50*kpc) & (np.abs(gas_coords[:, 1]) < fov_limit + 50*kpc)
        gas_indices = np.where(gas_fov_mask)[0]
        
        num_gas_in_fov = len(gas_indices)
        if num_gas_in_fov > 0:
            num_to_sample = num_gas_in_fov
            if particle_limit < float('inf'):
                num_to_sample = min(num_to_sample, int(particle_limit))
            
            sampled_gas_indices = np.random.choice(gas_indices, num_to_sample, replace=False)
            opt_gas = Gas(
                masses=target_gas.masses[sampled_gas_indices],
                metallicities=target_gas.metallicities[sampled_gas_indices],
                coordinates=target_gas.coordinates[sampled_gas_indices],
                smoothing_lengths=target_gas.smoothing_lengths[sampled_gas_indices],
                dust_masses=target_gas.dust_masses[sampled_gas_indices],
                redshift=target_gas.redshift,
                centre=target_gas.centre
            )
    
    # 3. Calculate Spectra
    original_stars, original_gas = target_galaxy.stars, target_galaxy.gas
    target_galaxy.stars, target_galaxy.gas = opt_stars, opt_gas
    
    tau_v = np.zeros(num_stars)
    if target_galaxy.gas is not None and len(target_galaxy.gas.masses) > 0:
        print(f"  Calculating tau_v (nthreads={opt['nthreads_tau_v']})...", flush=True)
        tau_v = target_galaxy.get_stellar_los_tau_v(
            kappa=mod['kappa'], 
            kernel=Kernel(name="cubic", binsize=1000).get_kernel(), 
            nthreads=opt['nthreads_tau_v']
        )
    else:
        if target_gas is not None and len(target_gas.masses) > 0:
            print("  INFO: Gas removed by FOV. Dust set to zero.", flush=True)
        else:
            print("  INFO: No gas in subhalo. Dust set to zero.", flush=True)

    # --- 3. Calculate Spectra (Chunked for Memory Efficiency) ---
    # chunk_size and num_stars already defined above
    
    # Aggregators
    hist_total = np.zeros((resolution, resolution))
    lnu_fiber_total = None
    lam_obs = None
    
    # Pre-calculate global coordinates and fiber mask
    coords_for_img = opt_stars.coordinates - (opt_stars.centre if opt_stars.centre is not None else 0)
    r_arcsec = np.sqrt(coords_for_img[:, 0].to(kpc).value**2 + coords_for_img[:, 1].to(kpc).value**2) / scale_kpc_per_arcsec
    desi_mask_global = r_arcsec <= (desi_conf['fiber_diameter_arcsec'] / 2.0) if desi_conf.get('enabled', False) else None

    # Pre-calculate Euclid VIS transmission curve for interpolation
    # We'll do this once outside the loop
    # dummy_grid loaded just to get wavelength bins
    vis_filter_lam = vis_filter.lam.to(Angstrom).value
    vis_filter_trans = vis_filter.transmission
    
    # Pre-calculate filter transmission at the spectral grid points (redshifted)
    # This is constant for all stars in the galaxy
    lam_rest_grid = grid.lam.to(Angstrom).value
    t_rest = np.interp(lam_rest_grid * (1 + z_obs), vis_filter_lam, vis_filter_trans, left=0, right=0)
    
    print(f"  Calculating spectra in chunks of {chunk_size} (total {num_stars} stars)...", flush=True)
    
    for i in range(0, num_stars, chunk_size):
        end = min(i + chunk_size, num_stars)
        print(f"    Processing stars {i}-{end}...", flush=True)
        
        # Subset stars and tau_v
        chunk_stars = Stars(
            initial_masses=opt_stars.initial_masses[i:end],
            ages=opt_stars.ages[i:end],
            metallicities=opt_stars.metallicities[i:end],
            coordinates=opt_stars.coordinates[i:end],
            current_masses=opt_stars.current_masses[i:end],
            smoothing_lengths=opt_stars.smoothing_lengths[i:end],
            velocities=opt_stars.velocities[i:end] if opt_stars.velocities is not None else None,
            redshift=opt_stars.redshift,
            centre=opt_stars.centre
        )
        chunk_tau_v = tau_v[i:end]
        
        # get_particle_spectra for this chunk
        spectra_dict = chunk_stars.get_particle_spectra(emission_model=model, tau_v=chunk_tau_v, nthreads=opt['nthreads_spectra'], verbose=False)
        particle_spectra = spectra_dict.get('attenuated', list(spectra_dict.values())[0]) if isinstance(spectra_dict, dict) else spectra_dict
        
        # --- Euclid VIS Integration ---
        nu_rest = particle_spectra.nu.to('Hz').value
        lnu_rest = particle_spectra.lnu.to('erg/s/Hz').value
        
        # Integrate and scale by weights
        chunk_flux_in_band = (np.abs(np.trapezoid(lnu_rest * t_rest, x=nu_rest, axis=-1)) * star_weight_scale) / (4 * np.pi * d_lum.value**2)
        
        # Histogram accumulation
        chunk_coords = coords_for_img[i:end]
        hist_chunk, _, _ = np.histogram2d(
            chunk_coords[:, 0].to(kpc).value, 
            chunk_coords[:, 1].to(kpc).value, 
            bins=resolution, 
            range=[[-fov_kpc/2, fov_kpc/2], [-fov_kpc/2, fov_kpc/2]], 
            weights=chunk_flux_in_band
        )
        hist_total += hist_chunk
        
        # --- DESI Fiber Integration ---
        if desi_mask_global is not None:
            chunk_desi_mask = desi_mask_global[i:end]
            if np.sum(chunk_desi_mask) > 0:
                chunk_lnu_fiber = np.sum(particle_spectra.lnu[chunk_desi_mask], axis=0) * star_weight_scale
                if lnu_fiber_total is None:
                    lnu_fiber_total = chunk_lnu_fiber
                    lam_obs = particle_spectra.lam.to(Angstrom).value * (1 + z_obs)
                else:
                    lnu_fiber_total += chunk_lnu_fiber
        
        # Clean up chunk to save memory
        del spectra_dict, particle_spectra, lnu_rest, chunk_stars, chunk_flux_in_band
        print(f"    Chunk complete. Current progress: {end/num_stars*100:.1f}%. ({time.ctime()})", flush=True)
    
    print("  Chunked calculation complete.", flush=True)

    # --- 4. Imaging Output ---
    # Save Euclid
    euclid_dir = os.path.join(paths['output_path'], f"sn{config['simulation']['snap_number']}", "Euclid")
    os.makedirs(euclid_dir, exist_ok=True)
    
    # Raw Image
    fits.PrimaryHDU(hist_total).writeto(os.path.join(euclid_dir, f"euclid_vis_{subhalo_id}_raw.fits"), overwrite=True)
    
    # Convolved Image
    sigma_pixels = (obs['fwhm_arcsec'] / 2.355) / pixel_scale_arcsec
    img_convolved = gaussian_filter(hist_total, sigma=sigma_pixels)
    hdu = fits.PrimaryHDU(img_convolved)
    hdu.header['OBJECT'] = f"Subhalo {subhalo_id}"
    hdu.header['REDSHIFT'] = z_obs
    hdu.header['SUBHALO'] = subhalo_id
    hdu.header['MASS'] = stellar_mass
    hdu.writeto(os.path.join(euclid_dir, f"euclid_vis_{subhalo_id}.fits"), overwrite=True)
    print(f"  Euclid images saved to {euclid_dir}", flush=True)

    # --- 5. DESI Output ---
    if desi_conf.get('enabled', False) and lnu_fiber_total is not None:
        fnu_fiber = (lnu_fiber_total / (4 * np.pi * d_lum**2)).to('erg/s/cm**2/Hz').value
        
        # Save DESI
        desi_dir = os.path.join(paths['output_path'], f"sn{config['simulation']['snap_number']}", "DESI")
        os.makedirs(desi_dir, exist_ok=True)
        
        # Raw Spectrum
        cols_raw = [fits.Column(name='wavelength', format='E', array=lam_obs), fits.Column(name='flux', format='E', array=fnu_fiber)]
        fits.BinTableHDU.from_columns(fits.ColDefs(cols_raw)).writeto(os.path.join(desi_dir, f"desi_spectrum_{subhalo_id}_raw.fits"), overwrite=True)
        
        # Convolved and Resampled
        lam_desi = np.arange(desi_conf['lam_min'], desi_conf['lam_max'] + desi_conf['d_lam'], desi_conf['d_lam'])
        fnu_interp = interp1d(lam_obs, fnu_fiber, bounds_error=False, fill_value=0.0)(lam_desi)
        R_desi = desi_conf['R_min'] + (desi_conf['R_max'] - desi_conf['R_min']) * (lam_desi - desi_conf['lam_min']) / (desi_conf['lam_max'] - desi_conf['lam_min'])
        fnu_conv = gaussian_filter1d(fnu_interp, sigma=np.mean((lam_desi / (2.355 * R_desi)) / desi_conf['d_lam']))
        
        cols_conv = [fits.Column(name='wavelength', format='E', array=lam_desi), fits.Column(name='flux', format='E', array=fnu_conv)]
        hdu_desi = fits.BinTableHDU.from_columns(fits.ColDefs(cols_conv))
        hdu_desi.header['REDSHIFT'] = z_obs
        hdu_desi.header['SUBHALO'] = subhalo_id
        hdu_desi.header['MASS'] = stellar_mass
        hdu_desi.writeto(os.path.join(desi_dir, f"desi_spectrum_{subhalo_id}.fits"), overwrite=True)
        print(f"  DESI spectra saved to {desi_dir}", flush=True)

    # Revert target_galaxy after processing (optional but cleaner)
    target_galaxy.stars, target_galaxy.gas = original_stars, original_gas

    # 4. Update Catalog
    cat_path = os.path.join(paths['output_path'], f"sn{config['simulation']['snap_number']}", "catalog.csv")
    write_header = not os.path.exists(cat_path)
    with open(cat_path, 'a') as f:
        if write_header:
            f.write("subhalo_id,redshift,stellar_mass\n")
        f.write(f"{subhalo_id},{z_obs:.6f},{stellar_mass:.4e}\n")

def generate_euclid_vis_image(config):
    paths, sim = config['paths'], config['simulation']
    
    print(f"Loading TNG data for snap {sim['snap_number']}...", flush=True)
    limit = float(sim.get('stellar_mass_limit', 1e10))
    subhalo_ids = sim.get('subhalo_ids')
    if sim.get('batch', False):
        subhalo_ids = None # Ignore specific IDs if batch mode is on
    
    galaxies, subhalo_mask = load_IllustrisTNG_fixed(
        directory=paths['tng_path'], snap_number=sim['snap_number'], 
        stellar_mass_limit=limit,
        subhalo_ids=subhalo_ids, 
        max_galaxies=sim.get('max_galaxies'),
        verbose=True
    )

    if not galaxies:
        print("No galaxies found!"); return

    # Load shared resources
    print("Loading shared resources (grid, filter, model)...", flush=True)
    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])
    
    vis_filter = None
    local_filter = os.path.join(paths['grid_dir'], paths.get('filter_file', 'Euclid_VIS.vis.dat'))
    if os.path.exists(local_filter):
        vis_filter = Filter("Euclid/VIS_local", transmission=np.loadtxt(local_filter)[:, 1], new_lam=np.loadtxt(local_filter)[:, 0] * Angstrom)
        vis_filter._interpolate_wavelength(grid.lam)
    else:
        vis_filter = FilterCollection(filter_codes=["Euclid/VIS.vis"], new_lam=grid.lam)[0]

    model = AttenuatedEmission(grid=grid, dust_curve=Calzetti2000(), apply_to=ReprocessedEmission(grid=grid), emitter="stellar")
    
    # Get the actual subhalo IDs from the mask
    all_subhalo_indices = np.where(subhalo_mask)[0]
    
    print(f"Starting batch processing of {len(galaxies)} galaxies...", flush=True)
    obs_conf = config['observation']
    for i, galaxy in enumerate(galaxies):
        subhalo_id = all_subhalo_indices[i]
        
        # Randomize redshift if requested
        if obs_conf.get('randomize_redshift', False):
            z_min, z_max = obs_conf.get('z_min', 0.0), obs_conf.get('z_max', 1.5)
            z_rand = float(np.random.uniform(z_min, z_max))
            obs_conf['z_obs'] = z_rand
            print(f"  [{i+1}/{len(galaxies)}] Subhalo {subhalo_id}: Assigning random redshift z={z_rand:.4f}", flush=True)
            
        try:
            process_galaxy(galaxy, subhalo_id, grid, vis_filter, model, config)
        except Exception as e:
            print(f"ERROR: Failed to process subhalo {subhalo_id}: {e}")

if __name__ == "__main__":
    config = load_config()
    generate_euclid_vis_image(config)
    print("\nSUCCESS: All galaxies in the snapshot have been processed.", flush=True)
