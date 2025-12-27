import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
from unyt import Myr, kpc, arcsec, Angstrom, Msun, pc, km, s, unyt_quantity, unyt_array, rad
from synthesizer.load_data.load_illustris import load_IllustrisTNG
from synthesizer.grid import Grid
from synthesizer.imaging import Image
from synthesizer.instruments.filters import FilterCollection, Filter
from synthesizer.emission_models import AttenuatedEmission, ReprocessedEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.kernel_functions import Kernel
from synthesizer.particle.gas import Gas
from synthesizer.particle.stars import Stars
from scipy.ndimage import gaussian_filter, gaussian_filter1d
from scipy.interpolate import interp1d
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

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

    # Gas / Dust
    if not hasattr(target_galaxy.gas, 'dust_masses'):
        target_galaxy.gas.dust_masses = target_galaxy.gas.masses * target_galaxy.gas.metallicities * mod['dust_to_metal']
    
    gas_coords = target_galaxy.gas.coordinates
    if target_galaxy.gas.centre is not None:
        gas_coords -= target_galaxy.gas.centre
    gas_fov_mask = (np.abs(gas_coords[:, 0]) < fov_limit + 50*kpc) & (np.abs(gas_coords[:, 1]) < fov_limit + 50*kpc)
    gas_indices = np.where(gas_fov_mask)[0]
    sampled_gas_indices = np.random.choice(gas_indices, min(len(gas_indices), int(particle_limit)), replace=False)

    opt_gas = Gas(
        masses=target_galaxy.gas.masses[sampled_gas_indices],
        metallicities=target_galaxy.gas.metallicities[sampled_gas_indices],
        coordinates=target_galaxy.gas.coordinates[sampled_gas_indices],
        smoothing_lengths=target_galaxy.gas.smoothing_lengths[sampled_gas_indices],
        dust_masses=target_galaxy.gas.dust_masses[sampled_gas_indices],
        redshift=target_galaxy.gas.redshift,
        centre=target_galaxy.gas.centre
    )
    
    # Spectra
    original_stars, original_gas = target_galaxy.stars, target_galaxy.gas
    target_galaxy.stars, target_galaxy.gas = opt_stars, opt_gas
    
    tau_v = target_galaxy.get_stellar_los_tau_v(kappa=mod['kappa'], kernel=Kernel(name="cubic", binsize=1000).get_kernel(), nthreads=opt['nthreads_tau_v'])
    spectra_dict = target_galaxy.stars.get_particle_spectra(model, tau_v=tau_v, nthreads=opt['nthreads_spectra'])
    
    target_galaxy.stars, target_galaxy.gas = original_stars, original_gas
    particle_spectra = spectra_dict.get('attenuated', list(spectra_dict.values())[0]) if isinstance(spectra_dict, dict) else spectra_dict

    # 2. Imaging
    nu_rest = particle_spectra.nu.to('Hz').value
    lnu_rest = particle_spectra.lnu.to('erg/s/Hz').value
    t_rest = np.interp(particle_spectra.lam.to(Angstrom).value * (1 + z_obs), vis_filter.lam.to(Angstrom).value, vis_filter.transmission, left=0, right=0)
    flux_in_band = (np.abs(np.trapezoid(lnu_rest * t_rest, x=nu_rest, axis=-1)) * star_weight_scale) / (4 * np.pi * d_lum.value**2)
    
    coords_for_img = opt_stars.coordinates - (opt_stars.centre if opt_stars.centre is not None else 0)
    hist, _, _ = np.histogram2d(coords_for_img[:, 0].to(kpc).value, coords_for_img[:, 1].to(kpc).value, bins=resolution, range=[[-fov_kpc/2, fov_kpc/2], [-fov_kpc/2, fov_kpc/2]], weights=flux_in_band)
    
    # Save Euclid
    euclid_dir = os.path.join(paths['output_path'], f"sn{config['simulation']['snap_number']}", "Euclid")
    os.makedirs(euclid_dir, exist_ok=True)
    
    # Raw Image
    fits.PrimaryHDU(hist).writeto(os.path.join(euclid_dir, f"euclid_vis_{subhalo_id}_raw.fits"), overwrite=True)
    
    # Convolved Image
    sigma_pixels = (obs['fwhm_arcsec'] / 2.355) / pixel_scale_arcsec
    img_convolved = gaussian_filter(hist, sigma=sigma_pixels)
    hdu = fits.PrimaryHDU(img_convolved)
    hdu.header['OBJECT'] = f"Subhalo {subhalo_id}"
    hdu.header['REDSHIFT'] = z_obs
    hdu.header['SUBHALO'] = subhalo_id
    hdu.writeto(os.path.join(euclid_dir, f"euclid_vis_{subhalo_id}.fits"), overwrite=True)

    # 3. DESI
    if desi_conf.get('enabled', False):
        r_arcsec = np.sqrt(coords_for_img[:, 0].to(kpc).value**2 + coords_for_img[:, 1].to(kpc).value**2) / scale_kpc_per_arcsec
        mask = r_arcsec <= (desi_conf['fiber_diameter_arcsec'] / 2.0)
        
        if np.sum(mask) > 0:
            lnu_fiber = np.sum(particle_spectra.lnu[mask], axis=0) * star_weight_scale
            fnu_fiber = (lnu_fiber / (4 * np.pi * d_lum**2)).to('erg/s/cm**2/Hz').value
            lam_obs = particle_spectra.lam.to(Angstrom).value
            
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
            hdu_desi.writeto(os.path.join(desi_dir, f"desi_spectrum_{subhalo_id}.fits"), overwrite=True)

def generate_euclid_vis_image(config):
    paths, sim = config['paths'], config['simulation']
    
    print(f"Loading TNG data for snap {sim['snap_number']}...", flush=True)
    limit = sim.get('stellar_mass_limit', 1e10)
    subhalo_ids = sim.get('subhalo_ids')
    
    galaxies, subhalo_mask = load_IllustrisTNG(
        directory=paths['tng_path'], snap_number=sim['snap_number'], 
        stellar_mass_limit=limit if not subhalo_ids else 8.5e6, # fallback if ids provided
        subhalo_ids=subhalo_ids, verbose=True
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
    for i, galaxy in enumerate(galaxies):
        subhalo_id = all_subhalo_indices[i]
        try:
            process_galaxy(galaxy, subhalo_id, grid, vis_filter, model, config)
        except Exception as e:
            print(f"ERROR: Failed to process subhalo {subhalo_id}: {e}")

if __name__ == "__main__":
    config = load_config()
    generate_euclid_vis_image(config)
