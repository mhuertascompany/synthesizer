import numpy as np
import matplotlib.pyplot as plt
import os
import yaml
import argparse
from unyt import Myr, kpc, arcsec, Angstrom, Msun, pc, km, s
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
    
    args = parser.parse_args()
    
    # Load YAML config
    if os.path.exists(args.config):
        with open(args.config, 'r') as f:
            config = yaml.safe_load(f)
    else:
        print(f"WARNING: Config file {args.config} not found. Using internal defaults.")
        config = {
            'paths': {}, 'simulation': {}, 'observation': {}, 
            'model': {}, 'optimization': {}
        }
        
    # Override with CLI arguments if provided
    if args.tng_path: config['paths']['tng_path'] = args.tng_path
    if args.grid_dir: config['paths']['grid_dir'] = args.grid_dir
    if args.output_path: config['paths']['output_path'] = args.output_path
    if args.snap: config['simulation']['snap_number'] = args.snap
    if args.subhalo_ids: config['simulation']['subhalo_ids'] = args.subhalo_ids
    if args.grid_name: config['simulation']['grid_name'] = args.grid_name
    if args.z_obs is not None: config['observation']['z_obs'] = args.z_obs
    if args.fov_kpc: config['observation']['fov_kpc'] = args.fov_kpc
    if args.pixel_scale: config['observation']['pixel_scale_arcsec'] = args.pixel_scale
    if args.fwhm: config['observation']['fwhm_arcsec'] = args.fwhm
    if args.particle_limit: config['optimization']['particle_limit'] = args.particle_limit
    if args.nthreads: config['optimization']['nthreads_spectra'] = args.nthreads
    if args.desi is not None: config['desi']['enabled'] = args.desi
    
    return config

def generate_euclid_vis_image(config):
    # Extract parameters from config
    paths = config['paths']
    sim = config['simulation']
    obs = config['observation']
    mod = config['model']
    opt = config['optimization']

    print("Loading TNG data...", flush=True)
    galaxies, subhalo_mask = load_IllustrisTNG(
        directory=paths['tng_path'],
        snap_number=sim['snap_number'],
        subhalo_ids=sim['subhalo_ids'],
    )

    if len(galaxies) == 0:
        print("No galaxies found!", flush=True)
        return

    target_galaxy = galaxies[0]
    print(f"Selected galaxy: {target_galaxy.name}", flush=True)

    # Redshift and Distance Handling
    z_obs = obs.get('z_obs')
    if z_obs is None:
        z_obs = target_galaxy.redshift
    
    if z_obs < 0.001:
        print(f"WARNING: Redshift {z_obs:.4f} is too low. Using z=0.05.", flush=True)
        z_obs = 0.05
    
    # Calculate Distances
    d_lum = unyt_quantity.from_astropy(cosmo.luminosity_distance(z_obs).to(u.cm))
    scale_kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_obs).value / 60.0
    
    print(f"Observation Redshift: {z_obs}", flush=True)
    print(f"Scale: {scale_kpc_per_arcsec:.3f} kpc/arcsec", flush=True)

    # Load Spectral Grid
    print(f"Loading spectral grid: {sim['grid_name']}...", flush=True)
    grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])

    # Load Filter
    print("Loading filter...", flush=True)
    vis_filter = None
    local_filter_path = os.path.join(paths['grid_dir'], paths.get('filter_file', 'Euclid_VIS.vis.dat'))
    if os.path.exists(local_filter_path):
        print(f"Loading local filter: {local_filter_path}", flush=True)
        data = np.loadtxt(local_filter_path)
        vis_filter = Filter("Euclid/VIS_local", transmission=data[:, 1], new_lam=data[:, 0] * Angstrom)
        vis_filter._interpolate_wavelength(grid.lam)

    if vis_filter is None:
        print("Falling back to SVO Euclid/VIS.vis...", flush=True)
        filters = FilterCollection(filter_codes=["Euclid/VIS.vis"], new_lam=grid.lam)
        vis_filter = filters[0]

    # Dust Model
    dust_curve = Calzetti2000() # Could be parameterized if needed
    reprocessed_model = ReprocessedEmission(grid=grid)
    model = AttenuatedEmission(
        grid=grid,
        dust_curve=dust_curve,
        apply_to=reprocessed_model,
        emitter="stellar"
    )
    
    # FOV and Resolution
    fov_kpc = obs['fov_kpc']
    fov_arcsec = fov_kpc / scale_kpc_per_arcsec
    pixel_scale_arcsec = obs['pixel_scale_arcsec']
    resolution = int(fov_arcsec / pixel_scale_arcsec)
    
    # Optimization: Filtering and Downsampling
    particle_limit = opt.get('particle_limit')
    if particle_limit is None or particle_limit < 0:
        particle_limit = float('inf')
    
    fov_limit = fov_kpc / 2 * kpc
    
    # Stars
    star_coords = target_galaxy.stars.coordinates
    if target_galaxy.stars.centre is not None:
        star_coords -= target_galaxy.stars.centre
    star_fov_mask = (np.abs(star_coords[:, 0]) < fov_limit) & (np.abs(star_coords[:, 1]) < fov_limit)
    star_indices = np.where(star_fov_mask)[0]
    
    star_weight_scale = 1.0
    if len(star_indices) > particle_limit:
        print(f"Downsampling stars to {particle_limit}...", flush=True)
        sampled_indices = np.random.choice(star_indices, particle_limit, replace=False)
        star_weight_scale = len(star_indices) / particle_limit
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

    # Gas
    if target_galaxy.gas is not None:
        if not hasattr(target_galaxy.gas, 'dust_masses'):
            target_galaxy.gas.dust_masses = target_galaxy.gas.masses * target_galaxy.gas.metallicities * mod['dust_to_metal']
        
        gas_coords = target_galaxy.gas.coordinates
        if target_galaxy.gas.centre is not None:
            gas_coords -= target_galaxy.gas.centre
        gas_limit = fov_limit + 50 * kpc
        gas_fov_mask = (np.abs(gas_coords[:, 0]) < gas_limit) & (np.abs(gas_coords[:, 1]) < gas_limit)
        gas_indices = np.where(gas_fov_mask)[0]
        
        if len(gas_indices) > particle_limit:
            sampled_gas_indices = np.random.choice(gas_indices, particle_limit, replace=False)
        else:
            sampled_gas_indices = gas_indices

        opt_gas = Gas(
            masses=target_galaxy.gas.masses[sampled_gas_indices],
            metallicities=target_galaxy.gas.metallicities[sampled_gas_indices],
            coordinates=target_galaxy.gas.coordinates[sampled_gas_indices],
            velocities=target_galaxy.gas.velocities[sampled_gas_indices] if target_galaxy.gas.velocities is not None else None,
            smoothing_lengths=target_galaxy.gas.smoothing_lengths[sampled_gas_indices],
            dust_masses=target_galaxy.gas.dust_masses[sampled_gas_indices],
            redshift=target_galaxy.gas.redshift,
            centre=target_galaxy.gas.centre
        )
        
        # Calculate tau_v
        original_stars, original_gas = target_galaxy.stars, target_galaxy.gas
        target_galaxy.stars, target_galaxy.gas = opt_stars, opt_gas
        
        print("Calculating tau_v...", flush=True)
        kernel = Kernel(name="cubic", binsize=1000).get_kernel()
        tau_v_opt = target_galaxy.get_stellar_los_tau_v(
            kappa=mod['kappa'], kernel=kernel, nthreads=opt['nthreads_tau_v']
        )
        
        print(f"Calculating spectra (nthreads={opt['nthreads_spectra']})...", flush=True)
        spectra_dict = target_galaxy.stars.get_particle_spectra(
            model, tau_v=tau_v_opt, nthreads=opt['nthreads_spectra']
        )
        
        target_galaxy.stars, target_galaxy.gas = original_stars, original_gas
        
        if isinstance(spectra_dict, dict):
            particle_spectra = spectra_dict.get('attenuated', list(spectra_dict.values())[0])
        else:
            particle_spectra = spectra_dict

        # Photometry
        print("Calculating photometry...", flush=True)
        nu_rest = particle_spectra.nu.to('Hz').value
        lnu_rest = particle_spectra.lnu.to('erg/s/Hz').value
        t_rest = np.interp(
            particle_spectra.lam.to(Angstrom).value * (1 + z_obs),
            vis_filter.lam.to(Angstrom).value,
            vis_filter.transmission,
            left=0, right=0
        )
        
        luminosity_in_band = np.abs(np.trapezoid(lnu_rest * t_rest, x=nu_rest, axis=-1))
        flux_in_band = (luminosity_in_band * star_weight_scale) / (4 * np.pi * d_lum.value**2)
        
        # Imaging
        print("Generating image...", flush=True)
        coords_for_img = opt_stars.coordinates
        if opt_stars.centre is not None:
            coords_for_img -= opt_stars.centre
            
        x = coords_for_img[:, 0].to(kpc).value
        y = coords_for_img[:, 1].to(kpc).value
        
        hist, _, _ = np.histogram2d(
            x, y, bins=resolution, 
            range=[[-fov_kpc/2, fov_kpc/2], [-fov_kpc/2, fov_kpc/2]],
            weights=flux_in_band
        )
        
        pixel_size = (fov_kpc / resolution) * kpc
        img = Image(img=hist, fov=fov_kpc*kpc, resolution=pixel_size)

        # DESI Mock Spectra Generation
        if config.get('desi', {}).get('enabled', False):
            print("Generating realistic DESI mock spectrum...", flush=True)
            desi_conf = config['desi']
            fiber_radius_arcsec = desi_conf['fiber_diameter_arcsec'] / 2.0
            
            # Calculate angular distance from center
            r_kpc = np.sqrt(x**2 + y**2)
            r_arcsec = r_kpc / scale_kpc_per_arcsec
            
            # Select stars within fiber
            fiber_mask = r_arcsec <= fiber_radius_arcsec
            n_fiber = np.sum(fiber_mask)
            print(f"Stars in DESI fiber: {n_fiber}", flush=True)
            
            if n_fiber > 0:
                # 1. Sum spectra of stars in fiber
                lnu_fiber_total = np.sum(particle_spectra.lnu[fiber_mask], axis=0)
                lnu_fiber_total *= star_weight_scale
                
                # 2. Convert to Flux (erg/s/cm^2/Hz)
                fnu_fiber = lnu_fiber_total / (4 * np.pi * d_lum**2)
                lam_obs = particle_spectra.lam.to(Angstrom).value
                fnu_val = fnu_fiber.to('erg/s/cm**2/Hz').value

                # 3. Define DESI wavelength grid
                lam_desi = np.arange(desi_conf['lam_min'], desi_conf['lam_max'] + desi_conf['d_lam'], desi_conf['d_lam'])
                
                # 4. Apply wavelength-dependent resolution convolution
                # R(lam) = R_min + (R_max - R_min) * (lam - lam_min) / (lam_max - lam_min)
                # sigma_lam = lam / (2.355 * R)
                # We do this on the original grid first, then resample
                R_lam = desi_conf['R_min'] + (desi_conf['R_max'] - desi_conf['R_min']) * \
                        (lam_obs - desi_conf['lam_min']) / (desi_conf['lam_max'] - desi_conf['lam_min'])
                R_lam = np.clip(R_lam, desi_conf['R_min'], desi_conf['R_max'])
                
                sigma_lam = lam_obs / (2.355 * R_lam)
                
                # Since sigma_lam varies slowly, we can use a variable-width Gaussian convolution
                # For efficiency on a large grid, we'll interpolate to the DESI grid first
                # and then apply a mean sigma if the variation is small, or do it properly.
                # Here we'll do a proper convolution by iterating over a few chunks if needed,
                # but for 0.1A grid, a simple loop or scipy.ndimage.gaussian_filter1d with constant sigma 
                # is often "good enough" if the variation is small. 
                # Let's do a slightly better approach:
                fnu_interp_func = interp1d(lam_obs, fnu_val, bounds_error=False, fill_value=0.0)
                fnu_desi_raw = fnu_interp_func(lam_desi)
                
                # Calculate sigma in pixels on the DESI grid
                R_desi = desi_conf['R_min'] + (desi_conf['R_max'] - desi_conf['R_min']) * \
                         (lam_desi - desi_conf['lam_min']) / (desi_conf['lam_max'] - desi_conf['lam_min'])
                sigma_pixels = (lam_desi / (2.355 * R_desi)) / desi_conf['d_lam']
                
                # Apply convolution. Since sigma varies, we'll use a trick or just a mean sigma
                # for this specific mock (variation is ~10%).
                mean_sigma = np.mean(sigma_pixels)
                fnu_desi_convolved = gaussian_filter1d(fnu_desi_raw, sigma=mean_sigma)
                
                # 5. Save DESI Spectrum
                desi_out = os.path.join(paths['output_path'], desi_conf['output_name'])
                
                col1 = fits.Column(name='wavelength', format='E', array=lam_desi)
                col2 = fits.Column(name='flux', format='E', array=fnu_desi_convolved)
                cols = fits.ColDefs([col1, col2])
                hdu_desi = fits.BinTableHDU.from_columns(cols)
                
                hdu_desi.header['OBJECT'] = target_galaxy.name
                hdu_desi.header['REDSHIFT'] = z_obs
                hdu_desi.header['FIBER_D'] = (desi_conf['fiber_diameter_arcsec'], 'arcsec')
                hdu_desi.header['R_MIN'] = desi_conf['R_min']
                hdu_desi.header['R_MAX'] = desi_conf['R_max']
                hdu_desi.header['UNITS_W'] = 'Angstrom'
                hdu_desi.header['UNITS_F'] = 'erg/s/cm^2/Hz'
                
                hdu_desi.writeto(desi_out, overwrite=True)
                print(f"Realistic DESI spectrum saved to {desi_out}", flush=True)
            else:
                print("No stars found in DESI fiber aperture.", flush=True)

        # PSF
        sigma_pixels = (obs['fwhm_arcsec'] / 2.355) / pixel_scale_arcsec
        img_smoothed = gaussian_filter(img.arr, sigma=sigma_pixels)
        
        # Save
        if not os.path.exists(paths['output_path']):
            os.makedirs(paths['output_path'], exist_ok=True)
            
        out_file = os.path.join(paths['output_path'], 'euclid_vis_galaxy.fits')
        hdu = fits.PrimaryHDU(img_smoothed)
        hdu.header['OBJECT'] = target_galaxy.name
        hdu.header['REDSHIFT'] = z_obs
        hdu.header['PIXSCALE'] = (pixel_scale_arcsec, 'arcsec/pixel')
        hdu.writeto(out_file, overwrite=True)
        print(f"Done! Saved to {out_file}", flush=True)

if __name__ == "__main__":
    config = load_config()
    generate_euclid_vis_image(config)
