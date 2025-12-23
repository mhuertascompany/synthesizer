import numpy as np
import matplotlib.pyplot as plt
import os
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
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

# Define paths - USER MUST VERIFY THESE
TNG_PATH = "/virgotng/universe/IllustrisTNG/TNG50-1/output"  # e.g., /virgotng/universe/IllustrisTNG/TNG50-1/output
GRID_DIR = "/u/mhuertas/data/synthesizer"     # e.g., /home/user/synthesizer_data/grids
# Standard grid with nebular emission (required for ReprocessedEmission)
GRID_NAME = "bc03-2016-Miles_chabrier-0.1,100_cloudy-c23.01-sps"
OUTPUT_PATH="/u/mhuertas/data/euclid/tngmocks"

# Optimization Parameters
PARTICLE_LIMIT = 100000 # 100k is safe and provides good quality

def generate_euclid_vis_image():
    print("Loading TNG data...", flush=True)
    # Load TNG data
    # We load just one massive galaxy to accelerate debugging.
    # You can find subhalo IDs in the TNG group catalog.
    # For TNG50-1 snap 99, subhalo 0 is usually the most massive.
    galaxies, subhalo_mask = load_IllustrisTNG(
        directory=TNG_PATH,
        snap_number=99,  # z=0
        subhalo_ids=[0], # Load just the first subhalo
    )

    if len(galaxies) == 0:
        print("No galaxies found!", flush=True)
        return

    target_galaxy = galaxies[0]
    
    print(f"Selected galaxy: {target_galaxy.name}", flush=True)
    try:
        print(f"Stellar Mass: {target_galaxy.stellar_mass}", flush=True)
    except:
        pass

    # Redshift and Distance Handling
    z_obs = target_galaxy.redshift
    print(f"Snapshot Redshift: {z_obs:.4f}", flush=True)
    
    # If z is effectively 0, we cannot "observe" it at that redshift (distance=0).
    # We will place it at z=0.05 for demonstration if z < 0.001, or warn.
    if z_obs < 0.001:
        print("WARNING: Galaxy is at z ~ 0. Placing it at z = 0.05 for observation purposes.", flush=True)
        z_obs = 0.05
    
    # Calculate Distances
    d_lum = cosmo.luminosity_distance(z_obs).to(u.cm).value # Luminosity distance in cm
    d_ang = cosmo.angular_diameter_distance(z_obs).to(u.kpc).value # Angular diameter distance in kpc
    
    # Scale: kpc per arcsec
    scale_kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_obs).value / 60.0
    
    print(f"Observation Redshift: {z_obs}", flush=True)
    print(f"Scale: {scale_kpc_per_arcsec:.3f} kpc/arcsec", flush=True)

    # Load Spectral Grid
    print(f"Loading spectral grid: {GRID_NAME}...", flush=True)
    grid = Grid(GRID_NAME, grid_dir=GRID_DIR)

    # Load Euclid VIS Filter
    print("Loading Euclid VIS filter...", flush=True)
    vis_filter = None
    
    # 1. Try Local ASCII File (User provided)
    local_filter_path = os.path.join(GRID_DIR, 'Euclid_VIS.vis.dat')
    if os.path.exists(local_filter_path):
        print(f"Found local filter at {local_filter_path}. Loading...", flush=True)
        try:
            data = np.loadtxt(local_filter_path)
            lam_local = data[:, 0] * Angstrom
            trans_local = data[:, 1]
            
            vis_filter = Filter(
                "Euclid/VIS_local",
                transmission=trans_local,
                new_lam=lam_local
            )
            vis_filter._interpolate_wavelength(grid.lam)
            print("Successfully loaded local Euclid_VIS.vis.dat.", flush=True)
        except Exception as e:
            print(f"Failed to load local file: {e}", flush=True)

    # 2. Try SVO (Euclid/VIS.vis)
    if vis_filter is None:
        try:
            print("Attempting to load Euclid/VIS.vis from SVO...", flush=True)
            filters = FilterCollection(
                filter_codes=["Euclid/VIS.vis"], 
                new_lam=grid.lam
            )
            vis_filter = filters[0]
            print("Successfully loaded Euclid/VIS.vis from SVO.", flush=True)
        except Exception as e:
            print(f"Could not load Euclid/VIS.vis from SVO: {e}", flush=True)

    # 3. Fallback to Manual Definition
    if vis_filter is None:
        print("Falling back to manual definition...", flush=True)
        vis_lam = np.linspace(5000, 9500, 1000) * Angstrom
        vis_trans = np.zeros_like(vis_lam)
        mask = (vis_lam.value >= 5500) & (vis_lam.value <= 9000)
        vis_trans[mask] = 1.0 
        
        try:
            vis_filter = Filter(
                "Euclid/VIS_manual",
                transmission=vis_trans,
                new_lam=vis_lam
            )
            vis_filter._interpolate_wavelength(grid.lam)
        except Exception as e2:
            print(f"Could not create manual filter: {e2}", flush=True)
            print("Using a top-hat approximation (5500-9000 A)...", flush=True)
            filters = FilterCollection(
                tophat_dict={"Euclid_VIS_approx": {"lam_min": 5500 * Angstrom, "lam_max": 9000 * Angstrom}},
                new_lam=grid.lam
            )
            vis_filter = filters[0]

    # Define Dust Model
    print("Configuring dust model...", flush=True)
    dust_curve = Calzetti2000()
    reprocessed_model = ReprocessedEmission(grid=grid)
    model = AttenuatedEmission(
        grid=grid,
        dust_curve=dust_curve,
        apply_to=reprocessed_model,
        emitter="stellar"
    )
    
    # FOV Definition
    fov_kpc = 100.0
    fov_arcsec = fov_kpc / scale_kpc_per_arcsec
    pixel_scale_arcsec = 0.1 # Euclid VIS
    resolution = int(fov_arcsec / pixel_scale_arcsec)
    
    # Calculate Physical Optical Depth (tau_v)
    if target_galaxy.gas is not None:
        print("Calculating physical line-of-sight optical depth from gas...", flush=True)
        
        # 1. Get Kernel
        print("Generating kernel...", flush=True)
        kernel_obj = Kernel(name="cubic", binsize=1000)
        kernel = kernel_obj.get_kernel()
        print("Kernel generated.", flush=True)
        
        # 2. Define Kappa (Dust Opacity)
        kappa = 20.0 
        
        # 3. Calculate tau_v
        if not hasattr(target_galaxy.gas, 'dust_masses'):
            print("Calculating dust masses from metallicity (D/M = 0.4 * Z)...", flush=True)
            dust_to_metal = 0.4
            target_galaxy.gas.dust_masses = target_galaxy.gas.masses * target_galaxy.gas.metallicities * dust_to_metal
            print("Dust masses calculated.", flush=True)

        # OPTIMIZATION: Filter particles to FOV and DOWNSAMPLE
        print("Optimizing: Filtering and Downsampling particles...", flush=True)
        
        # FOV limits
        fov_limit = fov_kpc / 2 * kpc
        gas_buffer = 50 * kpc
        
        # --- STAR FILTERING ---
        star_coords = target_galaxy.stars.coordinates
        if target_galaxy.stars.centre is not None:
            star_coords -= target_galaxy.stars.centre
            
        star_fov_mask = (np.abs(star_coords[:, 0]) < fov_limit) & \
                        (np.abs(star_coords[:, 1]) < fov_limit)
        
        star_indices = np.where(star_fov_mask)[0]
        n_stars_fov = len(star_indices)
        print(f"Stars in FOV: {n_stars_fov} / {target_galaxy.stars.nparticles}", flush=True)

        # Downsample Stars
        star_weight_scale = 1.0
        if n_stars_fov > PARTICLE_LIMIT:
            print(f"Downsampling stars to {PARTICLE_LIMIT}...", flush=True)
            sampled_indices = np.random.choice(star_indices, PARTICLE_LIMIT, replace=False)
            star_weight_scale = n_stars_fov / PARTICLE_LIMIT
        else:
            sampled_indices = star_indices
            
        # Create Optimized Stars Object
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

        # --- GAS FILTERING ---
        gas_coords = target_galaxy.gas.coordinates
        if target_galaxy.gas.centre is not None:
            gas_coords -= target_galaxy.gas.centre
            
        gas_limit = fov_limit + gas_buffer
        gas_fov_mask = (np.abs(gas_coords[:, 0]) < gas_limit) & \
                       (np.abs(gas_coords[:, 1]) < gas_limit)
                       
        gas_indices = np.where(gas_fov_mask)[0]
        n_gas_fov = len(gas_indices)
        print(f"Gas in FOV+Buffer: {n_gas_fov} / {target_galaxy.gas.nparticles}", flush=True)

        # Downsample Gas
        if n_gas_fov > PARTICLE_LIMIT:
            print(f"Downsampling gas to {PARTICLE_LIMIT}...", flush=True)
            sampled_gas_indices = np.random.choice(gas_indices, PARTICLE_LIMIT, replace=False)
        else:
            sampled_gas_indices = gas_indices

        # Create Optimized Gas Object
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
        
        # Temporarily replace galaxy components for tau_v calculation
        original_stars = target_galaxy.stars
        original_gas = target_galaxy.gas
        target_galaxy.stars = opt_stars
        target_galaxy.gas = opt_gas
        
        # Calculate tau_v
        print("Starting get_stellar_los_tau_v calculation (this should be fast now)...", flush=True)
        tau_v_opt = target_galaxy.get_stellar_los_tau_v(
            kappa=kappa,
            kernel=kernel,
            nthreads=-1 
        )
        print("get_stellar_los_tau_v calculation complete.", flush=True)
        
        # Calculate Particle Spectra for Optimized Stars
        print(f"Calculating particle spectra for {opt_stars.nparticles} particles (nthreads=8)...", flush=True)
        spectra_dict = target_galaxy.stars.get_particle_spectra(
            model, 
            tau_v=tau_v_opt,
            nthreads=8
        )
        
        # Restore original galaxy components
        target_galaxy.stars = original_stars
        target_galaxy.gas = original_gas
        
        # Handle case where get_particle_spectra returns a single Sed or a dict
        if isinstance(spectra_dict, dict):
            spec_key = "attenuated" 
            if spec_key not in spectra_dict:
                spec_key = list(spectra_dict.keys())[0]
            particle_spectra = spectra_dict[spec_key]
        else:
            particle_spectra = spectra_dict # It's already an Sed object

        # Calculate Photometry (Flux)
        print("Calculating photometry (memory-efficient)...", flush=True)
        # We calculate photometry in the rest-frame and then scale to flux
        # This avoids creating a huge fnu array for 1M particles
        
        # 1. Get rest-frame frequency and luminosity
        # particle_spectra.lnu is (n_particles, n_lam)
        nu_rest = particle_spectra.nu.to('Hz').value
        lnu_rest = particle_spectra.lnu.to('erg/s/Hz').value
        
        # 2. Interpolate filter transmission to rest-frame
        # T_rest(nu_rest) = T_obs(nu_rest / (1+z))
        # Or T_rest(lam_rest) = T_obs(lam_rest * (1+z))
        t_rest = np.interp(
            particle_spectra.lam.to(Angstrom).value * (1 + z_obs),
            vis_filter.lam.to(Angstrom).value,
            vis_filter.transmission,
            left=0, right=0
        )
        
        # 3. Integrate L_nu * T over nu_rest
        # Note: nu_rest is descending, so we take absolute value of trapezoid
        luminosity_in_band = np.abs(np.trapezoid(
            lnu_rest * t_rest,
            x=nu_rest,
            axis=-1
        ))
        
        # 4. Convert to Flux (erg/s/cm^2)
        # F = L / (4 * pi * d_lum**2)
        # Note: d_lum is already in cm from line 64
        flux_in_band = luminosity_in_band / (4 * np.pi * d_lum**2)
        
        # SCALE FLUX to account for downsampling
        flux_in_band *= star_weight_scale
        
        # Use opt_stars coordinates for imaging
        coords_for_img = opt_stars.coordinates
        if opt_stars.centre is not None:
            coords_for_img -= opt_stars.centre
            
    else:
        print("No gas found. Using tau_v = 0 and full star population (if small) or downsampling.", flush=True)
        # Handle case with no gas (similar downsampling if needed)
        # For brevity, assuming gas exists in TNG50 subhalo 0.
        return

    # Generate Image
    print("Generating image...", flush=True)
    print(f"FOV: {fov_kpc:.1f} kpc ({fov_arcsec:.1f} arcsec)")
    print(f"Resolution: {resolution} x {resolution} pixels")
    
    x = coords_for_img[:, 0].to(kpc).value
    y = coords_for_img[:, 1].to(kpc).value
    weights = flux_in_band
    
    hist, _, _ = np.histogram2d(
        x, y, 
        bins=resolution, 
        range=[[-fov_kpc/2, fov_kpc/2], [-fov_kpc/2, fov_kpc/2]],
        weights=weights
    )
    
    # Create Image object
    # Note: synthesizer.Image expects resolution to be the pixel size with units
    pixel_size = (fov_kpc / resolution) * kpc
    img = Image(img=hist, fov=fov_kpc*kpc, resolution=pixel_size)

    # Apply PSF
    print("Applying PSF...", flush=True)
    fwhm_arcsec = 0.16
    sigma_arcsec = fwhm_arcsec / 2.355
    sigma_pixels = sigma_arcsec / pixel_scale_arcsec
    
    print(f"PSF Sigma: {sigma_pixels:.2f} pixels", flush=True)
    img_smoothed = gaussian_filter(img.arr, sigma=sigma_pixels)
    
    # Save as FITS
    print("Saving FITS image...", flush=True)
    hdu = fits.PrimaryHDU(img_smoothed)
    hdu.header['TELESCOP'] = 'Euclid'
    hdu.header['INSTRUME'] = 'VIS'
    hdu.header['OBJECT'] = target_galaxy.name
    hdu.header['REDSHIFT'] = z_obs
    hdu.header['PIXSCALE'] = (pixel_scale_arcsec, 'arcsec/pixel')
    hdu.header['UNITS'] = 'erg/s/cm^2'
    hdu.header['FOV_KPC'] = fov_kpc
    hdu.header['DOWNSAMP'] = (PARTICLE_LIMIT, 'Max particles sampled')
    hdu.header['WSCALE'] = (star_weight_scale, 'Weight scale factor')
    
    if not os.path.exists(OUTPUT_PATH):
        os.makedirs(OUTPUT_PATH, exist_ok=True)
    
    hdu.writeto(os.path.join(OUTPUT_PATH, 'euclid_vis_galaxy.fits'), overwrite=True)
    print(f"Done! Saved to {os.path.join(OUTPUT_PATH, 'euclid_vis_galaxy.fits')}", flush=True)

if __name__ == "__main__":
    generate_euclid_vis_image()
