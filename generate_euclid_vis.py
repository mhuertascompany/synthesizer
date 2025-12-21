import numpy as np
import matplotlib.pyplot as plt
from unyt import Myr, kpc, arcsec, Angstrom, Msun, pc
from synthesizer.load_data.load_illustris import load_IllustrisTNG
from synthesizer.grid import Grid
from synthesizer.imaging import Image
from synthesizer.instruments.filters import FilterCollection
from synthesizer.emission_models.models import AttenuatedEmission, ReprocessedEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.kernel_functions import Kernel
from scipy.ndimage import gaussian_filter
from astropy.io import fits
from astropy.cosmology import Planck15 as cosmo
import astropy.units as u

# Define paths - USER MUST VERIFY THESE
TNG_PATH = "/path/to/tng/data"  # e.g., /virgotng/universe/IllustrisTNG/TNG50-1/output
GRID_DIR = "/path/to/grids"     # e.g., /home/user/synthesizer_data/grids
# Standard grid with nebular emission (required for ReprocessedEmission)
GRID_NAME = "bc03-2016-Miles_chabrier-0.1,100_cloudy-c23.01-sps"

def generate_euclid_vis_image():
    print("Loading TNG data...")
    # Load TNG data
    # We assume the user wants the snapshot specified here.
    # If z=0 (snap 99), we might need to place it at a fictitious redshift for "observation".
    galaxy = load_IllustrisTNG(
        directory=TNG_PATH,
        snap_number=99,  # z=0
        stellar_mass_limit=1e10, 
    )

    if isinstance(galaxy, list):
        target_galaxy = galaxy[0]
    else:
        target_galaxy = galaxy
    
    print(f"Selected galaxy: {target_galaxy.name}")
    try:
        print(f"Stellar Mass: {target_galaxy.stellar_mass}")
    except:
        pass

    # Redshift and Distance Handling
    z_obs = target_galaxy.redshift
    print(f"Snapshot Redshift: {z_obs:.4f}")
    
    # If z is effectively 0, we cannot "observe" it at that redshift (distance=0).
    # We will place it at z=0.05 for demonstration if z < 0.001, or warn.
    if z_obs < 0.001:
        print("WARNING: Galaxy is at z ~ 0. Placing it at z = 0.05 for observation purposes.")
        z_obs = 0.05
    
    # Calculate Distances
    d_lum = cosmo.luminosity_distance(z_obs).to(u.cm).value # Luminosity distance in cm
    d_ang = cosmo.angular_diameter_distance(z_obs).to(u.kpc).value # Angular diameter distance in kpc
    scale_kpc_per_arcsec = d_ang * np.pi / 180 / 3600 * 1000 # kpc per arcsec (approx)
    # Actually astropy has kpc_proper_per_arcmin
    scale_kpc_per_arcsec = cosmo.kpc_proper_per_arcmin(z_obs).value / 60.0
    
    print(f"Observation Redshift: {z_obs}")
    print(f"Scale: {scale_kpc_per_arcsec:.3f} kpc/arcsec")

    # Load Spectral Grid
    print(f"Loading spectral grid: {GRID_NAME}...")
    grid = Grid(GRID_NAME, grid_dir=GRID_DIR)

    # Load Euclid VIS Filter
    print("Loading Euclid VIS filter...")
    try:
        filters = FilterCollection(filter_codes=["Euclid/VIS"], new_lam=grid.lam)
        vis_filter = filters[0]
    except Exception as e:
        print(f"Could not load Euclid/VIS from SVO: {e}")
        print("Using a top-hat approximation (5500-9000 A)...")
        filters = FilterCollection(
            tophat_dict={"Euclid_VIS_approx": {"lam_min": 5500, "lam_max": 9000}},
            new_lam=grid.lam
        )
        vis_filter = filters[0]

    # Define Dust Model
    print("Configuring dust model...")
    dust_curve = Calzetti2000()
    reprocessed_model = ReprocessedEmission(grid=grid)
    model = AttenuatedEmission(
        grid=grid,
        dust_curve=dust_curve,
        apply_to=reprocessed_model,
        emitter="stellar"
    )

    # Calculate Physical Optical Depth (tau_v)
    if target_galaxy.gas is not None:
        print("Calculating physical line-of-sight optical depth from gas...")
        
        # 1. Get Kernel
        # We use a cubic spline kernel, standard in SPH
        kernel_obj = Kernel(name="cubic")
        kernel = kernel_obj.get_kernel()
        
        # 2. Define Kappa (Dust Opacity)
        # Kappa = dust_to_gas_ratio * mass_extinction_coefficient
        # Synthesizer expects kappa in units of Msun / pc^2 (inverse surface density).
        # We use a value of 20 pc^2 / Msun which is typical for these simulations.
        kappa = 20.0 
        
        # 3. Calculate tau_v
        # We need to ensure gas has 'dust_masses'. 
        # If not, we calculate them from metallicity.
        if not hasattr(target_galaxy.gas, 'dust_masses'):
            print("Calculating dust masses from metallicity (D/M = 0.4 * Z)...")
            # Simple assumption: Dust-to-Metal ratio = 0.4 (approx MW)
            dust_to_metal = 0.4
            target_galaxy.gas.dust_masses = target_galaxy.gas.masses * target_galaxy.gas.metallicities * dust_to_metal
            
        tau_v = target_galaxy.get_stellar_los_tau_v(
            kappa=kappa,
            kernel=kernel,
        )
        print(f"Mean tau_v: {np.mean(tau_v):.3f}")
        print(f"Max tau_v: {np.max(tau_v):.3f}")
        
    else:
        print("No gas found. Cannot calculate physical dust. Using tau_v = 0.")
        tau_v = 0.0

    # Calculate Particle Spectra
    print("Calculating particle spectra...")
    spectra_dict = target_galaxy.stars.get_particle_spectra(
        model, 
        tau_v=tau_v
    )
    
    spec_key = "attenuated" 
    if spec_key not in spectra_dict:
        spec_key = list(spectra_dict.keys())[0]
    
    particle_spectra = spectra_dict[spec_key] # (n_particles, n_lam)

    # Calculate Photometry (Flux)
    print("Calculating photometry...")
    lam = grid.lam # Angstrom
    trans = vis_filter.transmission
    
    # Integrate to get Luminosity in band (erg/s)
    # To get observed flux:
    # 1. Redshift the spectrum: lam_obs = lam_rest * (1+z)
    # 2. Apply filter in observed frame.
    
    trans_effective = np.interp(lam * (1 + z_obs), vis_filter.lam, vis_filter.transmission, left=0, right=0)
    
    # Integrate L_rest * T_effective
    luminosity_in_band = np.trapz(particle_spectra * trans_effective, x=lam, axis=1)
    
    # Convert to Flux (erg/s/cm^2)
    flux_in_band = luminosity_in_band / (4 * np.pi * d_lum**2)
    
    # Generate Image
    print("Generating image...")
    # Pixel scale
    pixel_scale_arcsec = 0.1 # Euclid VIS
    pixel_scale_kpc = pixel_scale_arcsec * scale_kpc_per_arcsec
    
    # FOV
    # Let's define a FOV that covers the galaxy. 
    # TNG50 massive galaxies can be 50-100 kpc.
    fov_kpc = 100.0
    fov_arcsec = fov_kpc / scale_kpc_per_arcsec
    resolution = int(fov_arcsec / pixel_scale_arcsec)
    
    print(f"FOV: {fov_kpc:.1f} kpc ({fov_arcsec:.1f} arcsec)")
    print(f"Resolution: {resolution} x {resolution} pixels")
    
    coords = target_galaxy.stars.coordinates
    if target_galaxy.stars.centre is not None:
        coords -= target_galaxy.stars.centre
    
    # Select particles in FOV
    mask = (np.abs(coords[:, 0]) < fov_kpc/2 * kpc) & (np.abs(coords[:, 1]) < fov_kpc/2 * kpc)
    
    x = coords[mask, 0].to(kpc).value
    y = coords[mask, 1].to(kpc).value
    weights = flux_in_band[mask]
    
    hist, _, _ = np.histogram2d(
        x, y, 
        bins=resolution, 
        range=[[-fov_kpc/2, fov_kpc/2], [-fov_kpc/2, fov_kpc/2]],
        weights=weights
    )
    
    img = Image(img=hist, fov=fov_kpc*kpc, resolution=resolution)

    # Apply PSF
    print("Applying PSF...")
    # Euclid VIS PSF FWHM ~ 0.16 arcsec
    # Sigma = FWHM / 2.355
    fwhm_arcsec = 0.16
    sigma_arcsec = fwhm_arcsec / 2.355
    sigma_pixels = sigma_arcsec / pixel_scale_arcsec
    
    print(f"PSF Sigma: {sigma_pixels:.2f} pixels")
    img_smoothed = gaussian_filter(img.img, sigma=sigma_pixels)
    
    # Save as FITS
    print("Saving FITS image...")
    hdu = fits.PrimaryHDU(img_smoothed)
    hdu.header['TELESCOP'] = 'Euclid'
    hdu.header['INSTRUME'] = 'VIS'
    hdu.header['OBJECT'] = target_galaxy.name
    hdu.header['REDSHIFT'] = z_obs
    hdu.header['PIXSCALE'] = (pixel_scale_arcsec, 'arcsec/pixel')
    hdu.header['UNITS'] = 'erg/s/cm^2'
    hdu.header['FOV_KPC'] = fov_kpc
    
    hdu.writeto('euclid_vis_galaxy.fits', overwrite=True)
    print("Done! Saved to euclid_vis_galaxy.fits")

if __name__ == "__main__":
    generate_euclid_vis_image()
