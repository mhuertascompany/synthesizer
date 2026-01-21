
import numpy as np
from unyt import km, s, Msun, Mpc, Angstrom
from synthesizer.particle.stars import Stars
from synthesizer.grid import Grid
from synthesizer.emission_models import NebularEmission

def test_velocity_projection():
    # 1. Setup a simple grid
    print("Loading Grid...")
    grid = Grid("test_grid") # Assuming test_grid exists or this will fail. 
    # If test_grid doesn't exist, we might need another one. 
    # However, for checking C++ logic, any grid works.
    
    # 2. Setup Emission Model
    model = NebularEmission(grid, vel_shift=True)
    
    # 3. Create dummy stars with known properties
    # We use a single particle to makes things obvious
    n_stars = 1
    coordinates = np.zeros((n_stars, 3)) * Mpc
    masses = np.ones(n_stars) * 1e6 * Msun
    ages = np.array([10.]) * 1e6 # 10 Myr
    metallicities = np.array([0.01])
    
    # CASE A: Velocity strictly in X direction (should have NO Doppler shift if Z is LOS)
    vel_x = np.array([[1000.0, 0.0, 0.0]]) * km / s
    
    # CASE B: Velocity strictly in Z direction (should have Doppler shift)
    vel_z = np.array([[0.0, 0.0, 1000.0]]) * km / s
    
    print("\n--- TEST CASE A: Velocity in X (1000 km/s) ---")
    stars_x = Stars(
        coordinates=coordinates,
        velocities=vel_x,
        masses=masses,
        redshift=0.0,
        softening_lengths=0.0*Mpc,
        nparticles=n_stars,
        centre=coordinates[0]
    )
    # Mocking attributes typically loaded
    stars_x.p_initial_mass = masses
    stars_x.p_metallicity = metallicities
    stars_x.p_age = ages
    
    spectra_x = stars_x.get_spectra(model, vel_shift=True)['nebular']
    
    print("\n--- TEST CASE B: Velocity in Z (1000 km/s) ---")
    stars_z = Stars(
        coordinates=coordinates,
        velocities=vel_z,
        masses=masses,
        redshift=0.0,
        softening_lengths=0.0*Mpc,
        nparticles=n_stars,
        centre=coordinates[0]
    )
    # Mocking attributes
    stars_z.p_initial_mass = masses
    stars_z.p_metallicity = metallicities
    stars_z.p_age = ages
    
    spectra_z = stars_z.get_spectra(model, vel_shift=True)['nebular']
    
    # 4. Compare Peak Positions
    # Find peak wavelength
    wavs = spectra_x.lam
    
    peak_idx_x = np.argmax(spectra_x.lnu)
    peak_wav_x = wavs[peak_idx_x]
    
    peak_idx_z = np.argmax(spectra_z.lnu)
    peak_wav_z = wavs[peak_idx_z]
    
    print(f"\nResults:")
    print(f"Peak Wavelength (X-velocity): {peak_wav_x}")
    print(f"Peak Wavelength (Z-velocity): {peak_wav_z}")
    
    # Expected shift: 1000 km/s / c * lambda
    # c ~ 3e5 km/s. shift ~ 1/300 ~ 0.3%
    # For H-alpha (6563), shift is ~20 Angstroms.
    
    diff = peak_wav_z - peak_wav_x
    print(f"Shift (Z - X): {diff}")
    
    if diff > 1.0 * Angstrom:
        print("SUCCESS: Z-velocity caused a redshift compared to X-velocity.")
        print("Likely Conclusion: Library correctly projects 3D velocity onto Z-axis (LOS).")
    elif diff < -1.0 * Angstrom:
         print("SUCCESS: Z-velocity caused a blueshift compared to X-velocity.")
         print("Likely Conclusion: Library correctly projects 3D velocity onto Z-axis (LOS).")
    else:
        print("FAILURE/AMBIGUOUS: No significant shift detected.")
        print("Possible reasons: \n1. Library expects pre-rotated 1D LOS velocities.\n2. Grid resolution too low.\n3. Bug in test.")

    # 5. Check "My Fear" - Flattening
    # If 3D array is flattened:
    # Particle 0 (X-vel case): [1000, 0, 0]. Flattened: [1000, 0, 0]
    # If code iterates 0..N-1 (N=1), it picks index 0 -> 1000.
    # So X-velocity case WOULD show a shift if it was blindly flattening!
    
    # Particle 0 (Z-vel case): [0, 0, 1000]. Flattened: [0, 0, 1000]
    # If code iterates 0..N-1, it picks index 0 -> 0.
    # So Z-velocity case WOULD NOT show a shift if it was blindly flattening.
    
    # So if X shows shift and Z does not -> It is blindly flattening/indexing.
    # If Z shows shift and X does not -> It is correctly projecting.
    

if __name__ == "__main__":
    try:
        test_velocity_projection()
    except Exception as e:
        print(f"Test crashed: {e}")
