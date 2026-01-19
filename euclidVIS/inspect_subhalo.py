import argparse
import numpy as np
import illustris_python as il
from unyt import Msun, kpc
from generate_euclid_vis import load_IllustrisTNG_fixed
import yaml
import os

def load_simple_config():
    config_path = os.path.join(os.path.dirname(__file__), 'config.yaml')
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def inspect_subhalo(subhalo_id):
    config = load_simple_config()
    paths = config['paths']
    snap = config['simulation']['snap_number']
    
    print(f"Inspecting Subhalo {subhalo_id} from Snap {snap}...")
    
    # 1. Load Group Catalog Properties (Global Truth)
    print("Loading Group Catalog info...")
    basePath = paths['tng_path']
    fields = ['SubhaloMass', 'SubhaloMassType', 'SubhaloSFR', 'SubhaloGasMetallicity', 'SubhaloPos', 'SubhaloHalfmassRadType']
    subhalos = il.groupcat.loadSubhalos(basePath, snap, fields=fields)
    
    # Check if ID is valid
    if subhalo_id >= len(subhalos['SubhaloMass']):
        print(f"Error: Subhalo ID {subhalo_id} out of bounds.")
        return

    sfr = subhalos['SubhaloSFR'][subhalo_id]
    mass_type = subhalos['SubhaloMassType'][subhalo_id] * 1e10 / 0.6774 # h=0.6774 approx for TNG
    gas_mass = mass_type[0]
    stellar_mass = mass_type[4]
    half_light_rad_stars = subhalos['SubhaloHalfmassRadType'][subhalo_id, 4] # ckpc/h
    
    print("--- Catalog Properties ---")
    print(f"Stellar Mass: {stellar_mass:.2e} Msun")
    print(f"Gas Mass:     {gas_mass:.2e} Msun")
    print(f"SFR:          {sfr:.4f} Msun/yr")
    print(f"sSFR:         {sfr/stellar_mass:.2e} yr^-1" if stellar_mass > 0 else "sSFR: N/A")
    
    if sfr == 0:
        print("WARNING: Galaxy is quenched (SFR=0). No emission lines expected!")
    
    # 2. Check Particle Data (Resolution / distribution)
    print("\n--- Particle Data ---")
    # We use our fixed loader to see what the script actually "sees"
    galaxies, _ = load_IllustrisTNG_fixed(
        directory=basePath, 
        snap_number=snap, 
        stellar_mass_limit=0, 
        subhalo_ids=[subhalo_id], 
        verbose=False
    )
    
    if not galaxies:
        print("Error: Could not load galaxy particles.")
        return

    gal = galaxies[0]
    
    if gal.stars is not None:
        print(f"Loaded {len(gal.stars.initial_masses)} star particles.")
    else:
        print("No stars loaded.")

    if gal.gas is not None:
        print(f"Loaded {len(gal.gas.masses)} gas particles.")
        print(f"Gas Mass (Sum): {np.sum(gal.gas.masses):.2e}")
        
        # Check starforming gas
        if hasattr(gal.gas, 'star_forming'):
            sf_gas = gal.gas.masses[gal.gas.star_forming]
            print(f"Star-Forming Gas Mass: {np.sum(sf_gas):.2e} Msun")
            if np.sum(sf_gas) == 0:
                print("WARNING: No gas particles are flagged as star-forming!")
        
        # Check coordinates (is gas central or extended?)
        if len(gal.gas.coordinates) > 0:
             # coordinates are relative to centre in the loader
             r = np.sqrt(np.sum(gal.gas.coordinates**2, axis=1)).to('kpc')
             print(f"Gas Radius (Mean): {np.mean(r):.2f}")
             print(f"Gas Radius (Min):  {np.min(r):.2f}")
             print(f"Gas Radius (Max):  {np.max(r):.2f}")
             
             print("Gas Radial Distribution (percentiles):")
             print(f"  10%: {np.percentile(r, 10):.2f}")
             print(f"  50%: {np.percentile(r, 50):.2f}")
             print(f"  90%: {np.percentile(r, 90):.2f}")

    else:
        print("WARNING: No gas particles loaded via generate_euclid_vis loader.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("subhalo_id", type=int, help="ID of the subhalo to inspect")
    args = parser.parse_args()
    
    inspect_subhalo(args.subhalo_id)
