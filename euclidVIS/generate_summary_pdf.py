
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.colors import LogNorm
from astropy.io import fits
import pandas as pd
import yaml

# Try importing illustris_python for SFR
try:
    import illustris_python as il
    HAS_ILLUSTRIS = True
except ImportError:
    HAS_ILLUSTRIS = False
    print("WARNING: illustris_python not found. Ranking by SFR will be skipped.")

# --- Configuration ---
# Hardcoded paths based on user request and config
BASE_OUTPUT_PATH = "/u/mhuertas/data/euclid/tngmocks/sn99"
CATALOG_PATH = os.path.join(BASE_OUTPUT_PATH, "catalog.csv")
EUCLID_DIR = os.path.join(BASE_OUTPUT_PATH, "Euclid_fixed")
DESI_DIR = os.path.join(BASE_OUTPUT_PATH, "DESI_fixed")
TNG_BASE_PATH = "/virgotng/universe/IllustrisTNG/TNG50-1/output"
SNAP_NUM = 99
OUTPUT_PDF = "euclid_desi_summary.pdf"

# Emission Lines to mark (Rest Frame Angstroms)
EMISSION_LINES = {
    r"H$\alpha$": 6562.8,
    r"H$\beta$": 4861.3,
    r"[OIII]": 5006.8,
    r"[OII]": 3727.0, # Doublet blend
    r"[NII]": 6583.0
}

def get_sfrs(subhalo_ids):
    """Fetch SFRs for the given subhalos using illustris_python."""
    if not HAS_ILLUSTRIS:
        return np.zeros(len(subhalo_ids))
    
    print("Loading SFRs from TNG...")
    try:
        # Load all subhalos fields needed
        # Note: This loads the whole catalog subset, might be slow if huge, 
        # but TNG50-1 group cat is manageable.
        # Optimisation: Load specific IDs if possible? 
        # il.groupcat.loadSubhalos returns all unless filtered?
        # Actually loadSubhalos loads everything. 
        # For efficiency, we just load the SFR field.
        fields = ['SubhaloSFR']
        subhalos = il.groupcat.loadSubhalos(TNG_BASE_PATH, SNAP_NUM, fields=fields)
        
        if isinstance(subhalos, dict):
            all_sfrs = subhalos['SubhaloSFR']
        else:
            # Assume it returned the array directly (e.g. valid for single field request)
            # Check if it has the right shape/properties to be sure? 
            # For now assume it is the data.
            print(f"DEBUG: subhalos is not a dict, assuming it is the data array. Shape: {getattr(subhalos, 'shape', 'Unknown')}")
            all_sfrs = subhalos

        # Create a map
        # TNG SFR is usually distinct. 
        # We just need to map them.
        
        # Create a map
        # Ensure keys are ints!
        sfr_map = {int(i): sfr for i, sfr in enumerate(all_sfrs)}
        
        # Extract for our IDs (ensure IDs are ints)
        return np.array([sfr_map.get(int(sid), 0.0) for sid in subhalo_ids])
        
    except Exception as e:
        print(f"Failed to load SFRs: {e}")
        return np.zeros(len(subhalo_ids))

def plot_galaxy(pdf, subhalo_row, euclid_path, desi_path):
    """Plot a single galaxy to the PDF."""
    
    sid = int(subhalo_row['subhalo_id'])
    mass = subhalo_row['stellar_mass']
    z = subhalo_row['redshift']
    sfr = subhalo_row.get('sfr', 0.0) # Handle missing SFR
    
    # Setup Figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(f"Subhalo ID: {sid} | z={z:.3f} | logM*={np.log10(mass):.2f} | SFR={sfr:.2f}", fontsize=14)
    
    # 1. Euclid Image
    img_file = os.path.join(euclid_path, f"euclid_vis_{sid}.fits")
    if os.path.exists(img_file):
        try:
            img_data = fits.getdata(img_file)
            # Use LogNorm for better dynamic range visualization
            norm = LogNorm(vmin=np.percentile(img_data[img_data>0], 1) if np.any(img_data>0) else 1e-5, 
                           vmax=np.max(img_data) if np.any(img_data>0) else 1.0)
            
            im = axes[0].imshow(img_data, origin='lower', cmap='inferno', norm=norm)
            axes[0].set_title("Euclid VIS (Log Scale)")
            plt.colorbar(im, ax=axes[0], fraction=0.046, pad=0.04)
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error loading image:\n{e}", ha='center')
        except Exception as e:
            axes[0].text(0.5, 0.5, f"Error loading image:\n{e}", ha='center')
    else:
        axes[0].text(0.5, 0.5, f"Image not found:\n{img_file}", ha='center', fontsize=8)
        
    # 2. DESI Spectrum
    spec_file = os.path.join(desi_path, f"desi_spectrum_{sid}.fits")
    if os.path.exists(spec_file):
        try:
            with fits.open(spec_file) as hdul:
                data = hdul[1].data
                hdr = hdul[1].header
                wave = data['wavelength']
                flux = data['flux']
                
                # Check for velocity shift flag
                vshift = hdr.get('VELSHIFT', 'Unknown')
                
                axes[1].plot(wave, flux, lw=1, color='k', alpha=0.8)
                axes[1].set_xlabel(r"Wavelength [$\AA$]")
                axes[1].set_ylabel(r"Flux [$erg\ s^{-1}\ cm^{-2}\ Hz^{-1}$]")
                axes[1].set_title(f"DESI Spectrum (VelShift: {vshift})")
                axes[1].grid(True, alpha=0.3)
                
                # Plot Emission Lines
                ylim = axes[1].get_ylim()
                for name, rest_lam in EMISSION_LINES.items():
                    obs_lam = rest_lam * (1 + z)
                    if 3500 < obs_lam < 9900:
                        axes[1].axvline(obs_lam, color='r', linestyle='--', alpha=0.5, linewidth=0.8)
                        axes[1].text(obs_lam, ylim[1]*0.9, name, rotation=90, color='r', fontsize=8, ha='right')

                # Add zoom inset or just limits? 
                # Let's keep full range but adding min/max
                axes[1].set_xlim(3500, 9900)
                
        except Exception as e:
            axes[1].text(0.5, 0.5, f"Error loading spectrum:\n{e}", ha='center')
    else:
        axes[1].text(0.5, 0.5, "Spectrum not found", ha='center')

    pdf.savefig(fig)
    plt.close(fig)


def main():
    print(f"Reading catalog from {CATALOG_PATH}...")
    if not os.path.exists(CATALOG_PATH):
        print("Catalog not found! cannot proceed.")
        return
        
    df = pd.read_csv(CATALOG_PATH)
    print(f"Found {len(df)} entries.")
    
    if len(df) == 0:
        print("Catalog is empty.")
        return

    # Remove duplicates just in case (e.g. restarts or race conditions)
    df = df.drop_duplicates(subset='subhalo_id')

    # Add SFR if possible
    if HAS_ILLUSTRIS:
        df['sfr'] = get_sfrs(df['subhalo_id'].values)
        print(f"Max SFR found: {df['sfr'].max():.2f}")
    else:
        df['sfr'] = 0.0

    # Selection - PURELY RANDOM as requested
    selection = []
    
    # Helper to check if file exists
    def has_image(sid):
        return os.path.exists(os.path.join(EUCLID_DIR, f"euclid_vis_{sid}.fits"))

    print("Selecting random galaxies...")
    
    # Shuffle dataframe
    df_shuffled = df.sample(frac=1.0, random_state=42)
    
    count = 0
    TARGET_COUNT = 30
    
    for _, row in df_shuffled.iterrows():
        if count >= TARGET_COUNT: break
        
        sid = int(row['subhalo_id'])
        if has_image(sid):
            selection.append(row.to_dict())
            count += 1
            
    if count < TARGET_COUNT:
        print(f"WARNING: Could only find {count} valid random galaxies (target: {TARGET_COUNT}).")

    print(f"Generating PDF with {len(selection)} galaxies...")
        
    print(f"Generating PDF with {len(selection)} galaxies...")
    
    with PdfPages(OUTPUT_PDF) as pdf:
        # Title Page
        plt.figure(figsize=(8, 6))
        plt.text(0.5, 0.7, "Euclid VIS & DESI Mock Challenge", fontsize=24, ha='center')
        plt.text(0.5, 0.5, f"Snapshot: {SNAP_NUM}", fontsize=18, ha='center')
        plt.text(0.5, 0.4, f"Total Galaxies Processed: {len(df)}", fontsize=14, ha='center')
        plt.axis('off')
        pdf.savefig()
        plt.close()
        
        # Plots
        for i, galaxy in enumerate(selection):
            print(f"  Plotting {i+1}/{len(selection)}: Subhalo {int(galaxy['subhalo_id'])}")
            plot_galaxy(pdf, galaxy, EUCLID_DIR, DESI_DIR)
            
    print(f"Done! Saved to {OUTPUT_PDF}")

if __name__ == "__main__":
    main()
