
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
        fields = ['SubhaloStarFormationRate']
        subhalos = il.groupcat.loadSubhalos(TNG_BASE_PATH, SNAP_NUM, fields=fields)
        
        all_sfrs = subhalos['SubhaloStarFormationRate'] # In Msun/yr ?? No, check units.
        # TNG SFR is usually distinct. 
        # We just need to map them.
        
        # Create a map
        sfr_map = {i: sfr for i, sfr in enumerate(all_sfrs)}
        
        # Extract for our IDs
        return np.array([sfr_map.get(sid, 0.0) for sid in subhalo_ids])
        
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
    else:
        axes[0].text(0.5, 0.5, "Image not found", ha='center')
        
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
    else:
        df['sfr'] = 0.0

    # Sort
    df_sorted_mass = df.sort_values(by='stellar_mass', ascending=False)
    df_sorted_sfr = df.sort_values(by='sfr', ascending=False)
    
    # Selection
    selection = []
    
    # Top 10 Mass
    selection.extend(df_sorted_mass.head(10).to_dict('records'))
    
    # Top 10 SFR (if meaningful)
    if HAS_ILLUSTRIS and df['sfr'].max() > 0:
        # Avoid duplicates
        top_sfr = df_sorted_sfr.head(10)
        existing_ids = {item['subhalo_id'] for item in selection}
        for _, row in top_sfr.iterrows():
            if row['subhalo_id'] not in existing_ids:
                selection.append(row.to_dict())
                
    # Random 10
    # Avoid duplicates
    existing_ids = {item['subhalo_id'] for item in selection}
    remaining = df[~df['subhalo_id'].isin(existing_ids)]
    if len(remaining) > 0:
        n_random = min(10, len(remaining))
        random_sample = remaining.sample(n=n_random, random_state=42)
        selection.extend(random_sample.to_dict('records'))
        
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
