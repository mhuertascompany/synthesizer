
import os
import pandas as pd
import yaml
import argparse

def clean_catalog(config_file):
    # Load Configuration
    with open(config_file, 'r') as f:
        config = yaml.safe_load(f)

    # Paths
    paths = config['paths']
    snap_number = config['simulation']['snap_number']
    base_dir = os.path.join(paths['output_path'], f"sn{snap_number}")
    catalog_path = os.path.join(base_dir, "catalog.csv")
    cleaned_catalog_path = os.path.join(base_dir, "catalog_clean.csv")
    
    # Euclid Subdirectory
    euclid_subdir = paths.get('euclid_subdir', 'Euclid')
    euclid_dir = os.path.join(base_dir, euclid_subdir)

    print(f"Cleaning catalog at: {catalog_path}")
    print(f"Checking for images in: {euclid_dir}")

    if not os.path.exists(catalog_path):
        print("Catalog not found!")
        return

    # Load Catalog
    df = pd.read_csv(catalog_path)
    print(f"Original entries: {len(df)}")

    # Remove duplicates (keep last entry as it's likely the most recent run)
    df = df.drop_duplicates(subset='subhalo_id', keep='last')
    print(f"After removing duplicates: {len(df)}")

    # Check existence
    valid_indices = []
    
    # Pre-loading directory listing might be faster than os.path.exists for thousands of files
    # But os.path.exists is safer for race conditions. 
    # Given the scale (potentially thousands), let's use os.path.exists for simplicity first.
    
    files_found = 0
    files_missing = 0
    
    valid_rows = []
    
    for i, row in df.iterrows():
        sid = int(row['subhalo_id'])
        filename = f"euclid_vis_{sid}.fits"
        filepath = os.path.join(euclid_dir, filename)
        
        if os.path.exists(filepath):
            valid_rows.append(row)
            files_found += 1
        else:
            files_missing += 1
            
    # Create new DataFrame
    df_clean = pd.DataFrame(valid_rows)
    
    print(f"Valid entries found: {len(df_clean)}")
    print(f"Entries removed (missing files): {files_missing}")
    
    if len(df_clean) > 0:
        df_clean.to_csv(cleaned_catalog_path, index=False)
        print(f"Clean catalog saved to: {cleaned_catalog_path}")
    else:
        print("WARNING: No valid entries found! Clean catalog not saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean the galaxy catalog.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file")
    args = parser.parse_args()
    
    clean_catalog(args.config)
