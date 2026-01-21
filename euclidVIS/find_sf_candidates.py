
import illustris_python as il
import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

directory = config['paths']['tng_path']
snap = config['simulation']['snap_number']
h = 0.6774 # TNG50 h

print(f"Loading subhalos for snap {snap}...")
fields = ['SubhaloMassType', 'SubhaloStarFormationRate', 'SubhaloPos']
subs = il.groupcat.loadSubhalos(directory, snap, fields=fields)

m_star = subs['SubhaloMassType'][:, 4] * 1e10 / h
sfr = subs['SubhaloStarFormationRate']

# Filter for low mass (10^8.5 - 10^10) to keep it fast
# and high SFR (> 0.5 Msun/yr is high for these masses)
mask = (m_star > 10**8.5) & (m_star < 10**10) & (sfr > 0.5)
indices = np.where(mask)[0]

if len(indices) == 0:
    print("No candidates found with those exact cuts, relaxation SFR cut...")
    mask = (m_star > 10**8.5) & (m_star < 10**10) & (sfr > 0.1)
    indices = np.where(mask)[0]

# Sort by sSFR (SFR / M_star) to find the "bluest"
ssfr = sfr[indices] / m_star[indices]
sorted_idx = indices[np.argsort(ssfr)[::-1]]

print("\nTop 10 High-sSFR Candidates (Low Mass):")
print(f"{'ID':<10} {'M_star [Msun]':<15} {'SFR [Msun/yr]':<15} {'sSFR [yr^-1]':<15}")
for i in sorted_idx[:10]:
    print(f"{i:<10} {m_star[i]:<15.2e} {sfr[i]:<15.2f} {sfr[i]/m_star[i]:<15.2e}")
