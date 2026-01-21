
import illustris_python as il
import numpy as np
import yaml

with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

directory = config['paths']['tng_path']
snap = config['simulation']['snap_number']
h = 0.6774 # TNG50 h

print(f"Loading subhalos for snap {snap}...")
# Corrected field name: SubhaloSFR
fields = ['SubhaloMassType', 'SubhaloSFR', 'SubhaloPos']
subs = il.groupcat.loadSubhalos(directory, snap, fields=fields)

m_star = subs['SubhaloMassType'][:, 4] * 1e10 / h
sfr = subs['SubhaloSFR']

# Filter for low mass (10^9 - 10^10) for faster processing
# and significant SFR (> 0.2 Msun/yr)
mask = (m_star > 1e9) & (m_star < 1e10) & (sfr > 0.2)
indices = np.where(mask)[0]

if len(indices) == 0:
    print("No candidates found with those cuts, relaxing...")
    mask = (m_star > 10**8.5) & (m_star < 1e10) & (sfr > 0.05)
    indices = np.where(mask)[0]

# Sort by sSFR to find the most "active" galaxies
ssfr = sfr[indices] / m_star[indices]
sorted_idx = indices[np.argsort(ssfr)[::-1]]

print("\nTop Star-Forming Candidates (Mass 10^9 - 10^10):")
print(f"{'ID':<10} {'M_star':<15} {'SFR':<10} {'sSFR':<12}")
for i in sorted_idx[:15]:
    print(f"{i:<10} {m_star[i]:<15.2e} {sfr[i]:<10.2f} {sfr[i]/m_star[i]:<12.2e}")
