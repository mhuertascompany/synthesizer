from synthesizer.grid import Grid
import yaml
import os

# Load config to get paths
with open('config.yaml', 'r') as f:
    config = yaml.safe_load(f)

paths = config['paths']
sim = config['simulation']

grid_path = os.path.join(paths['grid_dir'], sim['grid_name'] + '.h5')
print(f"Inspecting grid: {grid_path}")

grid = Grid(sim['grid_name'], grid_dir=paths['grid_dir'])

print(f"Grid axes: {grid.axes}")
print(f"Grid extract axes: {grid._extract_axes}")
if hasattr(grid, 'available_spectra_emissions'):
    print(f"Available spectra: {grid.available_spectra_emissions}")
if hasattr(grid, 'available_line_emissions'):
    print(f"Available lines: {grid.available_line_emissions}")
