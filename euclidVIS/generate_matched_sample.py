"""Generate TNG50 mocks matched to Euclid deep-field redshift and mass."""

import argparse
import copy
import glob
import os
import re
from pathlib import Path

import illustris_python as il
import numpy as np
import pandas as pd
import yaml
from astropy.io import fits
from joblib import Parallel, delayed
from unyt import Angstrom

from synthesizer.emission_models import AttenuatedEmission, ReprocessedEmission
from synthesizer.emission_models.attenuation import Calzetti2000
from synthesizer.grid import Grid
from synthesizer.instruments.filters import Filter, FilterCollection

from generate_euclid_vis import process_single_galaxy_wrapper


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate a TNG50 sample matched in redshift and mass."
    )
    parser.add_argument(
        "--config",
        default=str(Path(__file__).with_name("config_matched.yaml")),
    )
    parser.add_argument("--n_mocks", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument(
        "--selection_only",
        action="store_true",
        help="Write the matched manifest without generating spectra/images.",
    )
    parser.add_argument("--force_overwrite", action="store_true")
    return parser.parse_args()


def reference_histogram(path, sample):
    """Stream the reference FITS catalog into a two-dimensional histogram."""
    z_edges = np.arange(
        sample["z_min"],
        sample["z_max"] + sample["redshift_bin_width"],
        sample["redshift_bin_width"],
    )
    mass_edges = np.arange(
        sample["log_mass_min"],
        sample["log_mass_max"] + sample["mass_bin_width"],
        sample["mass_bin_width"],
    )
    counts = np.zeros((len(z_edges) - 1, len(mass_edges) - 1), dtype=np.int64)

    with fits.open(path, memmap=True) as hdul:
        table = hdul[1].data
        for start in range(0, len(table), 1_000_000):
            stop = min(start + 1_000_000, len(table))
            z = np.asarray(table["REDSHIFT"][start:stop])
            mass = np.asarray(table["LOGMSTAR"][start:stop])
            valid = (
                np.isfinite(z)
                & np.isfinite(mass)
                & (z >= sample["z_min"])
                & (z <= sample["z_max"])
                & (mass >= sample["log_mass_min"])
                & (mass <= sample["log_mass_max"])
            )
            counts += np.histogram2d(
                z[valid], mass[valid], bins=(z_edges, mass_edges)
            )[0].astype(np.int64)

    if counts.sum() == 0:
        raise ValueError("No reference galaxies pass the redshift/mass cuts")
    return counts, z_edges, mass_edges


def draw_targets(counts, z_edges, mass_edges, n_mocks, rng):
    """Draw continuous target coordinates from the empirical 2D distribution."""
    probabilities = counts.ravel().astype(float)
    probabilities /= probabilities.sum()
    flat_bins = rng.choice(probabilities.size, n_mocks, p=probabilities)
    z_bin, mass_bin = np.unravel_index(flat_bins, counts.shape)
    return pd.DataFrame(
        {
            "target_redshift": rng.uniform(
                z_edges[z_bin], z_edges[z_bin + 1]
            ),
            "target_log_mass": rng.uniform(
                mass_edges[mass_bin], mass_edges[mass_bin + 1]
            ),
        }
    )


def discover_snapshots(tng_path, sample):
    """Discover locally available snapshots and read their redshifts."""
    patterns = [
        os.path.join(tng_path, "groups_*"),
        os.path.join(tng_path, "snapdir_*"),
    ]
    snapshot_numbers = set()
    for pattern in patterns:
        for directory in glob.glob(pattern):
            match = re.search(r"_(\d+)$", directory)
            if match:
                snapshot_numbers.add(int(match.group(1)))

    minimum = int(sample.get("snapshot_min", 0))
    maximum = int(sample.get("snapshot_max", 99))
    snapshot_numbers = sorted(
        snap for snap in snapshot_numbers if minimum <= snap <= maximum
    )
    if not snapshot_numbers:
        raise FileNotFoundError(f"No TNG snapshots found under {tng_path}")

    rows = []
    for snapshot in snapshot_numbers:
        try:
            header = il.groupcat.loadHeader(tng_path, snapshot)
            rows.append((snapshot, float(header["Redshift"])))
        except Exception as error:
            print(f"Skipping snapshot {snapshot}: {error}", flush=True)
    if not rows:
        raise RuntimeError("Could not read any TNG snapshot headers")
    return pd.DataFrame(rows, columns=["snapshot", "snapshot_redshift"])


def assign_snapshots(targets, snapshots):
    available_z = snapshots["snapshot_redshift"].to_numpy()
    nearest = np.abs(
        targets["target_redshift"].to_numpy()[:, None] - available_z[None, :]
    ).argmin(axis=1)
    targets["snapshot"] = snapshots["snapshot"].to_numpy()[nearest]
    targets["snapshot_redshift"] = available_z[nearest]
    return targets


def choose_subhalos(targets, tng_path, mass_floor, mass_bin_width, rng):
    """Match target masses to unique resolved subhalos in each snapshot."""
    targets["subhalo_id"] = -1
    targets["tng_log_mass"] = np.nan

    for snapshot, indices in targets.groupby("snapshot").groups.items():
        header = il.groupcat.loadHeader(tng_path, int(snapshot))
        hubble = float(header["HubbleParam"])
        mass_type = il.groupcat.loadSubhalos(
            tng_path, int(snapshot), fields=["SubhaloMassType"]
        )
        if isinstance(mass_type, dict):
            mass_type = mass_type["SubhaloMassType"]
        stellar_mass = mass_type[:, 4] * 1e10 / hubble
        log_mass = np.full(stellar_mass.shape, -np.inf, dtype=float)
        positive = stellar_mass > 0
        log_mass[positive] = np.log10(stellar_mass[positive])
        available = np.flatnonzero(log_mass >= mass_floor)
        if len(available) < len(indices):
            raise RuntimeError(
                f"Snapshot {snapshot} has {len(available)} resolved subhalos "
                f"but {len(indices)} unique mocks were requested"
            )

        # Bucket candidates by mass so selection remains fast for large mock
        # samples. Each bucket is shuffled to avoid repeatedly preferring the
        # same subhalo ordering from the group catalog.
        mass_bins = np.floor(
            (log_mass[available] - mass_floor) / mass_bin_width
        ).astype(int)
        candidates_by_bin = {}
        for mass_bin in np.unique(mass_bins):
            candidates = available[mass_bins == mass_bin].copy()
            rng.shuffle(candidates)
            candidates_by_bin[int(mass_bin)] = candidates.tolist()

        nonempty_bins = set(candidates_by_bin)
        for index in indices:
            target_mass = targets.at[index, "target_log_mass"]
            target_bin = int(
                np.floor((target_mass - mass_floor) / mass_bin_width)
            )
            selected_bin = min(nonempty_bins, key=lambda b: abs(b - target_bin))
            subhalo = candidates_by_bin[selected_bin].pop()
            if not candidates_by_bin[selected_bin]:
                nonempty_bins.remove(selected_bin)
            targets.at[index, "subhalo_id"] = subhalo
            targets.at[index, "tng_log_mass"] = log_mass[subhalo]
    return targets


def load_shared_resources(config):
    paths = config["paths"]
    grid = Grid(
        config["simulation"]["grid_name"],
        grid_dir=paths["grid_dir"],
    )
    local_filter = os.path.join(
        paths["grid_dir"], paths.get("filter_file", "Euclid_VIS.vis.dat")
    )
    if os.path.exists(local_filter):
        curve = np.loadtxt(local_filter)
        vis_filter = Filter(
            "Euclid/VIS_local",
            transmission=curve[:, 1],
            new_lam=curve[:, 0] * Angstrom,
        )
        vis_filter._interpolate_wavelength(grid.lam)
    else:
        vis_filter = FilterCollection(
            filter_codes=["Euclid/VIS.vis"], new_lam=grid.lam
        )[0]
    model = AttenuatedEmission(
        grid=grid,
        dust_curve=Calzetti2000(),
        apply_to=ReprocessedEmission(grid=grid),
        emitter="stellar",
    )
    return grid, vis_filter, model


def process_target(row, base_config, grid, vis_filter, model):
    config = copy.deepcopy(base_config)
    config["simulation"]["snap_number"] = int(row.snapshot)
    config["simulation"]["batch"] = False
    config["simulation"]["subhalo_ids"] = [int(row.subhalo_id)]
    config["observation"]["z_obs"] = float(row.target_redshift)
    config["observation"]["randomize_redshift"] = False
    return process_single_galaxy_wrapper(
        int(row.subhalo_id), config, grid, vis_filter, model
    )


def main():
    args = parse_args()
    with open(args.config) as stream:
        config = yaml.safe_load(stream)
    sample = config["sample"]
    n_mocks = args.n_mocks if args.n_mocks is not None else sample["n_mocks"]
    seed = args.seed if args.seed is not None else sample["seed"]
    if n_mocks <= 0:
        raise ValueError("n_mocks must be positive")
    if args.force_overwrite:
        config["optimization"]["force_overwrite"] = True
    rng = np.random.default_rng(seed)

    print("Building the Euclid reference distribution...", flush=True)
    counts, z_edges, mass_edges = reference_histogram(
        config["paths"]["reference_catalog"], sample
    )
    targets = draw_targets(counts, z_edges, mass_edges, n_mocks, rng)
    snapshots = discover_snapshots(config["paths"]["tng_path"], sample)
    targets = assign_snapshots(targets, snapshots)
    targets = choose_subhalos(
        targets,
        config["paths"]["tng_path"],
        sample["log_mass_min"],
        sample["mass_bin_width"],
        rng,
    )
    targets.insert(0, "mock_id", np.arange(n_mocks))
    targets["delta_log_mass"] = (
        targets["tng_log_mass"] - targets["target_log_mass"]
    )
    targets["delta_redshift"] = (
        targets["snapshot_redshift"] - targets["target_redshift"]
    )
    targets = targets.sort_values(
        ["snapshot_redshift", "target_log_mass"],
        ascending=[True, True],
    ).reset_index(drop=True)

    output_root = Path(config["paths"]["output_path"])
    output_root.mkdir(parents=True, exist_ok=True)
    manifest = output_root / sample.get(
        "selection_manifest", "matched_sample.csv"
    )
    targets.to_csv(manifest, index=False)
    print(f"Wrote selection manifest: {manifest}", flush=True)
    print(
        targets.groupby("snapshot").size().rename("n_mocks").to_string(),
        flush=True,
    )
    if args.selection_only:
        return

    grid, vis_filter, model = load_shared_resources(config)
    n_jobs = int(config["optimization"].get("n_jobs", 1))
    # Keep all work for one snapshot together. The Parallel context keeps the
    # worker pool alive between snapshot groups while each group benefits from
    # parallel galaxy processing and filesystem/cache locality.
    with Parallel(n_jobs=n_jobs) as parallel:
        snapshot_groups = targets.groupby("snapshot", sort=False)
        total_groups = targets["snapshot"].nunique()
        for group_number, (snapshot, group) in enumerate(
            snapshot_groups, start=1
        ):
            snapshot_redshift = group["snapshot_redshift"].iloc[0]
            print(
                f"Snapshot {int(snapshot)} (z={snapshot_redshift:.4f}): "
                f"generating {len(group)} mocks "
                f"[{group_number}/{total_groups}]",
                flush=True,
            )
            results = parallel(
                delayed(process_target)(
                    row, config, grid, vis_filter, model
                )
                for row in group.itertuples(index=False)
            )
            for result in results:
                print(result, flush=True)


if __name__ == "__main__":
    main()
