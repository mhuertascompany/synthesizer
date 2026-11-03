#!/bin/bash
# Submit the fixed-selection job followed by the restart-safe production array.
#
# Usage:
#   ./popeye_submit_matched.sh [n_mocks] [seed] [galaxies_per_node]
#
# Defaults generate 1000 mocks in shards of four, with at most four full gen
# nodes active at once (the documented Popeye general-partition user limit).

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
export POPEYE_SCRIPT_DIR="${SCRIPT_DIR}"
N_MOCKS="${1:-1000}"
SEED="${2:-42}"
GALAXIES_PER_NODE="${3:-4}"
MAX_CONCURRENT_NODES="${POPEYE_MAX_CONCURRENT_NODES:-4}"
PYTHON_BIN="${POPEYE_PYTHON:-python3}"

export POPEYE_TNG_PATH="${POPEYE_TNG_PATH:-/mnt/sdceph/users/sgenel/Illustris_IllustrisTNG_public_data_release/L35n2160TNG/output}"
export POPEYE_GRID_DIR="${POPEYE_GRID_DIR:-/mnt/sdceph/users/${USER}/synthesizer_grids}"
export POPEYE_REFERENCE_CATALOG="${POPEYE_REFERENCE_CATALOG:-/mnt/sdceph/users/${USER}/euclid_desi_mocks/reference/catalog_MER_DR1_DESI_DR1_combined_wide_deep_v1.1.fits}"
export POPEYE_OUTPUT_ROOT="${POPEYE_OUTPUT_ROOT:-/mnt/sdceph/users/${USER}/euclid_desi_mocks/tngmatched_kappa0p0795}"
export PYTHONPATH="${SCRIPT_DIR}/../src:${POPEYE_ILLUSTRIS_PYTHON_ROOT:-${HOME}/python/illustris_python}:${PYTHONPATH:-}"

GRID_NAME="bc03-2016-Miles_chabrier-0.1,100_cloudy-c25.00-sps_0.055res_1200-11000A.hdf5"
GRID_PATH="${POPEYE_GRID_DIR}/${GRID_NAME}"

if [[ ! -d "${POPEYE_TNG_PATH}" ]]; then
    echo "ERROR: TNG50 output directory not found: ${POPEYE_TNG_PATH}" >&2
    echo "Set POPEYE_TNG_PATH to the directory containing groups_XXX/snapdir_XXX." >&2
    exit 1
fi
if [[ ! -f "${GRID_PATH}" ]]; then
    echo "ERROR: Synthesizer grid not found: ${GRID_PATH}" >&2
    exit 1
fi
if [[ ! -f "${POPEYE_REFERENCE_CATALOG}" ]]; then
    echo "ERROR: reference catalog not found: ${POPEYE_REFERENCE_CATALOG}" >&2
    exit 1
fi
if ! command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
    echo "ERROR: Python executable not found: ${PYTHON_BIN}" >&2
    echo "Activate the Synthesizer environment or set POPEYE_PYTHON." >&2
    exit 1
fi
if ! "${PYTHON_BIN}" -c 'import astropy, h5py, illustris_python, joblib, numpy, pandas, scipy, unyt, yaml; import synthesizer' >/dev/null; then
    echo "ERROR: the active Python environment is missing required packages." >&2
    echo "Activate the Popeye Synthesizer environment before submitting." >&2
    exit 1
fi
if (( N_MOCKS <= 0 || GALAXIES_PER_NODE <= 0 || MAX_CONCURRENT_NODES <= 0 )); then
    echo "ERROR: counts and concurrency must be positive integers" >&2
    exit 1
fi

N_SHARDS=$(( (N_MOCKS + GALAXIES_PER_NODE - 1) / GALAXIES_PER_NODE ))
ARRAY_MAX=$(( N_SHARDS - 1 ))
LOG_DIR="${POPEYE_OUTPUT_ROOT}/logs"
MANIFEST="${POPEYE_OUTPUT_ROOT}/matched_sample.csv"
mkdir -p "${LOG_DIR}"

SELECT_JOB_ID="$(
    sbatch --parsable \
        --output="${LOG_DIR}/select.%j.out" \
        --error="${LOG_DIR}/select.%j.err" \
        "${SCRIPT_DIR}/popeye_select_matched.sb" \
        "${N_MOCKS}" "${SEED}"
)"

ARRAY_JOB_ID="$(
    sbatch --parsable \
        --dependency="afterok:${SELECT_JOB_ID}" \
        --array="0-${ARRAY_MAX}%${MAX_CONCURRENT_NODES}" \
        --output="${LOG_DIR}/mocks.%A_%a.out" \
        --error="${LOG_DIR}/mocks.%A_%a.err" \
        "${SCRIPT_DIR}/popeye_generate_matched_array.sb" \
        "${MANIFEST}" "${N_SHARDS}" "${SEED}"
)"

echo "Selection job: ${SELECT_JOB_ID}"
echo "Mock array:   ${ARRAY_JOB_ID} (0-${ARRAY_MAX}%${MAX_CONCURRENT_NODES})"
echo "Mocks:        ${N_MOCKS}"
echo "Output:       ${POPEYE_OUTPUT_ROOT}"
echo "Monitor:      squeue -j ${SELECT_JOB_ID},${ARRAY_JOB_ID}"
