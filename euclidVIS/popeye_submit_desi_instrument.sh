#!/bin/bash
# Submit DESI instrument/noise post-processing on Popeye.
#
# Usage:
#   ./popeye_submit_desi_instrument.sh MANIFEST OUTPUT_ROOT [n_jobs] [limit]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:?Pass the fixed selection manifest}"
OUTPUT_ROOT="${2:?Pass the root containing sn*/DESI/*_raw.fits}"
N_JOBS="${3:-8}"
LIMIT="${4:-}"

if (( N_JOBS <= 0 )); then
    echo "ERROR: n_jobs must be a positive integer" >&2
    exit 1
fi
if [[ -n "${LIMIT}" ]] && (( LIMIT <= 0 )); then
    echo "ERROR: limit must be a positive integer" >&2
    exit 1
fi

LOG_DIR="${OUTPUT_ROOT}/logs/desi_instrument"
mkdir -p "${LOG_DIR}"
export POPEYE_SCRIPT_DIR="${SCRIPT_DIR}"

JOB_ID="$(
    sbatch --parsable \
        --cpus-per-task="${N_JOBS}" \
        --output="${LOG_DIR}/desi_noise.%j.out" \
        --error="${LOG_DIR}/desi_noise.%j.err" \
        "${SCRIPT_DIR}/popeye_apply_desi_instrument.sb" \
        "${MANIFEST}" "${OUTPUT_ROOT}" "${N_JOBS}" "${LIMIT}"
)"

echo "DESI instrument/noise job: ${JOB_ID}"
echo "Input root: ${OUTPUT_ROOT}"
echo "Log: ${LOG_DIR}/desi_noise.${JOB_ID}.out"
echo "Monitor: squeue -j ${JOB_ID}"
