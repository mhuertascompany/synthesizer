#!/bin/bash
# Submit matched DESI DR1 coadd-noise post-processing on Popeye.
# Usage: ./popeye_submit_dr1_noise.sh MANIFEST OUTPUT_ROOT [n_jobs] [limit]

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
MANIFEST="${1:?Pass the DR1 donor-assigned manifest}"
OUTPUT_ROOT="${2:?Pass the raw-production root}"
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

LOG_DIR="${OUTPUT_ROOT}/logs/dr1_noise"
mkdir -p "${LOG_DIR}"
export POPEYE_SCRIPT_DIR="${SCRIPT_DIR}"

JOB_ID="$(
    sbatch --parsable \
        --cpus-per-task="${N_JOBS}" \
        --output="${LOG_DIR}/apply_dr1.%j.out" \
        --error="${LOG_DIR}/apply_dr1.%j.err" \
        "${SCRIPT_DIR}/popeye_apply_dr1_noise.sb" \
        "${MANIFEST}" "${OUTPUT_ROOT}" "${N_JOBS}" "${LIMIT}"
)"

echo "DESI DR1 noise job: ${JOB_ID}"
echo "Output root: ${OUTPUT_ROOT}"
echo "Log: ${LOG_DIR}/apply_dr1.${JOB_ID}.out"
echo "Monitor: squeue -j ${JOB_ID}"
