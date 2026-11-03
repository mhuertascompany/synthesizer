#!/bin/bash
# Submit the mock-property measurement needed for DESI DR1 donor matching.
# Usage: ./popeye_submit_mock_properties.sh OUTPUT_ROOT

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${1:?Pass the completed raw-production root}"
INPUT_MANIFEST="${OUTPUT_ROOT}/matched_sample.csv"
OUTPUT_MANIFEST="${OUTPUT_ROOT}/matched_sample_mock_properties.csv"
LOG_DIR="${OUTPUT_ROOT}/logs/dr1_noise"
mkdir -p "${LOG_DIR}"
export POPEYE_SCRIPT_DIR="${SCRIPT_DIR}"

JOB_ID="$(
    sbatch --parsable \
        --output="${LOG_DIR}/mock_properties.%j.out" \
        --error="${LOG_DIR}/mock_properties.%j.err" \
        "${SCRIPT_DIR}/popeye_measure_mock_properties.sb" \
        "${OUTPUT_ROOT}" "${INPUT_MANIFEST}" "${OUTPUT_MANIFEST}"
)"

echo "Mock-property job: ${JOB_ID}"
echo "Output manifest: ${OUTPUT_MANIFEST}"
echo "Monitor: squeue -j ${JOB_ID}"
