#!/bin/bash
# Build the isolated feasiBGS/DESI post-processing environment on Popeye.

set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
ENV_DIR="${POPEYE_DESI_ENV:-/mnt/home/${USER}/.venvs/desi-noise}"
FEASIBGS_ROOT="${POPEYE_FEASIBGS_ROOT:-/mnt/home/${USER}/python/feasiBGS}"
DESIMODEL_ROOT="${DESIMODEL:-/mnt/sdceph/users/${USER}/desimodel}"
FEASIBGS_COMMIT="b5f535f12cf64babc9e25bcec75edd45d8668f74"

module purge
module load python/3.11.11

if [[ ! -x "${ENV_DIR}/bin/python" ]]; then
    python -m venv "${ENV_DIR}"
fi
"${ENV_DIR}/bin/python" -m pip install --upgrade pip setuptools wheel
"${ENV_DIR}/bin/python" -m pip install \
    -r "${SCRIPT_DIR}/requirements-popeye-desi.txt"

if [[ ! -d "${FEASIBGS_ROOT}/.git" ]]; then
    git clone https://github.com/desi-bgs/feasiBGS.git "${FEASIBGS_ROOT}"
    git -C "${FEASIBGS_ROOT}" checkout --detach "${FEASIBGS_COMMIT}"
fi
ACTUAL_COMMIT="$(git -C "${FEASIBGS_ROOT}" rev-parse HEAD)"
if [[ "${ACTUAL_COMMIT}" != "${FEASIBGS_COMMIT}" ]]; then
    echo "ERROR: ${FEASIBGS_ROOT} is at ${ACTUAL_COMMIT}" >&2
    echo "Expected tested feasiBGS commit ${FEASIBGS_COMMIT}." >&2
    exit 1
fi
"${ENV_DIR}/bin/python" -m pip install -e "${FEASIBGS_ROOT}"

mkdir -p "${DESIMODEL_ROOT}"
export DESIMODEL="${DESIMODEL_ROOT}"
if [[ ! -d "${DESIMODEL_ROOT}/data" ]]; then
    "${ENV_DIR}/bin/python" -c \
        'import desimodel.install; desimodel.install.install()'
fi

export PYTHONPATH="${FEASIBGS_ROOT}:${SCRIPT_DIR}:${PYTHONPATH:-}"
"${ENV_DIR}/bin/python" "${SCRIPT_DIR}/check_feasibgs.py" --imports-only
echo "Popeye DESI-noise environment is ready: ${ENV_DIR}"
