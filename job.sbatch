#!/bin/bash
#SBATCH --gres=gpu:1
# SLURM splits an unfilled placeholder on the space and rejects the script, so this
# ships commented out; pass -C on the command line or set your class here
##SBATCH --constraint=a100
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

# device.py is imported by the system python3, before any venv exists
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)

# the system python3 on a compute node can be older than device.py expects; prefer any
# venv interpreter, which is pinned to 3.11, and fall back only if none is built yet
PROBE_PY=python3
for CANDIDATE in .venv/bin/python .venv-cu13/bin/python .venv-cu12/bin/python; do
    if [ -x "$CANDIDATE" ]; then PROBE_PY="$CANDIDATE"; break; fi
done
CUDA_MAJOR=$("$PROBE_PY" -c 'import sys, device; print(device.cuda_major(sys.argv[1], ((580, 13), (525, 12))))' "$DRIVER")
VENV=".venv-cu${CUDA_MAJOR}"
if [ ! -x "${VENV}/bin/python" ]; then
    echo "driver ${DRIVER} maps to CUDA ${CUDA_MAJOR}, but ${VENV} is not built here" >&2
    echo "on a login node: UV_PROJECT_ENVIRONMENT=${VENV} uv sync --extra cuda${CUDA_MAJOR} --python 3.11" >&2
    exit 1
fi
source "${VENV}/bin/activate"
# pytest reads pythonpath from pyproject.toml, but a bare "python -m" does not;
# the package lives in src/ while core.py and device.py stay flat at the root
export PYTHONPATH="src:${PYTHONPATH:-}"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS=--xla_gpu_deterministic_ops=true

# a quota-aborted uv sync leaves bin/python in place but the wheels half-extracted,
# so check the backend actually loads rather than that the venv exists
python -c 'import device; device.require_accelerator("gpu")' || {
    echo "node $(hostname): ${VENV} did not give a GPU backend (driver ${DRIVER}, CUDA ${CUDA_MAJOR})" >&2
    echo "rebuild it, and check quota: uv cache prune; du -sh ${VENV}" >&2
    exit 1
}

exec python -m "$@"
