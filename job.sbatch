#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --constraint=<gpu class>
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

# device.py is imported by the system python3, before any venv exists
cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
CUDA_MAJOR=$(python3 -c 'import sys, device; print(device.cuda_major(sys.argv[1], ((580, 13), (525, 12))))' "$DRIVER")
VENV=".venv-cu${CUDA_MAJOR}"
if [ ! -x "${VENV}/bin/python" ]; then
    echo "driver ${DRIVER} maps to CUDA ${CUDA_MAJOR}, but ${VENV} is not built here" >&2
    echo "on a login node: UV_PROJECT_ENVIRONMENT=${VENV} uv sync --extra cuda${CUDA_MAJOR} --python 3.11" >&2
    exit 1
fi
source "${VENV}/bin/activate"
export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
exec python -m "$@"
