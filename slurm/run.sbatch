#!/bin/bash
#SBATCH --job-name=fieldrefine
#SBATCH --gres=gpu:1
#SBATCH --constraint=<gpu class>
#SBATCH --cpus-per-task=8
#SBATCH --output=%x-%A_%a.out
set -euo pipefail

# One structure end to end: fetch -> prepare -> nullgate -> refine -> maps -> evaluate.
# Every argument is a Hydra override:
#   sbatch slurm/run.sbatch pdb_id=6LU7
#   sbatch slurm/run.sbatch pdb_id=6LU7 zero_field=true
# The grid becomes a job array without restructuring: submit with --array=0-99 and
# SLURM_ARRAY_TASK_ID is passed through as grid.indices.

cd "${SLURM_SUBMIT_DIR:-$PWD}"

DRIVER=$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -n 1)
CUDA_MAJOR=$(python3 -c 'import sys, device; print(device.cuda_major(sys.argv[1], ((580, 13), (525, 12))))' "$DRIVER")
echo "driver $DRIVER -> CUDA $CUDA_MAJOR"

VENV=".venv-cu${CUDA_MAJOR}"
if [ ! -d "$VENV" ]; then
  echo "missing $VENV; build it on a login node first (see docs/00_deploy.md)" >&2
  exit 1
fi
source "${VENV}/bin/activate"

export JAX_PLATFORMS=cuda
export JAX_ENABLE_X64=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
export XLA_FLAGS=--xla_gpu_deterministic_ops=true
# CUDA_VISIBLE_DEVICES belongs to SLURM; never set or unset it here.

# the fieldrefine package lives in src/; core.py and device.py stay flat at the root,
# so the working directory must remain the repository root.
export PYTHONPATH="src:${PYTHONPATH:-}"

if [ -n "${SLURM_ARRAY_TASK_ID:-}" ]; then
  exec python -m fieldrefine.cli "$@" "grid.indices=[${SLURM_ARRAY_TASK_ID}]"
fi
exec python -m fieldrefine.cli "$@"
