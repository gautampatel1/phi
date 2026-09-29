#!/bin/bash
# Prefetch a Hub dataset into $PROJECT/lerobot-data so training tasks do not race
# on the download.
#
#   sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:45:00 \
#     --job-name=phi-prefetch --output=/scratch/$USER/phi/logs/prefetch-%j.out \
#     configs/hpc/_prefetch_dataset.sh BrutalCaesar/phi_so101_cubes_cylinder_v1
#
# 🚨 Must be sbatch, not srun and not the login node:
#    - the login node KILLS the download partway through
#    - srun is tethered to your SSH session and dies with it
set -uo pipefail

REPO_ID="${1:?usage: _prefetch_dataset.sh <hf_repo_id>}"
PROJECT=/scratch/$USER/phi

export PYTHONNOUSERSITE=1
for _s in "${PHI_SITE:-}" "${PROJECT:-}/repo/configs/hpc/site.sh" "/scratch/$USER/phi/repo/configs/hpc/site.sh" "$HOME/phi/configs/hpc/site.sh"; do [ -n "$_s" ] && [ -f "$_s" ] && { . "$_s"; break; }; done
command -v phi_load_conda >/dev/null || { echo "FATAL: configs/hpc/site.sh not found. Sync the repo, or set PHI_SITE=/path/to/site.sh" >&2; exit 1; }
phi_set_proxy   # site.sh: probes; proxy on Northeastern, none on AICR
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data

phi_load_conda  # site.sh: anaconda3 on Northeastern, conda/latest on AICR
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

python - "$REPO_ID" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
ds = LeRobotDataset(sys.argv[1])
print("PREFETCH OK:", ds.meta.total_episodes, "episodes,", ds.meta.total_frames, "frames")
print("cameras:", ds.meta.camera_keys)
print("root:", ds.root)
PY
