#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — pull phi_so101_8bin_v1 onto /scratch ONCE, before the training array.
#
# WHY A SEPARATE JOB: six array tasks starting together would each try to
# download the same 1.25 GB and race on the same cache directory. Prefetch once,
# then every task reads a warm local copy.
#
# CPU partition — this is pure network + disk. No GPU.
#
# Submit:  sbatch configs/hpc/prefetch_dataset.sbatch
# ─────────────────────────────────────────────────────────────────────────────
#SBATCH --job-name=phi-prefetch
#SBATCH --partition=short
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=/scratch/%u/phi/logs/prefetch-%j.out
#SBATCH --error=/scratch/%u/phi/logs/prefetch-%j.err

set -uo pipefail
PROJECT=/scratch/$USER/phi
# Competition dataset by default; override with REPO_ID=... sbatch ...
REPO_ID=${REPO_ID:-BrutalCaesar/phi_so101_cubes_cylinder_v1}

# compute nodes have no direct route out
# 🚨 A stale torch in ~/.local/lib/python3.12/site-packages shadows the conda env
# (Python puts user-site ahead of it) and dies with `libnvJitLink.so.12: cannot
# open shared object file`. Every Φ job on Explorer must set this.
export PYTHONNOUSERSITE=1

for _s in "${PHI_SITE:-}" "${PROJECT:-}/repo/configs/hpc/site.sh" "/scratch/$USER/phi/repo/configs/hpc/site.sh" "$HOME/phi/configs/hpc/site.sh"; do [ -n "$_s" ] && [ -f "$_s" ] && { . "$_s"; break; }; done
command -v phi_load_conda >/dev/null || { echo "FATAL: configs/hpc/site.sh not found. Sync the repo, or set PHI_SITE=/path/to/site.sh" >&2; exit 1; }
phi_set_proxy   # site.sh: probes; proxy on Northeastern, none on AICR
export HF_HOME=$PROJECT/hf
export HF_LEROBOT_HOME=$PROJECT/lerobot-data      # every job must export this identically
mkdir -p "$HF_HOME" "$HF_LEROBOT_HOME"

phi_load_conda  # site.sh: anaconda3 on Northeastern, conda/latest on AICR
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

source "$PROJECT/repo/configs/hpc/_preflight.sh"

echo "node=$(hostname) date=$(date)"
echo "HF_LEROBOT_HOME=$HF_LEROBOT_HOME"

python - "$REPO_ID" <<'PY'
import sys
from lerobot.datasets.lerobot_dataset import LeRobotDataset
repo_id = sys.argv[1]
# Resolves the `v3.0` codebase-version TAG, not `main`. If this raises
# RevisionNotFoundError the tag is missing on the Hub — see
# docs/robots/so-arm101/03-teleop-and-data.md (§3c).
ds = LeRobotDataset(repo_id)
m = ds.meta
print(f"OK  episodes={m.total_episodes}  frames={m.total_frames}  fps={m.fps}")
print(f"    revision={m.revision}  robot={m.robot_type}")
print(f"    cameras={m.camera_keys}")
# Counts are per-DATASET, so they cannot be hardcoded here: this script is
# parameterised by REPO_ID. 119/72518 belong to 8bin_v1 and fired a false
# rc=1 on a perfectly good cubes_cylinder download (120/66,873).
# Opt in by passing the numbers you expect:
#   EXPECT_EPISODES=120 EXPECT_FRAMES=66873 sbatch ...
import os
for key, got in (("EXPECT_EPISODES", m.total_episodes),
                 ("EXPECT_FRAMES",   m.total_frames)):
    want = os.environ.get(key)
    if want:
        assert got == int(want), f"{key}: expected {want}, got {got}"
        print(f"    {key} ok ({got})")
PY
rc=$?

echo "--- on disk ---"
du -sh "$HF_LEROBOT_HOME"/* 2>/dev/null
echo "prefetch rc=$rc  $(date)"
exit $rc
