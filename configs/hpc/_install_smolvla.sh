#!/bin/bash
# One-off: add the SmolVLA extra to the Explorer env.
#
#   sbatch --partition=short --cpus-per-task=4 --mem=16G --time=00:40:00 \
#     --job-name=phi-install-smolvla \
#     --output=/scratch/$USER/phi/logs/install-smolvla-%j.out \
#     configs/hpc/_install_smolvla.sh
#
# 🚨 REFUSES TO RUN while any phi training job is queued or running. Installing
#    into site-packages under a live job can swap torch or lerobot out from under
#    it mid-run, and the failure would surface hours later as something
#    unrelated. The guard below is the whole point of this being a script.
#
# 🚨 lerobot on Explorer is a PyPI install, NOT editable — so it is
#    `pip install "lerobot[smolvla]==0.6.0"`, not `pip install -e ".[smolvla]"`
#    as the upstream docs say. The `==0.6.0` pin is load-bearing: without it pip
#    is free to upgrade lerobot itself and invalidate every model card and
#    checkpoint we have produced.
set -uo pipefail
PROJECT=/scratch/$USER/phi

# Count other phi jobs, EXCLUDING this one. Matching "^phi-" on the name alone
# makes the guard count itself -- this job is called phi-install-* -- so it
# refuses to run 100% of the time. That is what happened on 2026-08-12 (job
# 9088434, FAILED in 1 s). Filter by job id, not by name.
OTHERS=$(squeue -u "$USER" -h -o "%i %j" \
         | awk -v me="${SLURM_JOB_ID:-0}" '$1 != me && $2 ~ /^phi-/' | wc -l | tr -d ' ')
if [ "$OTHERS" -gt 0 ]; then
  echo "REFUSING: $OTHERS other phi job(s) queued/running. Installing now could break them."
  squeue -u "$USER" -o "%.14i %.18j %.8T"
  exit 1
fi

export PYTHONNOUSERSITE=1
for _s in "${PHI_SITE:-}" "${PROJECT:-}/repo/configs/hpc/site.sh" "/scratch/$USER/phi/repo/configs/hpc/site.sh" "$HOME/phi/configs/hpc/site.sh"; do [ -n "$_s" ] && [ -f "$_s" ] && { . "$_s"; break; }; done
command -v phi_load_conda >/dev/null || { echo "FATAL: configs/hpc/site.sh not found. Sync the repo, or set PHI_SITE=/path/to/site.sh" >&2; exit 1; }
phi_set_proxy   # site.sh: probes; proxy on Northeastern, none on AICR
export HF_HOME=$PROJECT/hf

phi_load_conda  # site.sh: anaconda3 on Northeastern, conda/latest on AICR
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate "$PROJECT/envs/lerobot"

echo "=== torch BEFORE (must not change) ==="
python -c "import torch; print(' torch', torch.__version__, '| cuda', torch.version.cuda)"

pip install "lerobot[smolvla]==0.6.0" || exit 1

echo "=== torch AFTER ==="
python -c "import torch; print(' torch', torch.__version__, '| cuda', torch.version.cuda)"

python - <<'PY' || exit 1
import torch
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy   # noqa: F401
from lerobot.policies.smolvla.configuration_smolvla import SmolVLAConfig
c = SmolVLAConfig()
print("smolvla imports OK")
print(f"  chunk_size {c.chunk_size} · n_action_steps {c.n_action_steps} · lr {c.optimizer_lr}")
print(f"  freeze_vision_encoder {c.freeze_vision_encoder} · train_expert_only {c.train_expert_only}")
assert torch.cuda.is_available() is False or True   # cuda check belongs to the GPU job
PY

echo "install finished rc=$? at $(date)"
