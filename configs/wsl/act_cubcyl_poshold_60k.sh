#!/usr/bin/env bash
# Φ — ACT on phi_so101_cubes_cylinder_v1 with the POSITION HOLDOUT, CVAE on vs off.
# Two 60k-step runs back to back on one consumer GPU (WSL2), then every checkpoint of
# both scored on the 30 held-out episodes. Experiment write-up:
#   experiments/2026-10-05_act-cubcyl-poshold-cvae-ab-wsl.md
#
#   bash configs/wsl/act_cubcyl_poshold_60k.sh            # both runs + scoring (~10.5 h on an RTX 5060 Ti)
#   bash configs/wsl/act_cubcyl_poshold_60k.sh score      # scoring only
#
# Run it from a wsl.exe that stays attached: WSL tears the distro down, and every nohup'd
# child with it, when the last wsl.exe session exits.
#
# Split: the Axis-2 position holdout from experiments/2026-08-06_act-cubes-cylinder-splits.md —
# 5 episodes from each of the 6 object×bin blocks held out (30), 90 train, 49,969 frames.
# This is the fix for eval_split=0.15 on this dataset, which holds out episodes 102-119:
# all white cube → white bin (experiments/2026-09-29_act-cubcyl-60k-vs-resume-wsl.md).
set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate phi
export HF_LEROBOT_HOME=~/lerobot-data
export HF_HOME=~/hf
export PYTHONUNBUFFERED=1
cd ~/phi
mkdir -p outputs/train outputs/eval

DS=BrutalCaesar/phi_so101_cubes_cylinder_v1
TRAIN_EPISODES="[5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64,70,71,72,73,74,75,76,77,78,79,80,81,82,83,84,85,86,87,88,89,95,96,97,98,99,100,101,102,103,104,105,106,107,108,109,115,116,117,118,119]"
HELDOUT_EPISODES="0,1,2,3,4,20,21,22,23,24,45,46,47,48,49,65,66,67,68,69,90,91,92,93,94,110,111,112,113,114"

# Competition config (same as act_cubcyl_60k) except the explicit split; eval_split=0 because
# lerobot would otherwise hold out the LAST 15% of the episodes we pass. No eval_loss is logged
# during training as a result — the held-out score comes from the scoring step below.
COMMON=(--dataset.repo_id=$DS "--dataset.episodes=$TRAIN_EPISODES" --dataset.eval_split=0.0
        --policy.type=act --policy.device=cuda --policy.push_to_hub=false
        --policy.chunk_size=50 --policy.n_action_steps=50
        --batch_size=8 --steps=60000 --save_freq=10000 --seed=1000 --num_workers=8 --wandb.enable=false)

train() {  # train <job_name> [extra lerobot-train flags]
  local job=$1; shift
  if [ -e "outputs/train/$job" ]; then echo "$(date +%T) $job: output dir exists, skipping training"; return 0; fi
  echo "$(date +%T) $job: start"
  lerobot-train "${COMMON[@]}" --output_dir="outputs/train/$job" --job_name="$job" "$@" > "outputs/train/$job.log" 2>&1
  local rc=$?
  echo "$(date +%T) $job: lerobot-train exit $rc"
  return $rc
}

score() {  # score <job_name>
  python -m phi.eval.loss_by_checkpoint --run "outputs/train/$1" --device cuda \
    --eval-episodes "$HELDOUT_EPISODES" --batch-size 32 --num-workers 8 \
    --out "outputs/eval/$1_ckpts.csv" 2>&1 | grep -v Fetching
}

if [ "${1:-}" != "score" ]; then
  train act_cubcyl_poshold_60k
  train act_cubcyl_poshold_novae_60k --policy.use_vae=false
fi
echo "$(date +%T) scoring on held-out episodes $HELDOUT_EPISODES"
score act_cubcyl_poshold_60k
score act_cubcyl_poshold_novae_60k
echo "$(date +%T) ALL_DONE"
