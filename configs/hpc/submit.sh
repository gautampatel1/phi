#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — submit an sbatch script with THIS cluster's partition and GPU type.
#
#   ./configs/hpc/submit.sh configs/hpc/train_cubes_cylinder.sbatch
#   ./configs/hpc/submit.sh configs/hpc/build_env.sbatch
#   ./configs/hpc/submit.sh --time=08:00:00 configs/hpc/train_pi05_cubcyl.sbatch
#
# WHY NOT JUST EDIT THE #SBATCH LINES
#
# SLURM parses `#SBATCH` before any shell runs, so those directives cannot read
# $USER or a config file -- which is exactly why `--partition=gpu --gres=gpu:h200:1`
# ended up baked in. Correct on Northeastern, nonexistent on AICR.
#
# Flags passed to `sbatch` on the COMMAND LINE override the directives in the
# file. So the file keeps a sensible default, and this wrapper supplies whatever
# the cluster you are actually sitting on calls its partitions.
#
# Anything you pass through is forwarded, and wins over what this computes.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail
cd "$(dirname "$0")/../.."
source configs/hpc/site.sh
phi_detect_partitions

# Last argument is the script; everything before it is passed through to sbatch.
SCRIPT="${@: -1}"
PASSTHRU=("${@:1:$#-1}")
[ -f "$SCRIPT" ] || { echo "not a file: $SCRIPT" >&2; exit 1; }

# Create whatever log dir the script asks for. SLURM will NOT create it, and a
# missing one is the nastiest failure here: the job dies instantly with
# ExitCode 0:53, writes no log anywhere, and just vanishes from squeue.
# Two scripts log under /home rather than /scratch, so read it from the file
# instead of assuming.
while read -r _p; do
  _p="${_p//%u/$USER}"
  mkdir -p "$(dirname "$_p")" 2>/dev/null || echo "submit: WARNING cannot create $(dirname "$_p")" >&2
done < <(grep -oE '^#SBATCH[[:space:]]+--(output|error)=\S+' "$SCRIPT" | sed 's/.*=//')
mkdir -p "$PHI_PROJECT/logs"

# A job is a GPU job iff it asks for a gres. CPU jobs must NOT get one.
ARGS=()
if grep -qE '^#SBATCH[[:space:]]+--gres=gpu' "$SCRIPT"; then
  [ -n "$PHI_PARTITION_GPU" ] || { echo "no GPU partition found by sinfo" >&2; exit 1; }
  ARGS+=(--partition="$PHI_PARTITION_GPU" --gres="$PHI_GRES")
  echo "submit: GPU  partition=$PHI_PARTITION_GPU gres=$PHI_GRES"
else
  [ -n "$PHI_PARTITION_CPU" ] || { echo "no CPU partition found by sinfo" >&2; exit 1; }
  ARGS+=(--partition="$PHI_PARTITION_CPU")
  echo "submit: CPU  partition=$PHI_PARTITION_CPU"
fi

echo "submit: logs -> $PHI_PROJECT/logs"
set -x
exec sbatch "${ARGS[@]}" ${PASSTHRU+"${PASSTHRU[@]}"} "$SCRIPT"
