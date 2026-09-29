#!/bin/bash
# ─────────────────────────────────────────────────────────────────────────────
# Φ — CLUSTER PROFILE. Sourced by every script in this directory.
#
# WHY THIS EXISTS
#
# Two different SLURM clusters are both reachable as `ssh explorer`, because the
# alias lives in each person's own ~/.ssh/config:
#
#   Northeastern "Explorer" (explorer-01)   partitions: gpu, short, courses-gpu
#                                           GPUs: v100 / a100 / h200 / t4
#                                           modules: Environment Modules 5.3
#                                           compute nodes: NO route to internet
#
#   AICR                                    partitions: cpu, rtx-batch, b200-batch,
#                                             b200-fullnode, rtx-devel, b200-devel,
#                                             preemptable
#                                           GPUs: rtx_pro_6000 / b200
#                                           modules: Lmod
#                                           compute nodes: direct internet
#
# Every script here was written against the first one. Run on the second, six
# separate things break (Tavish & Faisal's error log, 2026-09-28). The fix is NOT
# to sed the scripts to AICR values -- that breaks Northeastern, where every
# existing Φ checkpoint lives. Detect instead.
#
# Nothing here is cluster-specific by name. It asks SLURM and the module system
# what exists, so a third cluster works too.
#
#   source configs/hpc/site.sh
#   phi_load_conda      # whatever conda module this cluster has
#   phi_set_proxy       # only if this node actually needs one
# ─────────────────────────────────────────────────────────────────────────────

# ---- conda ------------------------------------------------------------------
# Northeastern has anaconda3/2024.06 and miniconda3/*. AICR has only conda/latest.
# Override with PHI_CONDA_MODULE=<name> if your cluster calls it something else.
phi_load_conda() {
  local m
  for m in ${PHI_CONDA_MODULE:-} anaconda3/2024.06 conda/latest miniconda3/25.9.1 \
           miniconda3/24.11.1 anaconda3 miniconda3 conda; do
    [ -n "$m" ] || continue
    if module load "$m" 2>/dev/null; then
      echo "site: conda module '$m'"
      return 0
    fi
  done
  if command -v conda >/dev/null 2>&1; then
    echo "site: conda already on PATH, no module needed"
    return 0
  fi
  echo "site: FATAL — no conda module found. Set PHI_CONDA_MODULE=<name>." >&2
  return 1
}

# ---- proxy ------------------------------------------------------------------
# Northeastern compute nodes cannot reach the internet and need 10.99.0.130:3128.
# AICR nodes route directly, and that address does not exist there -- setting it
# turns every download into a ConnectTimeoutError.
#
# A wrong proxy and a missing proxy both present as a hang, so PROBE rather than
# assume. ~8 s worst case, once per job, against hours of training.
phi_set_proxy() {
  if [ "${PHI_PROXY:-unset}" = "none" ]; then
    echo "site: proxy disabled by PHI_PROXY=none"; return 0
  fi
  if [ -n "${PHI_PROXY:-}" ]; then
    export http_proxy="$PHI_PROXY" https_proxy="$PHI_PROXY"
    echo "site: proxy $PHI_PROXY (forced via PHI_PROXY)"; return 0
  fi
  if command -v curl >/dev/null 2>&1 &&
     curl -fsS -o /dev/null -m 8 https://conda.anaconda.org 2>/dev/null; then
    echo "site: direct internet — no proxy"
    unset http_proxy https_proxy
    return 0
  fi
  export http_proxy=http://10.99.0.130:3128
  export https_proxy=http://10.99.0.130:3128
  echo "site: no direct route — proxy $http_proxy"
}

# ---- partitions -------------------------------------------------------------
# #SBATCH directives cannot read shell variables, so these are consumed by
# configs/hpc/submit.sh, which passes them to sbatch on the command line where
# they override the baked-in directives.
phi_detect_partitions() {
  local parts line want
  parts=$(sinfo -h -o "%P" 2>/dev/null | tr -d '*' | sort -u)

  PHI_PARTITION_CPU=""
  for want in short cpu preemptable batch; do
    if printf '%s\n' "$parts" | grep -qx "$want"; then PHI_PARTITION_CPU="$want"; break; fi
  done

  # Best generally-available accelerator first.
  PHI_PARTITION_GPU=""; PHI_GRES=""
  for want in h200 b200 a100 rtx_pro_6000 l40s v100-sxm2 v100-pcie t4; do
    line=$(sinfo -h -o "%P %G" 2>/dev/null | grep -m1 "gpu:${want}[:(]")
    if [ -n "$line" ]; then
      PHI_PARTITION_GPU=$(printf '%s' "$line" | awk '{print $1}' | tr -d '*')
      PHI_GRES="gpu:${want}:1"
      break
    fi
  done
  export PHI_PARTITION_CPU PHI_PARTITION_GPU PHI_GRES
}

# ---- paths ------------------------------------------------------------------
# /home and /scratch are separate filesystems on both clusters, and jobs may only
# write under /scratch. Never hardcode a username here: SLURM expands %u in
# --output/--error, and $USER works everywhere in the body.
export PHI_PROJECT="${PHI_PROJECT:-/scratch/$USER/phi}"

# A stale torch in ~/.local/lib/python3.*/site-packages shadows the conda env and
# dies with `libnvJitLink.so.12: cannot open shared object file`. Adding the env
# to PATH does not help; only disabling user-site does. Every Φ job needs this.
export PYTHONNOUSERSITE=1
