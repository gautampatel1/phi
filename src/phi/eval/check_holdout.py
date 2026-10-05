"""WHAT A CHECKPOINT'S HELD-OUT ERROR LOOKS LIKE IN DEGREES, joint by joint.

    python -m phi.eval.check_holdout --out outputs/eval/act_cubcyl_check.json \
        outputs/train/act_cubcyl_60k/checkpoints/060000 \
        outputs/train/act_cubcyl_60k/checkpoints/040000

    # runs trained with --dataset.episodes (no eval_split): name the holdout
    python -m phi.eval.check_holdout --eval-episodes 0,1,2,3,4,20,21,22,23,24 ... CKPT

WHY THIS EXISTS NEXT TO loss_by_checkpoint
------------------------------------------
`loss_by_checkpoint` ranks checkpoints by one normalized number. That is the
right tool for picking a step and the wrong tool for understanding what the
model does, because a normalized L1 of 0.25 says nothing about whether the
gripper is off by 2 or 20. This scores the same frames and reports:

  - per-joint mean absolute error in the dataset's own units (degrees for the
    five SO-101 arm joints, 0-100 for the gripper), at step 0, over the first
    10 steps, the last 10, and the whole chunk
  - a HOLD-STILL baseline: repeat the current joint state for the whole chunk.
    A policy that is not clearly below this is not predicting motion.
  - per-held-out-episode normalized L1, so a bad number can be traced to the
    episodes causing it

The normalized mean reproduces `loss_by_checkpoint` to four decimals on the
same frames (checked 2026-10-01 on act_cubcyl_60k/060000: 0.2537 both ways).

⚠️ Offline only. This is prediction error on recorded demonstrations, not
success rate, and the ordering between the two has inverted before
(experiments/2026-08-12). Found on 2026-10-01 with this tool: act_cubcyl_60k's
wrist_roll error equals the hold-still baseline on the held-out episodes.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import torch

from phi.eval.loss_by_checkpoint import _load


@torch.no_grad()
def check(ckpt: Path, device: str, batch_size: int, workers: int,
          eval_episodes: list[int] | None = None) -> dict:
    from lerobot.utils.constants import ACTION

    cfg, policy, ds, pre = _load(ckpt, device, eval_episodes)
    h = cfg.policy.chunk_size
    names = [n.replace(".pos", "") for n in ds.meta.features["action"]["names"]]
    loader = torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False,
                                         num_workers=workers)

    d = len(names)
    abs_phys = torch.zeros(h, d, dtype=torch.float64)   # policy, physical units
    abs_norm = torch.zeros(h, d, dtype=torch.float64)   # policy, normalized
    hold_phys = torch.zeros(h, d, dtype=torch.float64)  # hold-still baseline, physical
    hold_norm = torch.zeros(h, d, dtype=torch.float64)
    cnt = torch.zeros(h, dtype=torch.float64)
    ep_sum: dict[int, float] = defaultdict(float)
    ep_cnt: dict[int, float] = defaultdict(float)
    ep_frames: dict[int, int] = defaultdict(int)
    scale = None

    for batch in loader:
        raw_gt = batch[ACTION].double()                      # (B, h, d) physical
        raw_state = batch["observation.state"].double()      # (B, d)
        ep = batch["episode_index"].tolist()
        batch = {k: (v.float() / 255.0 if (torch.is_tensor(v) and v.dtype == torch.uint8) else v)
                 for k, v in batch.items()}
        batch = pre(batch)
        pred = policy.predict_action_chunk(batch)[:, :h].double().cpu()
        gt = batch[ACTION][:, :h].double().cpu()
        keep = (~batch["action_is_pad"][:, :h]).double().cpu()   # (B, h)

        if scale is None:
            # Normalization is affine per dim (raw = norm * scale + shift), so the
            # scale falls out of the data itself. No need to open the normalizer file.
            m = keep.bool()
            scale = torch.stack([raw_gt[..., j][m].std() / gt[..., j][m].std() for j in range(d)])

        e = (pred - gt).abs()                                # normalized
        hold = (raw_state[:, None, :] - raw_gt).abs()        # physical
        k3 = keep[..., None]
        abs_norm += (e * k3).sum(0)
        abs_phys += (e * scale * k3).sum(0)
        hold_phys += (hold * k3).sum(0)
        hold_norm += (hold / scale * k3).sum(0)
        cnt += keep.sum(0)

        per_sample = (e.mean(-1) * keep).sum(1)
        per_cnt = keep.sum(1)
        for i, ep_i in enumerate(ep):
            ep_sum[ep_i] += per_sample[i].item()
            ep_cnt[ep_i] += per_cnt[i].item()
            ep_frames[ep_i] += 1

    c = cnt.clamp(min=1)[:, None]
    phys, norm, hp, hn = abs_phys / c, abs_norm / c, hold_phys / c, hold_norm / c
    out = {
        "checkpoint": str(ckpt),
        "holdout_frames": int(sum(ep_frames.values())),
        "holdout_episodes": sorted(ep_frames),
        "joints": names,
        "mean_l1_norm": norm.mean(-1).mean().item(),   # same definition as loss_by_checkpoint
        "l1_norm_by_horizon": norm.mean(-1).tolist(),
        "hold_still_l1_norm": hn.mean(-1).mean().item(),
        "per_joint_phys_all": phys.mean(0).tolist(),
        "per_joint_phys_step0": phys[0].tolist(),
        "per_joint_phys_first10": phys[:10].mean(0).tolist(),
        "per_joint_phys_last10": phys[-10:].mean(0).tolist(),
        "hold_still_per_joint_phys_all": hp.mean(0).tolist(),
        "hold_still_per_joint_phys_first10": hp[:10].mean(0).tolist(),
        "per_episode_l1_norm": {str(k): ep_sum[k] / max(ep_cnt[k], 1) for k in sorted(ep_sum)},
    }
    del policy, pre, ds
    if device == "cuda":
        torch.cuda.empty_cache()
    return out


def _print(r: dict) -> None:
    def fmt(xs: list[float]) -> str:
        return " ".join(f"{x:7.2f}" for x in xs)

    eps = r["holdout_episodes"]
    print(f"\n== {r['checkpoint']}")
    print(f"  holdout: {r['holdout_frames']} frames, {len(eps)} episodes ({eps[0]}-{eps[-1]})")
    print(f"  mean normalized L1: {r['mean_l1_norm']:.4f}   "
          f"hold-still baseline: {r['hold_still_l1_norm']:.4f}")
    print(f"  joints:               {' '.join(f'{n[:7]:>7}' for n in r['joints'])}")
    print(f"  policy  step 0:       {fmt(r['per_joint_phys_step0'])}")
    print(f"  policy  first 10:     {fmt(r['per_joint_phys_first10'])}")
    print(f"  policy  whole chunk:  {fmt(r['per_joint_phys_all'])}")
    print(f"  policy  last 10:      {fmt(r['per_joint_phys_last10'])}")
    print(f"  hold    first 10:     {fmt(r['hold_still_per_joint_phys_first10'])}")
    print(f"  hold    whole chunk:  {fmt(r['hold_still_per_joint_phys_all'])}")
    per_ep = " ".join(f"{k}:{v:.3f}" for k, v in r["per_episode_l1_norm"].items())
    print(f"  per episode: {per_ep}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("ckpts", nargs="+", type=Path,
                    help="checkpoint dirs (or their pretrained_model)")
    ap.add_argument("--out", type=Path, required=True,
                    help="JSON with everything printed, all checkpoints")
    ap.add_argument("--eval-episodes", type=lambda s: [int(x) for x in s.split(",") if x],
                    help="comma-separated episode indices, for runs trained with "
                         "--dataset.episodes")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()

    results = []
    for c in args.ckpts:
        r = check(c, args.device, args.batch_size, args.workers, args.eval_episodes)
        results.append(r)
        _print(r)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(results, indent=1))
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
