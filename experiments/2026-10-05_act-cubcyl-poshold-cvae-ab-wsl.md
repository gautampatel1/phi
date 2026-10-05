# ACT on cubes+cylinder — position holdout, CVAE on vs off

**Status**: 🏃 running since 2026-10-05 00:25 · results section to be filled when the chain finishes
**Author**: Gautam · **Ladder**: L4
**Dataset**: [`phi_so101_cubes_cylinder_v1`](../datasets/phi_so101_cubes_cylinder_v1.md) — 120 episodes, [on the Hub](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_v1)
**Script**: [`configs/wsl/act_cubcyl_poshold_60k.sh`](../configs/wsl/act_cubcyl_poshold_60k.sh) — both runs and the scoring, in order
**Hardware**: one RTX 5060 Ti 16 GB, WSL2 · [`env/environment.cuda-cu128.yml`](../env/environment.cuda-cu128.yml) · lerobot 0.6.0
**Follows**: [2026-09-29](2026-09-29_act-cubcyl-60k-vs-resume-wsl.md), which found that `eval_split=0.15` holds out one object×bin condition only

_Written before the first checkpoint was saved. Predictions below are pre-registered so they cannot be retrofitted._

## Two questions

**1. What is the held-out error when the holdout actually covers the task?** The 2026-09-29 run's holdout was episodes 102–119, all white cube → white bin, a pairing with two training episodes. Its 0.254 says nothing about the red cube or the cylinder. This run uses the **Axis 2 position holdout** from [2026-08-06](2026-08-06_act-cubes-cylinder-splits.md): 5 episodes from each of the 6 object×bin blocks, so every condition is in both train and test and only the start positions are unseen.

**2. Does the CVAE do anything on this data?** The question [2026-08-06](2026-08-06_act-cubes-cylinder-splits.md) called the most valuable single run in its plan and nobody has run. The ACT paper's ablation: dropping the CVAE costs almost nothing on scripted data and drops human-demo success from 35% to 2%. Our data is human, and our latent was measured collapsed (KL ≈ 0 at `kl_weight=10`). Either the latent carries nothing and collapse is benign, or it was doing work and its collapse is a real defect in every ACT run in this repo.

## Setup

Identical to `act_cubcyl_60k` ([2026-09-29](2026-09-29_act-cubcyl-60k-vs-resume-wsl.md)) except the split and, in run 2, `use_vae`.

| | |
|---|---|
| Policy | ACT, chunk 50, `n_action_steps` 50, batch 8, lr 1e-5, seed 1000, no image aug, 3 cameras |
| Steps | 60,000 · `save_freq` 10,000 → 6 checkpoints per run |
| Train | 90 episodes, **49,969 frames** → 9.6 epochs |
| Held out | 30 episodes: `0-4, 20-24, 45-49, 65-69, 90-94, 110-114` |
| `eval_split` | **0.0** — lerobot would otherwise hold out the last 15% of the 90 we pass. No `eval_loss` is logged; the held-out score comes from `loss_by_checkpoint --eval-episodes` after training |

| Run | `use_vae` | Params | Output dir |
|---|---|---:|---|
| **C** | true | 51,571,590 | `outputs/train/act_cubcyl_poshold_60k` |
| **D** | **false** | 34,198,342 | `outputs/train/act_cubcyl_poshold_novae_60k` |

Run D's parameter count was confirmed by a 200-step smoke test before the chain started (3.53 step/s, loss 0.60 at step 200). The 34.20M matches the figure measured in 2026-08-06.

## Exact commands

```bash
bash configs/wsl/act_cubcyl_poshold_60k.sh          # run C, run D, then score both on the 30 held-out episodes
```

which expands to, per run:

```bash
lerobot-train --dataset.repo_id=BrutalCaesar/phi_so101_cubes_cylinder_v1 \
  --dataset.episodes="[5,...,19,25,...,44,50,...,64,70,...,89,95,...,109,115,...,119]" --dataset.eval_split=0.0 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --batch_size=8 --steps=60000 --save_freq=10000 --seed=1000 --num_workers=8 --wandb.enable=false \
  --output_dir=outputs/train/act_cubcyl_poshold_60k --job_name=act_cubcyl_poshold_60k
# run D adds:  --policy.use_vae=false
```

and the scoring:

```bash
python -m phi.eval.loss_by_checkpoint --run outputs/train/<run> --device cuda --batch-size 32 --num-workers 8 \
  --eval-episodes 0,1,2,3,4,20,21,22,23,24,45,46,47,48,49,65,66,67,68,69,90,91,92,93,94,110,111,112,113,114 \
  --out outputs/eval/<run>_ckpts.csv
```

The full episode list is in the script. No `lerobot-rollout` command yet.

## Pre-registered predictions

1. **Run C's best held-out L1 lands between 0.15 and 0.22** — below 0.254 because every held-out condition now has 15 training episodes, above the 0.079 the previous model scores on its own training episodes.
2. **The white-cube blocks (90–94, 110–114) are the worst held-out episodes** in `check_holdout`'s per-episode breakdown, for the contrast reason in the dataset card.
3. **Run D matches run C within 5% on held-out L1.** The latent is zeroed at inference either way (`modeling_act.py` gates the VAE encoder on `self.training`), so the only difference the holdout can see is what the KL term did to training. A collapsed latent should have done little.
4. If prediction 3 fails and D is **much worse**, that is the more important result: the latent was carrying information and its collapse elsewhere is a defect, not a curiosity.

## Results

_Pending. Run C ETA ~05:40, run D ~11:00, scoring ~11:30 on 2026-10-05._

| step | C (CVAE) | D (no CVAE) |
|---|---:|---:|
| 10,000 | | |
| 20,000 | | |
| 30,000 | | |
| 40,000 | | |
| 50,000 | | |
| 60,000 | | |

## What this run cannot tell us

- **Success rate.** Same caveat as every offline number in this repo: the ordering between held-out L1 and rollouts has inverted before ([2026-08-12](2026-08-12_dp-recovery-encoder-ab.md)).
- **Whether the CVAE matters on the arm.** The paper's 35% → 2% is a rollout number. If D matches C offline and still differs on the arm, the latent matters at execution time in a way L1 cannot see.
- **Object transfer.** Every object is in training here. That is [Axis 1](2026-08-06_act-cubes-cylinder-splits.md), not this run.

## Next

- [ ] Fill in the table; `check_holdout` on the best checkpoint of each run, per block
- [ ] Rollouts on both best checkpoints, 20 per condition, canonical `phi_follower` calibration
- [ ] If D ≈ C: measure the actual KL magnitude on run C rather than calling it "≈ 0"
