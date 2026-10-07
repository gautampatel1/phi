# ACT on cubes+cylinder — position holdout, CVAE on vs off

**Status**: ✅ both runs complete and scored offline (2026-10-07) · ❌ no rollouts yet
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

Run D's parameter count was confirmed by a 200-step smoke test before the chain started (3.53 step/s, loss 0.60 at step 200). The 34.20M matches the figure measured in 2026-08-06. The 17.4M difference is the VAE *encoder*, which only runs during training; at inference both policies are the same network with the latent zeroed, so this is a like-for-like comparison of what the CVAE objective did to the weights, not of model capacity.

**The chain took three nights because the PC slept twice.** Run C trained 00:25–03:28 on 10-05, the machine idle-slept and hibernated until 23:27, and it finished at 01:23 on 10-06. Run D started at once, the PC slept again at 01:30 and woke at 23:46, and it finished at 04:26 on 10-07. Both `lerobot-train` processes survived suspend and hibernate with their CUDA contexts intact and continued exactly where they stopped — no checkpoint was lost and no step was repeated. Pure compute time: run C 4 h 59 m (3.43 step/s), run D 4 h 47 m (3.50 step/s), scoring 1 h for 12 checkpoints. Two `SetThreadExecutionState(ES_SYSTEM_REQUIRED …)` holds did not prevent the idle sleep (15-minute AC timeout, S3 desktop). What kept it awake on the third night was a once-a-minute zero-distance `mouse_event`, which resets the user-idle timer; the power plan itself was never changed.

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

### Held-out L1 on the 30 position-holdout episodes (16,904 frames)

Normalized L1 over the 50-step chunk, padding excluded, identical frames for all 12 checkpoints. `first 10` / `last 10` are the mean over horizon steps 0–9 and 40–49.

| step | **C** (CVAE) | first 10 | last 10 | **D** (no CVAE) | first 10 | last 10 | D vs C |
|---|---:|---:|---:|---:|---:|---:|---:|
| 10,000 | 0.2243 | 0.1321 | 0.3115 | 0.2496 | 0.1779 | 0.3215 | +11.3% |
| 20,000 | 0.2216 | 0.1195 | 0.3141 | 0.2555 | 0.1668 | 0.3407 | +15.3% |
| 30,000 | 0.2190 | 0.1177 | 0.3085 | 0.2357 | 0.1466 | 0.3208 | +7.6% |
| 40,000 | 0.2127 | 0.1138 | 0.3028 | 0.2337 | 0.1367 | 0.3270 | +9.9% |
| 50,000 | 0.2136 | 0.1148 | 0.3009 | 0.2310 | 0.1367 | 0.3214 | +8.1% |
| **60,000** | **0.2100** | 0.1115 | 0.2999 | **0.2266** | 0.1287 | 0.3189 | **+7.9%** |

Final training loss: C 0.073, D 0.076 — the same fit on the training set.

- **Both runs are still improving at 60,000** (C: 0.224 → 0.210, D: 0.250 → 0.227, each with one small reversal). As in [2026-09-29](2026-09-29_act-cubcyl-60k-vs-resume-wsl.md), the last checkpoint is the best and the curve has not turned.
- **The CVAE run is better at every checkpoint, by 8–15%.** Per held-out episode at step 60,000, C beats D on **24 of 30**, loses 4, ties 2 (sign test p ≈ 0.0002). The gap is largest on the first 10 predicted steps (0.112 vs 0.129, 15%) and smallest on the last 10 (0.300 vs 0.319, 6%): the no-CVAE policy is worse at the *near-term* actions, which are the ones that get executed.
- **With equal training loss, the difference is generalisation.** Whatever the KL term did, it did it as a regulariser: D fits the 90 training episodes as well as C and predicts the 30 unseen ones worse.

### Per block, run C step 60,000 (mean of the 5 held-out episodes in each)

| block | red · cardboard | red · white bin | cylinder · cardboard | cylinder · white bin | white cube · cardboard | white cube · white bin |
|---|---:|---:|---:|---:|---:|---:|
| C | 0.197 | **0.167** | 0.223 | 0.219 | 0.231 | **0.233** |
| D | 0.220 | 0.176 | 0.244 | 0.221 | 0.259 | 0.251 |

The red cube is clearly the easiest condition; cylinder and white cube are close, white cube worst. Episode 0 is an outlier at 0.352 (D: 0.397) — the very first demonstration recorded, 23.6 s long against a 15–17 s norm, and the slowest. Without it the red·cardboard block scores 0.159.

Against the 2026-09-29 model: that run had seen only 2 white·white-bin episodes and scored 0.254 on the other 18; run C saw 15 of them and scores 0.233 on the 5 held out. More demonstrations of the condition helped, but less than one might hope — the white cube stays the hard object.

### Per-joint error in real units, step 60,000 of each run

`phi.eval.check_holdout` on the same 16,904 frames. Mean absolute error in **degrees** (gripper: 0–100 scale). "Hold still" repeats the current joint state for the whole chunk.

| whole chunk | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper | normalized |
|---|---:|---:|---:|---:|---:|---:|---:|
| **C** (CVAE) | **3.6** | **8.3** | **9.3** | **6.5** | 5.0 | **4.7** | **0.210** |
| D (no CVAE) | 3.7 | 8.6 | 9.7 | 7.6 | 6.0 | 4.9 | 0.227 |
| Hold still | 6.0 | 15.5 | 14.2 | 9.0 | **4.6** | 7.8 | 0.313 |

| steps 0–9 (0.33 s) | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| C | 2.0 | 3.8 | 4.1 | 3.4 | 2.5 | 3.0 |
| D | 2.1 | 4.6 | 4.8 | 4.3 | 3.3 | 3.1 |
| Hold still | 2.0 | 4.8 | 4.8 | 3.1 | 1.5 | 4.2 |

- **Both policies cut the hold-still error by 35–47% on the shoulder, elbow and gripper** over the chunk; C is 33% below the baseline overall.
- **`wrist_roll` is worse than doing nothing, again.** C's 5.0° against 4.6° for holding still over the chunk, and 2.5° against 1.5° over the first third of a second. The 2026-09-29 model showed the same thing on a different holdout. Two models, two holdouts, same joint: the policy adds noise on `wrist_roll` rather than tracking it. Worth remembering that this is the joint whose calibration the repo already flags as fragile ([02-setup §3b](../docs/robots/so-arm101/02-setup.md)).
- **Where the CVAE helps most is the wrist and the first steps.** D vs C: `wrist_roll` +19%, `wrist_flex` +15% over the chunk; `shoulder_lift` at step 0 is 3.8° vs 3.0°. The shoulder and gripper over the whole chunk are within 5%.
- Error still grows 2.5–3.5× from the first 10 steps to the last 10 on every arm joint. `--n-action-steps 15` remains the cheap mitigation.

### On the arm

**Not run.** No success rate. Everything above is prediction error on recorded demonstrations.

### Where the weights are

Local only, `pretrained_model` directories under `outputs/submit/`: `act_cubcyl_poshold_60k_step060000` and `act_cubcyl_poshold_novae_60k_step060000`. Per-checkpoint scores in `outputs/eval/act_cubcyl_poshold*_ckpts.csv`, per-joint detail in `outputs/eval/act_cubcyl_poshold_check.json`.

## Predictions vs outcome

| # | Prediction | Outcome |
|---|---|---|
| 1 | C's best held-out L1 in 0.15–0.22 | ✅ **0.210**, upper half of the range |
| 2 | White-cube blocks are the worst | ✅ barely — 0.231/0.233 against 0.219/0.223 for the cylinder. The clean finding is that the **red cube is much easier** (0.167/0.197), not that the white cube is much harder |
| 3 | D matches C within 5% | ❌ **D is 7.9% worse**, and worse at every checkpoint and on 24/30 episodes |
| 4 | If 3 fails, D is "much worse" | ❌ not that either — 8%, not the paper's 35% → 2% collapse |

## What we learned

**1. The CVAE is not dead weight on this data, and it is not what the paper describes either.** A latent that earlier runs measured as collapsed (KL ≈ 0; not re-measured here) still changed training enough to buy 8% held-out L1 at equal training loss. The honest reading is that the CVAE objective acts as a regulariser here rather than an information channel: the latent is zeroed at inference anyway, but training with it produced weights that generalise better to unseen start positions. That does not settle the paper's claim, which is about rollouts; it does mean the repo's working assumption that "collapsed = benign" was too strong.

**2. The position holdout is the right offline number for this dataset.** Every condition is in the test set, the per-block breakdown is readable, and the ranking it gives (red cube easy, white cube and cylinder harder) matches what the dataset card predicted from contrast. `eval_split=0.15` cannot produce any of this on a block-ordered dataset.

**3. 60k steps is still not the plateau.** Third run in a row where the last checkpoint is the best. One 100k run is overdue.

**4. The no-CVAE run is faster and smaller to train and that is all it buys.** 3.50 vs 3.43 step/s, 34M vs 52M trained parameters; identical inference cost. Not a reason to drop it.

**5. Training survives a PC going to sleep.** Both runs resumed after suspend-then-hibernate with nothing lost. The sleep itself is the problem to solve, not the training.

## What this run cannot tell us

- **Success rate.** Same caveat as every offline number in this repo: the ordering between held-out L1 and rollouts has inverted before ([2026-08-12](2026-08-12_dp-recovery-encoder-ab.md)).
- **Whether the CVAE matters on the arm.** The paper's 35% → 2% is a rollout number. If D matches C offline and still differs on the arm, the latent matters at execution time in a way L1 cannot see.
- **Object transfer.** Every object is in training here. That is [Axis 1](2026-08-06_act-cubes-cylinder-splits.md), not this run.

## Next

- [ ] **Rollouts** on C/60,000 and D/60,000, 20 per condition, canonical `phi_follower` calibration. The CVAE question is only answered on the arm.
- [ ] Measure the actual KL magnitude on run C — 8% from a latent we have been calling "≈ 0" says the number matters.
- [ ] Upload C/60,000 to the Hub and write its card in [`models/`](../models/).
- [ ] One 100k run on this split to find the plateau.
