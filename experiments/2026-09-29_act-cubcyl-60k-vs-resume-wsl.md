# ACT on cubes+cylinder — club-default 60k vs a 20k run resumed from a Mac, on a home PC

**Status**: ✅ both runs complete, scored offline · ❌ no rollouts yet · 2026-09-29 (re-scored 2026-10-01)
**Author**: Gautam · **Ladder**: L4
**Dataset**: [`phi_so101_cubes_cylinder_v1`](../datasets/phi_so101_cubes_cylinder_v1.md) — 120 episodes, 66,873 frames, [on the Hub](https://huggingface.co/datasets/BrutalCaesar/phi_so101_cubes_cylinder_v1)
**Hardware**: one RTX 5060 Ti 16 GB, Windows 11 + WSL2 (Ubuntu 26.04) · torch 2.11.0+cu128 · lerobot 0.6.0
**Weights**: local only, not on the Hub yet — see [Results](#results)

## Question

1. With the competition config held fixed, does the **club-default 60,000-step** run beat a **20,000-step** run, and which checkpoint should be submitted?
2. Can a LeRobot checkpoint that was started on **Apple-silicon MPS** be resumed on **CUDA** with nothing changed but `--policy.device`?
3. Is a single consumer GPU under WSL2 enough to do this without Explorer?

## Setup

Both runs share the competition config. Nothing below was varied.

| | |
|---|---|
| Policy | ACT, lerobot 0.6.0 defaults · `use_vae=true`, `kl_weight=10.0`, ResNet-18 ImageNet · **51,571,590 params** |
| `chunk_size` / `n_action_steps` | 50 / 50 |
| Batch / lr / seed | 8 / 1e-5 / 1000 |
| Cameras | `wrist`, `front`, `top` — 3 × 480×640, keys correct in this dataset |
| Image augmentation | off |
| Holdout | `eval_split=0.15` → **102 train episodes (58,446 frames) / 18 held out (8,427 frames)** |

The two runs:

| | **A** — `act_cubcyl_60k` | **B** — `act_cubcyl_m4` |
|---|---|---|
| Steps | 60,000 (8.2 epochs) | 20,000 (2.7 epochs) |
| Where it trained | PC, from scratch | steps 0–5,000 on a MacBook M4 (MPS), **5,000–20,000 on the PC (CUDA)** |
| `save_freq` | 10,000 → 6 checkpoints | 2,500 → 7 checkpoints (5,000 is the Mac's) |
| `eval_steps` | 10,000 | 0 — lerobot logged **no** `eval_loss` for this run |
| Wall time on the PC | 5 h 11 m of training | 1 h 17 m for 15,000 steps |
| Final train loss | 0.073 | 0.121 |

**Run A was interrupted once, on purpose.** It started at 13:33, saved its 10,000 checkpoint at 14:25, and was stopped near step 12,800 to give the GPU to run B (the guaranteed-model path in the handoff). It was continued at 15:58 with `--resume=true` from its own 10,000 checkpoint and finished at 20:17. About 2,800 steps (14 min) were trained twice; the optimizer, RNG and data order were restored from the checkpoint (`Resuming data order at epoch 1, sample 21552`).

## Exact commands

Environment (WSL2). The repo's `env/environment.cuda.yml` pins cu126, which has no kernels for the 50-series (`sm_120`), so this run used the cu128 variant now committed as [`env/environment.cuda-cu128.yml`](../env/environment.cuda-cu128.yml) — see Infrastructure below.

```bash
git clone https://github.com/gautampatel1/phi.git ~/phi && cd ~/phi     # trained at commit 1ef6c73
conda env create -f env/environment.cuda-cu128.yml && conda activate phi && pip install -e .
export HF_LEROBOT_HOME=~/lerobot-data
```

Run A, fresh:

```bash
lerobot-train --dataset.repo_id=BrutalCaesar/phi_so101_cubes_cylinder_v1 --dataset.eval_split=0.15 \
  --policy.type=act --policy.device=cuda --policy.push_to_hub=false \
  --policy.chunk_size=50 --policy.n_action_steps=50 \
  --batch_size=8 --steps=60000 --save_freq=10000 --eval_steps=10000 --seed=1000 --num_workers=8 \
  --output_dir=outputs/train/act_cubcyl_60k --job_name=act_cubcyl_60k --wandb.enable=false
```

Run A, continued after the interruption:

```bash
lerobot-train --config_path=outputs/train/act_cubcyl_60k/checkpoints/last/pretrained_model/train_config.json --resume=true
```

Run B, resumed from the Mac's `checkpoint_005000` (staged at `outputs/train/act_cubcyl/checkpoints/005000`, with `last -> 005000`):

```bash
lerobot-train --config_path=outputs/train/act_cubcyl/checkpoints/last/pretrained_model/train_config.json \
  --resume=true --policy.device=cuda --num_workers=8
```

Scoring, every checkpoint of each run on the **whole** holdout (`--max-batches 0` was needed at the time; it is the default now):

```bash
python -m phi.eval.loss_by_checkpoint --run outputs/train/act_cubcyl_60k --device cuda \
  --max-batches 0 --batch-size 32 --out outputs/eval/act_cubcyl_60k_ckpts_full.csv
python -m phi.eval.loss_by_checkpoint --run outputs/train/act_cubcyl_m4 --device cuda \
  --max-batches 0 --batch-size 32 --out outputs/eval/act_cubcyl_m4_ckpts_full.csv
python -m phi.eval.check_holdout --out outputs/eval/act_cubcyl_check.json \
  outputs/train/act_cubcyl_60k/checkpoints/040000 outputs/train/act_cubcyl_60k/checkpoints/050000 \
  outputs/train/act_cubcyl_60k/checkpoints/060000 outputs/train/act_cubcyl_m4/checkpoints/017500
```

No `lerobot-rollout` command: nothing has been run on the arm.

## Results

### Held-out L1, whole holdout (8,427 frames, 18 episodes)

Normalized L1 over the 50-step chunk, padding excluded, lower is better. Same frames for all 13 checkpoints.

| Run A (60k) | L1 | logged `eval_loss` | | Run B (resumed) | L1 |
|---|---:|---:|---|---|---:|
| 10,000 | 0.2750 | 0.2602 | | **5,000 (Mac's)** | **0.2724** |
| 20,000 | 0.2932 | 0.2790 | | 7,500 | 0.2757 |
| 30,000 | 0.2702 | 0.2558 | | 10,000 | 0.2772 |
| 40,000 | 0.2602 | 0.2459 | | 12,500 | 0.2870 |
| 50,000 | 0.2613 | 0.2466 | | 15,000 | 0.2780 |
| **60,000** | **0.2537** | **0.2394** | | 17,500 | 0.2781 |
| | | | | 20,000 | 0.2742 |

- **Best checkpoint overall: run A, step 60,000.** It is 2.5% better than 40,000, and run B's best is 7% worse than it.
- **The two independent measures agree on run A's ordering.** lerobot's logged `eval_loss` and the post-hoc L1 rank the six checkpoints identically. The logged number is about 5.5% lower throughout because ACT averages over padded positions too (explained in [2026-09-05](2026-09-05_act-pen-chunk-50-100.md)).
- **Run B never improved on held-out data after the Mac handed it over.** Train loss fell from 0.37 to 0.12 across steps 5,000–20,000 while held-out L1 went from 0.272 to 0.274. Run A needed 30,000 steps before it moved below that level.
- **The curve is not monotonic.** Run A's step 20,000 is its worst checkpoint (0.293), worse than 10,000. Do not read a trend off one checkpoint.

### 🚨 The first scoring pass picked the wrong checkpoint

On 2026-09-29 the same tool was run with `--max-batches 100` (the docs suggest 40). The loader is not shuffled, so that scored **only the first 800 of 8,427 held-out frames — under two episodes**:

| | first 800 frames | whole holdout |
|---|---|---|
| Best of run A | 40,000 (0.2582) | **60,000 (0.2537)** |
| 60,000 vs 40,000 | 3.8% **worse** | 2.5% **better** |
| Best of run B | 17,500 (0.2709) | 5,000 (0.2724) |

The partial pass reported that run A peaked early and warned against deploying the final step. That conclusion was wrong for this run, and step 40,000 had already been staged for submission on the strength of it.

### Per-joint error in real units — run A, step 60,000

Mean absolute error on the same 8,427 held-out frames, un-normalized: **degrees** for the five arm joints, the 0–100 scale for the gripper. "Hold still" is a do-nothing baseline that repeats the current joint state for the whole chunk.

| | shoulder_pan | shoulder_lift | elbow_flex | wrist_flex | wrist_roll | gripper |
|---|---:|---:|---:|---:|---:|---:|
| Policy, step 0 | 1.5 | 2.8 | 3.3 | 2.7 | 2.0 | 2.7 |
| Policy, steps 0–9 (0.33 s) | 2.1 | 3.8 | 4.6 | 3.8 | 3.0 | 3.6 |
| Policy, steps 40–49 | 5.2 | 13.9 | 15.8 | 12.4 | 12.1 | 6.9 |
| **Policy, whole chunk** | **3.8** | **8.7** | **10.3** | **8.4** | **7.9** | **5.3** |
| Hold still, whole chunk | 7.9 | 19.7 | 19.1 | 10.9 | 7.8 | 10.5 |
| Hold still, steps 0–9 | 2.6 | 6.0 | 6.3 | 3.7 | 2.5 | 5.8 |

- **It is predicting real motion.** Normalized L1 is 0.254 against 0.423 for holding still, 40% lower. On the shoulder, elbow and gripper the policy's error is about half the baseline's.
- **The wrist is the weak part.** On `wrist_roll` the policy is no better than not moving (7.9° vs 7.8°), and over the first 0.33 s both wrist joints are slightly *worse* than not moving. `wrist_flex` is only 23% better over the chunk.
- **Error grows about 3–4× across the chunk on the arm joints** (elbow 4.6° → 15.8°), the same open-loop drift measured in [2026-09-05](2026-09-05_act-pen-chunk-50-100.md). Executing all 50 actions before re-planning uses the worst predictions; `--n-action-steps 15` does not.
- **Per held-out episode** the normalized L1 runs from 0.206 to 0.344. Episodes 105, 109 and 118 are the worst for every checkpoint checked, so that spread is about the episodes, not the checkpoint.
- Against run B's step 17,500, run A's 60,000 is better on every joint, most on `elbow_flex` (10.3° vs 11.7°) and `wrist_roll` (7.9° vs 9.0°).

The tool is [`phi.eval.check_holdout`](../src/phi/eval/check_holdout.py), committed with this write-up; it reuses `loss_by_checkpoint._load` and reproduces that tool's 0.2537 exactly. Raw output: `outputs/eval/act_cubcyl_check.json`.

For scale: the same checkpoint scores **0.079** on training episodes 0–1 (`--eval-episodes 0,1`) against 0.254 held out — a 3.2× gap. It has fitted the demonstrations it saw far more closely than it predicts the ones it did not.

### On the arm

**Not run.** No success rate, no rollout scores, nothing in `outputs/rollout_scores.csv`. Everything above is prediction error on recorded demonstrations.

For expectation-setting only: the earlier ACT baseline on this dataset (`cvae_3cam`, 100k steps, see [models/](../models/README.md)) scored 27% on held-out rollouts and 0/4 on the white cube. Nothing measured here says this model will do better or worse than that.

### Where the weights are

Not uploaded, so there is no Hub id, no revision and no model card yet. Local copies of the `pretrained_model` directories:

| | Path under `outputs/submit/` |
|---|---|
| Recommended | `act_cubcyl_60k_step060000` |
| Staged earlier, now second choice | `act_cubcyl_60k_step040000` |
| Run B | `act_cubcyl_resume_step017500` |

## What we learned

**1. The holdout is one condition, not the task.** `eval_split` holds out the *last* `ceil(n × split)` episodes per task string. This dataset has one task string and is ordered by block, so the 18 held-out episodes are **102–119: all white cube → white bin**. Training saw exactly two episodes of that block (100, 101). Every held-out number on this page therefore measures a nearly unseen object-and-bin pairing — and the hardest one, since the white cube is the low-contrast object. It says nothing about red cube or cylinder, where the model may be much better. This applies to every run that uses `eval_split` on this dataset, including the competition script.

**2. `--max-batches N` was a prefix, not a sample.** With an unshuffled loader, N batches of 8 was the first 8N frames. At the then-documented `--max-batches 40` that is 320 frames, less than one episode. Here it flipped the checkpoint ranking in both runs. Fixed alongside this write-up: the cap now takes evenly spaced frames across the holdout and the default is the whole holdout, which cost 2.5 min per checkpoint at batch 32 on this GPU. A 320-frame spread sample of the same checkpoint scores 0.2565, within 1.1% of the full 0.2537.

**3. More steps helped; the 20k run was not enough.** Run A's held-out L1 kept falling through 60,000 with no sign of a turn, so the question "would 100k be better still" is open. Run B's flat curve is what under-training looks like here, not over-fitting.

**4. MPS → CUDA resume works.** The Mac checkpoint loaded and continued on CUDA with only `--policy.device=cuda` changed; loss picked up at 0.37 where the Mac left it. One quirk: lerobot honours the `output_dir` saved in the checkpoint's config (`act_cubcyl_m4`), not the folder the checkpoint was staged in, so the new checkpoints landed in `outputs/train/act_cubcyl_m4/`.

**5. A 16 GB consumer card is enough for ACT.** 3.28 step/s at batch 8, steady for five hours — 60k steps in 5 h 11 m. The M4 managed 2.63 s/step, about 8.6× slower, and crashed under memory pressure shortly after step 5,000.

### Infrastructure notes for the next person on Windows

- **50-series GPUs need the cu128 wheels.** cu126 installs cleanly and then `torch.cuda.is_available()` is `False`.
- **torchcodec from the cu128 index needs `libnppicc`** and fails to import. The plain PyPI build (`torchcodec==0.11.1`, `--no-deps`) decodes fine, with `ffmpeg<9`.
- **conda 26 refuses to run while `defaults` is in the channel list.** Removing the channel (`conda config --system --remove channels defaults`, `nodefaults` in the yml) fixes it without accepting anything.
- **WSL shuts the distro down when the last `wsl.exe` exits**, which kills `nohup`'d training. Keep one `wsl.exe` attached for the life of the run.
- **Windows sleeps mid-run on the default power plan.** Something has to hold the system awake while training.

## Next

- [ ] **Rollouts.** 20 per condition under the canonical `phi_follower` calibration, reported per object × bin, never averaged. Candidates: A/60,000, A/40,000, and B/20,000 as the under-trained reference. Try `--n-action-steps 15` as well as 50.
- [ ] Upload A/60,000 to the Hub and write its card in [`models/`](../models/).
- [x] Commit the cu128 env spec (`env/environment.cuda-cu128.yml`) and the per-joint check (`phi.eval.check_holdout`).
- [x] Make `loss_by_checkpoint` sample across the whole holdout and fix the `--max-batches 40` advice in [docs/training](../docs/training/README.md).
- [ ] A holdout that covers all six blocks — **running**: [2026-10-05](2026-10-05_act-cubcyl-poshold-cvae-ab-wsl.md) uses the position split from [2026-08-06](2026-08-06_act-cubes-cylinder-splits.md), with and without the CVAE.
- [ ] One longer run (100k) to find where run A's held-out curve actually turns.
