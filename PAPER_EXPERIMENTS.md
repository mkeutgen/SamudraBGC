# Paper Ablation Experiments

This document provides a quick reference to the ablation study described in the GRL manuscript.

For detailed documentation including config-to-experiment mappings, reproduction instructions, and seed sensitivity analysis, see **[docs/EXPERIMENTS.md](docs/EXPERIMENTS.md)**.

## Ablation Tree Summary

The paper evaluates 14 configurations across 5 sequential design choices:

```
#1 Velocity (u,v)           R²=-0.07  ──┐
                                        ├─► Phase 1: Circulation
#2 Helmholtz (ψ,φ)          R²=0.50  ──┘   Winner: #2
        │
        ▼
#2 Linear BGC               R²=0.50  ──┐
                                        ├─► Phase 1.5: BGC Transform
#3 Log BGC                  R²=0.63  ──┘   Winner: #3
        │
        ▼
#4 Grad Weight 0            R²=0.75  ──┐
#5 Grad Weight 0.10         R²=0.80  ──┼─► Phase 2: Gradient Penalty
#6 Grad Weight 0.25         R²=0.78  ──┤   Winner: #5
#7 Grad Weight 0.50         R²=0.76  ──┘
        │
        ▼
#8  5 components            R²=0.70  ──┐
#9  10 components           R²=0.78  ──┼─► Phase 5: PCA Compression
#10 15 components           R²=0.82  ──┤   Winner: #11
#11 20 components           R²=0.81  ──┘
        │
        ▼
#11 SamudraBGC (baseline)   R²=0.81  ──┐
#12 Wider                   R²=0.81  ──┼─► Phase 7: Architecture
#13 Much Wider              R²=0.78  ──┤   Winner: #11 (no gain)
#14 Wider+Deeper            R²=0.81  ──┘
```

## Config Files

| # | Experiment | Training Config | Eval Config |
|---|------------|-----------------|-------------|
| 1 | Velocity | `phase1_fullstate_nograd.yaml` | `phase1_velocity_nograd_eval.yaml` |
| 2 | Helmholtz | `phase1_helmholtz_nograd.yaml` | `phase1_helmholtz_nograd_eval.yaml` |
| 3 | Log BGC | `phase15_helmholtz_log_all.yaml` | `phase15_helmholtz_log_eval.yaml` |
| 4 | Grad 0 | `phase2_helmholtz_grad00.yaml` | `phase2_helmholtz_grad00_eval.yaml` |
| 5 | Grad 0.10 | `phase2_helmholtz_grad010.yaml` | `phase2_helmholtz_grad010_eval.yaml` |
| 6 | Grad 0.25 | `phase2_helmholtz_grad025.yaml` | `phase2_helmholtz_grad025_eval.yaml` |
| 7 | Grad 0.50 | `phase2_helmholtz_grad050.yaml` | `phase2_helmholtz_grad050_eval.yaml` |
| 8 | PCA 5 | `phase5_pca5_helmholtz_grad010.yaml` | `phase5_pca5_helmholtz_grad010_eval.yaml` |
| 9 | PCA 10 | `phase5_pca10_helmholtz_grad010.yaml` | `phase5_pca10_helmholtz_grad010_eval.yaml` |
| 10 | PCA 15 | `phase5_pca15_helmholtz_grad010.yaml` | `phase5_pca15_helmholtz_grad010_eval.yaml` |
| 11 | PCA 20 | `phase5_pca20_helmholtz_grad010.yaml` | `phase5_pca20_helmholtz_grad010_eval.yaml` |
| 12 | Wider | `phase7_pca20_arch_wider.yaml` | `phase7_pca20_arch_wider_eval_rollout2010_2014.yaml` |
| 13 | Much Wider | `phase7_pca20_arch_much_wider.yaml` | `phase7_pca20_arch_much_wider_eval_rollout2010_2014.yaml` |
| 14 | Wider+Deeper | `phase7_pca20_arch_wider_deeper.yaml` | `phase7_pca20_arch_wider_deeper_eval_rollout2010_2014.yaml` |

**Champion model** (trained on train+val, evaluated on test):
- Training: `phase5_pca20_helmholtz_grad010_full.yaml`
- Evaluation: `champion_model_eval_rollout2015_2019.yaml`

## Evaluation Periods

| Period | Years | Purpose |
|--------|-------|---------|
| Training | 1960-2009 | Model training |
| Validation | 2010-2014 | Model selection (ablation comparisons) |
| Test | 2015-2019 | Final evaluation (holdout) |

## Running All Ablations

```bash
# Phase 1: Circulation
sbatch scripts/slurm/train_phase1_fullstate_nograd.sh
sbatch scripts/slurm/train_phase1_helmholtz_nograd.sh

# Phase 1.5: BGC Transform
sbatch scripts/slurm/train_phase15_helmholtz_log.sh

# Phase 2: Gradient Weight
sbatch scripts/slurm/train_phase2_helmholtz_grad00.sh
sbatch scripts/slurm/train_phase2_helmholtz_grad010.sh
sbatch scripts/slurm/train_phase2_helmholtz_grad025.sh
sbatch scripts/slurm/train_phase2_helmholtz_grad050.sh

# Phase 5: PCA Compression
sbatch scripts/slurm/train_phase5_pca5_helmholtz_grad010.sh
sbatch scripts/slurm/train_phase5_pca10_helmholtz_grad010.sh
sbatch scripts/slurm/train_phase5_pca15_helmholtz_grad010.sh
sbatch scripts/slurm/train_phase5_pca20_helmholtz_grad010.sh

# Phase 7: Architecture
sbatch scripts/slurm/train_phase7_pca20_arch_wider.sh
sbatch scripts/slurm/train_phase7_pca20_arch_much_wider.sh
sbatch scripts/slurm/train_phase7_pca20_arch_wider_deeper.sh

# Champion (train+val)
sbatch scripts/slurm/train_phase5_pca20_helmholtz_grad010_full.sh
```
