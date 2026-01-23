# JRA Forcing Experiments

## Overview
Baseline and gradient-weighted experiments using JRA-55 forcing data on 360×360 domain.
Hindcast emulator for reproducing numerical ensembles with perturbed initial conditions.

## Experiments
- **mae_baseline**: MAE loss, 60 epochs
- **mae_grad_w05**: Gradient weighting (α=0.5), 60 epochs
- **mae_grad_w025**: Aggressive gradient weighting (α=0.25), 60 epochs

## Purpose
Evaluate model performance with realistic atmospheric forcing (JRA-55) instead of climatological forcing.
Test ability to reproduce numerical true ensembles with perturbed ICs starting from year 1990.

## Data Configuration

### Dataset
- **Data source**: `MOM6_CobaltDG_JRA_FULL`
- **Total extent**: 1960-2019 (60 years)
- **Domain**: 360×360

### Time Splits
- **Training**: 1960-01-01 to 1989-12-31 (30 years)
- **Year 1990**: EXCLUDED (reserved for ensemble IC perturbation tests, prevents data leakage)
- **Validation**: 1991-01-01 to 1995-12-31 (5 years)
- **Development buffer**: 1996-01-01 to 2000-12-31 (5 years, available for intermediate testing)
- **Paper holdout**: 2001-01-01 to 2019-12-31 (19 years, untouched for final paper results)

### Variables
- **Prognostic**: `full_state_all` (DIC, O2, NO3, PP, CHL, temp, salt, u, v at all depth levels + SSH)
- **Boundary forcing**: `standard_forcing` (Qnet, tauuo, tauvo, PRCmE)

## Wandb Configuration

### Setup for Compute Nodes Without Internet

The training uses **offline mode** since compute nodes lack internet access:

1. **Config settings** ([mae_grad_w05.yaml](mae_grad_w05.yaml)):
   ```yaml
   experiment:
     wandb:
       mode: offline
       project: bgc-emulator
       entity: mkeutgen
       group: mom6-bgc-training
   ```

2. **Authentication**:
   - Wandb API key stored in `~/.bashrc`
   - Training script sources `~/.bashrc` before execution

3. **Offline logging**:
   - Logs saved locally to `./outputs/<experiment_name>/wandb/`
   - No internet connection needed during training

### Syncing Logs After Training

From a machine with internet access:

```bash
# Sync a specific run
wandb sync ./outputs/mae_grad_w05_jra/wandb/offline-run-*

# Or sync all offline runs
wandb sync ./outputs/mae_grad_w05_jra/wandb --sync-all
```

## Running Experiments

```bash
# Submit training job
sbatch scripts/experiments/jra/train_mae_grad_w05.sh
```

The slurm script:
- Uses 8 nodes × 1 L40S GPU per node
- Sources `~/.bashrc` for wandb API key
- Runs in offline mode (logs synced later)
