# Paper Ablation Experiments

This document describes the experiments created for the paper analysis.

## Research Questions

### Question 1: What is the optimal velocity representation?
**Comparison of model architectures with minimal forcing and grad_weight=0.5**

| Experiment | Prognostic Variables | Description |
|------------|---------------------|-------------|
| `jra_helmholtz_min_grad05` | `helmholtz_only_all` (psi, phi) | **Helmholtz-only** - Uses decomposition |
| `jra_fullstate_min_grad05` | `full_state_all` (u, v) | **Full state** - Direct velocity prediction |
| `jra_fullstate_helmholtz_min_grad05` | `full_state_and_helmholtz_all` (u, v, psi, phi) | **Redundant** - Both representations |

**Hypothesis**: Helmholtz decomposition should perform best due to enforced divergence-free constraint.

**Expected Outcome**: 
- Best: `helmholtz_only` (cleaner representation)
- Worse: `full_state` (hard to enforce div-free)
- Worst/Middle: `full_state_helmholtz` (redundancy may help or hurt)

---

### Question 2: What is the optimal gradient penalty weight?
**Ablation of gradient penalty using Helmholtz-only with minimal forcing**

| Experiment | Gradient Weight | Description |
|------------|----------------|-------------|
| `jra_helmholtz_min_grad05` | 0.5 | **Baseline** - Current best practice |
| `jra_helmholtz_min_grad025` | 0.25 | **Reduced** - Less spatial regularization |
| `jra_helmholtz_min_grad010` | 0.10 | **Minimal** - Very weak regularization |
| `jra_helmholtz_min_grad000` | 0.0 | **None** - Pure MAE loss |

**Hypothesis**: Moderate gradient penalty (0.25-0.5) balances point-wise accuracy and spatial smoothness.

**Expected Outcome**:
- Best: 0.25 or 0.5 (good balance)
- Worse: 0.10 (too little regularization)
- Worst: 0.0 (checkerboard artifacts, poor spatial structure)

---

## Experiment Configuration

### Common Settings
- **Dataset**: MOM6_CobaltDG_JRA_FULL_POC_Helmholtz (60 years, 1958-2019)
- **Training Period**: 1960-2009 (50 years)
- **Validation Period**: 2010-2014 (5 years)
- **Test Period**: 2015-2019 (5 years holdout)
- **Boundary Variables**: `minimal_forcing` (Qnet, tauuo, tauvo - no PRCmE)
- **Model**: ConvNeXt U-Net with gradient-weighted MAE loss
- **Hardware**: 8 nodes × 1 L40S GPU (8 GPUs total)
- **Training Time**: ~72 hours per experiment

### Variable Counts

| Config | Prognostic Vars | Output Channels |
|--------|----------------|-----------------|
| `helmholtz_only_all` | dic, o2, no3, chl, temp, salt, psi, phi @ 50 levels + SSH | ~401 |
| `full_state_all` | dic, o2, no3, pp, chl, temp, salt, uo, vo @ 50 levels + SSH | ~451 |
| `full_state_and_helmholtz_all` | dic, o2, no3, chl, temp, salt, uo, vo, psi, phi, poc @ 50 levels + SSH | ~551 |

---

## Running the Experiments

### Training

```bash
# Question 1: Model architecture comparison
sbatch scripts/experiments/paper_ablations/train_jra_helmholtz_min_grad05.sh
sbatch scripts/experiments/paper_ablations/train_jra_fullstate_min_grad05.sh
sbatch scripts/experiments/paper_ablations/train_jra_fullstate_helmholtz_min_grad05.sh

# Question 2: Gradient penalty ablation
sbatch scripts/experiments/paper_ablations/train_jra_helmholtz_min_grad025.sh
sbatch scripts/experiments/paper_ablations/train_jra_helmholtz_min_grad010.sh
sbatch scripts/experiments/paper_ablations/train_jra_helmholtz_min_grad000.sh
```

### Evaluation

```bash
# Question 1: Model architecture comparison
sbatch scripts/experiments/paper_ablations/eval_jra_helmholtz_min_grad05.sh
sbatch scripts/experiments/paper_ablations/eval_jra_fullstate_min_grad05.sh
sbatch scripts/experiments/paper_ablations/eval_jra_fullstate_helmholtz_min_grad05.sh

# Question 2: Gradient penalty ablation
sbatch scripts/experiments/paper_ablations/eval_jra_helmholtz_min_grad025.sh
sbatch scripts/experiments/paper_ablations/eval_jra_helmholtz_min_grad010.sh
sbatch scripts/experiments/paper_ablations/eval_jra_helmholtz_min_grad000.sh
```

---

## Evaluation Metrics

All experiments will be evaluated on:

### Primary Metrics (Test Period 2015-2019)
- **RMSE** (Root Mean Square Error) per variable
- **Bias** (Mean Error) per variable  
- **Correlation** (Pattern correlation) per variable
- **Spatial gradients** (Smoothness metrics)

### Secondary Metrics
- **Ocean Heat Content (OHC)** - Conservation properties
- **ENSO Metrics** - Nino3.4 index correlation
- **Basin-specific statistics** - Regional performance
- **Rollout stability** - 25-day autoregressive performance

### Visualization
- Spatial maps (mean, bias, RMSE)
- Time series (global mean evolution)
- PDFs (distribution matching)
- Spectral diagnostics (spatial scales)

---

## File Locations

### Configurations
```
configs/experiments/paper_ablations/
├── jra_helmholtz_min_grad05.yaml
├── jra_fullstate_min_grad05.yaml
├── jra_fullstate_helmholtz_min_grad05.yaml
├── jra_helmholtz_min_grad025.yaml
├── jra_helmholtz_min_grad010.yaml
└── jra_helmholtz_min_grad000.yaml

configs/eval/paper_ablations/
├── jra_helmholtz_min_grad05_eval.yaml
├── jra_fullstate_min_grad05_eval.yaml
├── jra_fullstate_helmholtz_min_grad05_eval.yaml
├── jra_helmholtz_min_grad025_eval.yaml
├── jra_helmholtz_min_grad010_eval.yaml
└── jra_helmholtz_min_grad000_eval.yaml
```

### Scripts
```
scripts/experiments/paper_ablations/
├── train_jra_helmholtz_min_grad05.sh
├── train_jra_fullstate_min_grad05.sh
├── train_jra_fullstate_helmholtz_min_grad05.sh
├── train_jra_helmholtz_min_grad025.sh
├── train_jra_helmholtz_min_grad010.sh
├── train_jra_helmholtz_min_grad000.sh
├── eval_jra_helmholtz_min_grad05.sh
├── eval_jra_fullstate_min_grad05.sh
├── eval_jra_fullstate_helmholtz_min_grad05.sh
├── eval_jra_helmholtz_min_grad025.sh
├── eval_jra_helmholtz_min_grad010.sh
└── eval_jra_helmholtz_min_grad000.sh
```

### Outputs
```
outputs/
├── jra_helmholtz_min_grad05/
├── jra_fullstate_min_grad05/
├── jra_fullstate_helmholtz_min_grad05/
├── jra_helmholtz_min_grad025/
├── jra_helmholtz_min_grad010/
└── jra_helmholtz_min_grad000/
```

---

## Notes

- All experiments use the same random seed (42) for reproducibility
- Models checkpoint every 5 epochs
- EMA (Exponential Moving Average) checkpoints are used for evaluation
- Evaluation produces zarr files with full rollout predictions
- W&B logging is set to offline mode (group: mom6-bgc-training-jra60)

