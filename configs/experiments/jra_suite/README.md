# JRA 60-Year Experiment Suite

Comprehensive experiment suite for training ocean biogeochemical emulators on the 60-year JRA-forced MOM6 Double Gyre dataset.

## Overview

**Total experiments: 8**
- **Phase 1** (4 experiments): Prognostic variables & forcing comparison
- **Phase 2** (4 experiments): Loss functions with second-order penalties

**Goal**: Build the best hindcast emulator capable of reproducing numerical ensembles with perturbed initial conditions.

## Dataset Configuration

All experiments use the same dataset splits to ensure fair comparison:

- **Dataset**: `MOM6_CobaltDG_JRA_FULL` (1960-2019, 60 years)
- **Domain**: 360×360
- **Training**: 1960-1989 (30 years)
- **Year 1990**: EXCLUDED (reserved for ensemble IC tests, prevents data leakage)
- **Validation**: 1991-1995 (5 years)
- **Paper Holdout**: 2001-2019 (19 years, untouched for final paper results)

### Fixed Settings

```yaml
experiment:
  rand_seed: 42  # Fixed across ALL experiments for reproducibility
  wandb:
    mode: offline
    project: bgc-emulator
    entity: mkeutgen
    group: mom6-bgc-training-jra60

epochs: 60
batch_size: 1
learning_rate: 0.0002
scheduler:
  type: cosine
```

## Phase 1: Prognostic Variables & Forcing (4 Experiments)

### 1.1 Full State with Raw Velocities
**Config**: `jra_fullstate_grad05.yaml`
**SLURM**: `train_jra_fullstate_grad05.sh`

```yaml
prognostic_vars_key: full_state_all
boundary_vars_key: standard_forcing
gradient_weight: 0.5
second_order_weight: 0.0
```

**Hypothesis**: Baseline performance with raw u,v velocities
**Expected**: Moderate performance; velocities are hard for UNet to predict
**Channels**: ~451 (dic, o2, no3, pp, chl, temp, salt, uo, vo @ 50 levels + SSH)

---

### 1.2 Helmholtz + Standard Forcing ⭐ EXPECTED WINNER
**Config**: `jra_helmholtz_std_grad05.yaml`
**SLURM**: `train_jra_helmholtz_std_grad05.sh`

```yaml
prognostic_vars_key: optimized_helmholtz_all
boundary_vars_key: standard_forcing
gradient_weight: 0.5
second_order_weight: 0.0
```

**Hypothesis**: Smoother psi,phi representation improves velocity prediction
**Expected**: **WINNER** based on CLIM 10yr results
**Channels**: ~301 (dic, o2, temp, salt, psi, phi @ 50 levels + SSH + chl_0)

---

### 1.3 Helmholtz + Minimal Forcing
**Config**: `jra_helmholtz_min_grad05.yaml`
**SLURM**: `train_jra_helmholtz_min_grad05.sh`

```yaml
prognostic_vars_key: optimized_helmholtz_all
boundary_vars_key: minimal_forcing  # No PRCmE
gradient_weight: 0.5
second_order_weight: 0.0
```

**Hypothesis**: Test if corrected PRCmE normalization helps on 60yr dataset
**Expected**: Slightly worse than standard_forcing, especially for salinity
**Purpose**: Ablation study

---

### 1.4 Full State + Helmholtz 🃏 WILD CARD
**Config**: `jra_fullstate_helmholtz_grad05.yaml`
**SLURM**: `train_jra_fullstate_helmholtz_grad05.sh`

```yaml
prognostic_vars_key: full_state_and_helmholtz_all
boundary_vars_key: standard_forcing
gradient_weight: 0.5
second_order_weight: 0.0
```

**Hypothesis**: Model can use psi,phi for smooth patterns, u,v for sharp features
**Expected**: Wild card - might win if redundancy helps, or fail if too complex
**Channels**: ~551 (dic, o2, no3, pp, chl, temp, salt, uo, vo, psi, phi @ 50 levels + SSH)

---

## Phase 2: Loss Functions with Second-Order Penalties (4 Experiments)

**Goal**: Preserve fine-scale structures (eddies, fronts) that remain blurry with first-order gradient loss.

**Note**: Update `prognostic_vars_key` and `boundary_vars_key` with Phase 1 winner before running Phase 2.

### 2.1 First-Order Only (Baseline)
**Config**: `jra_best_grad05_so00.yaml`
**SLURM**: `train_jra_best_grad05_so00.sh`

```yaml
gradient_weight: 0.5
second_order_weight: 0.0  # Disabled
```

**Hypothesis**: Reproduces CLIM result showing persistent blurriness
**Expected**: Good R² but blurry eddies/fronts
**Purpose**: Reference point for Phase 2 comparison

---

### 2.2 Conservative Second-Order ⭐ RECOMMENDED
**Config**: `jra_best_grad05_so005.yaml`
**SLURM**: `train_jra_best_grad05_so005.sh`

```yaml
gradient_weight: 0.5
second_order_weight: 0.05  # Light curvature penalty
```

**Hypothesis**: Light curvature penalty sharpens features without stability issues
**Expected**: **Best balance** of accuracy and sharpness
**Purpose**: Recommended starting point for second-order experiments

---

### 2.3 Aggressive Second-Order
**Config**: `jra_best_grad05_so01.yaml`
**SLURM**: `train_jra_best_grad05_so01.sh`

```yaml
gradient_weight: 0.5
second_order_weight: 0.1  # Moderate/aggressive
```

**Hypothesis**: Stronger Laplacian penalty for maximum sharpness
**Expected**: Sharpest features but might sacrifice R² or stability
**Purpose**: Test upper limit of second-order penalty

---

### 2.4 Balanced Penalties
**Config**: `jra_best_grad025_so025.yaml`
**SLURM**: `train_jra_best_grad025_so025.sh`

```yaml
gradient_weight: 0.25   # Reduced first-order
second_order_weight: 0.25  # Equal second-order
```

**Hypothesis**: Equal weighting of gradient and curvature gives best balance
**Expected**: Alternative winner - might balance accuracy and sharpness better
**Purpose**: Test if reducing first-order while adding second-order improves balance

---

## Running Experiments

### Quick Start

```bash
# Phase 1: Run all prognostic variable experiments
cd scripts/experiments/jra_suite
sbatch train_jra_fullstate_grad05.sh
sbatch train_jra_helmholtz_std_grad05.sh
sbatch train_jra_helmholtz_min_grad05.sh
sbatch train_jra_fullstate_helmholtz_grad05.sh

# After Phase 1 completes, analyze results and pick winner
# Update Phase 2 configs with winner's prognostic_vars_key and boundary_vars_key

# Phase 2: Run all loss function experiments
sbatch train_jra_best_grad05_so00.sh
sbatch train_jra_best_grad05_so005.sh
sbatch train_jra_best_grad05_so01.sh
sbatch train_jra_best_grad025_so025.sh
```

### Evaluation

```bash
# After training completes
sbatch eval_jra_fullstate_grad05.sh
sbatch eval_jra_helmholtz_std_grad05.sh
# ... etc
```

### Syncing Wandb Logs

From a machine with internet access:

```bash
# Sync all offline runs
wandb sync outputs/*/wandb/offline-run-* --sync-all
```

## File Structure

```
configs/experiments/jra_suite/
├── README.md                              # This file
├── jra_fullstate_grad05.yaml              # 1.1 Full state + raw velocities
├── jra_helmholtz_std_grad05.yaml          # 1.2 Helmholtz + standard forcing ⭐
├── jra_helmholtz_min_grad05.yaml          # 1.3 Helmholtz + minimal forcing
├── jra_fullstate_helmholtz_grad05.yaml    # 1.4 Full + Helmholtz wild card
├── jra_best_grad05_so00.yaml              # 2.1 First-order only (baseline)
├── jra_best_grad05_so005.yaml             # 2.2 Conservative second-order ⭐
├── jra_best_grad05_so01.yaml              # 2.3 Aggressive second-order
└── jra_best_grad025_so025.yaml            # 2.4 Balanced penalties

scripts/experiments/jra_suite/
├── train_jra_*.sh                         # Training scripts (8)
├── eval_jra_*.sh                          # Evaluation scripts (8)
└── generate_slurm_scripts.py              # Script generator
```

## Expected Outcomes

### Phase 1 Winner
- **Most likely**: `jra_helmholtz_std_grad05` (Helmholtz decomposition + standard forcing)
- Based on previous CLIM 10yr experiments showing superior performance

### Phase 2 Winner
- **Most likely**: `jra_best_grad05_so005` (Conservative second-order penalty)
- Should provide sharpness improvements without sacrificing stability

## Technical Notes

### Second-Order Penalty Implementation
The `mae_gradient_weighted` loss function supports a `second_order_weight` parameter:

```python
Loss = MAE(pred, target)
     + α * gradient_penalty(pred, target)
     + β * laplacian_penalty(pred, target)
```

Where:
- α = `gradient_weight` (first-order spatial gradients)
- β = `second_order_weight` (Laplacian/curvature)

The second-order term helps preserve:
- Eddy centers (local extrema)
- Curvature of fronts
- Spatial smoothness structure

### Hardware Requirements
- 8 nodes × 1 L40S GPU per node
- 400GB RAM per node
- ~72 hours per training run

## Next Steps After Experiments

1. **Analyze Phase 1 results**
   - Compare validation metrics
   - Identify best prognostic_vars_key and boundary_vars_key

2. **Update Phase 2 configs**
   - Replace `optimized_helmholtz_all` with Phase 1 winner
   - Update `boundary_vars_key` if needed

3. **Run Phase 2 experiments**
   - Test second-order penalties
   - Identify best loss configuration

4. **Final evaluation**
   - Use paper holdout period (2001-2019)
   - Test ensemble reproduction from year 1990
   - Generate publication figures
