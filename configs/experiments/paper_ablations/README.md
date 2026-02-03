# Paper Ablation Experiments

This directory contains the **clean, final experiments** for the paper.

## Organization

All paper experiments are organized here to keep them separate from exploratory/archived work.

### Research Questions

**Question 1: Velocity Representation**
- `jra_helmholtz_min_grad05.yaml` - Helmholtz only (psi, phi)
- `jra_fullstate_min_grad05.yaml` - Full state (u, v)
- `jra_fullstate_helmholtz_min_grad05.yaml` - Both (u, v, psi, phi)

**Question 2: Gradient Penalty**
- `jra_helmholtz_min_grad05.yaml` - grad=0.5 (baseline)
- `jra_helmholtz_min_grad025.yaml` - grad=0.25
- `jra_helmholtz_min_grad010.yaml` - grad=0.10
- `jra_helmholtz_min_grad000.yaml` - grad=0.0 (pure MAE)

## Related Directories

- **Training configs**: `configs/experiments/paper_ablations/`
- **Evaluation configs**: `configs/eval/paper_ablations/`
- **Training scripts**: `scripts/experiments/paper_ablations/`
- **Evaluation scripts**: `scripts/experiments/paper_ablations/`
- **Output logs**: `scripts/experiments/paper_ablations/logs/`

## Archived Experiments

Old exploratory experiments from `jra_suite/` have been archived to:
- `configs/experiments/archived_jra_suite/`
- `configs/eval/archived_jra_suite/`

## Documentation

See [PAPER_EXPERIMENTS.md](../../../PAPER_EXPERIMENTS.md) in the root directory for:
- Full experiment descriptions
- Running instructions
- Expected outcomes
- Evaluation metrics
