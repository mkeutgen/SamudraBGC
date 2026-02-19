# Training Configurations

Training experiment configs for the paper ablation studies.

## Research Questions

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

- **Evaluation configs**: `configs/eval/`
- **SLURM scripts**: `scripts/slurm/`

## Documentation

See [PAPER_EXPERIMENTS.md](../../PAPER_EXPERIMENTS.md) for full experiment descriptions.
