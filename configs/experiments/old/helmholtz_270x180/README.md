# Helmholtz Decomposition Experiments (270×180)

## Overview
Testing Helmholtz decomposition (ψ/φ) on reduced domain with different loss configurations.

## Experiments
- **mae_grad_w01**: Conservative gradient weighting (α=0.1)
- **mae_grad_w025**: Aggressive gradient weighting (α=0.25)
- **mae_grad_60ep**: Unweighted MAE+gradient, extended training
- **mae_control_60ep**: MAE only, extended training (control)

## Key Questions
1. Can weighted gradient loss preserve sharp features without bias?
2. Does extended training help convergence for multi-objective losses?
