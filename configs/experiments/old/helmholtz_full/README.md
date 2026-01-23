# Helmholtz Full Domain Experiments (360×360)

## Overview
Scaling Helmholtz decomposition to full 360×360 domain with memory optimizations.

## Experiments
- **mae_control_25lev**: 25 depth levels (skip2), MAE only
- **mae_grad_w01_25lev**: 25 depth levels (skip2), α=0.1
- **mae_grad_w025_25lev**: 25 depth levels (skip2), α=0.25

## Architecture Changes
- Reduced channel widths: [160, 220, 300] vs [320, 440, 600]
- Gradient checkpointing enabled
- Variable selection: removed PP, kept surface Chl only
