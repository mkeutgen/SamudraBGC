# JRA Forcing Experiments

## Overview
Baseline and gradient-weighted experiments using JRA-55 forcing data on 270×180 domain.

## Experiments
- **mae_baseline**: MAE loss, 60 epochs
- **mae_grad_w025**: Aggressive gradient weighting (α=0.25), 60 epochs

## Purpose
Evaluate model performance with realistic atmospheric forcing (JRA-55) instead of climatological forcing.

## Data
- Training period: 1960-01-01 to 1988-12-31
- Validation period: 1989-01-01 to 1989-12-31
- Inference: 1992-01-01 to 1992-06-30
- Data source: MOM6_CobaltDG_JRA
