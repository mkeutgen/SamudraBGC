# Changelog

All notable changes to SamudraBGC will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.0.0] - 2026-07-10

### Added

- Initial public release accompanying the GRL paper
- ConvNeXt U-Net architecture for ocean biogeochemistry emulation
- Support for Helmholtz decomposition (streamfunction + velocity potential)
- PCA-based vertical compression (5-20 components)
- Log-transform preprocessing for biogeochemical tracers
- Gradient penalty loss for improved spatial coherence
- 50-member ensemble generation via perturbed initial conditions
- Comprehensive evaluation metrics (RMSE, bias, R², correlations)
- Paper figure generation scripts in `code_paper/`

### Model Architecture

- Based on Samudra (Subel et al., 2024) with modifications for BGC
- Channel widths: [320, 440, 600] (baseline configuration)
- Input: Ocean state at t-1, t plus atmospheric forcing
- Output: Predicted state at t+1, t+2

### Training Configuration

- 14 ablation configurations exploring design choices
- Champion model: Helmholtz + Log BGC + Gradient weight 0.10 + 20 PCA components
- Training: 50 epochs, Adam optimizer, cosine annealing, lr=2e-4

### Data

- DG-MOM6-COBALTv2 double gyre simulation (1960-2019)
- Training: 1960-2009, Validation: 2010-2014, Test: 2015-2019
- Variables: T, S, velocities, SSH, DIC, O2, NO3, Chl
