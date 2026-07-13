# Paper Figures — SamudraBGC

Scripts for generating figures in the SamudraBGC manuscript (GRL submission).

## Naming Conventions

Per CLAUDE.md conventions:
- **Ground Truth**: MOM6-COBALTv2 simulation (never "MOM6-DG" or "GT")
- **SamudraBGC**: ML emulator predictions (never "ML Emulator" or "Haddock")
- Ground Truth panels on the left, SamudraBGC on the right

## Main Figures

| Script | Description | Paper Figure |
|--------|-------------|--------------|
| `fig01_3d_schematic.py` | Architecture schematic with 3D ocean state visualization | Fig. 1a |
| `fig03_ablation_tree.py` | Ablation tree showing sequential design choices | Fig. 1b |
| `fig02.py` | Best model BGC performance: O₂ snapshots, DIC sections, time series, PDFs | Fig. 2 |
| `figure04_combined/fig04_combined.py` | Design choices: DIC/O₂/NO₃ snapshots, spectra, bias, RMSE vs depth | Fig. 3-4 |
| `fig05.py` | Ensemble comparison: spread maps + biome trajectories | Fig. 5 |

## Supporting Figures

| Script | Description | SI Figure |
|--------|-------------|-----------|
| `fig02_bis.py` | Extended time series by biome (surface + interior) | S1-S2 |
| `fig02_ter.py` | Seasonal Hovmoller: MLD + surface Chl | S5 |
| `fig05_multivar.py` | Multi-variable ensemble spread comparison | S10+ |
| `fig06_conservation.py` | Tracer drift diagnostic | S9 |
| `figS_mesoscale_multivar.py` | Mesoscale structure across variables | S10 |
| `figS_ensemble_snapshots.py` | Ensemble member snapshots | S11 |

## Utility Scripts

| Script | Description |
|--------|-------------|
| `biomes_utils.py` | Biome definitions and helper functions |
| `env_setup.sh` | Environment setup for SLURM scripts |

## Running Figures

**Always submit via SLURM** (never run `.py` directly):

```bash
# Main figures
sbatch code_paper/fig02.sh
sbatch code_paper/fig03_ablation_tree.sh
sbatch code_paper/figure04_combined/fig04_combined.sh
sbatch code_paper/fig05.sh

# Supporting figures
sbatch code_paper/fig02_bis.sh
sbatch code_paper/fig02_ter.sh
sbatch code_paper/figS_mesoscale_multivar.sh
sbatch code_paper/figS_ensemble_snapshots.sh
```

## Output Directories

Figures are saved to `code_paper/figures/{script_name}/`:
- `figures/fig02/` — main figure panels
- `figures/fig03_ablation_tree/` — ablation tree
- `figures/figure04_combined/` — circulation + BGC ablation
- `figures/fig05/` — ensemble comparison

## Dependencies

Most scripts require the standard `ocean-emulator` conda environment. Exceptions:
- `fig02_ter.py` requires `preprocess_env` (has `gsw` for TEOS-10 MLD computation)

## Data Sources

- **Ground Truth**: `$OCEAN_EMU_DATA_ROOT/bgc_data.zarr`
- **Best Model Predictions**: `outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr`
- **Ablation Predictions**: `outputs/phase{N}_*/predictions_depth.zarr`
- **Ensemble Predictions**: `outputs/champion_model_eval_ensemble*/`

## Reproducibility Notes

**From a clean clone**, only `fig03_ablation_tree.py` runs without data (hardcoded metrics).

Other figures require:
1. Download champion weights from HuggingFace
2. Download evaluation data subset from Zenodo
3. Run eval to generate `predictions_depth.zarr`

See the top-level README for data download instructions.

## Style Guidelines

Font sizes (consistent across all figures):
- Panel titles `(a), (b), ...`: fontsize=17-18, fontweight="bold"
- Axis labels: fontsize=15
- Tick labels: labelsize=13
- Annotations (R²/RMSE boxes): fontsize=13-14
- Legend: fontsize=13
- Colorbar labels: fontsize=15

Legend placement:
- Time series: lower left/right (data peaks in upper half)
- PDFs: lower right (distributions peak on left)
- Power spectra: lower left (power decays to right)
