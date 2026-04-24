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
| `fig04.py` | Ocean circulation representation: DIC snapshots + power spectrum | Fig. 3 |
| `fig04_bis.py` | BGC representation ablation: time series, bias, RMSE vs depth | Fig. 4 |
| `fig05.py` | Ensemble comparison: spread maps + pointwise trajectories | Fig. 5 |

## Supporting Figures

| Script | Description | SI Figure |
|--------|-------------|-----------|
| `fig02_bis.py` | Extended time series by biome (surface + interior) | S1-S2 |
| `fig02_ter.py` | Seasonal Hovmoller: MLD + surface Chl | S5 |
| `fig05_multivar.py` | Multi-variable ensemble spread comparison | S10+ |

## Utility Scripts

| Script | Description |
|--------|-------------|
| `fig01_panels.py` | Individual panels for Fig. 1 schematic |
| `fig03_lollipop.py` | Alternative lollipop-style ablation visualization |
| `fig04_bgc_pdf.py` | BGC variable PDF comparisons |
| `fig04_design_choices.py` | Design choice comparison plots |
| `fig05_diagnostics.py` | Ensemble diagnostic plots |
| `env_setup.sh` | Environment setup for SLURM scripts |

## Running Figures

**Always submit via SLURM** (never run `.py` directly):

```bash
sbatch code_paper/fig02.sh
sbatch code_paper/fig03_ablation_tree.sh
sbatch code_paper/fig04.sh
sbatch code_paper/fig04_bis.sh
sbatch code_paper/fig05.sh
```

## Output Directories

Figures are saved to `code_paper/figures/{script_name}/`:
- `figures/fig02/` — main figure panels
- `figures/fig03_ablation_tree/` — ablation tree
- `figures/fig04/` — circulation representation
- `figures/fig04_bis/` — BGC ablation panels
- `figures/fig05/` — ensemble comparison

## Dependencies

Most scripts require the standard `ocean-emulator` conda environment. Exceptions:
- `fig02_ter.py` requires `preprocess_env` (has `gsw` for TEOS-10 MLD computation)

## Data Sources

- **Ground Truth**: `$OCEAN_EMU_DATA_ROOT/MOM6_CobaltDG_JRA_FULL_POC_Helmholtz/bgc_data.zarr`
- **Best Model Predictions**: `outputs/champion_model_eval_rollout2015_2019/predictions_depth.zarr`
- **Ablation Predictions**: `outputs/phase{N}_*/predictions_depth.zarr`
- **Ensemble Predictions**: `outputs/champion_model_eval_ensemble*/`

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
