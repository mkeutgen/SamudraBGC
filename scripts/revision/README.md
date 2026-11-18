# Revised figures and numbers (GRL revision, November 2026)

The scripts in this folder draw the figures that changed in the revision of
"SamudraBGC: Machine Learning Emulation of Regional Mesoscale Ocean Biogeochemistry" and
compute the numbers the revised text quotes from them. The figures that did not change come
from `code_paper/` (see `code_paper/README.md`).

Every comparison pairs the dynamical model with SamudraBGC by date
(`common.gt_indices_for`, `common.date_index`). Some of the submitted figure scripts paired
them by position from 1 January, while the rollouts start on 3 January, so those figures
compared each emulator day with the dynamical model two days earlier. Training, model
selection and the test evaluation (`ocean_emulators.eval`) always paired by date and are
unaffected.

## Which script draws which figure

Figure numbers are those of the revised manuscript. Several file names keep the submitted
Supporting Information numbering (for example `figS10_no_o2.py` draws the revised
Figure S5).

| Revised figure | Script | Output in `revision_figures/` | Change from the submitted version |
|---|---|---|---|
| Figure 2 | `fig02_minor_fixes.py` | `fig02_minor_fixes.png` | Paired by date; R² to two decimals; KS on every point |
| Figure S2 (Taylor diagram) | `plot_taylor.py` | `fig_taylor.png` | New |
| Figure S3 (meridional sections) | `figS_sections.py` | `figS_sections.png` | New |
| Figure S5 (mesoscale structure, 1 April 2015) | `figS10_no_o2.py` | `figS10_no_o2.png` | O₂ row removed; salinity in psu |
| Figure S6 (seasonal cycle of chlorophyll and mixed layer depth) | `figS05_chl_capped.py` | `fixed/figS05_chl_capped.png` | Paired by date; colour scale capped at the data range |
| Figures S7, S8 (time series, 0–100 m and 100–500 m) | `figS01_04_by_date.py` | `fixed/figS01.png`, `fixed/figS02.png` | Paired by date |
| Figure S9 (tracer drift) | `figS09_relabel.py` | `figS09_relabel.png` | Labels "Ground Truth" and "SamudraBGC" |
| Figure S10 (spatial spectra) | `plot_spectra.py` | `fig_spectra_finescale.png` | New |
| Figures S11, S12 (distributions, 0–100 m and 100–500 m) | `figS01_04_by_date.py` | `fixed/figS03.png`, `fixed/figS04.png` | Paired by date |
| Figure S15 (ensemble snapshots) | `figS_ensemble_snapshots.py` | `fixed/figS_ensemble_snapshots.png` | Emulator members shown on 15 April (were 17 April) |

Numbers quoted in the text:

| Script | Numbers | Output |
|---|---|---|
| `figS09_relabel.py` | Upper-500 m trends of Section 3.1 and Figure S9 (temperature, DIC, O₂), printed to the log | log only |
| `figS05_numbers.py` | Timing of the chlorophyll bloom at 40°N and 55°N, mixed-layer depth ranges (Section 3.1) | `data/figS05_numbers.json` |
| `chl_min.py` | Smallest chlorophyll value of the dynamical model, per level, 1960–2019 ("never zero", Section 2.4) | `chl_min.json` |
| `drift_analysis.py` | Supporting checks: the DIC drift at 100–200 m in the validation and test rollouts, and the upper-500 m trends with area and cos(latitude) weights | `data/drift_metrics.json`, `fig_drift_dic_100_200m.png` |

## Inputs

| Input | Where | Used by |
|---|---|---|
| Dynamical model store | `$OCEAN_EMU_DATA_ROOT/bgc_data.zarr` | all |
| PCA basis | `$OCEAN_EMU_DATA_ROOT/pca_params.npz` | `figS_ensemble_snapshots.py` |
| Test rollout, 2015–2019 | `$OCEAN_EMU_OUTPUTS_PCA/champion_model_eval_rollout2015_2019/predictions_depth.zarr`, from `configs/eval/champion_model_eval_rollout2015_2019.yaml` | `prep_test_fields.py`, `drift_series.py`, `spectra.py`, `figS05_chl_capped.py`, `figS10_no_o2.py`, and the time axis in `fig02_minor_fixes.py` and `figS01_04_by_date.py` |
| Validation rollout, 2010–2014 | `$OCEAN_EMU_OUTPUTS_PCA/phase5_pca20_helmholtz_grad010_eval_rollout2010_2014/predictions_depth.zarr`, from the config of the same name | `drift_series.py` |
| 50-member ensemble, 2015 | `$OCEAN_EMU_OUTPUTS_PCA/champion_model_eval_ensemble50_tsonly_std05_2015/`, from the config of the same name | `figS_ensemble_snapshots.py` |
| Dynamical model ensemble, surface chlorophyll | `code_paper/figures/fig05/_cache_chl_surface.pkl`, written by `code_paper/fig05.py` | `figS_ensemble_snapshots.py` |

`OCEAN_EMU_OUTPUTS_PCA` defaults to `outputs`, where `ocean_emulators.eval` writes. The
SLURM wrappers also need `OCEAN_EMU_CONDA_ENV` and `OCEAN_EMU_PROJECT_DIR` (the repository
root), as for `code_paper/` (`code_paper/env_setup.sh`). `figS05_chl_capped.py` needs `gsw`,
which `run_figS05.sh` takes from a conda environment named `preprocess_env`, as
`code_paper/fig02_ter.sh` does.

## Run order

Submit from the repository root after setting `YOUR_PARTITION` and `YOUR_ACCOUNT` in the
wrappers. The first four jobs read the zarr stores and can run in parallel; everything
after them reads only the arrays they cache in `revision_figures/data/`.

```bash
mkdir -p logs
# Read the stores (hours each)
sbatch scripts/revision/run_prep.sh           # layer means, time means, wet mask, dates
sbatch scripts/revision/run_drift.sh          # daily per-level domain means, test and validation
sbatch scripts/revision/run_spectra.sh        # weekly spectra of 2019
sbatch scripts/revision/run_figS05.sh         # Figure S6 and its arrays
sbatch scripts/revision/run_plot.sh scripts/revision/figS10_no_o2.py   # Figure S5
sbatch scripts/revision/chl_min.sh            # smallest chlorophyll value
# code_paper/fig05.sh must have run first (its cache), then:
sbatch scripts/revision/run_plot.sh scripts/revision/figS_ensemble_snapshots.py

# From the cached arrays (minutes each)
for s in fig02_minor_fixes plot_taylor figS_sections figS01_04_by_date \
         figS09_relabel plot_spectra drift_analysis figS05_numbers; do
    sbatch scripts/revision/run_plot.sh scripts/revision/$s.py
done
```

`plot_taylor.py --replot` and `figS05_chl_capped.py --replot` redraw from the statistics
already saved. `figS01_04_by_date.py DIR` also renders the submitted position-based pairing
into `DIR`, for comparison.
