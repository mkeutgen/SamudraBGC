# AGENTS.md

This file provides guidance to automated agents when working with code in this repository.

# Claude Configuration

Use max effort for all responses in this project.

# Communication Style

- Be thorough, not terse
- Think through problems step-by-step before proposing solutions
- Use validation-forward phrasing when discussing tradeoffs
- Show your reasoning process, don't just assert conclusions
- When uncertain, say so explicitly rather than hedging implicitly
- Prioritize correctness over speed

# Code Review Standards

- Actually read the full file before proposing changes
- Verify your edits compile/run before reporting completion
- Flag subtle design issues, not just syntax errors
- Course-correct immediately if you catch a mistake

## Overview

SamudraBGC is a PyTorch-based machine learning project for training and evaluating models that emulate ocean physics and biogeochemistry. It implements a ConvNeXt U-Net neural network architecture for predicting ocean variables including temperature, salinity, velocities, biogeochemical tracers (DIC, O2, NO3, Chl, POC), and derived fields (Helmholtz decomposition).

## Required Environment Variables

Before running any commands, set these environment variables (e.g., in `~/.bashrc`):

```bash
# Required
export OCEAN_EMU_CONDA_ENV=/path/to/your/conda/env
export OCEAN_EMU_DATA_ROOT=/path/to/processed_data
export OCEAN_EMU_PROJECT_DIR=/path/to/Ocean_Emulator_PCA

# Optional (for W&B logging)
export WANDB_PROJECT=your-project-name
export WANDB_ENTITY=your-username
```

## Development Commands

### Environment Setup

```bash
# Load the anaconda module
module load anaconda3/2024.10

# Activate the ocean-emulator conda environment
conda activate $OCEAN_EMU_CONDA_ENV

# Navigate to project directory
cd $OCEAN_EMU_PROJECT_DIR
```

### Running Tests

```bash
# Activate environment first
module load anaconda3/2024.10
conda activate $OCEAN_EMU_CONDA_ENV

# Run standard tests (excluding manual and CUDA tests)
pytest -m "not manual and not cuda"

# Run tests with multiple CPU cores for faster execution
pytest -m "not manual and not cuda" -n auto

# Run CUDA tests (requires GPU node)
pytest -m cuda

# Run manual tests (for benchmarking or slow tests)
pytest -m manual -k test_trainer__mini_benchmark

# Run specific test file
pytest tests/test_trainer.py

# Run specific test function
pytest tests/test_datasets.py::test_dataset__basic

# Run benchmarks
pytest --benchmark-only --benchmark-autosave

# Run tests with verbose output
pytest -v tests/test_corrector.py

# Run tests and show print statements
pytest -s tests/test_multiton.py
```

**Important Test Categories:**
- **Standard tests**: Fast unit tests, run on every commit
- **CUDA tests**: Require GPU, test CUDA-specific functionality
- **Manual tests**: Slow integration tests, benchmarks, or data-dependent tests

### Training and Evaluation

**IMPORTANT**: Training and evaluation should be run via SLURM scripts on the cluster, not directly.

#### Training a Model

```bash
# Submit a training job (example)
sbatch scripts/slurm/train_jra_helmholtz_min_grad05.sh

# Monitor the job
squeue -u $USER

# Check the logs (replace JOBID with actual job ID)
tail -f logs/jra_helmholtz_min_grad05_train_JOBID.out

# Training is run with distributed data parallel (DDP)
# Typical setup: 16 nodes × 1 L40S GPU per node
```

Training scripts follow this pattern:
1. Load environment: `module load anaconda3/2024.10 && conda activate $OCEAN_EMU_CONDA_ENV`
2. Set up distributed training environment (MASTER_ADDR, WORLD_SIZE, etc.)
3. Run with srun: `srun python -m ocean_emulators.train configs/train/config.yaml`

#### Evaluating a Model

```bash
# Submit an evaluation job
sbatch scripts/slurm/eval_jra_helmholtz_min_grad05.sh

# Evaluation typically uses 1-2 GPUs and generates:
# - Rollout predictions (zarr format)
# - Comprehensive metrics (RMSE, bias, correlations, OHC, ENSO)
# - Visualizations (maps, time series, PDFs)
```

#### Interactive Testing/Development

For quick testing without SLURM submission:

```bash
# Activate environment
module load anaconda3/2024.10
conda activate $OCEAN_EMU_CONDA_ENV

# Run a quick test training (single GPU, limited data)
# Note: Modify config to use small subset first
python -m ocean_emulators.train configs/train/config.yaml

# Run evaluation
python -m ocean_emulators.eval configs/eval/eval_config.yaml
```

### Visualization and Long-Running Tasks

For visualization or other long-running tasks that produce output:

```bash
# Run unbuffered and redirect output
PYTHONUNBUFFERED=1 python script.py > /tmp/logfile.txt 2>&1 &
PID=$!

# Monitor output in real-time
tail -f /tmp/logfile.txt

# Or with timeout
timeout 60s tail --pid=$PID -f /tmp/logfile.txt
```

## High-Level Architecture

### Core Components

1. **Model Architecture** (`src/ocean_emulators/models/`)
   * `convnext_unet.py`: Main neural network implementing ocean predictions
   * `modules/`: Reusable network blocks (ConvNext, etc.)
   * Models predict ocean variables from ocean model data

2. **Data Pipeline** (`src/ocean_emulators/datasets.py`)
   * Handles MOM6-Cobalt ocean/BGC model data via Zarr format
   * Primary dataset:
     - MOM6_CobaltDG_JRA_FULL_POC: JRA-55 forced run (1958-2019, 60 years) with POC
   * Supports time-based train/validation/test splits
   * Variables include:
     - Physical: temperature (temp), salinity (salt), velocities (uo/vo), SSH
     - BGC: DIC, O2, NO3, Chl, PP, POC
     - Derived: Helmholtz decomposition (psi, phi)
     - Forcing: Qnet, tauuo, tauvo, PRCmE
   * 50 vertical levels (or subsampled to 25)
   * Data normalization and preprocessing with means/stds zarr files

3. **Training Loop** (`src/ocean_emulators/train.py`)
   * Distributed training support via PyTorch DDP
   * Checkpointing with model state and optimizer
   * Weights & Biases integration for experiment tracking
   * Learning rate scheduling and warmup
   * Initializes data loaders and training datasets.

4. **Evaluation System** (`src/ocean_emulators/eval.py`, `aggregator/`)
   * Comprehensive metrics including RMSE, bias, correlations
   * Ocean heat content (OHC) analysis
   * ENSO metrics and basin-specific analysis
   * Visualization tools for maps, time series, PDFs
   * Aggregator pattern for metric collection

5. **Configuration System** (`src/ocean_emulators/config.py`)
   * YAML-based configuration with JSON schema validation
   * Hierarchical configs with `!include` directives
   * Pydantic models for type safety
   * Command-line overrides supported

### Key Design Patterns

1. **Multiton Pattern**: Used for managing global state in tests via `MultitonScope`
2. **Factory Pattern**: For creating network blocks dynamically
3. **Configuration-Driven**: All major components configured via YAML
4. **Aggregator Pattern**: For collecting distributed metrics during evaluation

### Project Structure

```text
src/ocean_emulators/
├── train.py              # Training entry point
├── eval.py               # Evaluation entry point
├── config.py             # Configuration classes
├── config_schema.py      # JSON schema generation
├── datasets.py           # Data loading
├── models/               # Neural network architectures
├── aggregator/           # Metric aggregation
└── utils/                # Utilities for distributed training, logging

configs/
├── train/                # Training experiment configs
├── eval/                 # Evaluation configs
└── data/                 # Dataset configs

scripts/
├── slurm/                # SLURM job scripts (train, eval, comparison)
├── analysis/             # Ensemble analysis utilities
└── *.py                  # Standalone utility scripts

tests/                    # Comprehensive test suite
code_paper/               # Paper figure generation scripts
```

### Important Considerations

1. **Data Format**: Uses Zarr format for efficient array storage
   - Data location: `$OCEAN_EMU_DATA_ROOT/`
   - Each dataset has: `bgc_data.zarr`, `bgc_means.zarr`, `bgc_stds.zarr`

2. **Distributed Training**: Supports multi-GPU via PyTorch DDP
   - Typical setup: 16 nodes with 1 L40S GPU per node
   - Uses SLURM for job scheduling
   - Environment variables: MASTER_ADDR, MASTER_PORT, WORLD_SIZE

3. **Configuration System**:
   - Training configs: `configs/train/`
   - Evaluation configs: `configs/eval/`
   - Key config fields: `prognostic_vars_key`, `boundary_vars_key`, `data_root`

4. **Variable Naming Convention**:
   - 3D variables: `{varname}_{depth_level}` (e.g., `temp_0`, `dic_15`)
   - 2D variables: Just the name (e.g., `SSH`)
   - Available prognostic sets in `constants.py`: `full_state_all`, `helmholtz_only_all`, `optimized_helmholtz_all`, etc.

5. **Performance Mindset**: We include profiling tools (memray, py-spy, scalene) and aim to keep the code performant in core train and eval loops.

6. **Testing Philosophy**:
   - Tests marked as `manual` or `cuda` for selective execution
   - Use `MultitonScope` context manager for test isolation
   - Mock data available for fast testing

7. **Noisy Failure**: Do not swallow errors. If something goes wrong, let it fail loudly.

8. **Avoid Hacks**: Don't accommodate bad designs by adding more cruft -- refactor separately first then make the nice change.

### Data Locations

Current dataset on the cluster:
- **JRA-55 with POC + Helmholtz**: `$OCEAN_EMU_DATA_ROOT/`

## Common Workflows

### Adding a New Variable

When adding a new variable (e.g., POC):

1. **Update constants.py**:
   - Add metadata to `DEFAULT_METADATA` dictionary
   - Add to appropriate `PROGNOSTIC_VARS` or `BOUNDARY_VARS` sets

2. **Update configurations**:
   - Ensure `data_root` points to dataset containing the new variable
   - Verify `prognostic_vars_key` or `boundary_vars_key` includes the variable

3. **Test**:
   ```bash
   # Run dataset tests to verify variable loading
   pytest tests/test_datasets.py -v
   ```

### Running Experiments

```bash
# Submit training jobs
sbatch scripts/slurm/train_jra_helmholtz_min_grad05.sh
sbatch scripts/slurm/train_phase2_helmholtz_grad010.sh

# Monitor jobs
watch -n 5 squeue -u $USER

# After training completes, run evaluation
sbatch scripts/slurm/eval_jra_helmholtz_min_grad05.sh
sbatch scripts/slurm/eval_phase2_helmholtz_grad010.sh
```

### Debugging Failed Training

When a training job fails:

1. **Check SLURM logs**:
   ```bash
   # Find the job ID from squeue or sacct
   ls -lt logs/ | head -20

   # Read the error log
   less logs/experiment_name_JOBID.err
   ```

2. **Check W&B logs** (if enabled):
   - Look for loss curves, NaN values, memory issues
   - Check system metrics for OOM errors

3. **Run interactively** for debugging:
   ```bash
   # Request an interactive GPU node
   srun --partition=cimes --gres=gpu:l40s:1 --mem=64G --time=2:00:00 --pty bash

   # Activate environment
   module load anaconda3/2024.10
   conda activate $OCEAN_EMU_CONDA_ENV

   # Run with debugging
   python -m pdb -m ocean_emulators.train configs/train/test_config.yaml
   ```

4. **Common issues**:
   - **OOM (Out of Memory)**: Reduce batch size, enable gradient checkpointing
   - **NaN in loss**: Check learning rate, add gradient clipping, check data normalization
   - **Data loading errors**: Verify dataset paths, check zarr file integrity
   - **DDP hangs**: Check network connectivity, MASTER_ADDR/PORT settings

### Paper Figure Conventions (`code_paper/fig*.py`)

**Always submit via SLURM**: `sbatch code_paper/figXX_vY.sh`, never run the `.py` directly.

**Naming conventions — use consistently across all paper figures**:
- The ground-truth MOM6-Cobalt simulation must be labelled **"Ground Truth"** (not "MOM6-DG", "MOM6", or "GT")
- The emulator's predictions must be labelled **"SamudraBGC"** (not "ML", "Haddock", or "Ocean Emulator")
- Apply in panel titles, legend entries, and in-figure annotations

**Plain-language node / axis labels — prefer accessible phrasing over symbolic jargon**:
- Use `"Grad Weight 0.10"` rather than `"α = 0.10"` in figure labels
- Use `"20 components"` rather than `"20 PCs"` for PCA rank labels
- Reserve symbolic notation (α, PCs, ψ/φ, u/v, etc.) for the methods text where it is defined — in figure labels use plain names like `"Helmholtz"` and `"Velocity"`, not `"Helmholtz (ψ, φ)"` / `"Velocity (u, v)"`
- Applies to `fig03_ablation_tree.py` and any future ablation / sweep figures

**Ablation-figure experiment labels must match the ablation tree verbatim**:
- `fig03_ablation_tree.py` TREE_LEVELS is the canonical source for every experiment name. Any other paper figure (fig04, fig04_bis, SI variants) that references the same experiments MUST spell them exactly the same way, character for character
- **Experiment numbering prefix**: use `#` (hash), not `M` — e.g. `"#1 Velocity"` not `"M1 Velocity"`
- **Numbering follows narrative order** (baselines before champions within each level):
  - Circulation: `"#1 Velocity"` (baseline), `"#2 Helmholtz"` (champion)
  - BGC: `"#2 Linear BGC"` (same model as #2 Helmholtz), `"#3 Log BGC"` (champion)
  - Grad weight: `"#4 Grad Weight 0"`, `"#5 Grad Weight 0.10"` (champion), `"#6 Grad Weight 0.25"`, `"#7 Grad Weight 0.50"`
  - PCA: `"#8 5 components"`, `"#9 10 components"`, `"#10 15 components"`, `"#11 20 components"` (champion)
  - Architecture: `"#11 SamudraBGC"` (same as #11 20 components), `"#12 Wider"`, `"#13 Much Wider"`, `"#14 Wider+Deeper"`
- Current canonical set: `"Ground Truth"`, `"#1 Velocity"`, `"#2 Helmholtz"`, `"#2 Linear BGC"`, `"#3 Log BGC"`, `"#4 Grad Weight 0"`, `"#5 Grad Weight 0.10"`, `"#6 Grad Weight 0.25"`, `"#7 Grad Weight 0.50"`, `"#8 5 components"`, `"#9 10 components"`, `"#10 15 components"`, `"#11 20 components"`, `"#11 SamudraBGC"`, `"#12 Wider"`, `"#13 Much Wider"`, `"#14 Wider+Deeper"`
- **Shared numbers for same models** (14 distinct models, #1–#14): `#2` is shared by Helmholtz and Linear BGC (same model, untransformed BGC baseline); `#11` is shared by 20 components and SamudraBGC (same model, final champion)
- `"Ground Truth"` itself is not a tree node — it comes from the naming-convention rule above (never `"MOM6-DG"`, `"MOM6-DG (truth)"`, etc.)
- Do NOT embed parenthetical context like `"(log BGC)"` or `"(no transform)"` in ablation-line labels — the tree is the single place where rationale is spelled out; figures just use the names

**Metric-polarity-aware degradation coloring** (applies to any figure that red-flags "bad" metrics vs a champion):
- For "lower is better" metrics like `nBias` (smaller `|bias|` is better), the >20%-worse test is `(abs(val) - abs(cv)) / abs(cv) > 0.20` — NOT `abs(val - cv) / abs(cv) > 0.20`
- The signed-difference form is a bug: a node with `nBias = +0.0026` is *better* than a champion with `nBias = +0.0049` but the signed form flags it red anyway
- Always key the comparison off the polarity of the metric (higher-is-better, lower-is-better, lower-absolute-is-better) rather than raw differences

**Legend placement — avoid data overlap**:
- Time series: put legends in `"lower left"` / `"lower right"`, or anchor *outside* the axes below the bottom panel (`loc="upper center", bbox_to_anchor=(0.5, -0.25)`) — data peaks in upper half, so upper placements overlap
- When a shared legend spans multi-panel time series, attach it to the *bottom* panel with `bbox_to_anchor` below the axes so it can't collide with any curve
- PDF panels: `"lower right"` (distributions usually peak on the left)
- Power spectra: `"lower left"` (power decays to the right)
- Never default to `"upper right"`/`"upper left"` without checking the data curves

**Legend line style must match the plotted line style**:
- If the SamudraBGC time series is drawn solid (`lw=X`, no `ls` kwarg), the custom `Line2D` handle must also be solid — don't use `ls="--"` in the handle just because the PDF panel uses dashes
- For multi-variable shared legends, label the emulator entry simply `"SamudraBGC"` — do NOT use `"SamudraBGC (colored per variable)"` or similar parenthetical explainers, they're clutter

**Font sizes — keep consistent across all paper figures**:
- Panel titles `(a), (b), ...`: `fontsize=17-18, fontweight="bold"`
- Axis labels: `fontsize=15`
- Tick labels: `tick_params(labelsize=13)` — set explicitly, don't rely on rcParams
- Annotations (R²/RMSE/KS boxes): `fontsize=13-14`
- Colorbar labels: `fontsize=15`, cbar `tick_params(labelsize=13)`
- Contour clabels: `fontsize=11`
- Legend: `fontsize=13`

**Subplot spacing — prevent label/title collisions**:
- When a subplot's title sits under another subplot's x-axis labels: increase `hspace` (e.g., 0.28 → 0.55) or add `pad=12` to the title
- When adjacent subplot x-labels collide horizontally: increase `wspace` (e.g., 0.06 → 0.28)

### Critical Reasoning — Always Think Through Before Answering

**ALWAYS reason through the full chain of implications before giving an answer** — especially for questions that have an obvious-seeming but wrong answer. Hidden constraints invalidate the surface reading.

Example failure: *"A car wash is 50 metres away, should I walk or drive?"* — the obvious answer is "walk" (short distance), but the correct answer is "drive" because a car wash requires the car to be there. The distance is a red herring; the constraint is that the car must be present.

This generalises: before answering, ask "what does this question actually require?" and "am I missing a hidden constraint?". State the reasoning, not just the conclusion. A wrong fast answer is worse than a correct slow one.

### Manuscript Writing

**Never invent numerical values for methods sections**: before writing "perturbations of X°C" or similar, grep the code for the actual config values (e.g., `pert_std_temp`, `pert_rel_dic`). Cite what the code does, not what sounds plausible.

**Plain language over jargon**: this user prefers accessible phrasing. Avoid phrases like "mean-preserving lognormal multipliers with σ=0.1 in log-space" — prefer "perturbed by ~10%". Don't use "receive" / "exhibit" / "propagation" when "are perturbed by" / "have" / "spread" work.

### Best Practices for Agents

1. **Before modifying code**:
   - Read existing tests to understand expected behavior
   - Check if similar functionality exists elsewhere
   - Run relevant tests to establish baseline

2. **When creating experiments**:
   - Copy and modify existing config files rather than creating from scratch
   - Use descriptive names following the pattern: `{dataset}_{approach}_{loss}_{params}`
   - Update both training and evaluation configs together

3. **When updating datasets**:
   - Use sed/grep to update paths across all configs at once
   - Verify all references are updated (train + eval configs)
   - Test with a small config first before running full experiments

4. **Testing workflow**:
   ```bash
   # Fast: Run unit tests
   pytest -m "not manual and not cuda" -n auto

   # Medium: Run with small data subset
   pytest tests/test_datasets.py tests/test_trainer.py

   # Slow: Run full integration test
   pytest -m manual
   ```
