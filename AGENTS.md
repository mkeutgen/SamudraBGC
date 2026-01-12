# AGENTS.md

This file provides guidance to automated agents when working with code in this repository.

## Overview

Ocean Emulator is a PyTorch-based machine learning project for training and evaluating models that emulate ocean physics and biogeochemistry. It implements a ConvNeXt U-Net neural network architecture for predicting ocean variables including temperature, salinity, velocities, biogeochemical tracers (DIC, O2, NO3, Chl, POC), and derived fields (Helmholtz decomposition).

## Development Commands

### Environment Setup

```bash
# Load the anaconda module
module load anaconda3/2024.10

# Activate the ocean-emulator conda environment
conda activate /scratch/cimes/maximek/envs/ocean-emulator

# Navigate to project directory
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator
```

### Running Tests

```bash
# Activate environment first
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

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
sbatch scripts/experiments/jra_suite/train_jra_classical_mae.sh

# Monitor the job
squeue -u $USER

# Check the logs (replace JOBID with actual job ID)
tail -f logs/jra_classical_mae_train_JOBID.out

# Training is run with distributed data parallel (DDP)
# Typical setup: 8 nodes × 1 GPU per node = 8 GPUs total
```

Training scripts follow this pattern:
1. Load environment: `module load anaconda3/2024.10 && conda activate /scratch/cimes/maximek/envs/ocean-emulator`
2. Set up distributed training environment (MASTER_ADDR, WORLD_SIZE, etc.)
3. Run with srun: `srun python -m ocean_emulators.train configs/path/to/config.yaml`

#### Evaluating a Model

```bash
# Submit an evaluation job
sbatch scripts/experiments/jra_suite/eval_jra_classical_mae.sh

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
conda activate /scratch/cimes/maximek/envs/ocean-emulator

# Run a quick test training (single GPU, limited data)
# Note: Modify config to use small subset first
python -m ocean_emulators.train configs/path/to/test_config.yaml

# Run evaluation
python -m ocean_emulators.eval configs/eval/path/to/eval_config.yaml
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
   * Primary datasets:
     - MOM6_CobaltDG_JRA_FULL_POC: JRA-55 forced run (1958-2019, 60 years) with POC
     - MOM6_CobaltDG_Clim_FULL: Climatological forcing runs
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

configs/                  # YAML configuration files
tests/                    # Comprehensive test suite
scripts/                  # Data download and preprocessing
```

### Important Considerations

1. **Data Format**: Uses Zarr format for efficient array storage
   - Data location: `/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/`
   - Each dataset has: `bgc_data.zarr`, `bgc_means.zarr`, `bgc_stds.zarr`

2. **Distributed Training**: Supports multi-GPU via PyTorch DDP
   - Typical setup: 8 nodes with 1 L40S GPU per node
   - Uses SLURM for job scheduling
   - Environment variables: MASTER_ADDR, MASTER_PORT, WORLD_SIZE

3. **Configuration System**:
   - All experiments defined in YAML configs under `configs/experiments/` and `configs/eval/`
   - Current active suite: `jra_suite/` (JRA-55 forced experiments)
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

Current datasets on the cluster:
- **JRA-55 with POC**: `/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/`
- **Climatological**: `/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_Clim_FULL/`

All JRA suite experiments use the JRA_FULL_POC dataset as of the latest update.

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

### Running an Experiment Suite

For the JRA suite experiments:

```bash
# Navigate to the JRA suite scripts
cd scripts/experiments/jra_suite

# Submit training jobs
sbatch train_jra_classical_mae.sh
sbatch train_jra_fullstate_grad05.sh
sbatch train_jra_helmholtz_std_grad05.sh

# Monitor jobs
watch -n 5 squeue -u $USER

# After training completes, run evaluation
sbatch eval_jra_classical_mae.sh
sbatch eval_jra_fullstate_grad05.sh
sbatch eval_jra_helmholtz_std_grad05.sh
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
   conda activate /scratch/cimes/maximek/envs/ocean-emulator

   # Run with debugging
   python -m pdb -m ocean_emulators.train configs/experiments/test_config.yaml
   ```

4. **Common issues**:
   - **OOM (Out of Memory)**: Reduce batch size, enable gradient checkpointing
   - **NaN in loss**: Check learning rate, add gradient clipping, check data normalization
   - **Data loading errors**: Verify dataset paths, check zarr file integrity
   - **DDP hangs**: Check network connectivity, MASTER_ADDR/PORT settings

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
