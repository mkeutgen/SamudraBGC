# Experiment Scripts

This directory contains SLURM scripts for training and evaluating models.

## Structure

```
scripts/experiments/
├── baseline/
├── helmholtz_270x180/
└── helmholtz_full/
```

## Usage

### Training
```bash
cd scripts/experiments/<category>
sbatch train_<experiment>.sh
```

### Evaluation
```bash
cd scripts/experiments/<category>
sbatch eval_<experiment>.sh
```

## Monitoring

Check logs in `logs/` directory:
```bash
tail -f logs/<experiment>_train_<jobid>.out
tail -f logs/<experiment>_eval_<jobid>.err
```

## Configuration

SLURM parameters can be adjusted in each script:
- `--nodes`: Number of nodes
- `--gpus-per-node`: GPUs per node
- `--time`: Wall time limit
- `--mem`: Memory per node

## Experiment Categories

See README.md in each category directory for experiment details.
