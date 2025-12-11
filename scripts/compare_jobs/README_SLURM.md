# SLURM Job Scripts for Comparison Analysis

This directory contains SLURM batch scripts for running comparison analysis on HPC clusters.

## Available Scripts

### 1. Quick Comparison (Recommended for Testing)
**File**: `compare_rollouts_quick.slurm`
- **Time**: ~30 minutes
- **Memory**: 16GB
- **Features**: Global metrics only (no regional analysis)
- **Use case**: Quick validation, testing configuration

```bash
sbatch scripts/slurm/compare_rollouts_quick.slurm
```

### 2. Full Comparison (Complete Analysis)
**File**: `compare_rollouts_full.slurm`
- **Time**: ~2-4 hours
- **Memory**: 64GB
- **Features**: All variables + regional analysis
- **Use case**: Complete 10-year rollout comparison

```bash
sbatch scripts/slurm/compare_rollouts_full.slurm
```

### 3. Visualization
**File**: `visualize_comparison.slurm`
- **Time**: ~1 hour
- **Memory**: 32GB
- **Features**: Generate all plots from computed metrics
- **Use case**: Run AFTER comparison completes

```bash
sbatch scripts/slurm/visualize_comparison.slurm
```

### 4. Custom Comparison
**File**: `compare_rollouts.slurm`
- **Flexible**: Set via environment variables
- **Use case**: Custom configurations

```bash
# Example: Custom config file
CONFIG_FILE=configs/eval/my_custom.yaml sbatch scripts/slurm/compare_rollouts.slurm

# Example: Skip regional analysis
EXTRA_ARGS="--skip-regional" sbatch scripts/slurm/compare_rollouts.slurm

# Example: Specific variables only
EXTRA_ARGS="--variables temp_0 salt_0 chl_0" sbatch scripts/slurm/compare_rollouts.slurm
```

## Quick Start

### Step 1: Edit Configuration (if needed)

```bash
# Edit your comparison config
vim configs/eval/jra_suite/jra_comparison.yaml

# Or use the default
```

### Step 2: Submit Job

For first-time users, start with quick comparison:

```bash
# Test run (fast)
sbatch scripts/slurm/compare_rollouts_quick.slurm
```

Or go straight to full analysis:

```bash
# Full analysis (slow but complete)
sbatch scripts/slurm/compare_rollouts_full.slurm
```

### Step 3: Monitor Job

```bash
# Check job status
squeue -u $USER

# Watch job output in real-time
tail -f logs/slurm/compare_quick_*.out

# Or for full analysis
tail -f logs/slurm/compare_full_*.out
```

### Step 4: Generate Visualizations

After comparison completes:

```bash
sbatch scripts/slurm/visualize_comparison.slurm
```

## Job Output

SLURM logs are saved to:
```
logs/slurm/
├── compare_quick_12345.out    # Standard output
├── compare_quick_12345.err    # Error output
├── compare_full_67890.out
├── compare_full_67890.err
└── visualize_*.out
```

Analysis results are saved to:
```
outputs/jra_comparison/
├── metrics/
│   ├── global_metrics.txt
│   └── regional_metrics.txt
├── data/
│   └── time_series/*.csv
└── figures/
    └── *.png
```

## Customization

### Adjust Resources

Edit the SLURM directives in the script:

```bash
#SBATCH --time=04:00:00      # Increase time limit
#SBATCH --mem=128G           # Increase memory
#SBATCH --cpus-per-task=16   # More CPUs
#SBATCH --partition=gpu      # Use different partition
```

### Change Partition

Your cluster may have different partitions. Common options:
- `all` - General compute
- `short` - Short jobs (<4 hours)
- `medium` - Medium jobs (<24 hours)
- `long` - Long jobs (>24 hours)
- `gpu` - GPU nodes (not needed for this)

Check available partitions:
```bash
sinfo
```

Edit the script:
```bash
#SBATCH --partition=YOUR_PARTITION
```

### Process Specific Variables

Create a custom script or use environment variables:

```bash
# Only analyze surface variables
EXTRA_ARGS="--variables temp_0 salt_0 uo_0 vo_0 SSH" \
sbatch scripts/slurm/compare_rollouts.slurm
```

### Use Different Time Slice

```bash
EXTRA_ARGS="--time-slice-start 1990-01-01 --time-slice-end 1992-12-31" \
sbatch scripts/slurm/compare_rollouts.slurm
```

## Troubleshooting

### Job Failed

```bash
# Check error log
cat logs/slurm/compare_full_12345.err

# Common issues:
# - Out of memory: Increase --mem in script
# - Timeout: Increase --time in script
# - Module not found: Check module load command
# - Conda env: Verify conda environment name
```

### Job Pending Forever

```bash
# Check queue
squeue -u $USER

# Check job details
scontrol show job JOBID

# Possible reasons:
# - Partition unavailable
# - Resource request too high
# - Queue full
```

### Memory Issues

If job runs out of memory:

1. Edit script to request more memory:
   ```bash
   #SBATCH --mem=128G
   ```

2. Or reduce workload:
   ```bash
   EXTRA_ARGS="--skip-regional --variables temp_0 salt_0" \
   sbatch scripts/slurm/compare_rollouts.slurm
   ```

### Slow Performance

Jobs are slow because:
- Large datasets (10 years, 360x360 grid)
- Many variables (10-50 variables)
- Regional analysis (4x slower)

Solutions:
- Use `compare_rollouts_quick.slurm` (no regional)
- Reduce time slice in config
- Process fewer variables

## Job Dependencies

Run jobs in sequence (visualization after comparison):

```bash
# Submit comparison
JOB1=$(sbatch --parsable scripts/slurm/compare_rollouts_full.slurm)

# Submit visualization (starts after JOB1 completes)
sbatch --dependency=afterok:$JOB1 scripts/slurm/visualize_comparison.slurm
```

## Email Notifications

Add to your SLURM script for email alerts:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=your.email@princeton.edu
```

## Useful Commands

```bash
# Submit job
sbatch script.slurm

# Check queue
squeue -u $USER

# Cancel job
scancel JOBID

# Cancel all your jobs
scancel -u $USER

# View job details
scontrol show job JOBID

# View completed job info
sacct -j JOBID --format=JobID,JobName,Elapsed,State,MaxRSS

# Monitor output in real-time
tail -f logs/slurm/compare_*.out

# Check partition availability
sinfo -p all
```

## Example Workflow

Complete workflow for comparing a 10-year rollout:

```bash
# 1. Navigate to Ocean_Emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# 2. Test with quick run first
sbatch scripts/slurm/compare_rollouts_quick.slurm

# 3. Monitor progress
tail -f logs/slurm/compare_quick_*.out

# 4. Once successful, run full analysis
JOB1=$(sbatch --parsable scripts/slurm/compare_rollouts_full.slurm)

# 5. Auto-submit visualization after comparison completes
sbatch --dependency=afterok:$JOB1 scripts/slurm/visualize_comparison.slurm

# 6. Monitor full analysis (this will take a while)
tail -f logs/slurm/compare_full_*.out

# 7. Check results when complete
ls outputs/jra_comparison/metrics/
ls outputs/jra_comparison/figures/
```

## Tips

1. **Start with quick test**: Always test with `compare_rollouts_quick.slurm` first
2. **Monitor logs**: Use `tail -f` to watch progress
3. **Check resources**: Review completed jobs with `sacct` to optimize resource requests
4. **Use job dependencies**: Chain jobs with `--dependency`
5. **Save configs**: Keep a copy of your config in the output directory
6. **Adjust time limits**: Based on your dataset size and variables

## Getting Help

If jobs fail or you need assistance:

1. Check error logs in `logs/slurm/`
2. Review [PERFORMANCE_TIPS.md](../PERFORMANCE_TIPS.md)
3. See [README_comparison.md](../README_comparison.md) for detailed usage
4. Contact your HPC support team for cluster-specific issues
