#!/bin/bash
#SBATCH --job-name=compare_full
#SBATCH --output=logs/slurm/compare_full_%j.out
#SBATCH --error=logs/slurm/compare_full_%j.err
#SBATCH --time=04:00:00           # 4 hours for full analysis
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16         # More CPUs for full analysis
#SBATCH --mem=780G                # Maximum memory for 3 experiments
#SBATCH --partition=cimes
#SBATCH --account=cimes3

# Full comparison with all variables and regional analysis
# Usage: sbatch scripts/slurm/compare_rollouts_full.slurm

echo "=========================================="
echo "Full Comparison Analysis"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "Node: $SLURM_NODELIST"
echo "=========================================="
echo

# Load conda environment
module load anaconda3/2024.10
conda activate preprocess_env

# Print environment details
echo "Python: $(python --version)"
echo "Working directory: $(pwd)"
echo

# Navigate to working directory
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator || exit 1

# Run full comparison with regional analysis
echo "Running full comparison (all variables + regional analysis)..."
echo

time python scripts/compare_rollouts.py \
    --config configs/eval/paper_ablations/jra_comparison.yaml

exit_code=$?

echo
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Full analysis completed successfully!"
    echo "Results saved to: outputs/jra_comparison/"
    echo ""
    echo "Next steps:"
    echo "  1. Review metrics in outputs/jra_comparison/metrics/"
    echo "  2. Generate plots: sbatch scripts/slurm/visualize_comparison.slurm"
else
    echo "Analysis failed with exit code: $exit_code"
    echo "Check error log: logs/slurm/compare_full_${SLURM_JOB_ID}.err"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit $exit_code
