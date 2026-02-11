#!/bin/bash
#SBATCH --job-name=visualize
#SBATCH --output=logs/slurm/visualize_%j.out
#SBATCH --error=logs/slurm/visualize_%j.err
#SBATCH --time=01:00:00           # 1 hour for visualization
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=500G
#SBATCH --partition=cpu
#SBATCH --account=cimes3

# Generate visualizations from comparison data
# Run this AFTER compare_rollouts.py has completed
# Usage: sbatch scripts/slurm/visualize_comparison.slurm

echo "=========================================="
echo "Generating Visualizations"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Started at: $(date)"
echo "=========================================="
echo

# Load conda environment
module load anaconda3/2024.10
conda activate preprocess_env

# Navigate to working directory
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator || exit 1

# Generate all visualizations
echo "Creating plots (using plot_types from config)..."
echo

python scripts/visualize_comparison.py \
    --config configs/eval/paper_ablations/jra_comparison.yaml \
    --batch-size 20

exit_code=$?

echo
echo "=========================================="
if [ $exit_code -eq 0 ]; then
    echo "Visualizations completed successfully!"
    echo "Figures saved to: outputs/jra_comparison/figures/"
    echo ""
    echo "To view figures:"
    echo "  ls outputs/jra_comparison/figures/"
else
    echo "Visualization failed with exit code: $exit_code"
fi
echo "Completed at: $(date)"
echo "=========================================="

exit $exit_code
