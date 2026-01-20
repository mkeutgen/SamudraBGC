#!/bin/bash
#SBATCH --job-name=jra_full_helm_ens
#SBATCH --output=logs/jra_fullstate_helmholtz_ensemble_eval_%j.out
#SBATCH --error=logs/jra_fullstate_helmholtz_ensemble_eval_%j.err
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=12:00:00

# Load environment
module load anaconda3/2024.10
source activate /scratch/cimes/maximek/envs/ocean-emulator

# Navigate to project directory
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Create logs directory if it doesn't exist
mkdir -p logs

# Run ensemble evaluation
echo "Starting ensemble evaluation at $(date)"
echo "Config: configs/eval/jra_suite/jra_fullstate_helmholtz_grad05_ensemble_eval.yaml"
echo "Checkpoint: outputs/jra_fullstate_helmholtz_grad05/saved_nets/ema_ckpt.pt"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo "Running on node: $HOSTNAME"
echo "GPU: $CUDA_VISIBLE_DEVICES"
echo ""

python -m ocean_emulators.eval configs/eval/jra_suite/jra_fullstate_helmholtz_grad05_ensemble_eval.yaml

echo ""
echo "Ensemble evaluation completed at $(date)"
