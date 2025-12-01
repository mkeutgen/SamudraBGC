#!/usr/bin/env python3
"""Generate SLURM training and evaluation scripts for JRA experiment suite."""

experiments = [
    # Phase 1: Prognostic variables & forcing
    {
        "name": "jra_fullstate_grad05",
        "description": "Full State with Raw Velocities",
        "phase": "1.1",
    },
    {
        "name": "jra_helmholtz_std_grad05",
        "description": "Helmholtz + Standard Forcing (Expected Winner)",
        "phase": "1.2",
    },
    {
        "name": "jra_helmholtz_min_grad05",
        "description": "Helmholtz + Minimal Forcing (Ablation)",
        "phase": "1.3",
    },
    {
        "name": "jra_fullstate_helmholtz_grad05",
        "description": "Full State + Helmholtz (Wild Card)",
        "phase": "1.4",
    },
    # Phase 2: Loss functions
    {
        "name": "jra_best_grad05_so00",
        "description": "First-Order Only (Baseline Blurry)",
        "phase": "2.1",
    },
    {
        "name": "jra_best_grad05_so005",
        "description": "Conservative Second-Order (Recommended)",
        "phase": "2.2",
    },
    {
        "name": "jra_best_grad05_so01",
        "description": "Aggressive Second-Order (Max Sharpness)",
        "phase": "2.3",
    },
    {
        "name": "jra_best_grad025_so025",
        "description": "Balanced Penalties (Alternative)",
        "phase": "2.4",
    },
]

train_template = """#!/bin/bash
#SBATCH --job-name={name}_train
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=72:00:00
#SBATCH --output=logs/{name}_train_%j.out
#SBATCH --error=logs/{name}_train_%j.err

# Experiment: {description}
# Phase: {phase}
# Suite: JRA 60-year BGC Emulator Training

set -e

# Source bashrc for wandb API key
source ~/.bashrc

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Distributed training environment (canonical)
GPUS_PER_NODE=$(echo $SLURM_GPUS_ON_NODE | tr ',' '\\n' | wc -l)
[ -z "$GPUS_PER_NODE" ] || [ "$GPUS_PER_NODE" -eq 0 ] && GPUS_PER_NODE=1
export MASTER_ADDR=$(scontrol show hostname $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$((SLURM_NNODES * GPUS_PER_NODE))

# Training
echo "Starting training: {name}"
echo "Config: configs/experiments/jra_suite/{name}.yaml"

srun --ntasks=8 \\
     --ntasks-per-node=1 \\
     --gpus-per-node=1 \\
     python -m ocean_emulators.train \\
     configs/experiments/jra_suite/{name}.yaml

echo "Training complete!"
"""

eval_template = """#!/bin/bash
#SBATCH --job-name={name}_eval
#SBATCH --partition=cimes
#SBATCH --account=cimes3
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH --time=24:00:00
#SBATCH --output=logs/{name}_eval_%j.out
#SBATCH --error=logs/{name}_eval_%j.err

# Experiment: {description}
# Phase: {phase}
# Suite: JRA 60-year BGC Emulator Evaluation

set -e

# Source bashrc for wandb API key
source ~/.bashrc

# Load modules
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator
cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

# Evaluation
echo "Starting evaluation: {name}"
echo "Config: configs/experiments/jra_suite/{name}.yaml"

# TODO: Update with correct checkpoint path after training
CHECKPOINT_PATH="outputs/{name}/checkpoints/checkpoint_epoch_60.pt"

python -m ocean_emulators.eval \\
    configs/experiments/jra_suite/{name}.yaml \\
    --ckpt_path $CHECKPOINT_PATH

echo "Evaluation complete!"
"""

# Generate training scripts
for exp in experiments:
    filename = f"train_{exp['name']}.sh"
    with open(filename, 'w') as f:
        f.write(train_template.format(**exp))
    print(f"Created {filename}")

# Generate evaluation scripts
for exp in experiments:
    filename = f"eval_{exp['name']}.sh"
    with open(filename, 'w') as f:
        f.write(eval_template.format(**exp))
    print(f"Created {filename}")

print(f"\\n✓ Generated {len(experiments) * 2} SLURM scripts")
