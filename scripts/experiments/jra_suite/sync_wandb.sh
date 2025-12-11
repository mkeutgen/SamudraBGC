#!/bin/bash
# Sync all offline W&B runs for JRA suite experiments to W&B cloud

set -e

# Activate environment
source ~/.bashrc
module purge
module load anaconda3/2024.10
conda activate /scratch/cimes/maximek/envs/ocean-emulator

cd /scratch/cimes/maximek/INMOS/Ocean_Emulator

echo "========================================="
echo "Syncing JRA Suite W&B Offline Runs"
echo "========================================="
echo ""

# Check if logged in to wandb
if ! python -m wandb login --verify 2>/dev/null; then
    echo "⚠ Not logged in to W&B. Please login first:"
    echo ""
    echo "Run: python -m wandb login"
    echo ""
    echo "You can find your API key at: https://wandb.ai/authorize"
    exit 1
fi

echo "✓ W&B authentication verified"
echo ""

# Define experiments
EXPERIMENTS=(
    "jra_fullstate_grad05"
    "jra_helmholtz_std_grad05"
    "jra_helmholtz_min_grad05"
    "jra_fullstate_helmholtz_grad05"
)

for exp in "${EXPERIMENTS[@]}"; do
    echo "----------------------------------------"
    echo "Experiment: $exp"
    echo "----------------------------------------"

    # Find latest offline run
    latest=$(ls -td outputs/$exp/wandb/offline-run-* 2>/dev/null | head -1)

    if [ -n "$latest" ]; then
        echo "  Found offline run: $(basename $latest)"
        echo "  Syncing to W&B cloud..."
        python -m wandb sync --entity m2lines --project bgc-emulator "$latest"
        echo "  ✓ Synced successfully"
    else
        echo "  ⚠ No offline runs found"
    fi
    echo ""
done

echo "========================================="
echo "Sync Complete!"
echo "========================================="
echo ""
echo "View your runs at: https://wandb.ai/m2lines/bgc-emulator"
