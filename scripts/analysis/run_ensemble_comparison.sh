#!/bin/bash
# Run ensemble comparison analysis
# This script compares ensemble predictions with ground truth

# Get the project root directory
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"

# Configuration (relative to project root)
ENSEMBLE_DIR="$PROJECT_ROOT/outputs/jra_helmholtz_min_grad05_ensemble_eval"
GROUND_TRUTH="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL_POC/bgc_data.zarr"
OUTPUT_DIR="$PROJECT_ROOT/outputs/ensemble_analysis"
# Auto-detect number of ensemble members (set to 0 for auto-discovery)
N_MEMBERS=0
SNAPSHOT_DAYS="0 90 180 270 360"

# Variables to analyze (space-separated)
VARIABLES="temp_0 salt_0 dic_0 o2_0 chl_0"

# Regional boundaries (degrees North)
SUBTROPICAL_JET=37.0
JET_SUBPOLAR=43.0

echo "========================================="
echo "Ensemble Comparison Analysis"
echo "========================================="
echo "Ensemble directory: $ENSEMBLE_DIR"
echo "Ground truth: $GROUND_TRUTH"
echo "Output directory: $OUTPUT_DIR"
echo "Number of members: $N_MEMBERS"
echo "Snapshot days: $SNAPSHOT_DAYS"
echo "Variables: $VARIABLES"
echo "Regional boundaries: subtropical_jet=${SUBTROPICAL_JET}°N, jet_subpolar=${JET_SUBPOLAR}°N"
echo "========================================="
echo ""

# Create output directory
mkdir -p "$OUTPUT_DIR"

# Run analysis
python "$SCRIPT_DIR/compare_ensemble_with_groundtruth.py" \
    --ensemble_dir "$ENSEMBLE_DIR" \
    --ground_truth "$GROUND_TRUTH" \
    --output_dir "$OUTPUT_DIR" \
    --n_members "$N_MEMBERS" \
    --include_unperturbed \
    --variables $VARIABLES \
    --snapshot_days $SNAPSHOT_DAYS \
    --subtropical_jet "$SUBTROPICAL_JET" \
    --jet_subpolar "$JET_SUBPOLAR"

echo ""
echo "========================================="
echo "Analysis complete!"
echo "Results saved to: $OUTPUT_DIR"
echo "========================================="
