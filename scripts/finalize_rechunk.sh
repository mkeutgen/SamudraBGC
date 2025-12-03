#!/bin/bash
# Finalize rechunking by moving data back to CIMES scratch
# Run this AFTER rechunk_jra.sh completes successfully

set -e

echo "=================================================="
echo "Finalizing JRA rechunk - Moving data back"
echo "=================================================="
echo "Started: $(date)"
echo ""

# Paths
JRA_ORIGINAL="/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL/bgc_data.zarr"
JRA_BACKUP="${JRA_ORIGINAL}.backup"
RECHUNKED_DATA="/scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_output/bgc_data.zarr"
TEMP_DIR="/scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_temp"

# Verify rechunked data exists
if [ ! -d "$RECHUNKED_DATA" ]; then
    echo "ERROR: Rechunked data not found at $RECHUNKED_DATA"
    echo "Make sure rechunk_jra.sh completed successfully first!"
    exit 1
fi

echo "Rechunked data found at: $RECHUNKED_DATA"
echo ""

# Create backup of original
if [ ! -d "$JRA_BACKUP" ]; then
    echo "Creating backup of original data..."
    mv "$JRA_ORIGINAL" "$JRA_BACKUP"
    echo "✓ Backup created at: $JRA_BACKUP"
else
    echo "Backup already exists, removing original..."
    rm -rf "$JRA_ORIGINAL"
fi

echo ""
echo "Moving rechunked data to final location..."
mv "$RECHUNKED_DATA" "$JRA_ORIGINAL"
echo "✓ Data moved to: $JRA_ORIGINAL"

echo ""
echo "Cleaning up temporary files..."
rm -rf "$TEMP_DIR"
rmdir "/scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_output" 2>/dev/null || true
echo "✓ Temp files removed"

echo ""
echo "=================================================="
echo "SUCCESS! Rechunking finalized"
echo "=================================================="
echo "New data location: $JRA_ORIGINAL"
echo "Backup location: $JRA_BACKUP"
echo ""
echo "To remove the backup (after verifying new data works):"
echo "  rm -rf $JRA_BACKUP"
echo ""
echo "Finished: $(date)"
