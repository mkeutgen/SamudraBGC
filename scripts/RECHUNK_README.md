# JRA Rechunking Solution - Inode Limit Workaround

## Problem
The original rechunking script was hitting disk quota errors due to the **inode (file count) limit** on `/scratch/cimes` (20M files max). The JRA dataset has 2.7M files, and rechunking creates temporary copies, which would exceed the limit.

## Solution
Use `/scratch/gpfs/GEOCLIM` scratch space for temporary and output storage during rechunking. This filesystem has a 60M inode limit with only 520K files currently used, providing plenty of headroom.

## Storage Quotas

### CIMES Scratch (Source - TIGHT LIMITS)
- **Files**: 7.0M / 20M (35% used) - **RISKY for rechunking**
- **Space**: 55.4TB / 100TB

### GEOCLIM Scratch (Temp Storage - PLENTY OF ROOM)
- **Files**: 520K / 60M (0.9% used) - **SAFE for rechunking**
- **Space**: 8.8TB / 1024TB

## Workflow

### Step 1: Run Rechunking Job
```bash
# Submit the rechunking job
sbatch scripts/rechunk_jra.sh
```

This will:
- Read from: `/scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL/bgc_data.zarr`
- Write temp files to: `/scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_temp/`
- Write output to: `/scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_output/bgc_data.zarr`
- Take ~2-4 hours with 800GB memory

### Step 2: Verify Success
Check the log file after completion:
```bash
tail -100 logs/rechunk-JRA-<jobid>.out
```

Look for:
- `✓ SUCCESS: Rechunking completed successfully!`
- No errors about disk quota or file limits

### Step 3: Finalize (Move Data Back)
```bash
# After verifying success, move the rechunked data back to CIMES scratch
bash scripts/finalize_rechunk.sh
```

This will:
1. Create backup: `bgc_data.zarr` → `bgc_data.zarr.backup`
2. Move rechunked data from GEOCLIM → CIMES scratch
3. Clean up temp files on GEOCLIM

### Step 4: Clean Up Backup (Optional)
After verifying the rechunked data works correctly with your training:
```bash
rm -rf /scratch/cimes/maximek/INMOS/processed_data/MOM6_CobaltDG_JRA_FULL/bgc_data.zarr.backup
```

This will free up ~4.4TB and ~2.7M inodes.

## Key Changes Made

### 1. Updated `rechunk_jra_to_daily.py`
- Added `--output-path` option for custom output location
- Added `--temp-path` option for custom temporary storage location

### 2. Updated `rechunk_jra.sh`
- Uses GEOCLIM scratch for temp/output storage
- Prevents hitting CIMES inode limits

### 3. Created `finalize_rechunk.sh`
- Automates moving rechunked data back to original location
- Creates backup of original data
- Cleans up temporary files

## Monitoring During Rechunking

```bash
# Watch the job progress
tail -f logs/rechunk-JRA-<jobid>.out

# Check inode usage on GEOCLIM (should stay well under 60M)
find /scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_temp -type f | wc -l
find /scratch/gpfs/GEOCLIM/LRGROUP/maximek/rechunk_output -type f | wc -l
```

## Alternative: No-Backup Mode
If you're confident and want to avoid the backup step entirely:
```bash
# Edit rechunk_jra.sh and add --no-backup flag
python rechunk_jra_to_daily.py \
    --zarr-path "$JRA_ZARR" \
    --max-mem 800GB \
    --compression 1 \
    --time-chunk-size 1 \
    --temp-path "$TEMP_DIR/bgc_data.zarr.rechunk_temp" \
    --output-path "$OUTPUT_DIR/bgc_data.zarr" \
    --no-backup
```

**Warning**: This will delete the original data without creating a backup! Only use if you have another copy elsewhere.
