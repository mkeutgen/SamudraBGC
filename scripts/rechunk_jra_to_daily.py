#!/usr/bin/env python3
"""
Standalone script to rechunk JRA data to daily (time=1) chunks.

Usage:
    python scripts/rechunk_jra_to_daily.py --zarr-path /path/to/bgc_data.zarr
"""
import argparse
import logging
import shutil
import sys
from pathlib import Path

import xarray as xr
import zarr
from numcodecs import Blosc

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def rechunk_to_daily(
    zarr_path: Path,
    max_mem: str = "60GB",
    compression_level: int = 1,
    backup: bool = True,
    time_chunk_size: int = 5,
    output_path: Path | None = None,
    temp_path: Path | None = None
):
    """
    Efficiently rechunk zarr store to specified time chunks using rechunker.

    Args:
        zarr_path: Path to the zarr store to rechunk
        max_mem: Maximum memory for rechunking operations
        compression_level: Compression level (1-9)
        backup: Whether to keep a backup of the original
        time_chunk_size: Time chunk size in days (default: 5)
        output_path: Optional custom output path (default: zarr_path.parent/zarr_path.name.rechunked)
        temp_path: Optional custom temp path (default: zarr_path.parent/zarr_path.name.rechunk_temp)
    """
    try:
        from rechunker import rechunk
    except ImportError:
        logger.error("rechunker not installed. Install with: pip install rechunker")
        sys.exit(1)

    logger.info("=" * 60)
    logger.info(f"Rechunking to {time_chunk_size}-day time chunks with compression")
    logger.info("=" * 60)

    # Open existing zarr
    logger.info(f"Opening {zarr_path}")
    source_store = str(zarr_path)
    ds = xr.open_zarr(source_store, consolidated=False)

    logger.info(f"Current chunks: {list(ds.chunks.items())[:5]}...")
    logger.info(f"Dataset size: {ds.nbytes / 1e9:.2f} GB")
    logger.info(f"Time dimension size: {ds.sizes['time']}")

    # Check current time chunk
    time_vars = [v for v in ds.data_vars if 'time' in ds[v].dims]
    if time_vars:
        current_time_chunk = ds[time_vars[0]].chunks[ds[time_vars[0]].dims.index('time')][0]
        logger.info(f"Current time chunk size: {current_time_chunk}")

        if current_time_chunk == time_chunk_size:
            logger.warning(f"Data is already chunked with time={time_chunk_size}. Nothing to do!")
            return

    # Define target chunks: specified time chunk size, full spatial
    target_chunks = {}
    for var in ds.data_vars:
        if var in ["mask", "wetmask"]:
            continue  # Skip masks
        var_chunks = []
        for dim in ds[var].dims:
            if dim == "time":
                var_chunks.append(time_chunk_size)
            else:
                var_chunks.append(ds.sizes[dim])  # Full dimension
        target_chunks[var] = tuple(var_chunks)

    logger.info(f"Target chunking: time={time_chunk_size}, spatial dimensions=-1 (full)")

    # Setup paths
    if output_path is None:
        target_store = str(zarr_path.parent / f"{zarr_path.name}.rechunked")
    else:
        target_store = str(output_path)
        logger.info(f"Using custom output path: {target_store}")

    if temp_path is None:
        temp_store = str(zarr_path.parent / f"{zarr_path.name}.rechunk_temp")
    else:
        temp_store = str(temp_path)
        logger.info(f"Using custom temp path: {temp_store}")

    # Clean up any existing temp/target stores from previous runs
    for path in [target_store, temp_store]:
        if Path(path).exists():
            logger.info(f"Removing existing {path}")
            shutil.rmtree(path)

    # Define compression for target store
    compressor = Blosc(cname="zstd", clevel=compression_level, shuffle=Blosc.BITSHUFFLE)
    target_options = {
        var: {"compressor": compressor}
        for var in ds.data_vars
        if var not in ["mask", "wetmask"]
    }

    # Create rechunk plan
    logger.info(f"Creating rechunk plan with max_mem={max_mem}, compression_level={compression_level}")
    rechunk_plan = rechunk(
        ds,
        target_chunks=target_chunks,
        max_mem=max_mem,
        target_store=target_store,
        temp_store=temp_store,
        target_options=target_options,
    )

    # Execute
    logger.info("Executing rechunk...")
    logger.info(f"  This may take 30-90 minutes depending on dataset size")
    logger.info(f"  Temp storage: {temp_store}")
    logger.info(f"  Output: {target_store}")

    rechunk_plan.execute()

    # Replace original with rechunked version
    logger.info("Replacing original zarr with rechunked version...")
    if backup:
        backup_store = str(zarr_path) + ".backup"
        logger.info(f"Creating backup at {backup_store}")
        if Path(backup_store).exists():
            logger.info(f"Removing existing backup...")
            shutil.rmtree(backup_store)
        shutil.move(source_store, backup_store)
    else:
        logger.info("Removing original (no backup)")
        shutil.rmtree(source_store)

    shutil.move(target_store, source_store)

    # Cleanup temp
    logger.info("Cleaning up temporary files...")
    shutil.rmtree(temp_store, ignore_errors=True)

    logger.info("✓ Rechunking complete!")

    # Consolidate metadata
    logger.info("Consolidating metadata...")
    zarr.consolidate_metadata(source_store)

    # Verify
    ds_new = xr.open_zarr(source_store, consolidated=True)
    logger.info(f"New chunks (first 3 vars): {list(ds_new.chunks.items())[:3]}")
    logger.info(f"New dataset size: {ds_new.nbytes / 1e9:.2f} GB")

    if backup:
        logger.info(f"\nBackup stored at: {backup_store}")
        logger.info(f"Remove backup with: rm -rf {backup_store}")


def main():
    parser = argparse.ArgumentParser(
        description="Rechunk Zarr store to daily (time=1) chunks"
    )
    parser.add_argument(
        "--zarr-path",
        type=Path,
        required=True,
        help="Path to the zarr store (e.g., /path/to/bgc_data.zarr)"
    )
    parser.add_argument(
        "--max-mem",
        type=str,
        default="60GB",
        help="Maximum memory for rechunking (default: 60GB)"
    )
    parser.add_argument(
        "--compression",
        type=int,
        default=1,
        help="Compression level 1-9 (default: 1)"
    )
    parser.add_argument(
        "--no-backup",
        action="store_true",
        help="Do not keep a backup of the original data"
    )
    parser.add_argument(
        "--time-chunk-size",
        type=int,
        default=5,
        help="Time chunk size in days (default: 5)"
    )
    parser.add_argument(
        "--output-path",
        type=Path,
        default=None,
        help="Custom output path for rechunked data (default: same directory as input)"
    )
    parser.add_argument(
        "--temp-path",
        type=Path,
        default=None,
        help="Custom temporary storage path (default: same directory as input)"
    )

    args = parser.parse_args()

    if not args.zarr_path.exists():
        logger.error(f"Zarr store not found: {args.zarr_path}")
        sys.exit(1)

    if not args.zarr_path.is_dir():
        logger.error(f"Path is not a directory: {args.zarr_path}")
        sys.exit(1)

    logger.info(f"Zarr store: {args.zarr_path}")
    logger.info(f"Max memory: {args.max_mem}")
    logger.info(f"Compression level: {args.compression}")
    logger.info(f"Time chunk size: {args.time_chunk_size} days")
    logger.info(f"Backup: {not args.no_backup}")
    if args.output_path:
        logger.info(f"Custom output path: {args.output_path}")
    if args.temp_path:
        logger.info(f"Custom temp path: {args.temp_path}")

    try:
        rechunk_to_daily(
            args.zarr_path,
            max_mem=args.max_mem,
            compression_level=args.compression,
            backup=not args.no_backup,
            time_chunk_size=args.time_chunk_size,
            output_path=args.output_path,
            temp_path=args.temp_path
        )
        logger.info("\n✓ SUCCESS: Rechunking completed successfully!")
    except Exception as e:
        logger.error(f"\n✗ FAILED: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main()
