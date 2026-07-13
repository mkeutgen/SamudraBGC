#!/usr/bin/env python3
"""
Zip a zarr store into a single .zarr.zip file for easy distribution.

The output can be opened directly without extraction:
    import zarr
    ds = zarr.open("eval_subset.zarr.zip", mode="r")

Usage:
    python scripts/zip_zarr.py --input eval_subset.zarr --output eval_subset.zarr.zip
"""

import argparse
import os

import zarr


def zip_zarr(input_path: str, output_path: str):
    """Copy a directory zarr store into a ZipStore."""
    if os.path.exists(output_path):
        os.remove(output_path)

    source = zarr.open(input_path, mode="r")

    # ZipStore with no extra compression (chunks are already compressed)
    import zipfile
    dest_store = zarr.ZipStore(output_path, mode="w", compression=zipfile.ZIP_STORED)

    zarr.copy_store(source.store, dest_store)
    dest_store.close()

    in_size = sum(
        os.path.getsize(os.path.join(dp, f))
        for dp, _, fn in os.walk(input_path)
        for f in fn
    )
    out_size = os.path.getsize(output_path)
    print(f"Input:  {input_path} ({in_size / 1e9:.1f} GB, many files)")
    print(f"Output: {output_path} ({out_size / 1e9:.1f} GB, 1 file)")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    zip_zarr(args.input, args.output)


if __name__ == "__main__":
    main()
