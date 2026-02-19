#!/usr/bin/env python3
"""
Convert log-transformed predictions back to linear space for comparison.

Transforms: linear_var = exp(log_var) - epsilon

Usage:
    python scripts/analysis/convert_log_to_linear.py \
        --input outputs/phase15_helmholtz_log_eval/predictions.zarr \
        --output outputs/phase15_helmholtz_log_eval_linear/predictions.zarr
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import xarray as xr
from dask.diagnostics import ProgressBar

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Epsilon values used during log transformation
EPSILON_MAP = {
    "dic": 1e-10,   # mol/kg — DIC never near zero (min ~2e-3)
    "o2": 1e-10,    # mol/kg — O2 rarely exactly zero (min ~2e-4)
    "chl": 1e-8,    # µg/kg — 2 orders below min (min ~6e-6)
    "no3": 1e-14,   # mol/kg — below observed min (min ~7e-13)
}


def convert_log_to_linear(input_path: str, output_path: str):
    """
    Convert log-transformed predictions back to linear space.

    Args:
        input_path: Path to input zarr with log-transformed variables
        output_path: Path to output zarr with linear variables
    """
    logger.info("="*80)
    logger.info("Converting log-transformed predictions to linear space")
    logger.info("="*80)
    logger.info(f"Input:  {input_path}")
    logger.info(f"Output: {output_path}")

    # Load the dataset
    logger.info("\nLoading dataset...")
    ds = xr.open_zarr(input_path)
    logger.info(f"  Loaded {len(ds.data_vars)} variables")

    # Create output directory
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Find log-transformed variables
    log_vars = [v for v in ds.data_vars if v.startswith('log_')]
    logger.info(f"\nFound {len(log_vars)} log-transformed variables")

    # Convert each log variable to linear space
    logger.info("\nConverting variables:")
    new_vars = {}

    for log_var in log_vars:
        # Extract base variable name and level (e.g., "log_dic_0" -> "dic", "0")
        base_var = log_var.replace('log_', '').rsplit('_', 1)[0]
        level = log_var.rsplit('_', 1)[1] if '_' in log_var.replace('log_', '') else None
        linear_var_name = log_var.replace('log_', '')

        # Get epsilon for this variable type
        epsilon = EPSILON_MAP.get(base_var, 1e-10)

        logger.info(f"  {log_var} -> {linear_var_name} (epsilon={epsilon})")

        # Transform: linear = exp(log) - epsilon
        linear_data = np.exp(ds[log_var]) - epsilon

        # Copy attributes
        attrs = ds[log_var].attrs.copy()
        attrs['transformation'] = f'Converted from log space: exp(log_{base_var}) - {epsilon}'

        new_vars[linear_var_name] = (ds[log_var].dims, linear_data, attrs)

    # Copy non-log variables as-is
    logger.info("\nCopying non-log variables:")
    for var in ds.data_vars:
        if not var.startswith('log_'):
            logger.info(f"  {var}")
            # Extract data, dims, and attrs properly
            new_vars[var] = (ds[var].dims, ds[var].data, ds[var].attrs)

    # Create new dataset
    logger.info("\nCreating output dataset...")
    ds_linear = xr.Dataset(
        {name: xr.DataArray(data, dims=dims, attrs=attrs)
         for name, (dims, data, attrs) in new_vars.items()},
        coords=ds.coords,
        attrs=ds.attrs
    )

    # Add metadata about conversion
    ds_linear.attrs['log_to_linear_conversion'] = 'true'
    ds_linear.attrs['epsilon_values'] = str(EPSILON_MAP)

    logger.info(f"  Output dataset: {len(ds_linear.data_vars)} variables")

    # Save to zarr
    logger.info("\nSaving to zarr...")
    logger.info(f"  Output: {output_path}")

    with ProgressBar():
        ds_linear.to_zarr(output_path, mode='w', consolidated=True)

    logger.info("\n" + "="*80)
    logger.info("Conversion complete!")
    logger.info("="*80)
    logger.info(f"\nConverted variables:")
    for log_var in sorted(log_vars)[:10]:
        linear_var = log_var.replace('log_', '')
        logger.info(f"  {log_var} -> {linear_var}")
    if len(log_vars) > 10:
        logger.info(f"  ... and {len(log_vars) - 10} more")


def main():
    parser = argparse.ArgumentParser(
        description="Convert log-transformed predictions to linear space"
    )
    parser.add_argument(
        '--input',
        type=str,
        required=True,
        help='Path to input zarr with log-transformed variables'
    )
    parser.add_argument(
        '--output',
        type=str,
        required=True,
        help='Path to output zarr with linear variables'
    )

    args = parser.parse_args()

    convert_log_to_linear(args.input, args.output)


if __name__ == '__main__':
    main()
