#!/usr/bin/env python
"""
Merge yearly zarr files into a single consolidated dataset with proper rechunking.
"""
import xarray as xr
import zarr
import numpy as np
from pathlib import Path
from numcodecs import Blosc
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def merge_yearly_zarr_files(input_dir: Path, output_file: str = "bgc_data.zarr",
                            time_chunk: int = 365, lev_chunk: int = 50,
                            y_chunk: int = 68, x_chunk: int = 45):
    """Merge individual yearly zarr files into single consolidated file with rechunking."""
    input_dir = Path(input_dir)
    
    # Find all yearly zarr files
    yearly_files = sorted(input_dir.glob("bgc_data_*.zarr"))
    if not yearly_files:
        raise FileNotFoundError(f"No bgc_data_*.zarr files found in {input_dir}")
    
    logger.info(f"Found {len(yearly_files)} yearly files to merge")
    
    # Open all datasets lazily
    datasets = []
    for f in yearly_files:
        logger.info(f"Opening {f.name}")
        ds = xr.open_zarr(f)
        datasets.append(ds)
    
    # Concatenate along time dimension
    logger.info("Concatenating datasets along time dimension...")
    combined = xr.concat(datasets, dim="time", combine_attrs="drop_conflicts")
    
    # Identify and drop datetime/timedelta variables (except main 'time')
    datetime_vars = []
    timedelta_vars = []
    
    for var in combined.variables:
        if var == 'time':
            continue
        if np.issubdtype(combined[var].dtype, np.datetime64):
            datetime_vars.append(var)
            logger.info(f"Found datetime variable: {var}")
        elif np.issubdtype(combined[var].dtype, np.timedelta64):
            timedelta_vars.append(var)
            logger.info(f"Found timedelta variable: {var}")
    
    vars_to_drop = datetime_vars + timedelta_vars
    if vars_to_drop:
        logger.info(f"Dropping time metadata variables: {vars_to_drop}")
        combined = combined.drop_vars(vars_to_drop, errors='ignore')
    
    # Determine chunk sizes based on data dimensions
    chunk_dict = {'time': time_chunk}
    
    # Check which dimensions exist and set appropriate chunks
    if 'lev' in combined.dims:
        chunk_dict['lev'] = min(lev_chunk, combined.sizes['lev'])
    if 'y' in combined.dims:
        chunk_dict['y'] = min(y_chunk, combined.sizes['y'])
    if 'x' in combined.dims:
        chunk_dict['x'] = min(x_chunk, combined.sizes['x'])
    
    logger.info(f"Rechunking with: {chunk_dict}")
    
    # Rechunk to uniform sizes
    combined = combined.chunk(chunk_dict)
    
    # Write consolidated file
    output_path = input_dir / output_file
    logger.info(f"Writing consolidated file to {output_path}")
    
    compressor = Blosc(cname="zstd", clevel=1, shuffle=Blosc.BITSHUFFLE)
    
    # Create encoding dict
    encoding = {}
    for v in combined.data_vars:
        if not np.issubdtype(combined[v].dtype, np.datetime64) and \
           not np.issubdtype(combined[v].dtype, np.timedelta64):
            encoding[v] = {"compressor": compressor, "dtype": "float32"}
    
    logger.info(f"Encoding {len(encoding)} variables with float32 compression")
    logger.info("Writing to zarr (this may take a while)...")
    
    combined.to_zarr(output_path, mode="w", consolidated=False, 
                     zarr_version=2, encoding=encoding)
    zarr.consolidate_metadata(str(output_path))
    
    logger.info(f"Successfully created {output_path}")
    logger.info(f"Final dataset shape: {combined.sizes}")
    logger.info(f"Final chunking: {combined.chunks}")
    
    return output_path

if __name__ == "__main__":
    import sys
    if len(sys.argv) < 2:
        print("Usage: python merge_yearly_zarr.py <data_directory> [time_chunk] [lev_chunk] [y_chunk] [x_chunk]")
        print("Example: python merge_yearly_zarr.py /path/to/data 365 50 68 45")
        sys.exit(1)
    
    data_dir = Path(sys.argv[1])
    
    # Optional chunk size arguments
    time_chunk = int(sys.argv[2]) if len(sys.argv) > 2 else 365
    lev_chunk = int(sys.argv[3]) if len(sys.argv) > 3 else 50
    y_chunk = int(sys.argv[4]) if len(sys.argv) > 4 else 68
    x_chunk = int(sys.argv[5]) if len(sys.argv) > 5 else 45
    
    merge_yearly_zarr_files(data_dir, "bgc_data.zarr", 
                           time_chunk, lev_chunk, y_chunk, x_chunk)