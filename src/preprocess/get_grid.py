import xarray as xr
import numpy as np
from xgcm import Grid

# ------------------------------------------
# 1. Load the hgrid (supergrid)
# ------------------------------------------
hgrid_path = "/scratch/cimes/maximek/MOM6_Double_Gyre/DG-MOM6-COBALTv2/ice_ocean_SIS2/OM4_DG_COBALT/INPUT/DG_hgrid_011deg.nc"
hgrid = xr.open_dataset(hgrid_path)
print("Loaded hgrid:")
print(hgrid)