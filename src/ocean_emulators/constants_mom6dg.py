"""
MOM6-DG specific constants for BGC emulator
===========================================
Defines prognostic, boundary, and diagnostic variables 
specific to MOM6 Double Gyre biogeochemical configuration.
"""

import numpy as np
from typing import Dict, List

# MOM6-DG depth levels (modify based on your actual configuration)
# This example uses 50 levels similar to OM4 but adjust to your MOM6-DG setup
MOM6DG_DEPTH_LEVELS = np.array([
    1.0, 3.0, 5.0, 7.0, 9.0, 11.0, 13.0, 15.005, 17.015, 19.03,
    21.055, 23.095, 25.16, 27.255, 29.385, 31.565, 33.81, 36.135,
    38.56, 41.105, 43.795, 46.655, 49.715, 53.015, 56.6, 60.515,
    64.805, 69.525, 74.74, 80.515, 86.92, 94.04, 101.96, 110.77,
    120.575, 131.485, 143.615, 157.095, 172.06, 188.655, 207.035,
    227.365, 249.82, 274.585, 301.86, 400.915, 483.69, 582.335,
    699.24, 998.605
])

# Generate depth index levels
MOM6DG_DEPTH_I_LEVELS = [str(i) for i in range(len(MOM6DG_DEPTH_LEVELS))]

# Mask variables for each depth level
MOM6DG_MASK_VARS = [f"mask_{i}" for i in range(len(MOM6DG_DEPTH_LEVELS))]

# ===================================================================
# PROGNOSTIC VARIABLES
# These are the state variables that the model predicts
# ===================================================================

MOM6DG_PROG_VARS_MAP = {
    # Full state including dynamics and biogeochemistry
    "mom6dg_full": [
        k + str(j) for k in [
            # Biogeochemical tracers
            "dic_", "o2_", "no3_", "po4_", "alk_", "chl_", "pp_",
            # Physical state (using native temp/salt)
            "temp_", "salt_",  
            # Velocity 
            "uo_", "vo_"
        ]
        for j in MOM6DG_DEPTH_I_LEVELS
    ] + ["SSH"],  # Sea surface height
    
    # Biogeochemistry and thermodynamics only (no velocity)
    "mom6dg_bgc_thermo": [
        k + str(j) for k in [
            "dic_", "o2_", "no3_", "po4_", "alk_", "chl_", "pp_",
            "temp_", "salt_"
        ] 
        for j in MOM6DG_DEPTH_I_LEVELS
    ] + ["SSH"],
    
    # Core biogeochemical tracers only
    "mom6dg_bgc_core": [
        k + str(j) for k in ["dic_", "o2_", "no3_", "po4_", "alk_"] 
        for j in MOM6DG_DEPTH_I_LEVELS
    ] + ["SSH"],
    
    # Minimal configuration for testing
    "mom6dg_minimal": [
        k + str(j) for k in ["temp_", "salt_", "o2_", "dic_"] 
        for j in MOM6DG_DEPTH_I_LEVELS
    ] + ["SSH"],
    
    # Physical variables only (no biogeochemistry)
    "mom6dg_physics": [
        k + str(j) for k in ["temp_", "salt_", "uo_", "vo_"] 
        for j in MOM6DG_DEPTH_I_LEVELS
    ] + ["SSH"],
}

# ===================================================================
# BOUNDARY/FORCING VARIABLES
# These are the external forcings that drive the system
# ===================================================================

MOM6DG_BOUND_VARS_MAP = {
    # Full forcing including biogeochemical fluxes
    "mom6dg_full_forcing": [
        "Qnet",      # Net surface heat flux
        "tauuo",     # Zonal wind stress
        "tauvo",     # Meridional wind stress
        "PRCmE",     # Precipitation minus evaporation
        "sfc_co2",   # Surface CO2 for carbon cycle
        "iron_dep",  # Iron deposition for productivity
    ],
    
    # Standard MOM6 forcing
    "mom6dg_forcing": [
        "Qnet", 
        "tauuo", 
        "tauvo", 
        "PRCmE"
    ],
    
    # Minimal forcing for testing
    "mom6dg_minimal_forcing": [
        "Qnet", 
        "tauuo", 
        "tauvo"
    ],
    
    # Wind-only forcing
    "mom6dg_wind": [
        "tauuo", 
        "tauvo"
    ],
}

# ===================================================================
# METADATA
# Variable descriptions and units
# ===================================================================

MOM6DG_METADATA = {
    # Physical variables (using native MOM6 temperature and salinity)
    "temp": {
        "long_name": "Potential Temperature",
        "units": "°C",
    },
    "salt": {
        "long_name": "Practical Salinity", 
        "units": "psu",
    },
    "uo": {
        "long_name": "Zonal Velocity",
        "units": "m/s",
    },
    "vo": {
        "long_name": "Meridional Velocity", 
        "units": "m/s",
    },
    "SSH": {
        "long_name": "Sea Surface Height",
        "units": "m",
    },
    
    # Biogeochemical variables
    "dic": {
        "long_name": "Dissolved Inorganic Carbon",
        "units": "mol/m³",
    },
    "o2": {
        "long_name": "Dissolved Oxygen",
        "units": "mol/m³",
    },
    "no3": {
        "long_name": "Nitrate",
        "units": "mol/m³",
    },
    "po4": {
        "long_name": "Phosphate",
        "units": "mol/m³",
    },
    "alk": {
        "long_name": "Alkalinity",
        "units": "mol eq/m³",
    },
    "chl": {
        "long_name": "Chlorophyll Concentration",
        "units": "mg/m³",
    },
    "pp": {
        "long_name": "Primary Production",
        "units": "mgC/m³/day",
    },
    
    # Forcing variables
    "Qnet": {
        "long_name": "Net Surface Heat Flux (positive downward)",
        "units": "W/m²",
    },
    "tauuo": {
        "long_name": "Surface Zonal Wind Stress",
        "units": "N/m²",
    },
    "tauvo": {
        "long_name": "Surface Meridional Wind Stress",
        "units": "N/m²",
    },
    "PRCmE": {
        "long_name": "Precipitation minus Evaporation",
        "units": "kg/m²/s",
    },
    "sfc_co2": {
        "long_name": "Surface CO2 Concentration",
        "units": "ppm",
    },
    "iron_dep": {
        "long_name": "Iron Deposition Flux",
        "units": "mol/m²/s",
    },
}

# ===================================================================
# HELPER FUNCTIONS
# ===================================================================

def get_mom6dg_variable_info(var_name: str) -> Dict:
    """
    Get metadata for a MOM6-DG variable.
    
    Args:
        var_name: Variable name (with or without depth index)
    
    Returns:
        Dictionary with variable metadata
    """
    # Strip depth index if present
    base_name = var_name.split('_')[0] if '_' in var_name else var_name
    
    return MOM6DG_METADATA.get(base_name, {
        "long_name": f"Unknown variable: {base_name}",
        "units": "unknown"
    })


def construct_mom6dg_metadata(data):
    """
    Construct metadata dictionary for all variables in dataset.
    
    Args:
        data: xarray Dataset
    
    Returns:
        Dictionary mapping variable names to metadata
    """
    import xarray as xr
    
    metadata = {}
    for var in data.variables:
        var_str = str(var)
        
        # Try to get from dataset attributes first
        try:
            metadata[var_str] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            # Fall back to our definitions
            metadata[var_str] = get_mom6dg_variable_info(var_str)
    
    return metadata


def get_mom6dg_tensor_indices(prognostic_vars_key: str, boundary_vars_key: str) -> Dict:
    """
    Get tensor channel indices for variables.
    
    Args:
        prognostic_vars_key: Key for prognostic variables set
        boundary_vars_key: Key for boundary variables set
    
    Returns:
        Dictionary mapping variable names to tensor indices
    """
    prog_vars = MOM6DG_PROG_VARS_MAP[prognostic_vars_key]
    bound_vars = MOM6DG_BOUND_VARS_MAP[boundary_vars_key]
    
    indices = {}
    
    # Prognostic variables
    for i, var in enumerate(prog_vars):
        indices[var] = i
    
    # Boundary variables (offset by prognostic count)
    offset = len(prog_vars)
    for i, var in enumerate(bound_vars):
        indices[var] = offset + i
    
    return indices


# ===================================================================
# VALIDATION
# ===================================================================

def validate_mom6dg_config(prognostic_vars_key: str, boundary_vars_key: str):
    """
    Validate that the configuration keys exist.
    
    Args:
        prognostic_vars_key: Key for prognostic variables
        boundary_vars_key: Key for boundary variables
    
    Raises:
        KeyError: If keys are not found
    """
    if prognostic_vars_key not in MOM6DG_PROG_VARS_MAP:
        raise KeyError(
            f"Prognostic vars key '{prognostic_vars_key}' not found. "
            f"Available: {list(MOM6DG_PROG_VARS_MAP.keys())}"
        )
    
    if boundary_vars_key not in MOM6DG_BOUND_VARS_MAP:
        raise KeyError(
            f"Boundary vars key '{boundary_vars_key}' not found. "
            f"Available: {list(MOM6DG_BOUND_VARS_MAP.keys())}"
        )


# ===================================================================
# EXPORT FOR COMPATIBILITY
# For backward compatibility with existing BGC emulator code
# ===================================================================

# These can be imported by the training script
PROG_VARS_MAP = MOM6DG_PROG_VARS_MAP
BOUND_VARS_MAP = MOM6DG_BOUND_VARS_MAP
DEPTH_LEVELS = MOM6DG_DEPTH_LEVELS
DEPTH_I_LEVELS = MOM6DG_DEPTH_I_LEVELS
MASK_VARS = MOM6DG_MASK_VARS