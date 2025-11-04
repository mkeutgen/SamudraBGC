import enum
import logging
from typing import TypeAlias, TypeVar

logger = logging.getLogger(__name__)

import torch
import xarray as xr
from jaxtyping import Bool, Float

from ocean_emulators.utils.multiton import Multiton

# Common Type Aliases
# See "Existing jaxtyping annotations" section of
#  https://docs.kidger.site/jaxtyping/api/array/#array

# Our Arrays will be either `torch.Tensor`s or `xarray.DataArray`s.
Array = TypeVar("Array", torch.Tensor, xr.DataArray)

Grid = Float[Array, "lat lon"]
Prognostic = Float[
    Grid, "*batch prognostic_vars"
]  # equivalent to "*batch prognostic_vars lat lon"
Boundary = Float[Grid, "*batch boundary_vars"]
# A note from jaxtyping (why we can't do "prognostic_vars+boundary_vars"):
#   In practice you should usually only use symbolic axes in annotations
#   for return types, referring only to axes annotated for arguments.
# So, we'll leave this default and use symbolic axes locally.
Input: TypeAlias = Float[Grid, "*batch total_vars"]

Example = tuple[Input, Prognostic] | tuple[xr.Dataset, xr.Dataset]

GridMask = Bool[Array, "lat lon"]
PrognosticMask = Bool[GridMask, "prognostic_vars"]

SingleChannelVar = Float[torch.Tensor, "batch time lat lon"]
DictSingleChannelVar = dict[str, SingleChannelVar]
SinglePrognosticTimeSeries = Float[Grid, "*batch time"]

SingleTimeSeriesOutput = Float[torch.Tensor, "batch=1 time prognostic_vars lat lon"]
BatchTimeSeriesOutput = Float[
    torch.Tensor, "batch time=(hist+1) prognostic_vars lat lon"
]
HistBatched = Float[torch.Tensor, "batch_hist prognostic_vars lat lon"]
HistChanneled = Float[torch.Tensor, "batch hist_prognostic_vars lat lon"]


MAX_TRAIN_MODEL_STEPS_FORWARD = 200

# Experiment prognostic and boundary variables
# Assumption that all 3D variables are appended with depth_i_levels
# and all 2D variables do not have any digits / underscores in their names

# These represent depth centers for MOM6-Cobalt (50 levels)
DEPTH_LEVELS = [
    1.0,
    3.0,
    5.0,
    7.0,
    9.0,
    11.0,
    13.0,
    15.005,
    17.015,
    19.03,
    21.055,
    23.095,
    25.16,
    27.255,
    29.385,
    31.565,
    33.81,
    36.135,
    38.56,
    41.105,
    43.795,
    46.655,
    49.715,
    53.015,
    56.6,
    60.515,
    64.805,
    69.525,
    74.74,
    80.515,
    86.92,
    94.04,
    101.96,
    110.77,
    120.575,
    131.485,
    143.615,
    157.095,
    172.06,
    188.655,
    207.035,
    227.365,
    249.82,
    274.585,
    301.86,
    400.915,
    483.69,
    582.335,
    699.24,
    998.605,
]

# Depth thicknesses - computed from level interfaces
DEPTH_THICKNESS = [
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.0,
    2.01,
    2.01,
    2.02,
    2.03,
    2.05,
    2.08,
    2.11,
    2.15,
    2.21,
    2.28,
    2.37,
    2.48,
    2.61,
    2.77,
    2.95,
    3.17,
    3.43,
    3.74,
    4.09,
    4.49,
    4.95,
    5.48,
    6.07,
    6.74,
    7.5,
    8.34,
    9.28,
    10.33,
    11.49,
    12.77,
    14.19,
    15.74,
    17.45,
    19.31,
    21.35,
    23.56,
    25.97,
    28.58,
    31.41,
    34.47,
    37.77,
    41.32,
    45.14,
]

# Generate depth index levels for 50 levels
DEPTH_I_LEVELS = [str(i) for i in range(50)]

# Mask variables for 50 levels
MASK_VARS = [f"mask_{i}" for i in range(50)]

RHO_0 = 1035.0  # DENSITY_OF_WATER kg/m^3
CP_SW = 3992.0  # SPECIFIC_HEAT_OF_WATER J/kg/K
SECONDS_PER_5DAY = 5 * 24 * 60 * 60  # 5 day average
TIME_DELTA = 5  # Time delta in days

PrognosticVarNames = list[str]
PROGNOSTIC_VARS: dict[str, PrognosticVarNames] = {
    # Full state including dynamics
    "full_state": [
        k + str(j)
        for k in [
            "dic_",
            "o2_",
            "no3_",
            "pp_",
            "chl_",  # Biogeochem
            "temp_",
            "salt_",  # Thermo
            "uo_",
            "vo_",
        ]  # Dynamic
        for j in DEPTH_I_LEVELS
    ]
    + ["SSH"],  # Using SSH
    # Without dynamics
    "bgc_thermo_all": [
        k + str(j)
        for k in ["dic_", "o2_", "no3_", "pp_", "chl_", "temp_", "salt_"]
        for j in DEPTH_I_LEVELS
    ]
    + ["SSH"],
    # Minimal for testing
    "minimal_all": [
        k + str(j) for k in ["temp_", "salt_", "o2_", "dic_"] for j in DEPTH_I_LEVELS
    ]
    + ["SSH"],
}

BoundaryVarNames = list[str]
BOUNDARY_VARS: dict[str, BoundaryVarNames] = {
    # Full forcing with surface chl for satellite assimilation
    "full_forcing": ["Qnet", "tauuo", "tauvo", "chl_surface", "PRCmE"],
    # Standard forcing
    "standard_forcing": ["Qnet", "tauuo", "tauvo", "PRCmE"],
    # Minimal
    "minimal_forcing": ["Qnet", "tauuo", "tauvo"],
}

DEFAULT_METADATA = {
    "CT": {
        "long_name": "Conservative Temperature",
        "units": "°C",
    },
    "SA": {
        "long_name": "Absolute Salinity",
        "units": "g/kg",
    },
    "uo": {
        "long_name": "Sea Water X Velocity",
        "units": "m/s",
    },
    "vo": {
        "long_name": "Sea Water Y Velocity",
        "units": "m/s",
    },
    "SSH": {
        "long_name": "Sea surface height above geoid",
        "units": "m",
    },
    "o2": {
        "long_name": "Dissolved Oxygen",
        "units": "mol/kg",
    },
    "dic": {
        "long_name": "Dissolved Inorganic Carbon",
        "units": "mol/kg",
    },
    "no3": {
        "long_name": "Nitrate",
        "units": "mol/kg",
    },
    "chl": {
        "long_name": "Chlorophyll Concentration",
        "units": "mg/m3",
    },
    "pp": {
        "long_name": "Primary Production",
        "units": "mol C/m3/day",
    },
    "tauuo": {
        "long_name": "Surface Downward X Stress",
        "units": "N/m^2",
    },
    "tauvo": {
        "long_name": "Surface Downward Y Stress",
        "units": "N/m^2",
    },
    "Qnet": {
        "long_name": "Net Surface Heat Flux",
        "units": "W/m^2",
    },
    "PRCmE": {
        "long_name": "Precipitation minus Evaporation",
        "units": "kg m-2 s-1",
    },
    "chl_surface": {
        "long_name": "Surface Chlorophyll (satellite)",
        "units": "mg/m3",
    },
}


def construct_metadata(data: xr.Dataset) -> dict[str, dict[str, str]]:
    metadata = {}
    for var in data.variables:
        try:
            metadata[str(var)] = {
                "long_name": data[var].long_name,
                "units": data[var].units,
            }
        except AttributeError:
            if var in DEFAULT_METADATA.keys():
                metadata[str(var)] = DEFAULT_METADATA[str(var)]
            elif (key := str(var).split("_")[0]) in DEFAULT_METADATA.keys():
                metadata[str(var)] = DEFAULT_METADATA[key]
            else:
                logger.info(f"{var} does not have any default metadata")
                metadata[str(var)] = {
                    "long_name": "Unknown",
                    "units": "Unknown",
                }

    return metadata


class LoaderVersion(enum.Enum):
    OM4_EAGER = "om4-eager"
    OM4_TORCH = "om4-torch"


# TODO(#95): See if this can be removed and replaced.
class TensorMap(Multiton):
    def _initialize(self, prognostic_vars_key: str, boundary_vars_key: str):
        """
        Maps input variables / depth levels to their indices in the input tensor.

        VAR_3D_IDX maps the input variables to their indices in the input tensor
        DP_3D_IDX maps the depth levels to their indices in the input tensor
        """
        self.prognostic_vars_key = prognostic_vars_key
        self.VAR_3D_IDX: dict[str, torch.Tensor] = {}
        self.DP_3D_IDX: dict[str, torch.Tensor] = {}

        self.INPT_BOUNDARY_IDX: dict[str, torch.Tensor] = {}
        self.VAR_SET_2D = []
        self.VAR_SET_3D = []
        for out in PROGNOSTIC_VARS[prognostic_vars_key]:
            var_split = out.split("_")
            if len(var_split) == 1:
                self.VAR_SET_2D.append(var_split[0])
            else:
                self.VAR_SET_3D.append(var_split[0])

        # Consistent order of variables
        self.VAR_SET = list(
            dict.fromkeys(
                [out.split("_")[0] for out in PROGNOSTIC_VARS[prognostic_vars_key]]
            )
        )

        levels_str = prognostic_vars_key.split("_")[-1]
        if "all" in levels_str:
            levels = 19
        else:
            levels = int(levels_str)

        self.DEPTH_SET = DEPTH_I_LEVELS[:levels]
        self.prognostic_var_names = PROGNOSTIC_VARS[prognostic_vars_key]
        self.boundary_var_names = BOUNDARY_VARS[boundary_vars_key]
        self.dz = torch.tensor(DEPTH_THICKNESS[:levels])

        self._populate_var_3d_idx()
        self._populate_dp_3d_idx()
        self._populate_boundary_idx()

    def _populate_var_3d_idx(self):
        for kt in self.VAR_SET:
            self.VAR_3D_IDX[kt] = torch.tensor([])
            for i, k in enumerate(self.prognostic_var_names):
                if kt in k:
                    self.VAR_3D_IDX[kt] = torch.cat(
                        [self.VAR_3D_IDX[kt], torch.tensor([i])]
                    )
            self.VAR_3D_IDX[kt] = self.VAR_3D_IDX[kt].to(torch.int32)

    def _populate_dp_3d_idx(self):
        for d in self.DEPTH_SET:
            self.DP_3D_IDX[d] = torch.tensor([])
            for i, k in enumerate(self.prognostic_var_names):
                k_split = k.split("_")
                if len(k_split) == 1:
                    continue
                elif d == k_split[-1]:
                    self.DP_3D_IDX[d] = torch.cat(
                        [self.DP_3D_IDX[d], torch.tensor([i])]
                    )
            self.DP_3D_IDX[d] = self.DP_3D_IDX[d].to(torch.int32)

        self.DP_3D_IDX[self.DEPTH_SET[0]] = torch.cat(
            [
                self.DP_3D_IDX[self.DEPTH_SET[0]],
                torch.tensor([self.VAR_3D_IDX[var_2D] for var_2D in self.VAR_SET_2D]),
            ]
        )

    def _populate_boundary_idx(self):
        """
        Populates the indices of the boundary variables in the input tensor.

        We assume the indices INPT_BOUNDARY_IDX will be used after the boundary
        condition is extracted from the input tensor
        """
        for i, k in enumerate(self.boundary_var_names):
            self.INPT_BOUNDARY_IDX[k] = torch.tensor([i])
