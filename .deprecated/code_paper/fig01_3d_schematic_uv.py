"""Render only zonal (uo) and meridional (vo) velocity for fig01_3d_schematic."""
import sys
sys.path.insert(0, "src")

import cmocean
import code_paper.fig01_3d_schematic as m

m.STATE_VARS = [
    ("uo", cmocean.cm.balance, "twoslope", None, None),
    ("vo", cmocean.cm.balance, "twoslope", None, None),
]
m.FORCING_VARS = []

if __name__ == "__main__":
    m.main()
