"""Render only the chlorophyll variable for fig01_3d_schematic."""
import sys
sys.path.insert(0, "src")

import cmocean
import code_paper.fig01_3d_schematic as m

m.STATE_VARS = [("chl", cmocean.cm.algae, "log", None, None)]
m.FORCING_VARS = []

if __name__ == "__main__":
    m.main()
