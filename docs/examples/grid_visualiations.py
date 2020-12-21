"""
Plotting trajectories on a grid
-----------------------------------
traja allows comparing trajectories using various methods.
"""
import traja

df = traja.generate()

###############################################################################
# Plot a heat map of the trajectory
# =================================
#
df.traja.trip_grid()

