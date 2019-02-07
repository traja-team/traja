"""
Plotting trajectories on a grid
-------------------------------
traja allows comparing trajectories using various methods.
"""
import traja

df = traja.generate()

###############################################################################
# Plot a heat map of the trajectory
# =================================
# A heat map can be generated using :func:`~traja.trajectory.trip_grid`.
df.traja.trip_grid()

###############################################################################
# Increase the grid resolution
# ============================
# Number of bins can be specified with the `bins` parameter.
df.traja.trip_grid(bins=40)

