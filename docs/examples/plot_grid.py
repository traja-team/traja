"""
Plotting trajectories on a grid
-------------------------------
traja allows comparing trajectories using various methods.
"""
import traja

df = traja.generate(seed=0)

###############################################################################
# Plot a heat map of the trajectory
# =================================
# A heat map can be generated using :func:`~traja.trajectory.trip_grid`.
df.traja.trip_grid()

###############################################################################
# Increase the grid resolution
# ============================
# Number of bins can be specified with the ``bins`` parameter.
df.traja.trip_grid(bins=40)

###############################################################################
# Convert coordinates to grid indices
# ===================================
# Number of x and y bins can be specified with the ``bins``` parameter.

from traja.trajectory import grid_coordinates

grid_coords = grid_coordinates(df, bins=32)
print(grid_coords.head())

###############################################################################
# Transitions as Markov first-order Markov model
# ==============================================
# Probability of transitioning between cells is computed using :func:`traja.trajectory.transitions`.

transitions_matrix = traja.trajectory.transitions(df, bins=32)
print(transitions_matrix[:10])
