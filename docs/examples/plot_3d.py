"""
3D Plotting with traja
----------------------
Plot trajectories with time in the vertical axis.
Note: Adjust matplotlib args ``dist``, ``labelpad``, ``aspect`` and ``adjustable```
as needed.
"""
import traja

df = traja.TrajaDataFrame({"x": [0, 1, 2, 3, 4], "y": [1, 3, 2, 4, 5]})

trj = traja.generate()
ax = trj.traja.plot_3d(dist=15, labelpad=32, title="Traja 3D Plot")

########
# Colors
# -------
#
# `Matplotlib cmaps<https://matplotlib.org/examples/color/colormaps_reference.html>`_ are available

trj.traja.plot_3d(dist=15, labelpad=32, title="Traja 3D Plot", cmap="jet")
