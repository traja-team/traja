"""
Average direction for each grid cell
====================================
See the flow between grid cells.
"""
import traja

df = traja.generate(seed=0)

###############################################################################
# Average Flow (3D)
# -----------------
# Flow can be plotted by specifying the `kind` parameter of :func:`traja.plotting.plot_flow`
# or by calling the respective functions.

import traja

traja.plotting.plot_surface(df, bins=32)

###############################################################################
# Quiver
# ------
# Quiver plot
# Additional arguments can be specified as a dictionary to `quiverplot_kws`.

traja.plotting.plot_quiver(df, bins=32)

###############################################################################
# Contour
# -------
# Parameters `filled` and `quiver` are both enabled by default and can be
# disabled.
# Additional arguments can be specified as a dictionary to `contourplot_kws`.

traja.plotting.plot_contour(df, filled=False, quiver=False, bins=32)

###############################################################################
# Contour (Filled)
# ----------------

traja.plotting.plot_contour(df, bins=32, contourfplot_kws={"cmap": "coolwarm"})

###############################################################################
# Stream
# ------
# 'cmap' can be specified, eg, 'coolwarm', 'viridis', etc.
# Additional arguments can be specified as a dictionary to 'streamplot_kws'.

traja.plotting.plot_stream(df, cmap="jet", bins=32)

###############################################################################
# Polar bar
# ---------
traja.plotting.polar_bar(df)

###############################################################################
# Polar bar (histogram)
# ---------------------
traja.plotting.polar_bar(df, overlap=False)
