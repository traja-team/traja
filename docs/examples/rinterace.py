"""
R interface
-----------
traja allows interfacing with R packages like ltraj.
"""
import traja
from traja import rutils

df = traja.generate()

###############################################################################
# Convert objects to adehabitat class ltraj for further analysis with R
# =====================================================================
#
# `adehabitat <https://www.rdocumentation.org/packages/adehabitat/versions/1.8.20>`_
# is a widely used R library for animal tracking and trajectory
# analysis.
ltraj = rutils.to_ltraj(df)
rutils.plot_ltraj(ltraj)

###############################################################################
# Perform further analysis in Python
# ==================================
# Data frame is stored in first index.
print(ltraj[0].head())