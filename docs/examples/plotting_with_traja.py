"""
Plotting with traja
-----------------------------------
`traja  <https://traja.readthedocs.io>`_ is a Python
library providing a selection of easy-to-use spatial visualizations. It is
built on top of pandas and is designed to work with a range of libraries.
For more details on the library refer to its documentation.
First we'll load in data using traja.
"""
import traja

df = traja.TrajaDataFrame({'x':[0,1,2,3,4],'y':[1,3,2,4,5]})

###############################################################################
# Plotting with Traja
# =====================
#
# We start out by plotting a basic sime series trajectory using the ``traja``
# accessor.
df.traja.plot(figsize=(8,4))

###############################################################################
# Generate Random Walks
# =====================
#
# Also, random walks can be generated using ``generate``.
df = traja.generate(n=1000, random=True, fps=30)
df.traja.plot(figsize=(8,4))

###############################################################################
# Traja can re-scale data with any units

df.traja.scale(100)
df.traja.plot(figsize=(8, 4), xlabel='x (cm)', ylabel='y (cm)')

###############################################################################
# Rediscretize step lengths
# =========================
#
# ``rediscretize`` method allows resample the trajectory into an arbitrary step
# length.
rt = df.traja.rediscretize(R=10)
rt.traja.plot(figsize=(8, 4))

###############################################################################
# Calculate derivatives
# =====================
#
# Derivatives can be calculated with ``derivatives`` and histograms can be
# plotted using pandas built-in :meth:`pandas.DataFrame.plot()` method.
derivs = df.traja.get_derivatives()
speed = derivs['speed']
speed.hist()

###############################################################################
# Again, these are just some of the plots you can make with Traja. There are
# several other possibilities not covered in this brief introduction. For more
# examples, refer to the
# `Gallery <https://traja.readthedocs.io/en/latest/gallery/index.html>`_ in the
# traja  documentation.
