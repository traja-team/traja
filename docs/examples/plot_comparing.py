"""
Comparing
---------
traja allows comparing trajectories using various methods.
"""
import traja

df = traja.generate(seed=0)
df.traja.plot()

###############################################################################
# Fast Dynamic Time Warping of Trajectories
# =========================================
#
# Fast dynamic time warping can be performed using ``fastdtw``.
# Source article: `link <https://cs.fit.edu/~pkc/papers/tdm04.pdf>`_.
import numpy as np

rotated = traja.rotate(df, angle=np.pi / 10)
rotated.traja.plot()

###############################################################################
# Compare trajectories hierarchically
# ===================================
# Hierarchical agglomerative clustering allows comparing trajectories as actograms
# and finding nearest neighbors. This is useful for comparing circadian rhythms,
# for example.

# Generate random trajectories
trjs = [traja.generate(seed=i) for i in range(20)]

# Calculate displacement
displacements = [trj.traja.calc_displacement() for trj in trjs]

traja.plot_clustermap(displacements)

###############################################################################
# Compare trajectories point-wise
# ===============================
dist = traja.distance_between(df.traja.xy, rotated.traja.xy)

print(f"Distance between the two trajectories is {dist}")
