"""
Comparing
---------
traja allows comparing trajectories using various methods.
"""
import traja

df = traja.generate()
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
# Compare trajectories point-wise
# ===============================
dist = traja.distance(df.traja.xy, rotated.traja.xy)

print(f"Distance between the two trajectories is {dist}")
