"""
Comparing
---------
traja allows comparing trajectories using various methods.
"""
import traja

df = traja.generate(angular_error_sd=0.5)

###############################################################################
# Fast Dynamic Time Warping of Trajectories
# =========================================
#
# Fast dynamic time warping can be performed using the optional package 
# ``fastdtw``. It can be installed with ``pip install fastdtw``. 
# Source article: `link <https://cs.fit.edu/~pkc/papers/tdm04.pdf>`_.
import numpy as np
rotated = traja.utils.rotate(df, angle=np.pi/10)
rotated.traja.plot()

###############################################################################
# Compare trajectories point-wise 
# ===============================
dist = traja.utils.distance(df.traja.xy, rotated.traja.xy)

print(f"Distance between the two trajectories is {dist}")
