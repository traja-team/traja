"""
Predicting Trajectories
-----------------------
Predicting trajectories with `traja` can be done with an LSTM neural network
via :class:`~traja.nn.TrajectoryLSTM`.
"""
import traja

df = traja.generate(n=1000)

###############################################################################
# Train and visualize predictions
# ===============================
# Recommended training is over 5000 epochs.
from traja.models.nn import TrajectoryLSTM
lstm = TrajectoryLSTM(xy = df.traja.xy)
lstm.train()
lstm.plot(interactive=False)
