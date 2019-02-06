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
.. code-block:: python

    from traja.models.nn import TrajectoryLSTM

    lstm = TrajectoryLSTM(xy = df.traja.xy, epochs=10)
    lstm.train()
    lstm.plot(interactive=False)

.. image:: https://raw.githubusercontent.com/justinshenk/traja/master/docs/source/_static/rnn_prediction.png