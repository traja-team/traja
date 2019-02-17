Predicting Trajectories
=======================

Predicting trajectories with `traja` can be done with an LSTM neural network
via :class:`~traja.models.nn.TrajectoryLSTM`.

.. code-block:: python

    import traja

    df = traja.generate(n=1000)

Train and visualize predictions

.. note::

    Recommended training is over 5000 epochs. This example only uses 10 epochs for demonstration.

.. code-block:: python

    from traja.models.nn import TrajectoryLSTM

    lstm = TrajectoryLSTM(xy=df.traja.xy, epochs=10)
    lstm.train()
    lstm.plot(interactive=True)

.. image:: _static/rnn_prediction.png