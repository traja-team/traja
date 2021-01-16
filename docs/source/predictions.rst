Predicting Trajectories
=======================

Predicting trajectories with `traja` can be done with a recurrent neural network (RNN). `Traja` includes
the Long Short Term Memory (LSTM), LSTM Autoencoder (LSTM AE) and LSTM Variational Autoencoder (LSTM VAE)
RNNs. Traja also supports custom RNNs.

To model a trajectory using RNNs, one needs to fit the network to the model. `Traja` includes the MultiTaskRNNTrainer
that can solve a prediction, classification and regression problem with `traja` DataFrames.

`Traja` also includes a DataLoader that handles `traja` dataframes.

Below is an example with a prediction LSTM:
via :class:`~traja.models.predictive_models.lstm.LSTM`.

.. code-block:: python

    import traja

    df = traja.datasets.example.jaguar()

.. note::
    LSTMs work better with data between -1 and 1. Therefore the data loader
    scales the data. To view the data in the original coordinate system,
    you need to invert the scaling with the returned `scaler`.

.. code-block:: python

    batch_size = 10 # How many sequences to train every step. Constrained by GPU memory.
    num_past = 10 # How many time steps from which to learn the time series
    num_future = 5 # How many time steps to predict


    data_loaders, scalers = dataset.MultiModalDataLoader(df,
                                                         batch_size=batch_size,
                                                         n_past=num_past,
                                                         n_future=num_future,
                                                         num_workers=1)

.. note::

    The width of the hidden layers and depth of the network are the two main way in which
    one tunes the performance of the network. More complex datasets require wider and deeper
    networks. Below are sensible defaults.

.. code-block:: python

    from traja.models.predictive_models.lstm import LSTM
    input_size = 2 # Number of input dimensions (normally x, y)
    output_size = 2 # Same as input_size when predicting
    num_layers = 2 # Number of LSTM layers. Deeper learns more complex patterns but overfits.
    hidden_size = 32 # Width of layers. Wider learns bigger patterns but overfits. Try 32, 64, 128, 256, 512
    dropout = 0.1 # Ignore some network connections. Improves generalisation.

    model = LSTM(input_size=input_size,
                 hidden_size=hidden_size,
                 num_layers=num_layers,
                 output_size=output_size,
                 dropout=dropout,
                 batch_size=batch_size,
                 num_future=num_future)

.. note::

    Recommended training is over 50 epochs. This example only uses 10 epochs for demonstration.

.. code-block:: python

    from traja.models.train import HybridTrainer

    optimizer_type = 'Adam' # Nonlinear optimiser with momentum
    loss_type = 'huber'

    # Trainer
    trainer = HybridTrainer(model=model,
                            optimizer_type=optimizer_type,
                            loss_type=loss_type)
    # Train the model
    trainer.fit(data_loaders, model_save_path, epochs=10, training_mode='forecasting')

After training, you can determine the network's final performance with test data, if you want to pick
the best model, or with validation data, if you want to determine the performance of your model.

The data_loaders dictionary contains the 'sequential_test_loader' and 'sequential_validation_loader,
that preserve the order of the original data. The dictionary also contains the 'test_loader' and
'validation_loader' data loaders, where the order of the time series is randomised.

.. code-block:: python
    validation_loader = data_loaders['sequential_validation_loader']

    trainer.validate(validation_loader)

Finally, you can display your training results using the built-in plotting libraries.

.. code-block:: python
    from traja.plotting import plot_prediction

    batch_index = 0  # The batch you want to plot
    plot_prediction(model, validation_loader, batch_index)

.. image:: _static/rnn_prediction.png