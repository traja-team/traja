from traja.datasets import dataset
from traja.datasets.example import jaguar
from traja.models.generative_models.vae import MultiModelVAE
from traja.models.predictive_models.ae import MultiModelAE
from traja.models.predictive_models.lstm import LSTM
from traja.models.train import HybridTrainer

# Sample data
df = jaguar()


def test_aevae():
    """
    Test Autoencoder and variational auto encoder models for training/testing/generative network and
    classification networks

    """
    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    # Prepare the dataloader
    data_loaders, scalers = dataset.MultiModalDataLoader(df,
                                                         batch_size=batch_size,
                                                         n_past=num_past,
                                                         n_future=num_future,
                                                         num_workers=1)

    model_save_path = './model.pt'

    model = MultiModelVAE(input_size=2,
                          output_size=2,
                          lstm_hidden_size=32,
                          num_lstm_layers=2,
                          num_classes=9,
                          latent_size=10,
                          dropout=0.1,
                          num_classifier_layers=4,
                          classifier_hidden_size=32,
                          batch_size=batch_size,
                          num_future=num_future,
                          num_past=num_past,
                          bidirectional=False,
                          batch_first=True,
                          reset_state=True)

    # Model Trainer
    # Model types; "ae" or "vae"
    trainer = HybridTrainer(model=model,
                            optimizer_type='Adam',
                            loss_type='huber')

    # Train the model
    trainer.fit(data_loaders, model_save_path, epochs=10, training_mode='forecasting')
    trainer.fit(data_loaders, model_save_path, epochs=10, training_mode='classification')


def test_ae():
    """
    Test Autoencoder and variational auto encoder models for training/testing/generative network and
    classification networks

    """
    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 5
    # Prepare the dataloader
    data_loaders, scalers = dataset.MultiModalDataLoader(df,
                                                         batch_size=batch_size,
                                                         n_past=num_past,
                                                         n_future=num_future,
                                                         num_workers=1)

    model_save_path = './model.pt'

    model = MultiModelAE(input_size=2, num_past=num_past, batch_size=batch_size, num_future=num_future,
                         lstm_hidden_size=32, num_lstm_layers=2, output_size=2, latent_size=10, batch_first=True,
                         dropout=0.1, reset_state=True, bidirectional=False, num_classifier_layers=4,
                         classifier_hidden_size=32, num_classes=9)

    # Model Trainer
    # Model types; "ae" or "vae"
    trainer = HybridTrainer(model=model,
                            optimizer_type='Adam',
                            loss_type='huber')

    # Train the model
    trainer.fit(data_loaders, model_save_path, epochs=10, training_mode='forecasting')
    trainer.fit(data_loaders, model_save_path, epochs=10, training_mode='classification')


def test_lstm():
    """
    Testing method for lstm model used for forecasting.
    """
    # Hyperparameters
    batch_size = 10
    num_past = 10
    num_future = 10

    # For timeseries prediction
    assert num_past == num_future

    # Prepare the dataloader
    data_loaders, scalers = dataset.MultiModalDataLoader(df,
                                                         batch_size=batch_size,
                                                         n_past=num_past,
                                                         n_future=num_future,
                                                         num_workers=1)

    model_save_path = './model.pt'

    # Model init
    model = LSTM(input_size=2,
                 hidden_size=32,
                 num_layers=2,
                 output_size=2,
                 dropout=0.1,
                 batch_size=batch_size,
                 num_future=num_future,
                 bidirectional=False,
                 batch_first=True,
                 reset_state=True)

    # Model Trainer
    trainer = HybridTrainer(model=model,
                            optimizer_type='Adam',
                            loss_type='huber')
    # Train the model
    trainer.fit(data_loaders, model_save_path, epochs=10, training_mode='forecasting')
