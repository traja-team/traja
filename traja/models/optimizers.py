import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau
from traja.models.generative_models.vae import MultiModelVAE
from traja.models.predictive_models.ae import MultiModelAE
from traja.models.predictive_models.lstm import LSTM
from traja.models.predictive_models.irl import MultiModelIRL
from traja.models.generative_models.vaegan import MultiModelVAEGAN


class Optimizer:
    def __init__(self, model_type, model, optimizer_type, classify=False):

        """
        Wrapper for setting the model optimizer and learning rate schedulers using ReduceLROnPlateau;
        If the model type is 'ae' or 'vae' - var optimizers is a dict with separate optimizers for encoder, decoder,
        latent and classifier. In case of 'lstm', var optimizers is an optimizer for lstm and TimeDistributed(linear layer)
        :param model_type: Type of model 'ae', 'vae' or 'lstm'
        :param model: Model instance
        :param optimizer_type: Optimizer to be used; Should be one in ['Adam', 'Adadelta', 'Adagrad', 'AdamW', 'SparseAdam', 'RMSprop', 'Rprop',
                                       'LBFGS', 'ASGD', 'Adamax']
        """

        assert isinstance(model, torch.nn.Module)
        assert str(optimizer_type) in [
            "Adam",
            "Adadelta",
            "Adagrad",
            "AdamW",
            "SparseAdam",
            "RMSprop",
            "Rprop",
            "LBFGS",
            "ASGD",
            "Adamax",
        ]

        self.model_type = model_type
        self.model = model
        self.optimizer_type = optimizer_type
        self.optimizers = {}
        self.forecasting_schedulers = {}
        self.classification_schedulers = {}

        self.forecasting_keys = ['encoder', 'decoder', 'latent']
        self.classification_keys = ['classifier']

    def get_optimizers(self, lr=0.0001):
        """Optimizers for each network in the model

        Args:

            lr (float, optional): Optimizer learning rate. Defaults to 0.0001.

        Returns:
            [type]: [description]
        """

        if self.model_type in ["lstm", "custom"]:
            self.optimizers['encoder'] = getattr(torch.optim, f"{self.optimizer_type}")(
                self.model.parameters(), lr=lr
            )

        elif self.model_type in ["ae", "vae"]:
            keys = ["encoder", "decoder", "latent", "classifier"]
            for key in keys:
                network = getattr(
                    torch.optim, f"{self.optimizer_type}"
                )(getattr(self.model, f"{key}").parameters(), lr=lr)
                if network is not None:
                    self.optimizers[key] = network

        elif self.model_type == "vaegan":
            return NotImplementedError

        else:  #  self.model_type == "irl":
            return NotImplementedError

        forecasting_optimizers = [self.optimizers[key] for key in self.forecasting_keys if key in self.optimizers]
        classification_optimizers = [self.optimizers[key] for key in self.classification_keys if key in self.optimizers]
        return forecasting_optimizers, classification_optimizers

    def get_lrschedulers(self, factor: float, patience: int):

        """Learning rate scheduler for each network in the model
        NOTE: Scheduler metric should be test set loss

        Args:
            factor (float, optional): [description]. Defaults to 0.1.
            patience (int, optional): [description]. Defaults to 10.

        Returns:
            [dict]: [description]
        """

        if self.model_type == "irl" or self.model_type == 'vaegan':
            return NotImplementedError

        forecasting_keys = [key for key in self.forecasting_keys if key in self.optimizers]
        classification_keys = [key for key in self.classification_keys if key in self.optimizers]

        for network in forecasting_keys:
            self.forecasting_schedulers[network] = ReduceLROnPlateau(
                self.optimizers[network],
                mode="max",
                factor=factor,
                patience=patience,
                verbose=True,
            )
        for network in classification_keys:
            self.classification_schedulers[network] = ReduceLROnPlateau(
                self.optimizers[network],
                mode="max",
                factor=factor,
                patience=patience,
                verbose=True,
            )

        return self.forecasting_schedulers, self.classification_schedulers


if __name__ == "__main__":
    # Test
    model_type = "custom"
    model = MultiModelAE(
        input_size=2,
        num_past=10,
        batch_size=5,
        num_future=5,
        lstm_hidden_size=32,
        num_lstm_layers=2,
        classifier_hidden_size=32,
        num_classifier_layers=4,
        output_size=2,
        num_classes=10,
        latent_size=10,
        batch_first=True,
        dropout=0.2,
        reset_state=True,
        bidirectional=True,
    )

    # Get the optimizers
    opt = Optimizer(model_type, model, optimizer_type="RMSprop")
    model_optimizers = opt.get_optimizers(lr=0.1)
    model_schedulers = opt.get_lrschedulers(factor=0.1, patience=10)

    print(model_optimizers, model_schedulers)
