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
        :param classify: If True, will return the Optimizer and scheduler for classifier

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
        self.classify = classify
        self.optimizers = {}
        self.schedulers = {}

    def get_optimizers(self, lr=0.0001):
        """Optimizers for each network in the model

        Args:

            lr (float, optional): Optimizer learning rate. Defaults to 0.0001.

        Returns:
            dict: Optimizers

        """

        if self.model_type in ["lstm", "custom"]:
            self.optimizers = getattr(torch.optim, f"{self.optimizer_type}")(
                self.model.parameters(), lr=lr
            )

        elif self.model_type in ["ae", "vae"]:
            keys = ["encoder", "decoder", "latent", "classifier"]
            for network in keys:
                if network != "classifier":
                    self.optimizers[network] = getattr(
                        torch.optim, f"{self.optimizer_type}"
                    )(getattr(self.model, f"{network}").parameters(), lr=lr)

            if self.classify:
                self.optimizers["classifier"] = getattr(
                    torch.optim, f"{self.optimizer_type}"
                )(getattr(self.model, "classifier").parameters(), lr=lr)
            else:
                self.optimizers["classifier"] = None

        elif self.model_type == "vaegan":
            return NotImplementedError

        else:  #  self.model_type == "irl":
            return NotImplementedError

        return self.optimizers

    def get_lrschedulers(self, factor: float, patience: int):

        """Learning rate scheduler for each network in the model
        NOTE: Scheduler metric should be test set loss

        Args:
            factor (float, optional): [description]. Defaults to 0.1.
            patience (int, optional): [description]. Defaults to 10.

        Returns:
            [dict]: Learning rate schedulers

        """
        if self.model_type in ["lstm", "custom"]:
            assert not isinstance(self.optimizers, dict)
            self.schedulers = ReduceLROnPlateau(
                self.optimizers,
                mode="max",
                factor=factor,
                patience=patience,
                verbose=True,
            )
        elif self.model_type in ["ae", "vae"]:
            for network in self.optimizers.keys():
                if self.optimizers[network] is not None:
                    self.schedulers[network] = ReduceLROnPlateau(
                        self.optimizers[network],
                        mode="max",
                        factor=factor,
                        patience=patience,
                        verbose=True,
                    )
            if not self.classify:
                self.schedulers["classifier"] = None

        elif self.model_type == "irl":
            return NotImplementedError

        else:  # self.model_type == 'vaegan':
            return NotImplementedError

        return self.schedulers


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
