import torch
from torch.optim.lr_scheduler import ReduceLROnPlateau


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
        self.forecasting_schedulers = {}
        self.classification_schedulers = {}
        self.regression_schedulers = {}

        self.forecasting_keys = ["encoder", "decoder", "latent"]
        self.classification_keys = ["classifier"]
        self.regression_keys = ["regressor"]

    def get_optimizers(self, lr=0.0001):
        """Optimizers for each network in the model

        Args:

            lr (float, optional): Optimizer learning rate. Defaults to 0.0001.

        Returns:
            dict: Optimizers

        """

        if self.model_type in ["lstm", "custom"]:
            self.optimizers["encoder"] = getattr(torch.optim, f"{self.optimizer_type}")(
                self.model.parameters(), lr=lr
            )

        elif self.model_type in ["ae", "vae"]:
            keys = ["encoder", "decoder", "latent", "classifier", "regressor"]
            for key in keys:
                network = getattr(self.model, f"{key}", None)
                if network is not None:
                    self.optimizers[key] = getattr(
                        torch.optim, f"{self.optimizer_type}"
                    )(network.parameters(), lr=lr)

        elif self.model_type == "vaegan":
            return NotImplementedError

        else:  # self.model_type == "irl":
            return NotImplementedError

        forecasting_optimizers = [
            self.optimizers[key]
            for key in self.forecasting_keys
            if key in self.optimizers
        ]
        classification_optimizers = [
            self.optimizers[key]
            for key in self.classification_keys
            if key in self.optimizers
        ]
        regression_optimizers = [
            self.optimizers[key]
            for key in self.regression_keys
            if key in self.optimizers
        ]
        return forecasting_optimizers, classification_optimizers, regression_optimizers

    def get_lrschedulers(self, factor: float, patience: int):

        """Learning rate scheduler for each network in the model
        NOTE: Scheduler metric should be test set loss

        Args:
            factor (float, optional): [description]. Defaults to 0.1.
            patience (int, optional): [description]. Defaults to 10.

        Returns:
            [dict]: Learning rate schedulers

        """

        if self.model_type == "irl" or self.model_type == "vaegan":
            return NotImplementedError

        forecasting_keys = [
            key for key in self.forecasting_keys if key in self.optimizers
        ]
        classification_keys = [
            key for key in self.classification_keys if key in self.optimizers
        ]
        regression_keys = [
            key for key in self.regression_keys if key in self.optimizers
        ]

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

        for network in regression_keys:
            self.regression_schedulers[network] = ReduceLROnPlateau(
                self.optimizers[network],
                mode="max",
                factor=factor,
                patience=patience,
                verbose=True,
            )

        return (
            self.forecasting_schedulers,
            self.classification_schedulers,
            self.regression_schedulers,
        )
