import torch
from torch.optim import optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from traja.models.ae import MultiModelAE


class Optimizer:

    def __init__(self, model_type, model, optimizer_type):

        assert isinstance(model, torch.nn.Module)
        assert str(optimizer_type) in ['Adam', 'Adadelta', 'Adagrad', 'AdamW', 'SparseAdam', 'RMSprop', 'Rprop',
                                       'LBFGS', 'ASGD', 'Adamax']

        self.model_type = model_type
        self.model = model
        self.optimizer_type = optimizer_type
        self.optimizers = {}
        self.schedulers = {}

    def get_optimizers(self, lr=0.0001):
        """Optimizers for each network in the model

        Args:
            model_type ([type]): [description]
            model ([type]): [description]
            lr (float, optional): [description]. Defaults to 0.0001.

        Returns:
            [type]: [description]
        """

        if self.model_type == 'ae' or 'vae':
            keys = ['encoder', 'decoder', 'latent', 'classifier']
            for network in keys:
                self.optimizers[network] = getattr(torch.optim, f'{self.optimizer_type}')(
                    getattr(self.model, f'{network}').parameters(), lr=lr)
            return self.optimizers
        if self.model_type == 'lstm':
            self.optimizers["lstm"] = torch.optim.Adam(self.model.parameters(), lr=lr)
            return self.optimizers

        elif self.model_type == 'vaegan':
            return NotImplementedError

        else:  # LSTM
            return NotImplementedError

    def get_lrschedulers(self, factor=0.1, patience=10):

        """Learning rate scheduler for each network in the model
        NOTE: Scheduler metric should be test set loss

        Args:
            factor (float, optional): [description]. Defaults to 0.1.
            patience (int, optional): [description]. Defaults to 10.

        Returns:
            [dict]: [description]
        """
        for network in self.optimizers.keys():
            self.schedulers[network] = ReduceLROnPlateau(self.optimizers[network], mode='max', factor=factor,
                                                         patience=patience, verbose=True)
        return self.schedulers


if __name__ == '__main__':
    # Test
    model_type = 'ae'
    model = MultiModelAE(input_size=2,
                         sequence_length=10,
                         batch_size=5,
                         num_future=5,
                         hidden_size=10,
                         num_layers=2,
                         output_size=2,
                         num_classes=10,
                         latent_size=10,
                         batch_first=True,
                         dropout=0.2,
                         reset_state=True,
                         bidirectional=True)

    # Get the optimizers
    opt = Optimizer(model_type, model, optimizer_type='RMSprop')
    model_optimizers = opt.get_optimizers(lr=0.1)
    model_schedulers = opt.get_lrschedulers(factor=0.1, patience=10)

    print(model_optimizers, model_schedulers)
