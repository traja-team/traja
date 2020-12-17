import torch


class Criterion:

    def __init__(self):
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.crossentropy_loss = torch.nn.CrossEntropyLoss()

    def ae_criterion(self, predicted, target, loss_type='huber'):

        if loss_type == 'huber':
            loss = self.huber_loss(predicted, target)
            return loss
        else:  # Root MSE
            return torch.sqrt(torch.mean((predicted - target) ** 2))

    def vae_criterion(self, predicted, target, mu, logvar, loss_type='huber'):
        """Time series generative model loss function
        """
        if loss_type == 'huber':
            dist_x = self.huber_loss(predicted, target)
        else:
            dist_x = torch.sqrt(torch.mean((predicted - target) ** 2))
        KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return dist_x + KLD

    def classifier_criterion(self, predicted, target):
        """Classifier loss function"""
        loss = self.crossentropy_loss(predicted, target)
        return loss

    def lstm_criterion(self, predicted, target):

        loss = self.huber_loss(predicted, target)
        return loss

    def vaegan_criterion(self):
        return NotImplementedError

# VAE loss

# VAE-GAN loss

# LSTM
