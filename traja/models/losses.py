import torch


class Criterion(object):

    def __init__(self, model_type):
        self.model_type = model_type
        self.huber_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.crossentropy_loss = torch.nn.CrossEntropyLoss()

    def ae_criterion(self, recon_x, x, loss_type='huber'):
        """[summary]

        Args:
            recon_x ([type]): [description]
            x ([type]): [description]
            loss_type(str): Type of Loss; huber or RMSE

        Returns:
            [type]: [description]
        """
        if loss_type == 'huber':
            dist_x = self.huber_loss(recon_x, x)
            return dist_x
        else:  # Root MSE
            return torch.sqrt(torch.mean((recon_x - x) ** 2))

    def vae_criterion(self, recon_x, x, mu, logvar, loss_type='huber'):
        """Time series generative model loss function

        Args:
            recon_x ([type]): [description]
            x ([type]): [description]
            mu ([type]): [description]
            logvar ([type]): [description]

        Returns:
            [type]: [description]
        """
        if loss_type == 'huber':
            dist_x = self.huber_loss(recon_x, x)
        else:
            dist_x = torch.sqrt(torch.mean((recon_x - x) ** 2))
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
