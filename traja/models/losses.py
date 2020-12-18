import torch


class Criterion:
    """Implements the loss functions of Autoencoders, Variational Autoencoders and LSTM models
    Huber loss is set as default for reconstruction loss, alternative is to use rmse,
    Cross entropy loss used for classification
    Variational loss used huber loss and unweighted KL Divergence loss"""

    def __init__(self):

        self.huber_loss = torch.nn.SmoothL1Loss(reduction='sum')
        self.crossentropy_loss = torch.nn.CrossEntropyLoss()

    def ae_criterion(self, predicted, target, loss_type='huber'):
        """ Implements the Autoencoder loss for time series forecasting
        :param predicted: Predicted time series by the model
        :param target: Target time series
        :param loss_type: Type of criterion; Defaults: 'huber'
        :return:
        """

        if loss_type == 'huber':
            loss = self.huber_loss(predicted, target)
            return loss
        else:  # Root MSE
            return torch.sqrt(torch.mean((predicted - target) ** 2))

    def vae_criterion(self, predicted, target, mu, logvar, loss_type='huber'):
        """ Time series generative model loss function
        :param predicted: Predicted time series by the model
        :param target: Target time series
        :param mu: Latent variable, Mean
        :param logvar: Latent variable, Log(Variance)
        :param loss_type: Type of criterion; Defaults: 'huber'
        :return: Reconstruction loss + KLD loss
        """
        if loss_type == 'huber':
            dist_x = self.huber_loss(predicted, target)
        else:
            dist_x = torch.sqrt(torch.mean((predicted - target) ** 2))
        KLD = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        return dist_x + KLD

    def classifier_criterion(self, predicted, target):
        """
        Classifier loss function
        :param predicted: Predicted label
        :param target: Target label
        :return: Cross entropy loss
        """

        loss = self.crossentropy_loss(predicted, target)
        return loss

    def lstm_criterion(self, predicted, target):

        loss = self.huber_loss(predicted, target)
        return loss

    def vaegan_criterion(self):
        return NotImplementedError
