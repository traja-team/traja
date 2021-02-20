import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

class Criterion:
    """Implements the loss functions of Autoencoders, Variational Autoencoders and LSTM models
    Huber loss is set as default for reconstruction loss, alternative is to use rmse,
    Cross entropy loss used for classification
    Variational loss used huber loss and unweighted KL Divergence loss"""

    def __init__(self):

        self.huber_loss = torch.nn.SmoothL1Loss(reduction="sum")
        self.manhattan_loss = torch.nn.L1Loss(reduction="sum")
        self.mse_loss = torch.nn.MSELoss()
        self.crossentropy_loss = torch.nn.CrossEntropyLoss()

    def forecasting_criterion(
        self, predicted, target, mu=None, logvar=None, loss_type="huber"
    ):
        """Time series forecasting model loss function
        Provides loss functions huber, manhattan, mse. Adds KL divergence if mu and logvar specified.
        and ae loss functions (huber_ae, manhattan_ae, mse_ae).
        :param predicted: Predicted time series by the model
        :param target: Target time series
        :param mu: Latent variable, Mean
        :param logvar: Latent variable, Log(Variance)
        :param loss_type: Type of criterion (huber, manhattan, mse, huber_ae, manhattan_ae, mse_ae); Defaults: 'huber'
        :return: Reconstruction loss + KLD loss (if not ae)
        """

        if mu is not None and logvar is not None:
            kld = -0.5 * torch.sum(1 + logvar - mu ** 2 - logvar.exp())
        else:
            kld = 0

        if loss_type == "huber":
            loss = self.huber_loss(predicted, target) + kld
        elif loss_type == "manhattan":
            loss = self.manhattan_loss(predicted, target) + kld
        elif loss_type == "mse":
            loss = self.mse_loss(predicted, target) + kld
        else:
            raise Exception("Loss type '{}' is unknown!".format(loss_type))
        return loss

    def classifier_criterion(self, predicted, target):
        """
        Classifier loss function
        :param predicted: Predicted label
        :param target: Target label
        :return: Cross entropy loss
        """

        predicted = predicted.to(device)
        target = target.to(device)
        loss = self.crossentropy_loss(predicted, target.view(-1))
        return loss

    def regressor_criterion(self, predicted, target):
        """
        Regressor loss function
        :param predicted: Predicted parameter value
        :param target: Target parameter value
        :return: MSE loss
        """

        loss = self.mse_loss(predicted, target)
        return loss
