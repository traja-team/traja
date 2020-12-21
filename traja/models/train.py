from .ae import MultiModelAE
from .vae import MultiModelVAE
from .lstm import LSTM
from . import utils
from .losses import Criterion
from .optimizers import Optimizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class HybridTrainer(object):
    """
    Wrapper for training and testing the LSTM model

            :param model_type: Type of model should be "LSTM"
            :param optimizer_type: Type of optimizer to use for training.Should be from ['Adam', 'Adadelta', 'Adagrad',
                                                                                'AdamW', 'SparseAdam', 'RMSprop', '
                                                                                Rprop', 'LBFGS', 'ASGD', 'Adamax'] 
            :param device: Selected device; 'cuda' or 'cpu' 
            :param input_size: The number of expected features in the input x
            :param output_size: Output feature dimension
            :param lstm_hidden_size: The number of features in the hidden state h
            :param num_lstm_layers: Number of layers in the LSTM model
            :param reset_state: If True, will reset the hidden and cell state for each batch of data
            :param num_classes: Number of categories/labels
            :param latent_size: Latent space dimension
            :param dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
            :param num_classifier_layers: Number of layers in the classifier
            :param epochs: Number of epochs to train the network
            :param batch_size: Number of samples in a batch 
            :param num_future: Number of time steps to be predicted forward
            :param num_past: Number of past time steps otherwise, length of sequences in each batch of data.
            :param bidirectional:  If True, becomes a bidirectional LSTM
            :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
            :param loss_type: Type of reconstruction loss to apply, 'huber' or 'rmse'. Default:'huber'
            :param lr_factor:  Factor by which the learning rate will be reduced
            :param scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced.
                                For example, if patience = 2, then we will ignore the first 2 epochs with no
                                improvement, and will only decrease the LR after the 3rd epoch if the loss still
                                hasn’t improved then.
            
            """

    def __init__(self, model_type: str,
                 optimizer_type: str,
                 device: str,
                 input_size: int,
                 output_size: int,
                 lstm_hidden_size: int,
                 num_lstm_layers: int,
                 reset_state: bool,
                 latent_size: int,
                 dropout: float,
                 epochs: int,
                 batch_size: int,
                 num_future: int,
                 num_past: int,
                 num_classes: int = None,
                 classifier_hidden_size: int =None,
                 num_classifier_layers: int = None,
                 bidirectional: bool = False,
                 batch_first: bool = True,
                 loss_type: str = 'huber',
                 lr_factor: float = 0.1,
                 scheduler_patience: int = 10):

        white_keys = ['ae', 'vae']
        assert model_type in white_keys, "Valid models are {}".format(white_keys)

        self.model_type = model_type
        self.device = device
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.classifier_hidden_size = classifier_hidden_size
        self.num_classifier_layers = num_classifier_layers
        self.batch_first = batch_first
        self.reset_state = reset_state
        self.output_size = output_size
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.num_classifier_layers = num_classifier_layers
        self.num_future = num_future
        self.epochs = epochs
        self.batch_size = batch_size
        self.num_past = num_past
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience
        self.model_hyperparameters = {'input_size': self.input_size,
                                      'num_past': self.num_past,
                                      'batch_size': self.batch_size,
                                      'lstm_hidden_size': self.lstm_hidden_size,
                                      'num_lstm_layers': self.num_lstm_layers,
                                      'classifier_hidden_size': self.classifier_hidden_size,
                                      'num_classifier_layers': self.num_classifier_layers,
                                      'num_future': self.num_future,
                                      'latent_size': self.latent_size,
                                      'output_size': self.output_size,
                                      'num_classes': self.num_classes,
                                      'batch_first': self.batch_first,
                                      'reset_state': self.reset_state,
                                      'bidirectional': self.bidirectional,
                                      'dropout': self.dropout
                                      }

        # Instantiate model instance based on model_type
        if self.model_type == 'ae':
            self.model = MultiModelAE(**self.model_hyperparameters)

        if self.model_type == 'vae':
            self.model = MultiModelVAE(**self.model_hyperparameters)

        # Model optimizer and the learning rate scheduler based on user defined optimizer_type
        # and learning rate parameters

        classify = True if self.classifier_hidden_size is not None else False

        optimizer = Optimizer(self.model_type, self.model, self.optimizer_type, classify = classify)
        self.model_optimizers = optimizer.get_optimizers(lr=0.001)
        self.model_lrschedulers = optimizer.get_lrschedulers(factor=self.lr_factor, patience=self.scheduler_patience)

    def __str__(self):
        return "Training model type {}".format(self.model_type)

    def train(self, train_loader, test_loader, model_save_path=None):
        """
        This method implements the batch- wise training and testing protocol for both time series forecasting and
        classification of the timeseriesis_classification

        :param train_loader: Dataloader object of train dataset with batch data [data,target,category]
        :param test_loader: Dataloader object of test dataset with [data,target,category]
        :param model_save_path: Directory path to save the model
        :return: None
        """

        assert self.model_type == 'ae' or 'vae'
        assert model_save_path is not None, "Model path unknown"
        self.model.to(device)

        encoder_optimizer, latent_optimizer, decoder_optimizer, classifier_optimizer = self.model_optimizers.values()
        encoder_scheduler, latent_scheduler, decoder_scheduler, classifier_scheduler = self.model_lrschedulers.values()

        # Training mode: Switch from Generative to classifier training mode
        training_mode = 'forecasting'

        if self.classifier_hidden_size is not None:
            self.epochs *= 2

        # Training
        for epoch in range(self.epochs):  # First half for generative model and next for classifier
            test_loss_forecasting = 0
            test_loss_classification = 0
            if epoch > 0:  # Initial step is to test and set LR schduler
                # Training
                self.model.train()
                total_loss = 0
                for idx, (data, target, category) in enumerate(train_loader):
                    # Reset optimizer states
                    encoder_optimizer.zero_grad()
                    latent_optimizer.zero_grad()
                    decoder_optimizer.zero_grad()
                    classifier_optimizer.zero_grad()

                    data, target, category = data.float().to(device), target.float().to(device), category.to(device)

                    if training_mode == 'forecasting':
                        if self.model_type == 'ae':
                            decoder_out, latent_out = self.model(data, training=True, classify=False)
                            loss = Criterion().ae_criterion(decoder_out, target)

                        else:  # vae
                            decoder_out, latent_out, mu, logvar = self.model(data, training=True,
                                                                             classify=False)
                            loss = Criterion().vae_criterion(decoder_out, target, mu, logvar)

                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        latent_optimizer.step()

                    elif self.classifier_hidden_size \
                            and training_mode is not 'forecasting':  # training_mode == 'classification'
                        if self.model_type == 'vae':
                            classifier_out, latent_out, mu, logvar = self.model(data, training=True,
                                                                                classify=True)
                        else:
                            classifier_out = self.model(data, training=True,
                                                        classify=True)
                        loss = Criterion().classifier_criterion(classifier_out, category - 1)
                        loss.backward()
                        classifier_optimizer.step()
                    total_loss += loss

                print('Epoch {} | {} loss {}'.format(epoch, training_mode, total_loss / (idx + 1)))

            if epoch + 1 == self.epochs:
                training_mode = 'classification'

            # Testing
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    for idx, (data, target, category) in enumerate(list(test_loader)):
                        data, target, category = data.float().to(device), target.float().to(device), category.to(device)
                        # Time series forecasting test
                        if self.model_type == 'ae':
                            out, latent = self.model(data, training=False, classify=False)
                            test_loss_forecasting += Criterion().ae_criterion(out, target).item()
                        else:
                            decoder_out, latent_out, mu, logvar = self.model(data, training=False,
                                                                             classify=False)
                            test_loss_forecasting += Criterion().vae_criterion(decoder_out, target, mu, logvar)

                        # Classification test
                        if self.classifier_hidden_size is not None:
                            if self.model_type == 'ae':
                                classifier_out = self.model(data, training=False,
                                                            classify=True)
                            else:
                                classifier_out, latent_out, mu, logvar = self.model(data, training=False,
                                                                                    classify=True)

                            test_loss_classification += Criterion().classifier_criterion(classifier_out,
                                                                                         category - 1).item()

                test_loss_forecasting /= len(test_loader.dataset)
                print(f'====> Mean test set generator loss: {test_loss_forecasting:.4f}')
                if test_loss_classification != 0:
                    test_loss_classification /= len(test_loader.dataset)
                    print(f'====> Mean test set classifier loss: {test_loss_classification:.4f}')

            # Scheduler metric is test set loss
            if training_mode == 'forecasting':
                encoder_scheduler.step(test_loss_forecasting)
                decoder_scheduler.step(test_loss_forecasting)
                latent_scheduler.step(test_loss_forecasting)
            else:
                if self.classifier_hidden_size is not None:
                    classifier_scheduler.step(test_loss_classification)

        # Save the model at target path
        utils.save_model(self.model, PATH=model_save_path)


class LSTMTrainer:
    """
    Wrapper for training and testing the LSTM model

    :param model_type: Type of model should be "LSTM"
    :param optimizer_type: Type of optimizer to use for training.Should be from ['Adam', 'Adadelta', 'Adagrad',
                                                                                'AdamW', 'SparseAdam', 'RMSprop', '
                                                                                Rprop', 'LBFGS', 'ASGD', 'Adamax']
    :param device: Selected device; 'cuda' or 'cpu'
    :param epochs: Number of epochs to train the network
    :param input_size: The number of expected features in the input x
    :param batch_size: Number of samples in a batch
    :param hidden_size: The number of features in the hidden state h
    :param num_future: Number of time steps to be predicted forward
    :param num_layers: Number of layers in the LSTM model
    :param output_size: Output feature dimension
    :param lr_factor:  Factor by which the learning rate will be reduced
    :param scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced.
                                For example, if patience = 2, then we will ignore the first 2 epochs with no
                                improvement, and will only decrease the LR after the 3rd epoch if the loss still
                                hasn’t improved then.
    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    :param dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    :param reset_state: If True, will reset the hidden and cell state for each batch of data
    :param bidirectional:  If True, becomes a bidirectional LSTM

    """

    def __init__(self,
                 model_type: str,
                 optimizer_type: str,
                 device: str,
                 epochs: int,
                 input_size: int,
                 batch_size: int,
                 hidden_size: int,
                 num_future: int,
                 num_layers: int,
                 output_size: int,
                 lr_factor: float,
                 scheduler_patience: int,
                 batch_first: True,
                 dropout: float,
                 reset_state: bool,
                 bidirectional: bool):
        self.model_type = model_type
        self.optimizer_type = optimizer_type
        self.device = device
        self.epochs = epochs
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_future = num_future
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        self.model_hyperparameters = {'input_size': self.input_size,
                                      'batch_size': self.batch_size,
                                      'hidden_size': self.hidden_size,
                                      'num_future': self.num_future,
                                      'num_layers': self.num_layers,
                                      'output_size': self.output_size,
                                      'batch_first': self.batch_first,
                                      'reset_state': self.reset_state,
                                      'bidirectional': self.bidirectional,
                                      'dropout': self.dropout
                                      }

        self.model = LSTM(**self.model_hyperparameters)
        optimizer = Optimizer(self.model_type, self.model, self.optimizer_type)
        self.optimizer = optimizer.get_optimizers(lr=0.001)
        self.scheduler = optimizer.get_lrschedulers(factor=self.lr_factor, patience=self.scheduler_patience)

    def train(self, train_loader, test_loader, model_save_path):

        """ Implements the batch wise training and testing for time series forecasting
        :param train_loader: Dataloader object of train dataset with batch data [data,target,category]
        :param test_loader: Dataloader object of test dataset with [data,target,category]
        :param model_save_path: Directory path to save the model
        :return: None"""

        assert self.model_type == 'lstm'
        self.model.to(device)

        for epoch in range(self.epochs):
            if epoch > 0:
                self.model.train()
                total_loss = 0
                for idx, (data, target, _) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    data, target = data.float().to(device), target.float().to(device)
                    output = self.model(data)
                    loss = Criterion().lstm_criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss

                print('Epoch {} | loss {}'.format(epoch, total_loss / (idx + 1)))

            # Testing
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    test_loss_forecasting = 0
                    for idx, (data, target, _) in enumerate(list(test_loader)):
                        data, target = data.float().to(device), target.float().to(device)
                        out = self.model(data)
                        test_loss_forecasting += Criterion().lstm_criterion(out, target).item()

                test_loss_forecasting /= len(test_loader.dataset)
                print(f'====> Test set generator loss: {test_loss_forecasting:.4f}')

                # Scheduler metric is test set loss
            self.scheduler.step(test_loss_forecasting)

        # Save the model at target path
        utils.save_model(self.model, PATH=model_save_path)


class VAEGANTrainer:
    def __init__(self):
        pass

    def train(self):
        return NotImplementedError


class IRLTrainer:
    def __init__(self):
        pass

    def train(self):
        return NotImplementedError
