from .ae import MultiModelAE
from .vae import MultiModelVAE
from .lstm import LSTM

from . import utils
from .losses import Criterion
from .optimizers import Optimizer
import torch

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LatentModelTrainer(object):

    def __init__(self, model_type: str,
                 optimizer_type: str,
                 device: str,
                 input_size: int,
                 output_size: int,
                 lstm_hidden_size: int,
                 lstm_num_layers: int,
                 reset_state: bool,
                 num_classes: int,
                 latent_size: int,
                 dropout: float,
                 num_layers: int,
                 epochs: int,
                 batch_size: int,
                 num_future: int,
                 sequence_length: int,
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
        self.lstm_num_layers = lstm_num_layers
        self.hidden_size = lstm_hidden_size  # For classifiers too
        self.batch_first = batch_first
        self.reset_state = reset_state
        self.output_size = output_size
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.num_layers = num_layers
        self.num_future = num_future
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience
        self.model_hyperparameters = {'input_size': self.input_size,
                                      'sequence_length': self.sequence_length,
                                      'batch_size': self.batch_size,
                                      'hidden_size': self.lstm_hidden_size,
                                      'num_future': self.num_future,
                                      'num_layers': self.lstm_num_layers,
                                      'latent_size': self.latent_size,
                                      'output_size': self.output_size,
                                      'num_classes': self.num_classes,
                                      'batch_first': self.batch_first,
                                      'reset_state': self.reset_state,
                                      'bidirectional': self.bidirectional,
                                      'dropout': self.dropout
                                      }

        if self.model_type == 'ae':
            self.model = MultiModelAE(**self.model_hyperparameters)

        if self.model_type == 'vae':
            self.model = MultiModelVAE(**self.model_hyperparameters)

        optimizer = Optimizer(self.model_type, self.model, self.optimizer_type)

        self.model_optimizers = optimizer.get_optimizers(lr=0.001)
        self.model_lrschedulers = optimizer.get_lrschedulers(factor=self.lr_factor, patience=self.scheduler_patience)

    def __str__(self):
        return "Training model type {}".format(self.model_type)

    def train(self, train_loader, test_loader, model_save_path):

        assert self.model_type == 'ae' or 'vae'
        self.model.to(device)

        encoder_optimizer, latent_optimizer, decoder_optimizer, classifier_optimizer = self.model_optimizers.values()
        encoder_scheduler, latent_scheduler, decoder_scheduler, classifier_scheduler = self.model_lrschedulers.values()

        # Training mode: Switch from Generative to classifier training mode
        training_mode = 'forecasting'

        # Training
        for epoch in range(self.epochs * 2):  # First half for generative model and next for classifier
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
                            decoder_out, latent_out = self.model(data, training=True, is_classification=False)
                            loss = Criterion.ae_criterion(decoder_out, target)

                        else:  # vae
                            decoder_out, latent_out, mu, logvar = self.model(data, training=True,
                                                                             is_classification=False)
                            loss = Criterion.vae_criterion(decoder_out, target, mu, logvar)

                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        latent_optimizer.step()

                    else:  # training_mode == 'classification'
                        if self.model_type == 'vae':
                            classifier_out, latent_out, mu, logvar = self.model(data, training=True, is_classification=True)
                        else:
                            classifier_out = self.model(data, training=True,
                                                        is_classification=True)
                        loss = Criterion.classifier_criterion(classifier_out, category - 1)
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
                            out, latent = self.model(data, training=False, is_classification=False)
                            test_loss_forecasting += Criterion().ae_criterion(out, target).item()
                        else:
                            decoder_out, latent_out, mu, logvar = self.model(data, training=False,
                                                                             is_classification=False)
                            test_loss_forecasting += Criterion().vae_criterion(decoder_out, target, mu, logvar)
                        # Classification test
                        if self.model_type == 'ae':
                            classifier_out = self.model(data, training=False,
                                                        is_classification=True)
                        else:
                            classifier_out, latent_out, mu, logvar = self.model(data, training=False,
                                                                                is_classification=True)

                        test_loss_classification += Criterion().classifier_criterion(classifier_out,
                                                                                     category - 1).item()

                test_loss_forecasting /= len(test_loader.dataset)
                print(f'====> Mean test set generator loss: {test_loss_forecasting:.4f}')
                test_loss_classification /= len(test_loader.dataset)
                print(f'====> Mean test set classifier loss: {test_loss_classification:.4f}')

            # Scheduler metric is test set loss
            if training_mode == 'forecasting':
                encoder_scheduler.step(test_loss_forecasting)
                decoder_scheduler.step(test_loss_forecasting)
                latent_scheduler.step(test_loss_forecasting)
            else:
                classifier_scheduler.step(test_loss_classification)

        # Save the model at target path
        utils.save_model(self.model, PATH=model_save_path)


class LSTMTrainer:

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
        self.lr_factor = lr_factor,
        self.scheduler_patience = scheduler_patience,
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
        self.optimizer = optimizer.get_optimizers(lr=0.001).values()
        self.scheduler = optimizer.get_lrschedulers(factor=self.lr_factor, patience=self.scheduler_patience).values()

    def train(self, train_loader, test_loader, model_save_path):

        assert self.model_type == 'lstm'
        self.model.to(device)

        for epoch in range(self.epochs):  # First half for generative model and next for classifier
            if epoch > 0:  # Initial step is to test and set LR schduler
                # Training
                self.model.train()
                total_loss = 0
                for idx, (data, target, _) in enumerate(train_loader):
                    # Reset optimizer states
                    self.optimizer.zero_grad()
                    data, target = data.float().to(device), target.float().to(device)

                    output = self.model(data)
                    loss = Criterion.lstm_criterion(output, target)
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
                        test_loss_forecasting += Criterion.lstm_criterion(out, target).item()

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
