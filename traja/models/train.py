from traja.models.generative_models.vae import MultiModelVAE
from traja.models.predictive_models.ae import MultiModelAE
from traja.models.predictive_models.lstm import LSTM
from traja.models.predictive_models.irl import MultiModelIRL
from traja.models.generative_models.vaegan import MultiModelVAEGAN
from . import utils
from .losses import Criterion
from .optimizers import Optimizer
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"


class HybridTrainer(object):
    """
    Wrapper for training and testing the LSTM model
        Args:
            model_type: Type of model should be "LSTM"
            optimizer_type: Type of optimizer to use for training.Should be from ['Adam', 'Adadelta', 'Adagrad',
                                                                                    'AdamW', 'SparseAdam', 'RMSprop', '
                                                                                    Rprop', 'LBFGS', 'ASGD', 'Adamax'] 
            device: Selected device; 'cuda' or 'cpu' 
            input_size: The number of expected features in the input x
            output_size: Output feature dimension
            lstm_hidden_size: The number of features in the hidden state h
            num_lstm_layers: Number of layers in the LSTM model
            reset_state: If True, will reset the hidden and cell state for each batch of data
            num_classes: Number of categories/labels
            latent_size: Latent space dimension
            dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                        with dropout probability equal to dropout
            num_classifier_layers: Number of layers in the classifier
            epochs: Number of epochs to train the network
            batch_size: Number of samples in a batch 
            num_future: Number of time steps to be predicted forward
            num_past: Number of past time steps otherwise, length of sequences in each batch of data.
            bidirectional:  If True, becomes a bidirectional LSTM
            batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
            loss_type: Type of reconstruction loss to apply, 'huber' or 'rmse'. Default:'huber'
            lr_factor:  Factor by which the learning rate will be reduced
            scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced.
                                    For example, if patience = 2, then we will ignore the first 2 epochs with no
                                    improvement, and will only decrease the LR after the 3rd epoch if the loss still
                                    hasn’t improved then.
            
        """

    def __init__(
        self,
        model_type: str,
        optimizer_type: str,
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
        classifier_hidden_size: int = None,
        num_classifier_layers: int = None,
        bidirectional: bool = False,
        batch_first: bool = True,
        loss_type: str = "huber",
        lr: float = 0.001,
        lr_factor: float = 0.1,
        scheduler_patience: int = 10,
    ):

        white_keys = ["ae", "vae"]

        assert model_type in white_keys, "Valid models are {}".format(white_keys)

        self.model_type = model_type
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
        self.lr = lr
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience

        self.model_hyperparameters = {
            "input_size": self.input_size,
            "num_past": self.num_past,
            "batch_size": self.batch_size,
            "lstm_hidden_size": self.lstm_hidden_size,
            "num_lstm_layers": self.num_lstm_layers,
            "classifier_hidden_size": self.classifier_hidden_size,
            "num_classifier_layers": self.num_classifier_layers,
            "num_future": self.num_future,
            "latent_size": self.latent_size,
            "output_size": self.output_size,
            "num_classes": self.num_classes,
            "batch_first": self.batch_first,
            "reset_state": self.reset_state,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }

        # Instantiate model instance based on model_type
        if self.model_type == "ae":
            self.model = MultiModelAE(**self.model_hyperparameters)

        if self.model_type == "vae":
            self.model = MultiModelVAE(**self.model_hyperparameters)

        # Classification task check
        self.classify = True if self.classifier_hidden_size is not None else False

        # Model optimizer and the learning rate scheduler
        optimizer = Optimizer(
            self.model_type, self.model, self.optimizer_type, classify=self.classify
        )

        self.model_optimizers = optimizer.get_optimizers(lr=self.lr)
        self.model_lrschedulers = optimizer.get_lrschedulers(
            factor=self.lr_factor, patience=self.scheduler_patience
        )

    def __str__(self):
        return "Training model type {}".format(self.model_type)

    def fit(self, train_loader, test_loader, model_save_path=None):
        """
        This method implements the batch- wise training and testing protocol for both time series forecasting and
        classification of the timeseriesis_classification


    train_loader: Dataloader object of train dataset with batch data [data,target,category]
    test_loader: Dataloader object of test dataset with [data,target,category]
    model_save_path: Directory path to save the model
        :return: None
        """

        assert self.model_type == "ae" or "vae"

        assert model_save_path is not None, "Model path unknown"

        self.model.to(device)

        (
            encoder_optimizer,
            latent_optimizer,
            decoder_optimizer,
            classifier_optimizer,
        ) = self.model_optimizers.values()
        (
            encoder_scheduler,
            latent_scheduler,
            decoder_scheduler,
            classifier_scheduler,
        ) = self.model_lrschedulers.values()

        # Training mode: Switch from Generative to classifier training mode
        training_mode = "forecasting"

        if self.classify:
            self.epochs *= 2  # Forecasting + Classification

        # Training

        for epoch in range(self.epochs):

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
                    if self.classify and classifier_optimizer is not None:
                        classifier_optimizer.zero_grad()

                    data, target, category = (
                        data.float().to(device),
                        target.float().to(device),
                        category.to(device),
                    )

                    if training_mode == "forecasting":
                        if self.model_type == "ae":
                            decoder_out, latent_out = self.model(
                                data, training=True, classify=False
                            )

                            loss = Criterion().ae_criterion(decoder_out, target)

                        else:  # vae
                            decoder_out, latent_out, mu, logvar = self.model(
                                data, training=True, classify=False
                            )
                            loss = Criterion().vae_criterion(
                                decoder_out, target, mu, logvar
                            )

                        loss.backward()
                        encoder_optimizer.step()
                        decoder_optimizer.step()
                        latent_optimizer.step()

                    elif self.classify and training_mode is not "forecasting":
                        if self.model_type == "vae":
                            classifier_out, latent_out, mu, logvar = self.model(
                                data, training=True, classify=True
                            )
                        else:  # "ae"
                            classifier_out = self.model(
                                data, training=True, classify=True
                            )
                        loss = Criterion().classifier_criterion(
                            classifier_out, category - 1
                        )

                        loss.backward()
                        classifier_optimizer.step()
                    total_loss += loss

                print(
                    "Epoch {} | {} loss {}".format(
                        epoch, training_mode, total_loss / (idx + 1)
                    )
                )

            if epoch == (self.epochs) // 2 and self.classify:
                training_mode = "classification"

            # Testing
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    for idx, (data, target, category) in enumerate(list(test_loader)):
                        data, target, category = (
                            data.float().to(device),
                            target.float().to(device),
                            category.to(device),
                        )
                        # Time series forecasting test
                        if self.model_type == "ae":
                            out, latent = self.model(
                                data, training=False, classify=False
                            )
                            test_loss_forecasting += (
                                Criterion().ae_criterion(out, target).item()
                            )

                        else:
                            decoder_out, latent_out, mu, logvar = self.model(
                                data, training=False, classify=False
                            )
                            test_loss_forecasting += Criterion().vae_criterion(
                                decoder_out, target, mu, logvar
                            )

                        # Classification test
                        if self.classify:
                            if self.model_type == "ae":
                                classifier_out = self.model(
                                    data, training=False, classify=True
                                )
                            else:
                                classifier_out, latent_out, mu, logvar = self.model(
                                    data, training=False, classify=True
                                )

                            test_loss_classification += (
                                Criterion()
                                .classifier_criterion(classifier_out, category - 1)
                                .item()
                            )

                test_loss_forecasting /= len(test_loader.dataset)
                print(
                    f"====> Mean test set generator loss: {test_loss_forecasting:.4f}"
                )
                if self.classify:
                    if test_loss_classification != 0:
                        test_loss_classification /= len(test_loader.dataset)
                        print(
                            f"====> Mean test set classifier loss: {test_loss_classification:.4f}"
                        )

            # Scheduler metric is test set loss
            if training_mode == "forecasting":
                encoder_scheduler.step(test_loss_forecasting)
                decoder_scheduler.step(test_loss_forecasting)
                latent_scheduler.step(test_loss_forecasting)
            else:
                if self.classify:
                    classifier_scheduler.step(test_loss_classification)

        # Save the model at target path
        utils.save_model(self.model, PATH=model_save_path)


class LSTMTrainer:
    """
    Wrapper for training and testing the LSTM model
    Parameters:
    -----------
        model_type: {'lstm'} 
            Type of model should be "LSTM"
        optimizer_type: {'Adam', 'Adadelta', 'Adagrad', 'AdamW', 'SparseAdam', 'RMSprop', 'Rprop', 'LBFGS', 'ASGD', 'Adamax'}
            Type of optimizer to use for training.
        device: {'cuda', 'cpu'}
            Target device to use for training the model
        epochs: int, default=100
            Number of epochs to train the network
        input_size: 
            The number of expected features in the input x
        batch_size: 
            Number of samples in a batch
        hidden_size: 
            The number of features in the hidden state h
        num_future: 
            Number of time steps to be predicted forward
        num_layers: 
            Number of layers in the LSTM model
        output_size: 
            Output feature dimension
        lr: 
            Optimizer learning rate
        lr_factor:  
            Factor by which the learning rate will be reduced
        scheduler_patience: 
            Number of epochs with no improvement after which learning rate will be reduced.
            For example, if patience = 2, then we will ignore the first 2 epochs with no
            improvement, and will only decrease the LR after the 3rd epoch if the loss still
            hasn’t improved then.
        batch_first: 
            If True, then the input and output tensors are provided as (batch, seq, feature)
        dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
        with dropout probability equal to dropout
        reset_state: 
            If True, will reset the hidden and cell state for each batch of data
        bidirectional:  
            If True, becomes a bidirectional LSTM

    """

    def __init__(
        self,
        model_type: str,
        optimizer_type: str,
        epochs: int,
        input_size: int,
        batch_size: int,
        hidden_size: int,
        num_future: int,
        num_layers: int,
        output_size: int,
        lr: float,
        lr_factor: float,
        scheduler_patience: int,
        batch_first: True,
        dropout: float,
        reset_state: bool,
        bidirectional: bool,
    ):
        self.model_type = model_type
        self.optimizer_type = optimizer_type
        self.epochs = epochs
        self.input_size = input_size
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_future = num_future
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.lr = lr
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        self.model_hyperparameters = {
            "input_size": self.input_size,
            "batch_size": self.batch_size,
            "hidden_size": self.hidden_size,
            "num_future": self.num_future,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "batch_first": self.batch_first,
            "reset_state": self.reset_state,
            "bidirectional": self.bidirectional,
            "dropout": self.dropout,
        }

        self.model = LSTM(**self.model_hyperparameters)
        optimizer = Optimizer(self.model_type, self.model, self.optimizer_type)
        self.optimizer = optimizer.get_optimizers(lr=0.001)
        self.scheduler = optimizer.get_lrschedulers(
            factor=self.lr_factor, patience=self.scheduler_patience
        )

    def fit(self, train_loader, test_loader, model_save_path):

        """ Implements the batch wise training and testing for time series forecasting. 
        Args:
            train_loader: Dataloader object of train dataset with batch data [data,target,category]
            test_loader: Dataloader object of test dataset with [data,target,category]
            model_save_path: Directory path to save the model
        Return: None"""

        assert self.model_type == "lstm"
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

                print("Epoch {} | loss {}".format(epoch, total_loss / (idx + 1)))

            # Testing
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    test_loss_forecasting = 0
                    for idx, (data, target, _) in enumerate(list(test_loader)):
                        data, target = (
                            data.float().to(device),
                            target.float().to(device),
                        )
                        out = self.model(data)
                        test_loss_forecasting += (
                            Criterion().lstm_criterion(out, target).item()
                        )

                test_loss_forecasting /= len(test_loader.dataset)
                print(f"====> Test set generator loss: {test_loss_forecasting:.4f}")

                # Scheduler metric is test set loss
            self.scheduler.step(test_loss_forecasting)

        # Save the model at target path
        utils.save_model(self.model, PATH=model_save_path)


class CustomTrainer:
    """
    Wrapper for training and testing user defined models
    Args:
        model: Custom/User-defined model
        optimizer_type: Type of optimizer to use for training.Should be from ['Adam', 'Adadelta', 'Adagrad',
                                                                                        'AdamW', 'SparseAdam', 'RMSprop', '
                                                                                        Rprop', 'LBFGS', 'ASGD', 'Adamax']
        device: Selected device; 'cuda' or 'cpu'
        epochs: Number of epochs to train the network
        lr:Optimizer learning rate
        lr_factor:  Factor by which the learning rate will be reduced
        scheduler_patience: Number of epochs with no improvement after which learning rate will be reduced.
                                        For example, if patience = 2, then we will ignore the first 2 epochs with no
                                        improvement, and will only decrease the LR after the 3rd epoch if the loss still
                                        hasn’t improved then.
    """

    def __init__(
        self,
        model: torch.nn.Module,
        optimizer_type: None,
        criterion: None,
        epochs: int,
        lr: float = 0.001,
        lr_factor: float = 0.001,
        scheduler_patience: int = 10,
    ):
        self.model = model
        self.optimizer_type = optimizer_type
        self.criterion = criterion
        self.epochs = epochs
        self.lr = lr
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience
        self.model_type == "custom"
        optimizer = Optimizer(self.model_type, self.model, self.optimizer_type)
        self.optimizer = optimizer.get_optimizers(lr=self.lr)
        self.scheduler = optimizer.get_lrschedulers(
            factor=self.lr_factor, patience=self.scheduler_patience
        )

    def fit(self, train_loader, test_loader, model_save_path):

        """ Implements the batch wise training and testing for time series forecasting
            Save train, test and validation performance in forecasting/classification tasks as a performance.csv
        Args:
            train_loader: Dataloader object of train dataset with batch data [data,target,category]
            test_loader: Dataloader object of test dataset with [data,target,category]
            model_save_path: Directory path to save the model
        Return: None
        """

        self.model.to(device)

        for epoch in range(self.epochs):
            if epoch > 0:
                self.model.train()
                total_loss = 0
                for idx, (data, target, _) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    data, target = data.float().to(device), target.float().to(device)
                    output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss

                print("Epoch {} | loss {}".format(epoch, total_loss / (idx + 1)))

            # Testing
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    test_loss_forecasting = 0
                    for idx, (data, target, _) in enumerate(list(test_loader)):
                        data, target = (
                            data.float().to(device),
                            target.float().to(device),
                        )
                        out = self.model(data)
                        test_loss_forecasting += self.criterion(out, target).item()

                test_loss_forecasting /= len(test_loader.dataset)
                print(f"====> Test set generator loss: {test_loss_forecasting:.4f}")

                # Scheduler metric is test set loss
            self.scheduler.step(test_loss_forecasting)

        # Save the model at target path
        utils.save_model(self.model, PATH=model_save_path)


class Trainer:

    """Wraps all the Trainers. Instantiate and return the Trainer of model type 
    
    Usage:
    ======
    
    trainer = Trainer(model_type='vae',  # "ae" or "vae"
                optimizer_type='Adam',   # ['Adam', 'Adadelta', 'Adagrad', 'AdamW', 'SparseAdam', 'RMSprop', 'Rprop','LBFGS', 'ASGD', 'Adamax']
                device='cuda', # 'cpu', 'cuda'
                input_size=2,  
                output_size=2, 
                lstm_hidden_size=32, 
                num_lstm_layers=2,
                reset_state=True, 
                latent_size=10, 
                dropout=0.1, 
                num_classes=9,  # Uncomment to create and train classifier network
                num_classifier_layers=4,
                classifier_hidden_size= 32, 
                epochs=10, 
                batch_size=batch_size, 
                num_future=num_future, 
                num_past=num_past,
                bidirectional=False, 
                batch_first=True,
                loss_type='huber') # 'rmse' or 'huber'
                  
    trainer.train(train_loader, test_loader, model_save_path)    
    """

    def __init__(self):
        pass

    # Check model type and instantiate corresponding trainer class:
    def __new__(cls):
        # Generative model trainer(model_type)

        # Predictive model trainer(model_type)

        # Custom trainer(classify=False)

        # Return the instance of the trainer
        return NotImplementedError


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

