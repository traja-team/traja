import matplotlib.pyplot as plt
import torch

from . import utils
from . import visualizer
from .losses import Criterion
from .optimizers import Optimizer

device = 'cuda' if torch.cuda.is_available() else 'cpu'


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

    valid_models = ['ae', 'vae', 'lstm']

    def __init__(
            self,
            model: torch.nn.Module,
            optimizer_type: str,
            loss_type: str = "huber",
            lr: float = 0.001,
            lr_factor: float = 0.1,
            scheduler_patience: int = 10,
    ):

        assert (
                model.model_type in HybridTrainer.valid_models
        ), "Model type is {model_type}, valid models are {}".format(
            HybridTrainer.valid_models
        )

        self.model_type = model.model_type
        self.loss_type = loss_type
        self.optimizer_type = optimizer_type
        self.lr = lr
        self.lr_factor = lr_factor
        self.scheduler_patience = scheduler_patience

        if model.model_type == 'lstm':
            self.model_hyperparameters = {
                "input_size": model.input_size,
                "batch_size": model.batch_size,
                "hidden_size": model.hidden_size,
                "num_future": model.num_future,
                "num_layers": model.num_layers,
                "output_size": model.output_size,
                "batch_first": model.batch_first,
                "reset_state": model.reset_state,
                "bidirectional": model.bidirectional,
                "dropout": model.dropout,
            }
        else:
            self.model_hyperparameters = {
                "input_size": model.input_size,
                "num_past": model.num_past,
                "batch_size": model.batch_size,
                "lstm_hidden_size": model.lstm_hidden_size,
                "num_lstm_layers": model.num_lstm_layers,
                "classifier_hidden_size": model.classifier_hidden_size,
                "num_classifier_layers": model.num_classifier_layers,
                "num_future": model.num_future,
                "latent_size": model.latent_size,
                "output_size": model.output_size,
                "num_classes": model.num_classes,
                "batch_first": model.batch_first,
                "reset_state": model.reset_state,
                "bidirectional": model.bidirectional,
                "dropout": model.dropout,
            }

        self.model = model

        # Classification, regression task checks
        self.classify = True if model.model_type != 'lstm' and model.classifier_hidden_size is not None else False
        self.regress = True if model.model_type != 'lstm' and model.regressor_hidden_size is not None else False

        # Model optimizer and the learning rate scheduler
        optimizer = Optimizer(
            self.model_type, self.model, self.optimizer_type, classify=self.classify
        )

        self.forecasting_optimizers, self.classification_optimizers, self.regression_optimizers = optimizer.get_optimizers(
            lr=self.lr)
        self.forecasting_schedulers, self.classification_schedulers, self.regression_schedulers = optimizer.get_lrschedulers(
            factor=self.lr_factor, patience=self.scheduler_patience
        )

    def __str__(self):
        return f"Training model type {self.model_type}"

    def fit(self, train_loader, test_loader, model_save_path=None, training_mode='forecasting', epochs=50):
        """
        This method implements the batch- wise training and testing protocol for both time series forecasting and
        classification of the timeseriesis_classification
        
        Parameters:
        -----------
        dataloaders: Dictionary containing train and test dataloaders
                train_loader: Dataloader object of train dataset with batch data [data,target,category]
                test_loader: Dataloader object of test dataset with [data,target,category]
        model_save_path: Directory path to save the model
        training_mode: Type of training ('forecasting', 'classification')
        epochs: Number of epochs to train
        """

        assert model_save_path is not None, f"Model path {model_save_path} unknown"
        assert training_mode in ['forecasting', 'classification',
                                 'regression'], f'Training mode {training_mode} unknown'

        self.model.to(device)

        train_loader, test_loader = dataloaders.values()
        # Training
        for epoch in range(epochs):
            test_loss_forecasting = 0
            test_loss_classification = 0
            test_loss_regression = 0
            if epoch > 0:  # Initial step is to test and set LR schduler
                # Training
                self.model.train()
                total_loss = 0
                for idx, (data, target, category, parameters) in enumerate(train_loader):
                    # Reset optimizer states
                    for optimizer in self.forecasting_optimizers:
                        optimizer.zero_grad()
                    if self.classify:
                        for optimizer in self.classification_optimizers:
                            optimizer.zero_grad()
                    if self.regress:
                        for optimizer in self.regression_optimizers:
                            optimizer.zero_grad()

                    data, target, category, parameters = (
                        data.float().to(device),
                        target.float().to(device),
                        category.to(device),
                        parameters.to(device)
                    )

                    if training_mode == "forecasting":
                        if self.model_type == "ae" or self.model_type == 'lstm':
                            decoder_out = self.model(
                                data, training=True, classify=False, latent=False
                            )
                            loss = Criterion().ae_criterion(decoder_out, target)
                        else:  # vae
                            decoder_out, latent_out, mu, logvar = self.model(data, training=True,
                                                                             is_classification=False)
                            loss = Criterion().vae_criterion(decoder_out, target, mu, logvar)

                        loss.backward()
                        for optimizer in self.forecasting_optimizers:
                            optimizer.step()

                    elif self.classify and training_mode == "classification":
                        if self.model_type == "vae":
                            classifier_out, latent_out, mu, logvar = self.model(
                                data, training=True, classify=True
                            )
                        else:  # 'ae', 'lstm'
                            classifier_out = self.model(
                                data, training=True, classify=True
                            )
                        loss = Criterion().classifier_criterion(
                            classifier_out, (category - 1).long()
                        )

                        loss.backward()
                        for optimizer in self.classification_optimizers:
                            optimizer.step()

                    elif self.regress and training_mode == 'regression':
                        regressor_out = self.model(data, training=True, regress=True, latent=False)
                        loss = Criterion().regressor_criterion(
                            regressor_out, parameters
                        )

                        loss.backward()
                        for optimizer in self.regression_optimizers:
                            optimizer.step()

                    total_loss += loss

                print('Epoch {} | {} loss {}'.format(epoch, training_mode, total_loss / (idx + 1)))

            # Testing
            if epoch % 10 == 0:
                with torch.no_grad():
                    if self.classify:
                        total = 0.0
                        correct = 0.0
                    self.model.eval()
                    for idx, (data, target, category, parameters) in enumerate(test_loader):
                        data, target, category, parameters = (
                            data.float().to(device),
                            target.float().to(device),
                            category.to(device),
                            parameters.to(device)
                        )
                        # Time series forecasting test
                        if self.model_type == 'ae' or self.model_type == 'lstm':
                            out = self.model(
                                data, training=False, classify=False, latent=False
                            )
                            test_loss_forecasting += (
                                Criterion().ae_criterion(out, target).item()
                            )

                        else:
                            decoder_out, latent_out, mu, logvar = self.model(data, training=False,
                                                                             is_classification=False)
                            test_loss_forecasting += Criterion().vae_criterion(decoder_out, target, mu, logvar)
                        # Classification test
                        if self.classify:
                            category = category.long()
                            if self.model_type == 'ae' or self.model_type == 'lstm':
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

                            # Compute number of correct samples
                            total += category.size(0)
                            _, predicted = torch.max(classifier_out.data, 1)
                            correct += (predicted == (category - 1)).sum().item()

                        if self.regress:
                            regressor_out = self.model(data, training=True, regress=True, latent=False)
                            test_loss_regression += Criterion().regressor_criterion(
                                regressor_out, parameters
                            )

                test_loss_forecasting /= len(test_loader.dataset)
                print(
                    f"====> Mean test set generator loss: {test_loss_forecasting:.4f}"
                )
                if self.classify:
                    accuracy = correct / total
                    if test_loss_classification != 0:
                        test_loss_classification /= len(test_loader.dataset)
                        print(
                            f"====> Mean test set classifier loss: {test_loss_classification:.4f}; accuracy: {accuracy:.2f}"
                        )

                if self.regress:
                    print(f'====> Mean test set regressor loss: {test_loss_regression:.4f}')

            # Scheduler metric is test set loss
            if training_mode == "forecasting":
                for scheduler in self.forecasting_schedulers.values():
                    scheduler.step(test_loss_forecasting)
            elif training_mode == 'classification':
                for scheduler in self.classification_schedulers.values():
                    scheduler.step(test_loss_classification)
            elif training_mode == 'regression':
                for scheduler in self.regression_schedulers.values():
                    scheduler.step(test_loss_regression)

        # Save the model at target path
        utils.save(self.model, self.model_hyperparameters, PATH=model_save_path)


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

        self.model_type = "custom"
        optimizer = Optimizer(self.model_type, self.model, self.optimizer_type)
        self.optimizer = optimizer.get_optimizers(lr=self.lr)
        self.scheduler = optimizer.get_lrschedulers(
            factor=self.lr_factor, patience=self.scheduler_patience
        )
        self.viz = True

    def fit(self, dataloaders, model_save_path):

        """ Implements the batch wise training and testing for time series forecasting
            Save train, test and validation performance in forecasting/classification tasks as a performance.csv
        Args:
            dataloaders: Dictionary containing train and test dataloaders
                train_loader: Dataloader object of train dataset with batch data [data,target,category]
                test_loader: Dataloader object of test dataset with [data,target,category]
            model_save_path: Directory path to save the model
        Return:
            None
        """

        # Init Visualization
        if self.viz == "True":
            self.fig = plt.figure(num="Latent Network Activity")
            self.plt_close = False
            self.directednetwork = visualizer.DirectedNetwork()

            self.fig2 = plt.figure(num="Local Linear Embedded Trajectory")
            self.plt2_close = False
            self.lle = visualizer.LocalLinearEmbedding()

            self.fig3 = plt.figure(num="Spectral Embedded Latent")
            self.plt3_close = False
            self.spectral_clustering = visualizer.SpectralEmbedding()

            plt.pause(0.00001)

        # Training loop
        self.model.to(device)
        train_loader, test_loader = dataloaders.values()
        for epoch in range(self.epochs):
            if epoch > 0:
                self.model.train()
                total_loss = 0
                for idx, (data, target, _) in enumerate(train_loader):
                    self.optimizer.zero_grad()
                    data, target = data.float().to(device), target.float().to(device)
                    activations, output = self.model(data)
                    loss = self.criterion(output, target)
                    loss.backward()
                    self.optimizer.step()
                    total_loss += loss

                    # TODO: Wrapper for visualization at visualizer.
                    if self.viz == "True":
                        # Visualize the network during training
                        if not self.plt_close:
                            # Get the hidden to hidden weights in the network and plot the connections
                            # TODO: Visualization of multiple layer activations in a window
                            hidden_weights = dict(
                                self.model.lstm.w_hhl0.clone().detach().numpy()
                            )

                            # Hidden activativations
                            hidden_activ = list(activations.clone().detach().numpy()[0])

                            try:
                                plt_close = self.directednetwork.show(
                                    hidden_activ, hidden_weights, self.fig4
                                )
                            except Exception:
                                plt_close = True
                                pass

                            plt_close = self.directednetwork.show(
                                hidden_activ, hidden_weights, self.fig
                            )

                        # # Visualize the network during training
                        if not self.plt2_close:
                            # Get the principle components
                            pc = self.lle.local_linear_embedding(
                                X=activations.clone().detach().numpy(),
                                d=3,
                                k=20,
                                alpha=0.1,
                            )
                            plt2_close = self.lle.show(pc, self.fig2)

                        # Visualize the graph embedding using spectral clusters
                        if not self.plt3_close:
                            # Get the principle components
                            embeddings = self.spectral_clustering.spectral_embedding(
                                X=activations.clone().detach().numpy(), rad=0.8
                            )
                            plt3_close = self.spectral_clustering.show(
                                activations.clone().detach().numpy(),
                                embeddings,
                                self.fig3,
                            )

                print("Epoch {} | loss {}".format(epoch, total_loss / (idx + 1)))

            # Testing loop
            if epoch % 10 == 0:
                with torch.no_grad():
                    self.model.eval()
                    self.test_loss_forecasting = 0
                    for idx, (data, target, _) in enumerate(list(test_loader)):
                        data, target = (
                            data.float().to(device),
                            target.float().to(device),
                        )
                        activations, out = self.model(data)
                        self.test_loss_forecasting += self.criterion(out, target).item()

                self.test_loss_forecasting /= len(test_loader.dataset)
                print(
                    f"====> Test set generator loss: {self.test_loss_forecasting:.4f}"
                )

            # Scheduler metric is test set loss
            self.scheduler.step(self.test_loss_forecasting)

        # Save the model at target path
        utils.save(self.model, hyperparameters=None, PATH=model_save_path)


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


# TODO
class Trainer:
    """Wraps all above Trainers. Instantiate and return the Trainer of model type """

    def __init__(self, *model_hyperparameters, **kwargs):
        self.model_type = model_hyperparameters["model_type"]
        self.TrainerType = None

    @property
    def TrainerType(self):
        return self.__TrainerType

    @TrainerType.setter
    def TrainerType(self, model_type):
        """[summary]

        Args:
            model_type ([type]): [description]
        """
        if model_type in ["ae", "vae"]:
            self.__TrainerType = HybridTrainer
        elif model_type in ["lstm"]:
            self.__TrainerType = LSTMTrainer
        else:
            self.__TrainerType = CustomTrainer

    # Check model type, instantiate and set corresponding trainer as traja trainer:
    def __new__(cls):
        # Generative model trainer(model_type)

        # Predictive model trainer(model_type)

        # Custom trainer(classify=False)

        # Return the instance of the trainer
        return NotImplementedError
