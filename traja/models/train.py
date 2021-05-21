import torch

from . import utils
from .losses import Criterion
from .optimizers import Optimizer

device = "cuda" if torch.cuda.is_available() else "cpu"


class HybridTrainer(object):
    """
    Wrapper for training and testing the LSTM model
    Args:
        optimizer_type: Type of optimizer to use for training.Should be from ['Adam', 'Adadelta', 'Adagrad',
                                                                                'AdamW', 'SparseAdam', 'RMSprop', '
                                                                                Rprop', 'LBFGS', 'ASGD', 'Adamax']
        device: Selected device; 'cuda' or 'cpu'
        input_size: The number of expected features in the input x
        output_size: Output feature dimension
        lstm_hidden_size: The number of features in the hidden state h
        num_lstm_layers: Number of layers in the LSTM model
        reset_state: If True, will reset the hidden and cell state for each batch of data
        output_size: Number of sequence_ids/labels
        latent_size: Latent space dimension
        dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
        num_layers: Number of layers in the classifier
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

    valid_models = ["ae", "vae", "lstm"]

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

        if model.model_type == "lstm":
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
        self.classify = (
            True
            if model.model_type != "lstm" and model.classifier_hidden_size is not None
            else False
        )
        self.regress = (
            True
            if model.model_type != "lstm" and model.regressor_hidden_size is not None
            else False
        )

        # Model optimizer and the learning rate scheduler
        optimizer = Optimizer(
            self.model_type, self.model, self.optimizer_type, classify=self.classify
        )

        (
            self.forecasting_optimizers,
            self.classification_optimizers,
            self.regression_optimizers,
        ) = optimizer.get_optimizers(lr=self.lr)
        (
            self.forecasting_schedulers,
            self.classification_schedulers,
            self.regression_schedulers,
        ) = optimizer.get_lrschedulers(
            factor=self.lr_factor, patience=self.scheduler_patience
        )

    def __str__(self):
        return f"Training model type {self.model_type}"

    def fit(
            self, dataloaders, model_save_path=None, training_mode="forecasting", epochs=50, test_every=10, validate_every=None
    ):
        """
        This method implements the batch- wise training and testing protocol for both time series forecasting and
        classification of the timeseriesis_classification

        Parameters:
        -----------
        dataloaders: Dictionary containing train and test dataloaders
                train_loader: Dataloader object of train dataset with batch data [data,target,ids]
                test_loader: Dataloader object of test dataset with [data,target,ids]
        model_save_path: Directory path to save the model
        training_mode: Type of training ('forecasting', 'classification')
        epochs: Number of epochs to train
        test_every: Run evaluation on the test set in multiple of 'test_every' epochs.
        validate_every: Run evaluation on the validation set in multiple of 'validate_every' epochs,
                if evaluation on test-set is not required for current epoch.
        """

        assert model_save_path is not None, f"Model path {model_save_path} unknown"
        assert training_mode in [
            "forecasting",
            "classification",
            "regression",
        ], f"Training mode {training_mode} unknown"

        self.model.to(device)

        train_loader = dataloaders["train_loader"]
        test_loader = dataloaders["test_loader"]
        if 'validation_loader' in dataloaders:
            validation_loader = dataloaders['validation_loader']
        else:
            validate_every = None

        # Training
        for epoch in range(epochs + 1):
            eval_loss_forecasting = 0
            eval_loss_classification = 0
            eval_loss_regression = 0
            if epoch > 0:  # Initial step is to test and set LR schduler
                # Training
                self.model.train()
                total_loss = 0
                for idx, (data, target, ids, parameters, classes) in enumerate(
                    train_loader
                ):
                    # Reset optimizer states
                    for optimizer in self.forecasting_optimizers:
                        optimizer.zero_grad()
                    if self.classify:
                        for optimizer in self.classification_optimizers:
                            optimizer.zero_grad()
                    if self.regress:
                        for optimizer in self.regression_optimizers:
                            optimizer.zero_grad()

                    if type(ids) == list:
                        ids = ids[0]
                    data, target, ids, parameters = (
                        data.float().to(device),
                        target.float().to(device),
                        ids.to(device),
                        parameters.float().to(device),
                    )

                    if training_mode == "forecasting":
                        if self.model_type == "ae" or self.model_type == "lstm":
                            decoder_out = self.model(
                                data, training=True, classify=False, latent=False
                            )
                            loss = Criterion().forecasting_criterion(
                                decoder_out, target, loss_type=self.loss_type
                            )
                        else:  # vae
                            decoder_out, latent_out, mu, logvar = self.model(
                                data, training=True, classify=False
                            )
                            loss = Criterion().forecasting_criterion(
                                decoder_out,
                                target,
                                mu=mu,
                                logvar=logvar,
                                loss_type=self.loss_type,
                            )

                        loss.backward()
                        for optimizer in self.forecasting_optimizers:
                            optimizer.step()

                    elif training_mode == "classification":
                        classifier_out = self.model(
                            data, training=True, classify=True, latent=False
                        )
                        loss = Criterion().classifier_criterion(classifier_out, classes)

                        loss.backward()
                        for optimizer in self.classification_optimizers:
                            optimizer.step()

                    elif training_mode == "regression":
                        regressor_out = self.model(
                            data, training=True, regress=True, latent=False
                        )
                        loss = Criterion().regressor_criterion(
                            regressor_out, parameters
                        )

                        loss.backward()
                        for optimizer in self.regression_optimizers:
                            optimizer.step()

                    total_loss += loss

                print(
                    "Epoch {} | {} loss {}".format(
                        epoch, training_mode, total_loss / len(train_loader.dataset)
                    )
                )

            # Testing & Validation
            evaluate_for_this_epoch = False
            data_loader_to_evaluate = test_loader
            current_set = "Test"
            if validate_every is not None:
                if epoch % validate_every == validate_every - 1 and epoch != 0:
                    data_loader_to_evaluate = validation_loader
                    evaluate_for_this_epoch = True
                    current_set = "Validation"
            if test_every is not None:
                if epoch % test_every == test_every - 1 and epoch != 0:
                    data_loader_to_evaluate = test_loader
                    evaluate_for_this_epoch = True
                    current_set = "Test"

            if evaluate_for_this_epoch:
                with torch.no_grad():
                    if self.classify:
                        total = 0.0
                        correct = 0.0
                    self.model.eval()
                    for idx, (data, target, ids, parameters, classes) in enumerate(
                            data_loader_to_evaluate
                    ):
                        if type(ids) == list:
                            ids = ids[0]
                        data, target, ids, parameters = (
                            data.float().to(device),
                            target.float().to(device),
                            ids.to(device),
                            parameters.float().to(device),
                        )
                        # Time series forecasting test
                        if self.model_type == "ae" or self.model_type == "lstm":
                            out = self.model(
                                data, training=False, classify=False, latent=False
                            )
                            eval_loss_forecasting += (
                                Criterion()
                                .forecasting_criterion(
                                    out, target, loss_type=self.loss_type
                                )
                                .item()
                            )

                        else:
                            decoder_out, latent_out, mu, logvar = self.model(
                                data, training=False, classify=False, latent=True
                            )
                            eval_loss_forecasting += Criterion().forecasting_criterion(
                                decoder_out,
                                target,
                                mu=mu,
                                logvar=logvar,
                                loss_type=self.loss_type,
                            )

                        # Classification test
                        if self.classify:
                            ids = ids.long()
                            classifier_out = self.model(
                                data, training=False, classify=True, latent=False
                            )

                            eval_loss_classification += (
                                Criterion()
                                .classifier_criterion(classifier_out, classes)
                                .item()
                            )

                            # Compute number of correct samples
                            total += ids.size(0)
                            _, predicted = torch.max(classifier_out.data, 1)

                            correct += (predicted.cpu() == classes.cpu().T).sum().item()

                        if self.regress:
                            regressor_out = self.model(
                                data, training=False, regress=True, latent=False
                            )
                            eval_loss_regression += Criterion().regressor_criterion(
                                regressor_out, parameters
                            )

                eval_loss_forecasting /= len(data_loader_to_evaluate.dataset)
                print(
                    f"====> Mean {current_set} set forecasting loss: {eval_loss_forecasting:.4f}"
                )
                if self.classify:
                    accuracy = correct / total
                    if eval_loss_classification != 0:
                        eval_loss_classification /= len(data_loader_to_evaluate.dataset)
                        print(
                            f"====> Mean {current_set} set classifier loss: {eval_loss_classification:.4f}; accuracy: {accuracy:.2f}"
                        )

                if self.regress:
                    print(
                        f"====> Mean {current_set} set regressor loss: {eval_loss_regression:.4f}"
                    )

            # Scheduler metric is test set loss
            if current_set == "Test" and training_mode == "forecasting":
                for scheduler in self.forecasting_schedulers.values():
                    scheduler.step(eval_loss_forecasting)
            elif current_set == "Test" and training_mode == "classification":
                for scheduler in self.classification_schedulers.values():
                    scheduler.step(eval_loss_classification)
            elif current_set == "Test" and training_mode == "regression":
                for scheduler in self.regression_schedulers.values():
                    scheduler.step(eval_loss_regression)

        # Save the model at target path
        utils.save(self.model, self.model_hyperparameters, path=model_save_path)

    def validate(self, validation_loader):
        # Perform model validation
        validation_loss_forecasting = 0.0
        validation_loss_classification = 0.0
        validation_loss_regression = 0.0
        with torch.no_grad():
            if self.classify:
                total = 0.0
                correct = 0.0
            self.model.eval()
            for idx, (data, target, ids, parameters, classes) in enumerate(
                validation_loader
            ):
                if type(ids) == list:
                    ids = ids[0]
                data, target, ids, parameters = (
                    data.float().to(device),
                    target.float().to(device),
                    ids.to(device),
                    parameters.float().to(device),
                )
                # Time series forecasting test
                if self.model_type == "ae" or self.model_type == "lstm":
                    out = self.model(data, training=False, classify=False, latent=False)
                    validation_loss_forecasting += (
                        Criterion()
                        .forecasting_criterion(out, target, loss_type=self.loss_type)
                        .item()
                    )

                else:
                    decoder_out, latent_out, mu, logvar = self.model(
                        data, training=False, classify=False
                    )
                    validation_loss_forecasting += Criterion().forecasting_criterion(
                        decoder_out,
                        target,
                        mu=mu,
                        logvar=logvar,
                        loss_type=self.loss_type,
                    )

                # Classification test
                if self.classify:
                    ids = ids.long()
                    classifier_out = self.model(
                        data, training=False, classify=True, latent=False
                    )

                    validation_loss_classification += (
                        Criterion().classifier_criterion(classifier_out, classes).item()
                    )

                    # Compute number of correct samples
                    total += ids.size(0)
                    _, predicted = torch.max(classifier_out.data, 1)
                    correct += (predicted.cpu() == classes.cpu().T).sum().item()

                if self.regress:
                    regressor_out = self.model(
                        data, training=True, regress=True, latent=False
                    )
                    validation_loss_regression += Criterion().regressor_criterion(
                        regressor_out, parameters
                    )

            validation_loss_forecasting /= len(validation_loader.dataset)
            print(
                f"====> Mean Validation set generator loss: {validation_loss_forecasting:.4f}"
            )
            if self.classify:
                accuracy = correct / total
                if validation_loss_classification != 0:
                    validation_loss_classification /= len(validation_loader.dataset)
                    print(
                        f"====> Mean Validation set classifier loss: {validation_loss_classification:.4f}; accuracy: {accuracy:.4f}"
                    )

            if self.regress:
                print(
                    f"====> Mean Validation set regressor loss: {validation_loss_regression:.4f}"
                )

        return (
            validation_loss_forecasting,
            validation_loss_regression,
            validation_loss_classification,
        )
