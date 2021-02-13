""" This module implement the Variational Autoencoder model for 
both forecasting and classification of time series data.
"""

import torch

from traja.models.base_models.MLPClassifier import MLPClassifier
from traja.models.base_models.MLPRegressor import MLPRegressor
from traja.models.utils import TimeDistributed

device = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMEncoder(torch.nn.Module):
    """Implementation of Encoder network using LSTM layers
    input_size: The number of expected features in the input x
    num_past: Number of time steps to look backwards to predict num_future steps forward
    batch_size: Number of samples in a batch
    hidden_size: The number of features in the hidden state h
    num_lstm_layers: Number of layers in the LSTM model

    batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    reset_state: If True, will reset the hidden and cell state for each batch of data
    bidirectional:  If True, becomes a bidirectional LSTM
    """

    def __init__(
        self,
        input_size: int,
        num_past: int,
        batch_size: int,
        hidden_size: int,
        num_lstm_layers: int,
        batch_first: bool,
        dropout: float,
        reset_state: bool,
        bidirectional: bool,
    ):
        super(LSTMEncoder, self).__init__()

        self.input_size = input_size
        self.num_past = num_past
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        self.lstm_encoder = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_lstm_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def _init_hidden(self):
        return (
            torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size)
            .requires_grad_()
            .to(device),
            torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size)
            .requires_grad_()
            .to(device),
        )

    def forward(self, x):
        (h0, c0) = self._init_hidden()
        enc_output, _ = self.lstm_encoder(x, (h0.detach(), c0.detach()))
        # RNNs obeys, Markovian. So, the last state of the hidden is the markovian state for the entire
        # sequence in that batch.
        enc_output = enc_output[:, -1, :]  # Shape(batch_size,hidden_dim)
        return enc_output


class DisentangledAELatent(torch.nn.Module):
    """Dense Dientangled Latent Layer between encoder and decoder"""

    def __init__(self, hidden_size: int, latent_size: int, dropout: float):
        super(DisentangledAELatent, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.latent = torch.nn.Linear(self.hidden_size, self.latent_size * 2)

    @staticmethod
    def reparameterize(mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return eps.mul(std).add_(mu)
        return mu

    def forward(self, x, training=True):
        z_variables = self.latent(x)  # [batch_size, latent_size*2]
        mu, logvar = torch.chunk(z_variables, 2, dim=1)  # [batch_size,latent_size]
        # Reparameterize
        z = self.reparameterize(
            mu, logvar, training=training
        )  # [batch_size,latent_size]
        return z, mu, logvar


class LSTMDecoder(torch.nn.Module):
    """Implementation of Decoder network using LSTM layers
    input_size: The number of expected features in the input x
    num_future: Number of time steps to be predicted given the num_past steps
    batch_size: Number of samples in a batch
    hidden_size: The number of features in the hidden state h
    num_lstm_layers: Number of layers in the LSTM model
    output_size: Number of expectd features in the output x_
    batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    reset_state: If True, will reset the hidden and cell state for each batch of data
    bidirectional:  If True, becomes a bidirectional LSTM
    """

    def __init__(
        self,
        batch_size: int,
        num_future: int,
        hidden_size: int,
        num_lstm_layers: int,
        output_size: int,
        latent_size: int,
        batch_first: bool,
        dropout: float,
        reset_state: bool,
        bidirectional: bool,
    ):
        super(LSTMDecoder, self).__init__()
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.num_future = num_future
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        # RNN decoder
        self.lstm_decoder = torch.nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.output = TimeDistributed(
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def _init_hidden(self):
        return (
            torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size)
            .requires_grad_()
            .to(device),
            torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size)
            .requires_grad_()
            .to(device),
        )

    def forward(self, x, num_future=None):

        # To feed the latent states into lstm decoder,
        # repeat the tensor n_future times at second dim
        (h0, c0) = self._init_hidden()
        decoder_inputs = x.unsqueeze(1)

        if num_future is None:
            decoder_inputs = decoder_inputs.repeat(1, self.num_future, 1)
        else:  # For multistep a prediction after training
            decoder_inputs = decoder_inputs.repeat(1, num_future, 1)

        # Decoder input Shape(batch_size, num_futures, latent_size)
        dec, _ = self.lstm_decoder(decoder_inputs, (h0.detach(), c0.detach()))

        # Map the decoder output: Shape(batch_size, sequence_len, hidden_dim)
        # to Time Dsitributed Linear Layer
        output = self.output(dec)
        return output


class MultiModelVAE(torch.nn.Module):
    """Implementation of Multimodel Variational autoencoders; This Module wraps the Variational Autoencoder
    models [Encoder,Latent[Sampler],Decoder]. If classify=True, then the wrapper also include classification layers

    input_size: The number of expected features in the input x
    num_future: Number of time steps to be predicted given the num_past steps
    batch_size: Number of samples in a batch
    hidden_size: The number of features in the hidden state h
    num_lstm_layers: Number of layers in the LSTM model
    output_size: Number of expectd features in the output x_
    batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    reset_state: If True, will reset the hidden and cell state for each batch of data
    bidirectional:  If True, becomes a bidirectional LSTM
    """

    def __init__(
        self,
        input_size: int,
        num_past: int,
        batch_size: int,
        num_future: int,
        lstm_hidden_size: int,
        num_lstm_layers: int,
        output_size: int,
        latent_size: int,
        batch_first: bool,
        dropout: float,
        reset_state: bool,
        bidirectional: bool = False,
        num_classifier_layers: int = None,
        classifier_hidden_size: int = None,
        num_classes: int = None,
        num_regressor_layers: int = None,
        regressor_hidden_size: int = None,
        num_regressor_parameters: int = None,
    ):

        super(MultiModelVAE, self).__init__()
        self.input_size = input_size
        self.num_past = num_past
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.num_future = num_future
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.classifier_hidden_size = classifier_hidden_size
        self.num_classifier_layers = num_classifier_layers
        self.output_size = output_size
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional
        self.num_regressor_layers = num_regressor_layers
        self.regressor_hidden_size = regressor_hidden_size
        self.num_regressor_parameters = num_regressor_parameters

        self.latent_output_disabled = False  # Manually override latent output

        # Let the trainer know what kind of model this is
        self.model_type = "vae"

        self.encoder = LSTMEncoder(
            input_size=self.input_size,
            num_past=self.num_past,
            batch_size=self.batch_size,
            hidden_size=self.lstm_hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            reset_state=True,
            bidirectional=self.bidirectional,
        )

        self.latent = DisentangledAELatent(
            hidden_size=self.lstm_hidden_size,
            latent_size=self.latent_size,
            dropout=self.dropout,
        )

        self.decoder = LSTMDecoder(
            batch_size=self.batch_size,
            num_future=self.num_future,
            hidden_size=self.lstm_hidden_size,
            num_lstm_layers=self.num_lstm_layers,
            output_size=self.output_size,
            latent_size=self.latent_size,
            batch_first=self.batch_first,
            dropout=self.dropout,
            reset_state=True,
            bidirectional=self.bidirectional,
        )

        if self.num_classes is not None:
            self.classifier = MLPClassifier(
                input_size=self.latent_size,
                hidden_size=self.classifier_hidden_size,
                output_size=self.num_classes,
                num_layers=self.num_classifier_layers,
                dropout=self.dropout,
            )

        if self.num_regressor_parameters is not None:
            self.regressor = MLPRegressor(
                input_size=self.latent_size,
                hidden_size=self.regressor_hidden_size,
                output_size=self.num_regressor_parameters,
                num_layers=self.num_regressor_layers,
                dropout=self.dropout,
            )

    def reset_classifier(self, classifier_hidden_size: int, num_classifier_layers: int):
        """Reset the classifier, with a new hidden size and depth.
        This is useful when parameter searching.

        classifier_hidden_size: The number of units in each classifier layer
        num_layers: Number of layers in the classifier
        """
        self.classifier_hidden_size = classifier_hidden_size
        self.num_classifier_layers = num_classifier_layers

        self.classifier = MLPClassifier(
            input_size=self.latent_size,
            hidden_size=self.classifier_hidden_size,
            output_size=self.num_classes,
            num_layers=self.num_classifier_layers,
            dropout=self.dropout,
        )

    def reset_regressor(self, regressor_hidden_size: int, num_regressor_layers: int):
        """Reset the regressor, with a new hidden size and depth.
        This is useful when parameter searching.

        regressor_hidden_size: The number of units in each classifier layer
        num_regressor_layers: Number of layers in the classifier
        """
        self.num_regressor_layers = num_regressor_layers
        self.regressor_hidden_size = regressor_hidden_size

        self.regressor = MLPRegressor(
            input_size=self.latent_size,
            hidden_size=self.regressor_hidden_size,
            output_size=self.num_regressor_parameters,
            num_layers=self.num_regressor_layers,
            dropout=self.dropout,
        )

    def disable_latent_output(self):
        """Disable latent output, to make the VAE behave like a standard autoencoder while training.
        This modifies the training loss computed."""
        self.latent_output_disabled = True

    def enable_latent_output(self):
        """Enable latent output, to make the VAE behave like a variational autoencoder while training.
        This modifies the training loss computed.
        NOTE: By default, latent output is enabled."""
        self.latent_output_disabled = False

    def forward(self, data, training=True, classify=False, regress=False, latent=True):
        """
        Parameters:
        -----------
            data: Train or test data
            training: If Training= False, latents are deterministic
            classify: If True, perform classification of input data using the latent embeddings
        Return:
        -------
            decoder_out,latent_out or classifier out
        """

        assert not (classify and regress), "Model cannot both classify and regress!"

        if not (classify or regress):
            # Set the classifier and regressor grads off
            if self.num_classes is not None:
                for param in self.classifier.parameters():
                    param.requires_grad = False
            if self.num_regressor_parameters is not None:
                for param in self.regressor.parameters():
                    param.requires_grad = False

            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.latent.parameters():
                param.requires_grad = True

            # Encoder -->Latent --> Decoder
            enc_out = self.encoder(data)
            latent_out, mu, logvar = self.latent(enc_out)
            decoder_out = self.decoder(latent_out)
            if latent:
                return decoder_out, latent_out, mu, logvar
            else:
                return decoder_out

        elif classify:
            # Unfreeze classifier and freeze the rest
            assert self.num_classes is not None, "Classifier not found"

            for param in self.classifier.parameters():
                param.requires_grad = True
            if self.num_regressor_parameters is not None:
                for param in self.regressor.parameters():
                    param.requires_grad = False
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.latent.parameters():
                param.requires_grad = False

            # Encoder -->Latent --> Classifier
            enc_out = self.encoder(data)
            latent_out, mu, logvar = self.latent(enc_out, training=training)

            classifier_out = self.classifier(mu)  # Deterministic
            if latent:
                return classifier_out, latent_out, mu, logvar
            else:
                return classifier_out

        elif regress:
            # Unfreeze classifier and freeze the rest
            assert self.num_regressor_parameters is not None, "Regressor not found"

            if self.num_classes is not None:
                for param in self.classifier.parameters():
                    param.requires_grad = False
            for param in self.regressor.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.latent.parameters():
                param.requires_grad = False

            # Encoder -->Latent --> Regressor
            enc_out = self.encoder(data)
            latent_out, mu, logvar = self.latent(enc_out, training=training)

            regressor_out = self.regressor(mu)  # Deterministic

            if self.latent_output_disabled:
                mu = None
                logvar = None

            if latent:
                return regressor_out, latent_out, mu, logvar
            else:
                return regressor_out
