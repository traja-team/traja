import torch
from traja.models.utils import TimeDistributed
from torch import nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'


class LSTMEncoder(torch.nn.Module):

    """ Implementation of Encoder network using LSTM layers
    :param input_size: The number of expected features in the input x
    :param num_past: Number of time steps to look backwards to predict num_future steps forward
    :param batch_size: Number of samples in a batch
    :param hidden_size: The number of features in the hidden state h
    :param num_lstm_layers: Number of layers in the LSTM model

    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    :param dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    :param reset_state: If True, will reset the hidden and cell state for each batch of data
    :param bidirectional:  If True, becomes a bidirectional LSTM
    """

    def __init__(self, input_size: int, num_past: int, batch_size: int,
                 hidden_size: int, num_lstm_layers: int,
                 batch_first: bool, dropout: float,
                 reset_state: bool, bidirectional: bool):

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

        self.lstm_encoder = torch.nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                                          num_layers=num_lstm_layers, dropout=dropout,
                                          bidirectional=self.bidirectional, batch_first=True)

    def _init_hidden(self):
        return (torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_lstm_layers, self.batch_size, self.hidden_size).to(device))

    def forward(self, x):
        enc_init_hidden = self._init_hidden()
        enc_output, _ = self.lstm_encoder(x, enc_init_hidden)
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
        self.latent = torch.nn.Linear(self.hidden_size, self.latent_size)

    def forward(self, x):
        z = self.latent(x)  # Shape(batch_size, latent_size*2)
        return z


class LSTMDecoder(torch.nn.Module):

    """ Implementation of Decoder network using LSTM layers
    :param input_size: The number of expected features in the input x
    :param num_future: Number of time steps to be predicted given the num_past steps
    :param batch_size: Number of samples in a batch
    :param hidden_size: The number of features in the hidden state h
    :param num_lstm_layers: Number of layers in the LSTM model
    :param output_size: Number of expectd features in the output x_
    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    :param dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    :param reset_state: If True, will reset the hidden and cell state for each batch of data
    :param bidirectional:  If True, becomes a bidirectional LSTM

    """

    def __init__(self, batch_size: int, num_future: int, hidden_size: int,
                 num_lstm_layers: int, output_size: int, latent_size: int,
                 batch_first: bool, dropout: float,
                 reset_state: bool, bidirectional: bool):
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
        self.lstm_decoder = torch.nn.LSTM(input_size=self.latent_size,
                                          hidden_size=self.hidden_size,
                                          num_layers=self.num_lstm_layers,
                                          dropout=self.dropout,
                                          bidirectional=self.bidirectional,
                                          batch_first=True)
        self.output = TimeDistributed(torch.nn.Linear(self.hidden_size,
                                                      self.output_size))

    def _init_hidden(self):
        return (torch.zeros(self.num_lstm_layers, self.batch_size,
                            self.hidden_size).to(device),
                torch.zeros(self.num_lstm_layers, self.batch_size,
                            self.hidden_size).to(device))

    def forward(self, x, num_future=None):

        # To feed the latent states into lstm decoder, 
        # repeat the tensor n_future times at second dim
        _init_hidden = self._init_hidden()
        decoder_inputs = x.unsqueeze(1)

        if num_future is None:
            decoder_inputs = decoder_inputs.repeat(1, self.num_future, 1)
        else:  # For multistep a prediction after training
            decoder_inputs = decoder_inputs.repeat(1, num_future, 1)

        # Decoder input Shape(batch_size, num_futures, latent_size)
        dec, _ = self.lstm_decoder(decoder_inputs, _init_hidden)

        # Map the decoder output: Shape(batch_size, sequence_len, hidden_dim) 
        # to Time Dsitributed Linear Layer
        output = self.output(dec)
        return output


class MLPClassifier(torch.nn.Module):

    """ MLP classifier: Classify the input data using the latent embeddings
            :param input_size: The number of expected latent size
            :param hidden_size: The number of features in the hidden state h
            :param num_classes: Size of labels or the number of categories in the data
            :param dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                            with dropout probability equal to dropout
            :param num_classifier_layers: Number of hidden layers in the classifier
            """
    def __init__(self, input_size: int, hidden_size:int, num_classes: int, latent_size: int, num_classifier_layers: int,
                 dropout: float):
        super(MLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_classifier_layers = num_classifier_layers
        self.dropout = dropout

        # Classifier layers
        self.hidden = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size)])
        self.hidden.extend([nn.Linear(self.hidden_size, self.hidden_size) for _ in range(1, self.num_classifier_layers - 1)])
        self.hidden = nn.Sequential(*self.hidden)
        self.out = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(self.hidden(x))
        out = self.out(x)
        return out


class MultiModelAE(torch.nn.Module):

    """Implementation of Multimodel  autoencoders; This Module wraps the  Autoencoder
    models [Encoder,Latent,Decoder]. If classify=True, then the wrapper also include classification layers

    :param input_size: The number of expected features in the input x
    :param num_future: Number of time steps to be predicted given the num_past steps
    :param batch_size: Number of samples in a batch
    :param hidden_size: The number of features in the hidden state h
    :param num_lstm_layers: Number of layers in the LSTM model
    :param output_size: Number of expectd features in the output x_
    :param batch_first: If True, then the input and output tensors are provided as (batch, seq, feature)
    :param dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    :param reset_state: If True, will reset the hidden and cell state for each batch of data
    :param bidirectional:  If True, becomes a bidirectional LSTM
    """
    def __init__(self, input_size: int, num_past: int, batch_size: int, num_future: int, lstm_hidden_size: int,
                 num_lstm_layers: int , output_size: int, latent_size: int, batch_first: bool, dropout: float,
                 reset_state: bool, bidirectional: bool=False, num_classifier_layers: int= None,
                 classifier_hidden_size: int=None, num_classes: int=None):


        super(MultiModelAE, self).__init__()
        self.input_size = input_size
        self.num_past = num_past
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.num_future = num_future
        self.lstm_hidden_size = lstm_hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.num_classifier_layers = num_classifier_layers
        self.classifier_hidden_size = classifier_hidden_size
        self.output_size = output_size
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        self.encoder = LSTMEncoder(input_size=self.input_size,
                                   num_past=self.num_past,
                                   batch_size=self.batch_size,
                                   hidden_size=self.lstm_hidden_size,
                                   num_lstm_layers=self.num_lstm_layers,
                                   batch_first=self.batch_first,
                                   dropout=self.dropout,
                                   reset_state=True,
                                   bidirectional=self.bidirectional)

        self.latent = DisentangledAELatent(hidden_size=self.lstm_hidden_size,
                                           latent_size=self.latent_size,
                                           dropout=self.dropout)

        self.decoder = LSTMDecoder(batch_size=self.batch_size,
                                   num_future=self.num_future,
                                   hidden_size=self.lstm_hidden_size,
                                   num_lstm_layers=self.num_lstm_layers,
                                   output_size=self.output_size,
                                   latent_size=self.latent_size,
                                   batch_first=self.batch_first,
                                   dropout=self.dropout,
                                   reset_state=True,
                                   bidirectional=self.bidirectional)

        if self.num_classes is not None:
            self.classifier = MLPClassifier(input_size=self.latent_size,
                                            hidden_size=self.classifier_hidden_size,
                                            num_classes=self.num_classes,
                                            latent_size=self.latent_size,
                                            num_classifier_layers=self.num_classifier_layers,
                                            dropout=self.dropout)

    def get_ae_parameters(self):
        """
        :return: Tuple of parameters of the encoder, latent and decoder networks
        """
        return [self.encoder.parameters(),self.latent.parameters(),self.decoder.parameters()]

    def get_classifier_parameters(self):
        """
        :return: Tuple of parameters of classifier network
        """
        assert self.classifier_hidden_size is not None,"Classifier not found"
        return [self.classifier.parameters()]

    def forward(self, data, classify=False, training=True):
        """
        :param data: Train or test data
        :param training: If Training= False, latents are deterministic; This arg is unused;
        :param classify: If True, perform classification of input data using the latent embeddings
        :return: decoder_out,latent_out or classifier out
        """
        if not classify:
            # Set the classifier grad off
            if self.num_classes is not None:
                for param in self.classifier.parameters():
                    param.requires_grad = False

            for param in self.encoder.parameters():
                param.requires_grad = True
            for param in self.decoder.parameters():
                param.requires_grad = True
            for param in self.latent.parameters():
                param.requires_grad = True


            # Encoder -->Latent --> Decoder
            enc_out = self.encoder(data)
            latent_out = self.latent(enc_out)
            decoder_out = self.decoder(latent_out)
            return decoder_out, latent_out

        else: # Classify
            # Unfreeze classifier and freeze the rest
            assert self.num_classifier_layers is not None, "Classifier not found"

            for param in self.classifier.parameters():
                param.requires_grad = True
            for param in self.encoder.parameters():
                param.requires_grad = False
            for param in self.decoder.parameters():
                param.requires_grad = False
            for param in self.latent.parameters():
                param.requires_grad = False

            # Encoder-->Latent-->Classifier
            enc_out = self.encoder(data)
            latent_out = self.latent(enc_out)

            classifier_out = self.classifier(latent_out)  # Deterministic
            return classifier_out
