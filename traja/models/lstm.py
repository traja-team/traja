"""Implementation of Multimodel LSTM"""

import torch


class LSTM(torch.nn.Module):
    """ Deep LSTM network. This implementation
    returns output_size outputs.
    Args:
        input_size: The number of expected features in the input `x`
        batch_size: 
        sequence_length: The number of in each sample
        hidden_size: The number of features in the hidden state `h`
        num_layers: Number of recurrent layers. E.g., setting ``num_layers=2``
            would mean stacking two LSTMs together to form a `stacked LSTM`,
            with the second LSTM taking in outputs of the first LSTM and
            computing the final results. Default: 1
        output_size: The number of output dimensions
        dropout: If non-zero, introduces a `Dropout` layer on the outputs of each
            LSTM layer except the last layer, with dropout probability equal to
            :attr:`dropout`. Default: 0
        bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
    """

    def __init__(self, batch_size: int, num_future: int, hidden_size: int, num_layers: int,
                 output_size: int, input_size: int, batch_first: bool, dropout: float,
                 reset_state: bool, bidirectional: bool):
        super(LSTM, self).__init__()

        self.batch_size = batch_size
        self.input_size = input_size
        self.num_future = num_future
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        # RNN decoder
        self.lstm_decoder = torch.nn.LSTM(input_size=self.input_size, hidden_size=self.hidden_size,
                                          num_layers=self.num_layers, dropout=self.dropout,
                                          bidirectional=self.bidirectional, batch_first=True)
        self.output = TimeDistributed(torch.nn.Linear(self.hidden_size, self.output_size))

    def _init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device),
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))

    def forward(self, x):
        # To feed the latent states into lstm decoder, repeat the tensor n_future times at second dim
        _init_hidden = self._init_hidden()

        # Decoder input Shape(batch_size, num_futures, latent_size)
        dec, (dec_hidden, dec_cell) = self.lstm_decoder(x, _init_hidden)

        # Map the decoder output: Shape(batch_size, sequence_len, hidden_dim) to Time Dsitributed Linear Layer
        output = self.output(dec)
        return output
