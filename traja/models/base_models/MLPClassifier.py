import torch
from torch import nn


class MLPClassifier(torch.nn.Module):
    """MLP classifier: Classify the input data using the latent embeddings
    input_size: The number of expected latent size
    hidden_size: The number of features in the hidden state h
    output_size: Size of labels or the number of sequence_ids in the data
    dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                    with dropout probability equal to dropout
    num_layers: Number of hidden layers in the classifier
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        num_layers: int,
        dropout: float,
    ):
        super(MLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = output_size
        self.num_layers = num_layers
        self.dropout = dropout

        # Classifier layers
        layers = list()

        layers.append(nn.Linear(self.input_size, self.hidden_size))
        layers.append(nn.ReLU())
        torch.nn.Dropout(p=dropout)

        for layer in range(1, self.num_layers):
            layers.append(nn.Linear(self.hidden_size, self.hidden_size))
            layers.append(nn.ReLU())
            torch.nn.Dropout(p=dropout)

        layers.append(nn.Linear(self.hidden_size, self.num_classes))

        self.hidden = nn.Sequential(*layers)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.hidden(x)
        output = self.sigmoid(x)
        return output
