"""This module contains the autoencoders and its variants
1. classic autoencoder
2. wasserstein autoencoder"""

import torch
from .utils import TimeDistributed
from .utils import load_model

# Encoder
class LSTMEncoder(torch.nn.Module):
    """ Deep LSTM network. This implementation
    returns output_size hidden size.
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

    def __init__(self, input_size: int, sequence_length:int, batch_size:int, 
                 hidden_size: int, num_layers: int,
                 batch_first: bool, dropout: float, 
                 reset_state: bool, bidirectional: bool):
        
        super(LSTMEncoder, self).__init__()
        
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        # RNN Encoder
        self.lstm_encoder = torch.nn.LSTM(input_size=input_size, hidden_size=self.hidden_size,
                            num_layers=num_layers, dropout=dropout,
                            bidirectional=self.bidirectional, batch_first=True)

    def _init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size), torch.zeros(self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, x):
        
        # Encoder
        enc_init_hidden = self._init_hidden()   
        enc_output,enc_states = self.lstm_encoder(x,enc_init_hidden)
        # RNNs obeys, Markovian. Consider the last state of the hidden is the markovian of the entire sequence in that batch.
        enc_output = enc_output[ : , -1, : ] # Shape(batch_size,hidden_dim) 
        return enc_output
# Latent
class DisentangledAELatent(torch.nn.Module):
    """Dense Latent Layer """

    def __init__(self,  hidden_size: int, latent_size:int, dropout: float):
        super(DisentangledAELatent, self).__init__()
    
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.latent = torch.nn.Linear(self.hidden_size,self.latent_size)
        
    def forward(self, x, training=True):
        # Feed it into the disentanled latent layer
        z = self.latent(x) # Shape(batch_size, latent_size*2)
        return z

# Decoder
class LSTMDecoder(torch.nn.Module):
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

    def __init__(self, batch_size:int, num_future:int, hidden_size: int, num_layers: int,
                 output_size: int, latent_size:int, batch_first: bool, dropout: float, 
                 reset_state: bool, bidirectional: bool):
        super(LSTMDecoder, self).__init__()
        
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.num_future = num_future
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        # RNN decoder
        self.lstm_decoder = torch.nn.LSTM(input_size=self.latent_size, hidden_size=self.hidden_size,
                            num_layers=self.num_layers, dropout=self.dropout,
                            bidirectional=self.bidirectional, batch_first=True)
        self.output = TimeDistributed(torch.nn.Linear(self.hidden_size , self.output_size))

    def _init_hidden(self):
        return (torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device), torch.zeros(self.num_layers, self.batch_size, self.hidden_size).to(device))

    def forward(self, x, num_future = None):
    
        # To feed the latent states into lstm decoder, repeat the tensor n_future times at second dim
        _init_hidden = self._init_hidden()
        decoder_inputs = x.unsqueeze(1)

        if num_future==None:
            decoder_inputs = decoder_inputs.repeat(1, self.num_future, 1)
        else: # For multistep a prediction after training
            decoder_inputs = decoder_inputs.repeat(1, num_future, 1)
            
        # Decoder input Shape(batch_size, num_futures, latent_size)
        dec,(dec_hidden,dec_cell) = self.lstm_decoder(decoder_inputs,_init_hidden)
    
        # Map the decoder output: Shape(batch_size, sequence_len, hidden_dim) to Time Dsitributed Linear Layer
        output = self.output(dec)
        return output   

class MLPClassifier(torch.nn.Module):
    """ MLP classifier
    """
    def __init__(self, hidden_size: int, num_classes:int, latent_size:int, dropout:float ):
        
        super(MLPClassifier, self).__init__()
        
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.dropout = dropout
        
        # Classifier
        self.classifier1 = torch.nn.Linear(self.latent_size , self.hidden_size)
        self.classifier2 = torch.nn.Linear(self.hidden_size , self.hidden_size)
        self.classifier3 = torch.nn.Linear(self.hidden_size , self.hidden_size)
        self.classifier4 = torch.nn.Linear(self.hidden_size , self.num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        
        classifier1 = self.dropout(self.classifier1(x))
        classifier2 = self.dropout(self.classifier2(classifier1))
        classifier3 = self.dropout(self.classifier3(classifier2))
        classifier4 = self.classifier4(classifier3)
        # classifier_out = F.softmax(classifier4)

        return classifier4
    
    
class MultiModelAE(torch.nn.Module):
    
    def __init__(self, *model_hyperparameters, **kwargs):
        
        super(MultiModelAE, self).__init__()
        # self.input_size = input_size
        # self.sequence_length = sequence_length
        # self.batch_size = batch_size
        # self.latent_size = latent_size
        # self.num_future = num_future
        # self.hidden_size = hidden_size
        # self.num_layers = num_layers
        # self.output_size = output_size
        # self.num_classes = num_classes
        # self.batch_first = batch_first
        # self.dropout = dropout
        # self.reset_state = reset_state
        # self.bidirectional = bidirectional
        for dictionary in model_hyperparameters:
            for key in dictionary:
                    setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

        # Network instances in the model
        self.encoder = LSTMEncoder(input_size=self.input_size, sequence_length=self.sequence_length, batch_size=self.batch_size, 
                    hidden_size=self.hidden_size, num_layers=self.num_layers,
                    batch_first=self.batch_first, dropout=self.dropout, 
                    reset_state=True, bidirectional=self.bidirectional)

        self.latent = DisentangledAELatent(hidden_size=self.hidden_size, latent_size=self.latent_size, dropout=self.dropout)

        self.decoder = LSTMDecoder(batch_size=self.batch_size, num_future=self.num_future, 
                        hidden_size=self.hidden_size, num_layers=self.num_layers, output_size=self.output_size, 
                        latent_size=self.latent_size, batch_first=self.batch_first, dropout=self.dropout, 
                        reset_state=True, bidirectional=self.bidirectional)

        self.classifier = MLPClassifier(hidden_size=self.hidden_size,num_classes=self.num_classes, latent_size=self.latent_size, dropout=self.dropout)

    def forward(self, data, training=True, is_classification=False):
            
            if not is_classification:
                # Set the classifier grad off
                for param in self.classifier.parameters():
                    param.requires_grad = False
                
                for param in self.encoder.parameters():
                    param.requires_grad = True
                
                for param in self.decoder.parameters():
                    param.requires_grad = True
                
                for param in self.latent.parameters():
                    param.requires_grad = True
                
                enc_out = self.encoder(data)
                # Latent
                latent_out= self.latent(enc_out,training=training)
                # Decoder
                decoder_out = self.decoder(latent_out)

                return decoder_out,latent_out

            else:
                # training_mode = 'classification'
                # Freeze decoder parameters; 
                for param in self.classifier.parameters():
                    param.requires_grad = True
                
                for param in self.encoder.parameters():
                    param.requires_grad = False
                
                for param in self.decoder.parameters():
                    param.requires_grad = False
                
                for param in self.latent.parameters():
                    param.requires_grad = False

                # Encoder
                enc_out = self.encoder(data)
                # Latent
                latent_out= self.latent(enc_out,training=training)
                #Classifier
                classifier_out = self.classifier(latent_out) # Deterministic
                return classifier_out

