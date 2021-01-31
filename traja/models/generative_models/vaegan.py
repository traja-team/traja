import torch
from torch import nn
from traja.models.utils import TimeDistributed
from torch.autograd import Variable
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import Dataset, DataLoader, Sampler

torch.autograd.set_detect_anomaly(True)
device = "cuda" if torch.cuda.is_available() else "cpu"


class LSTMEncoder(torch.nn.Module):
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

    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        batch_size: int,
        hidden_size: int,
        num_layers: int,
        batch_first: bool,
        dropout: float,
        reset_state: bool,
        bidirectional: bool,
    ):

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
        self.lstm_encoder = torch.nn.LSTM(
            input_size=input_size,
            hidden_size=self.hidden_size,
            num_layers=num_layers,
            dropout=dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

    def _init_hidden(self):
        return (
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).to(device),
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).to(device),
        )

    def forward(self, x):

        # Encoder
        enc_init_hidden = self._init_hidden()
        enc_output, enc_states = self.lstm_encoder(x, enc_init_hidden)
        # RNNs obeys, Markovian. Consider the last state of the hidden is the markovian of the entire sequence in that batch.
        enc_output = enc_output[:, -1, :]  # Shape(batch_size,hidden_dim)
        return enc_output


class LSTMEncoder(torch.nn.Module):
    """ Implementation of Encoder network using LSTM layers
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


class Sampler(torch.nn.Module):
    """Approximate Posterior Sampling over latent states

    Args:
        input (tensor): Latent variables, mu and log(variance)
    """

    def __init__(self):
        super(Sampler, self).__init__()

    def forward(self, input):
        mu = input[0]
        logvar = input[1]

        std = logvar.mul(0.5).exp_()  # calculate the STDEV
        if device == "cuda":
            eps = torch.cuda.FloatTensor(
                std.size()
            ).normal_()  # random normalized noise
        else:
            eps = torch.FloatTensor(std.size()).normal_()  # random normalized noise
        eps = Variable(eps)
        return eps.mul(std).add_(mu)


class DisentangledLatent(torch.nn.Module):
    """Dense Dientangled Latent Layer between encoder and decoder"""

    def __init__(self, hidden_size: int, latent_size: int, dropout: float):
        super(DisentangledLatent, self).__init__()
        self.latent_size = latent_size
        self.hidden_size = hidden_size
        self.dropout = dropout
        self.latent = torch.nn.Linear(self.hidden_size, self.latent_size * 2)
        self.sampler = Sampler()

    def reparameterize(self, mu, logvar, training=True):
        if training:
            std = logvar.mul(0.5).exp_()
            eps = std.data.new(std.size()).normal_()
            return self.sampler([mu, logvar])
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

    def __init__(
        self,
        batch_size: int,
        num_future: int,
        hidden_size: int,
        num_layers: int,
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
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        # RNN decoder
        self.lstm_decoder = torch.nn.LSTM(
            input_size=self.latent_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )
        self.output = TimeDistributed(
            torch.nn.Linear(self.hidden_size, self.output_size)
        )

    def _init_hidden(self):
        return (
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).to(device),
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).to(device),
        )

    def forward(self, x):

        # To feed the latent states into lstm decoder, repeat the tensor n_future times at second dim
        decoder_inputs = x.unsqueeze(1).repeat(1, self.num_future, 1)
        # Decoder input Shape(batch_size, num_futures, latent_size)
        dec, (dec_hidden, dec_cell) = self.lstm_decoder(
            decoder_inputs, self._init_hidden()
        )
        # dec,(dec_hidden,dec_cell) = self.lstm_decoder(decoder_inputs)

        # Map the decoder output: Shape(batch_size, sequence_len, hidden_dim) to Time Dsitributed Linear Layer
        output = self.output(dec)

        return output
        # return dec


class MLPClassifier(torch.nn.Module):
    """ MLP classifier: Classify the input data using the latent embeddings
            input_size: The number of expected latent size
            hidden_size: The number of features in the hidden state h
            num_classes: Size of labels or the number of categories in the data
            dropout:  If non-zero, introduces a Dropout layer on the outputs of each LSTM layer except the last layer,
                            with dropout probability equal to dropout
            num_classifier_layers: Number of hidden layers in the classifier
            """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_classes: int,
        latent_size: int,
        num_classifier_layers: int,
        dropout: float,
    ):
        super(MLPClassifier, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.num_classifier_layers = num_classifier_layers
        self.dropout = dropout

        # Classifier layers
        self.hidden = nn.ModuleList([nn.Linear(self.input_size, self.hidden_size)])
        self.hidden.extend(
            [
                nn.Linear(self.hidden_size, self.hidden_size)
                for _ in range(1, self.num_classifier_layers - 1)
            ]
        )
        self.hidden = nn.Sequential(*self.hidden)
        self.out = nn.Linear(self.hidden_size, self.num_classes)
        self.dropout = torch.nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(self.hidden(x))
        out = self.out(x)
        return out


class LSTMDiscriminator(torch.nn.Module):
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

    def __init__(
        self,
        input_size: int,
        batch_size: int,
        num_future: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        latent_size: int,
        batch_first: bool,
        dropout: float,
        reset_state: bool,
        bidirectional: bool,
    ):
        super(LSTMDiscriminator, self).__init__()

        self.batch_size = batch_size
        self.sequence_length = self.sequence_length
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional
        self.latent_size = latent_size
        self.input_size = input_size

        # RNN decoder
        self.lstm = torch.nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            dropout=self.dropout,
            bidirectional=self.bidirectional,
            batch_first=True,
        )

        self.fc1 = torch.nn.Linear(self.hidden_size, 10)
        self.fc2 = torch.nn.Linear(10, 10)
        self.fc3 = torch.nn.Linear(10, 10)
        self.fc4 = torch.nn.Linear(10, 1)
        self.relu = torch.nn.ReLU()
        self.sigmoid = torch.nn.Sigmoid()

    def _init_hidden(self):
        return (
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).to(device),
            Variable(
                torch.zeros(self.num_layers, self.batch_size, self.hidden_size)
            ).to(device),
        )

    def forward(self, x):

        # Encoder
        _init_hidden = self._init_hidden()
        lstm_out, _states = self.lstm(x)
        # Flatten the lstm output
        lstm_out = lstm_out[:, -1, :]  # batch_size, hidden_dim

        fc1 = self.fc1(lstm_out)
        fc2 = self.fc2(fc1)
        fc3 = self.fc3(fc2)
        fc4 = self.fc4(fc3)
        discriminator_out = self.sigmoid(fc4)  # Binary output layer/Real or Fake

        return discriminator_out


class MultiModelVAEGenerator(torch.nn.Module):
    def __init__(
        self,
        input_size: int,
        sequence_length: int,
        batch_size: int,
        num_future: int,
        hidden_size: int,
        num_layers: int,
        output_size: int,
        num_classes: int,
        latent_size: int,
        batch_first: bool,
        dropout: float,
        reset_state: bool,
        bidirectional: bool,
    ):

        super(MultiModelVAEGenerator, self).__init__()
        self.input_size = input_size
        self.sequence_length = sequence_length
        self.batch_size = batch_size
        self.latent_size = latent_size
        self.num_future = num_future
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.num_classes = num_classes
        self.batch_first = batch_first
        self.dropout = dropout
        self.reset_state = reset_state
        self.bidirectional = bidirectional

        # Network instances in the model
        self.encoder = LSTMEncoder(
            input_size=self.input_size,
            batch_size=self.batch_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            batch_first=self.batch_first,
            dropout=self.dropout,
            reset_state=True,
            bidirectional=self.bidirectional,
        )

        self.latent = DisentangledLatent(
            hidden_size=self.hidden_size,
            latent_size=self.latent_size,
            dropout=self.dropout,
        )

        self.decoder = LSTMDecoder(
            batch_size=self.batch_size,
            num_future=self.num_future,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            latent_size=self.latent_size,
            batch_first=self.batch_first,
            dropout=self.dropout,
            reset_state=True,
            bidirectional=self.bidirectional,
        )

        self.sampler = Sampler()

    def forward(self):
        pass


class MultiModelVAEGAN:
    """Wrap all the above defined model classes and train the model with respect to the defined loss function

        Args:
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

    def __init__(
        self,
        input_size: int,
        lstm_hidden_size: int,
        lstm_num_layers: int,
        num_classes: int,
        latent_size: int,
        dropout: float,
        epochs: int,
        batch_size: int,
        sequence_length: int,
        num_future: int,
    ):
        super(MultiModelVAEGAN, self).__init__()
        self.input_size = input_size
        self.lstm_hidden_size = lstm_hidden_size
        self.lstm_num_layers = lstm_num_layers
        self.num_classes = num_classes
        self.latent_size = latent_size
        self.dropout = dropout
        self.epochs = epochs
        self.batch_size = batch_size
        self.sequence_length = sequence_length
        self.num_future = num_future
        self.output_size = input_size
        self.reset_state = True
        self.bidirectional = False

        self.generator = MultiModelVAEGenerator(
            input_size=self.input_size,
            sequence_length=self.sequence_length,
            batch_size=self.batch_size,
            hidden_size=self.lstm_hidden_size,
            num_future=self.num_future,
            num_layers=self.lstm_num_layers,
            latent_size=self.latent_size,
            output_size=self.output_size,
            num_classes=self.num_classes,
            batch_first=True,
            dropout=self.dropout,
            reset_state=self.reset_state,
            bidirectional=False,
        )

        self.discriminator = LSTMDiscriminator(
            input_size=self.input_size,
            batch_size=self.batch_size,
            num_future=self.num_future,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            output_size=self.output_size,
            latent_size=self.latent_size,
            batch_first=self.batch_first,
            dropout=self.dropout,
            reset_state=self.reset_state,
            bidirectional=self.bidirectional,
        )

        self.classifier = MLPClassifier(
            hidden_size=self.hidden_size,
            num_classes=self.num_classes,
            latent_size=self.latent_size,
            dropout=self.dropout,
        )

        # Optimizers for each network in the model
        self.encoder_optimizer = torch.optim.Adam(self.generator.encoder.parameters())
        self.latent_optimizer = torch.optim.Adam(self.generator.latent.parameters())
        self.decoder_optimizer = torch.optim.Adam(self.generator.decoder.parameters())
        self.classifier_optimizer = torch.optim.Adam(self.classifier.parameters())
        self.discriminator_optimizer = torch.optim.Adam(self.discriminator.parameters())

        # Learning rate scheduler for each network in the model
        # NOTE: Scheduler metric is test set loss
        self.encoder_scheduler = ReduceLROnPlateau(
            self.encoder_optimizer, mode="max", factor=0.1, patience=2, verbose=True
        )
        self.decoder_scheduler = ReduceLROnPlateau(
            self.decoder_optimizer, mode="max", factor=0.1, patience=2, verbose=True
        )
        self.latent_scheduler = ReduceLROnPlateau(
            self.latent_optimizer, mode="max", factor=0.1, patience=2, verbose=True
        )
        self.classifier_scheduler = ReduceLROnPlateau(
            self.classifier_optimizer, mode="max", factor=0.1, patience=2, verbose=True
        )
        self.discriminator_scheduler = ReduceLROnPlateau(
            self.discriminator_optimizer,
            mode="max",
            factor=0.1,
            patience=2,
            verbose=True,
        )

        # Discriminator criterion
        self.discriminator_criterion = torch.nn.BCELoss()

        # Classifier loss function
        self.classifier_criterion = torch.nn.CrossEntropyLoss()

        # Decoder criterion
        self.huber_loss = torch.nn.SmoothL1Loss(reduction="sum")

        # Move the model to target device
        self.generator.to(device)
        self.discriminator.to(device)
        self.classifier.to(device)

        # Training mode: Switch from Generative to classifier training mode
        self.training_mode = "forecasting"

        # Training decoder and discriminator
        # Training
        self.real_label = 1
        self.fake_label = 0
        self.generated_label = 0

        # Noise and label
        self.noise = torch.FloatTensor(self.batch_size, self.latent_size)
        self.label = torch.FloatTensor(self.batch_size)
        self.noise = Variable(self.noise)
        self.label = Variable(self.label)

        # Discriminator loss constants
        self.gamma = 1.0

    def fit(self, train_loader, test_loader):

        for epoch in range(
            epochs * 2
        ):  # First half for generative model and next for classifier
            if epoch > 0:  # Initial step is to test and set LR schduler

                # Training
                self.generator.train()
                self.discriminator.train()
                self.classifier.train()

                discriminator_total_loss = 0  # Real + Fake
                vae_total_loss = 0  # VAE loss
                vae_disc_total_loss = 0  # VAE + Discriminator
                discriminator_total_vae_loss = 0  # Discriminator(VAE) loss
                total_classifier_loss = 0  # Classifier loss
                for idx, (data, target, category) in enumerate(train_loader):

                    data, target, category = (
                        data.float().to(device),
                        target.float().to(device),
                        category.to(device),
                    )

                    if self.training_mode == "forecasting":
                        for param in self.classifier.parameters():
                            param.requires_grad = False

                        for param in self.generator.encoder.parameters():
                            param.requires_grad = True

                        for param in self.generator.decoder.parameters():
                            param.requires_grad = True

                        for param in self.generator.latent.parameters():
                            param.requires_grad = True
                        ##########################################################
                        #  Update Discriminator network:
                        #  maximize log(D(x)) + log(D(G(z))) + log(1 - D(G(zp)))
                        ##########################################################
                        self.discriminator.zero_grad()
                        batch_size = data.size(0)

                        # Train Discriminator with Real data: log(D(x))
                        # (1) Feed the original data to the discriminator
                        output = self.discriminator(data)
                        # (2) The target label is real
                        label = torch.full(
                            (batch_size,),
                            self.real_label,
                            dtype=data.dtype,
                            device=device,
                        )
                        # (3) Measure the loss and backward pass the error using discriminator optimizer
                        discriminator_real_loss = self.discriminator_criterion(
                            output.squeeze(), label
                        )
                        discriminator_real_loss.backward()

                        # Train the discriminator with Real data: Autoencoded output: log(D(G(z)))
                        # Inference Network: input --> encoder-->latent-->decoder--> discriminator
                        # (1) Feed the input
                        encoder_out = self.generator.encoder(data)
                        latent_out_z, latent_out_zp, mu, logvar = self.generator.latent(
                            encoder_out, training=True
                        )
                        generator_out = self.generator.decoder(latent_out_zp)
                        # (2) Feed this generated data to the discriminator
                        output = self.discriminator(generator_out.detach())
                        # (3) The target label is fake!
                        label.data.fill_(self.real_label)
                        # (4) Measure the loss and backward pass the error
                        discriminator_real_aeloss = self.discriminator_criterion(
                            output.squeeze(), label
                        )
                        discriminator_real_aeloss.backward()

                        # Train the discriminator with Fake: Variational autoencoded output: log(1 - D(G(zp)))
                        # Generator network: noise --> latent-->sampler-->decoder--> discriminator
                        # (1) Feed the noise range N(0,1) to the latent and generate data.
                        self.noise.data.normal_(0, 1)
                        generator_out = self.generator.decoder(self.noise)
                        # (2) Feed this generated data to the discriminator
                        output = self.discriminator(generator_out.detach())
                        # (3) The target label is fake!
                        label.data.fill_(self.fake_label)
                        # (4) Measure the loss and backward pass the error
                        discriminator_fake_vaeloss = self.discriminator_criterion(
                            output.squeeze(), label
                        )
                        discriminator_fake_vaeloss.backward()
                        # (5) Optimize the discriminator parameters
                        self.discriminator_optimizer.step()
                        # For printing performance of discriminator
                        discriminator_loss = (
                            discriminator_real_loss
                            + discriminator_real_aeloss
                            + discriminator_fake_vaeloss
                        )

                        ##########################################################
                        # (2) Update Generative Network: VAE + Disc(VAE)
                        #  Maximize loglikelihood(P(X)) - KLD(Q(Z|X), P(Z)) - D(G(zp))
                        ##########################################################
                        self.generator.zero_grad()

                        # Forward step
                        # (1) Encoder
                        enc_out = self.generator.encoder(data)
                        # (2) Latent with sampler
                        z, zp, mu, logvar = self.generator.latent(
                            enc_out, training=True
                        )
                        # (3) Decoder
                        generator_out = self.generator.decoder(zp)
                        # (4) VAE Loss
                        KLD_element = (
                            mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                        )
                        KLD = torch.sum(KLD_element).mul_(
                            -0.5
                        )  # note mulitplied by -0.5
                        MSE = self.huber_loss(
                            generator_out, target
                        )  # Not true MSE loss
                        vae_loss = MSE + KLD
                        # (5) Backward pass
                        # vae_loss.backward()
                        # (6) Discriminator Loss
                        discriminator_output = self.discriminator(
                            generator_out.detach()
                        )
                        # (7) The target label is fake!
                        label.data.fill_(self.fake_label)
                        # (8) Measure the loss and backward pass the error;
                        # =1 if disc find it Fake, Punish the generator by discriminator's success weighted by constant gamma
                        discriminator_fake_vaeloss = self.discriminator_criterion(
                            discriminator_output.squeeze(), label
                        )

                        # VAE + Discriminator loss
                        vae_disc_loss = vae_loss + self.gamma * (
                            discriminator_fake_vaeloss
                        )
                        vae_disc_loss.backward()
                        # (9) Generator Optimizer step
                        self.encoder_optimizer.step()
                        self.latent_optimizer.step()
                        self.decoder_optimizer.step()

                        discriminator_total_loss += (
                            discriminator_loss.item()
                        )  # Real + Fake
                        vae_total_loss += vae_loss.item()  # VAE loss
                        vae_disc_total_loss += (
                            vae_disc_loss.item()
                        )  # VAE + Discriminator
                        discriminator_total_vae_loss += (
                            discriminator_fake_vaeloss.item()
                        )  # Discriminator(VAE) loss

                    print(
                        "Epoch {} | Discriminator Real+Fake Loss {} | VAE loss {} | VAE+Discriminator Loss {} | Discriminator(Generated) Loss {}".format(
                            epoch,
                            discriminator_total_loss / (idx + 1),
                            vae_total_loss / (idx + 1),
                            vae_disc_total_loss / (idx + 1),
                            discriminator_total_vae_loss / (idx + 1),
                        )
                    )
                    if self.training_mode != "forecasting":

                        self.classifier.zero_grad()

                        for param in self.classifier.parameters():
                            param.requires_grad = True

                        for param in self.generator.encoder.parameters():
                            param.requires_grad = False

                        for param in self.generator.decoder.parameters():
                            param.requires_grad = False

                        for param in self.generator.latent.parameters():
                            param.requires_grad = False

                        # input-->encoder-->latent-->classifier
                        # (1) Feed data to encoder
                        encoder_out = self.generator.encoder(data)
                        # (2) Latent without sampling
                        z, zp, mu, logvar = self.generator.latent(encoder_out)
                        # (3) Feed the latent vector to classifier
                        classifier_out = self.classifier(z.detach())
                        # (4) Cross entropy loss
                        classifier_loss = self.classifier_criterion(
                            classifier_out, category - 1
                        )
                        total_classifier_loss += classifier_loss.item()
                        # (5) Backward pass
                        classifier_loss.backward()
                        # (6) Classifier optimizer step
                        self.classifier_optimizer.step()

                        print(
                            "Epoch {} | {} loss {}".format(
                                epoch,
                                self.training_mode,
                                total_classifier_loss / (idx + 1),
                            )
                        )

                if epoch + 1 == epochs:
                    self.training_mode = "classification"

            # Testing
            if epoch % 2 == 0:
                with torch.no_grad():
                    self.generator.eval()
                    self.discriminator.eval()
                    self.classifier.eval()
                    test_loss_discrimination = 0  # Discriminator(VAE)
                    test_loss_forecasting = 0  # Huber(VAE, target)
                    test_loss_classification = (
                        0  # CrossEntropy(Classifier, target_category)
                    )

                    for idx, (data, target, category) in enumerate(list(test_loader)):
                        data, target, category = (
                            data.float().to(device),
                            target.float().to(device),
                            category.to(device),
                        )

                        # Forward step
                        # (1) Encoder
                        enc_out = self.generator.encoder(data)
                        # (2) Latent with sampler
                        z, zp, mu, logvar = self.generator.latent(
                            enc_out, training=False
                        )
                        # (3) Decoder
                        generator_out = self.generator.decoder(zp)
                        # (4) VAE Loss
                        KLD_element = (
                            mu.pow(2).add_(logvar.exp()).mul_(-1).add_(1).add_(logvar)
                        )
                        KLD = torch.sum(KLD_element).mul_(
                            -0.5
                        )  # note mulitplied by -0.5
                        MSE = self.huber_loss(
                            generator_out, target
                        )  # Not true MSE loss
                        vae_loss = MSE + KLD
                        # (5) Discriminator Loss
                        discriminator_output = self.discriminator(
                            generator_out.detach()
                        )
                        # (6) The target label is real!
                        self.label.data.fill_(self.real_label)
                        discriminator_real_vaeloss = self.discriminator_criterion(
                            discriminator_output.squeeze(), self.label
                        )

                        # VAE + Discriminator loss
                        vae_disc_loss = vae_loss - self.gamma * (
                            discriminator_real_vaeloss
                        )

                        # Classifier(z); Assumption, Discriminator should agree!
                        classifier_out = self.classifier(z.detach())

                        test_loss_discrimination += discriminator_real_vaeloss
                        test_loss_forecasting += vae_loss.item()
                        test_loss_classification += self.classifier_criterion(
                            classifier_out, category - 1
                        ).item()

                test_loss_forecasting /= len(test_loader.dataset)
                test_loss_discrimination /= len(test_loader.dataset)
                test_loss_classification /= len(test_loader.dataset)

                print(f"====> Test set Generator loss: {test_loss_forecasting:.4f}")
                print(f"Discriminator loss: {test_loss_forecasting:.4f}")
                print(f"Classifier loss: {test_loss_classification:.4f}")

