#! /usr/local/env python3
"""Pytorch visualization code modified from Chad Jensen's implementation
(https://discuss.pytorch.org/t/lstm-for-sequence-prediction/22021/3)."""
import logging

import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:
    raise ImportError(
        "Missing optional dependency 'pytorch'. Install it via pytorch.org"
    )
import torch.nn as nn
import torch.optim as optim
<<<<<<< Updated upstream

=======
import os
import pandas as pd
from time import time
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
>>>>>>> Stashed changes

nb_steps = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


<<<<<<< Updated upstream
=======
class TimeseriesDataset(Dataset):
    # Loads the dataset and splits it into equally sized chunks.
    # Whereas this can lead to uneven training data,
    # with sufficiently long sequence lengths the
    # bias should even out.

    def __init__(self, data_frame, sequence_length):
        self.data = data_frame
        self.sequence_length = sequence_length

    def __len__(self):
        return int((self.data.shape[0]) / self.sequence_length)

    def __getitem__(self, index):
        data = self.data[index * self.sequence_length: (index + 1) * self.sequence_length]
        return data
    
def get_transformed_timeseries_dataloaders(data_frame: pd.DataFrame, sequence_length: int, train_fraction: float, batch_size:int):
    """ Scale the timeseries dataset and generate train and test dataloaders

    Args:
        data_frame (pd.DataFrame): Dataset 
        sequence_length (int): Sequence length of time series for a single gradient step 
        train_fraction (float): train data vs test data ratio
        batch_size (int): Batch size of single gradient measure

    Returns:
        train_loader (Dataloader)
        validation_loader(Dataloader)
        scaler (instance): Data scaler instance
    """
    # Dataset transformation
    scaler = MinMaxScaler(copy=False)
    # scaler.fit(data_frame.values)
    # scaled_dataset = scaler.fit_transform(data_frame.values)
    dataset_length = int(data_frame.values.shape[0] / sequence_length)
    indices = list(range(dataset_length))
    split = int(np.floor(train_fraction * dataset_length))
    train_indices, val_indices = indices[split:], indices[:split]

    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    dataset = TimeseriesDataset(data_frame.values, sequence_length)
    
    train_loader = DataLoader(dataset, batch_size=batch_size,
                              sampler=train_sampler)
    validation_loader = DataLoader(dataset, batch_size=batch_size,
                                   sampler=valid_sampler)
    train_loader.name = "time_series"
    return train_loader, validation_loader, scaler

def get_transformed_timeseries_dataloaders_(data_frame: pd.DataFrame, sequence_length: int, train_fraction: float, batch_size:int):
    """ Scale the timeseries dataset and generate train and test dataloaders

    Args:
        data_frame (pd.DataFrame): Dataset 
        sequence_length (int): Sequence length of time series for a single gradient step 
        train_fraction (float): train data vs test data ratio
        batch_size (int): Batch size of single gradient measure

    Returns:
        train_loader (Dataloader)
        validation_loader(Dataloader)
        scaler (instance): Data scaler instance
    """
    # Split the data into train and test:
    train_dataset, test_dataset = train_test_split(data_frame.values, train_size=train_fraction)
    dataset_length = int(data_frame.values.shape[0] / sequence_length)
    indices = list(range(dataset_length))
    split = int(np.floor(train_fraction * dataset_length))
    train_indices, val_indices = indices[split:], indices[:split]
    
    # Creating PT data samplers and loaders:
    train_sampler = SubsetRandomSampler(train_indices)
    valid_sampler = SubsetRandomSampler(val_indices)

    # Dataset transformation; Train and test should have different scaler instances;
    train_dataset_scaler = MinMaxScaler(copy=False)
    scaled_train_dataset = train_dataset_scaler.fit_transform(train_dataset)
    test_dataset_scaler = MinMaxScaler(copy=False)
    scaled_test_dataset = test_dataset_scaler.fit_transform(test_dataset)

    # Convert transformed data into 3D tensor (batch_size,sequence_length, num_features)
    train_dataset_3d = TimeseriesDataset(scaled_train_dataset, sequence_length)
    test_dataset_3d = TimeseriesDataset(scaled_test_dataset, sequence_length)
    
    train_loader = DataLoader(train_dataset_3d, batch_size=batch_size,
                              sampler=train_sampler, drop_last=False )
    validation_loader = DataLoader(test_dataset_3d, batch_size=batch_size,
                                   sampler=valid_sampler, drop_last=False )
    train_loader.name = "time_series"
    return train_loader, validation_loader, train_dataset_scaler, test_dataset_scaler
class LossMseWarmup:
    """
    Calculate the Mean Squared Error between y_true and y_pred,
    but ignore the beginning "warmup" part of the sequences.

    y_true is the desired output.
    y_pred is the model's output.
    """
    def __init__(self, warmup_steps=50):
        self.warmup_steps = warmup_steps

    def __call__(self, y_pred, y_true):

        y_true_slice = y_true[:, self.warmup_steps:, :]
        y_pred_slice = y_pred[:, self.warmup_steps:, :]

        # Calculate the Mean Squared Error and use it as loss.
        mse = torch.mean(torch.square(y_true_slice - y_pred_slice))

        return mse


class Trainer:
    def __init__(self, model,
                 train_loader,
                 test_loader,
                 epochs=200,
                 batch_size=60,
                 run_id=0,
                 logs_dir='logs',
                 device='cpu',
                 optimizer='None',
                 plot=True,
                 downsampling=None,
                 warmup_steps=50):
        self.device = device
        self.model = model
        self.epochs = epochs
        self.plot = plot

        self.train_loader = train_loader
        self.test_loader = test_loader

        self.warmup_steps = warmup_steps

        self.criterion = LossMseWarmup(self.warmup_steps)
        print('Checking for optimizer for {}'.format(optimizer))
        if optimizer == "adam":
            print('Using adam')
            self.optimizer = optim.Adam(model.parameters())
        elif optimizer == "adam_lr":
            print("Using adam with higher learning rate")
            self.optimizer = optim.Adam(model.parameters(), lr=0.01)
        elif optimizer == 'adam_lr2':
            print('Using adam with to large learning rate')
            self.optimizer = optim.Adam(model.parameters(), lr=0.0001)
        elif optimizer == "SGD":
            print('Using SGD')
            self.optimizer = optim.SGD(model.parameters(), momentum=0.9, weight_decay=5e-4)
        elif optimizer == "LRS":
            print('Using LRS')
            self.optimizer = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=5e-4)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, self.epochs // 3)
        elif optimizer == "radam":
            print('Using radam')
            self.optimizer = RAdam(model.parameters())
        elif optimizer == "RMSprop":
            print('Using RMSprop')
            self.optimizer = optim.RMSprop(model.parameters())
        else:
            raise ValueError('Unknown optimizer {}'.format(optimizer))
        self.opt_name = optimizer
        save_dir = os.path.join(logs_dir, model.name, train_loader.name)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.savepath = os.path.join(save_dir, f'{model.name}_bs{batch_size}_e{epochs}_dspl{downsampling}_id{run_id}.csv')
        self.experiment_done = False
        if os.path.exists(self.savepath):
            trained_epochs = len(pd.read_csv(self.savepath, sep=';'))

            if trained_epochs >= epochs:
                self.experiment_done = True
                print(f'Experiment Logs for the exact same experiment with identical run_id was detected, training will be skipped, consider using another run_id')
        if os.path.exists((self.savepath.replace('.csv', '.pt'))):
            self.model.load_state_dict(torch.load(self.savepath.replace('.csv', '.pt'))['model_state_dict'])
            self.model = self.model.to(self.device)

            self.optimizer.load_state_dict(torch.load(self.savepath.replace('.csv', '.pt'))['optimizer'])
            self.start_epoch = torch.load(self.savepath.replace('.csv', '.pt'))['epoch'] + 1
        else:

            self.start_epoch = 0
            self.model = self.model.to(self.device)


    def _infer_initial_epoch(self, savepath):
        if not os.path.exists(savepath):
            return 0
        else:
            df = pd.read_csv(savepath, sep=';', index_col=0)
            print(len(df)+1)
            return len(df)

    def train(self):
        if self.experiment_done:
            return
        for epoch in range(self.start_epoch, self.epochs):

            print('Start training epoch', epoch)
            print("{} Epoch {}, training loss: {}".format(datetime.now(), epoch, self.train_epoch()))
            self.test(epoch=epoch)
            if self.opt_name == "LRS":
                print('LRS step')
                self.lr_scheduler.step()
        return self.savepath+'.csv'

    def train_epoch(self):
        self.model.train()
        total = 0
        running_loss = 0
        old_time = time()
        for batch, data in enumerate(self.train_loader):
            if batch % 10 == 0 and batch != 0:
                print(batch, 'of', len(self.train_loader), 'processing time', time()-old_time, 'loss:', running_loss/total)
                old_time = time()
            inputs= data.to(self.device).float()

            self.optimizer.zero_grad()
            outputs = self.model(inputs)
            # For time series step prediction, targets are inputs
            loss = self.criterion(outputs, inputs)
            loss.backward()
            self.optimizer.step()

            running_loss += loss.item()
            # Increment number of batches
            total += 1
        return running_loss/total

    def test(self, epoch, save=True):
        self.model.eval()
        total = 0
        test_loss = 0
        with torch.no_grad():
            for batch, data in enumerate(self.test_loader):
                if batch % 10 == 0:
                    print('Processing eval batch', batch,'of', len(self.test_loader))
                inputs = data.to(self.device).float()

                outputs = self.model(inputs)
                # For time series step prediction, targets are inputs 
                loss = self.criterion(outputs, inputs)
                total += 1
                test_loss += loss.item()

        if save:
            torch.save({
                'model_state_dict': self.model.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'epoch': epoch,
                'test_loss': test_loss / total
            }, self.savepath.replace('.csv', '.pt'))
        return test_loss / total


>>>>>>> Stashed changes
class LSTM(nn.Module):
    def __init__(self):
        super(LSTM, self).__init__()
        self.lstm = nn.LSTM(2, 100)
        self.head = nn.Linear(100, 2)

    def forward(self, x):
        outputs, states = self.lstm(x)
        outputs = outputs.reshape(x.shape[0] * x.shape[1], -1)
        pred = self.head(outputs)

        return pred


class TrajectoryLSTM:
    def __init__(
        self, xy, nb_steps=10, epochs=1000, batch_size=1, criterion=nn.MSELoss()
    ):
        fig, ax = plt.subplots(2, 1)
        self.fig = fig
        self.ax = ax
        assert xy.shape[1] is 2, f"xy should be an N x 2 array, but is {xy.shape}"
        self.xy = xy
        self.nb_steps = nb_steps
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = criterion
        self.rnn = LSTM()

    def load_batch(self, batch_size=32):
        t_1_b = np.zeros((self.nb_steps, self.batch_size, 2))
        t_b = np.zeros((self.nb_steps * self.batch_size, 2))

        inds = np.random.randint(0, len(self.xy) - self.nb_steps, (self.batch_size))
        for i, ind in enumerate(inds):
            t_1_b[:, i] = self.xy[ind : ind + self.nb_steps]
            t_b[i * nb_steps : (i + 1) * self.nb_steps] = self.xy[
                ind + 1 : ind + nb_steps + 1
            ]
        return torch.from_numpy(t_1_b).float(), torch.from_numpy(t_b).float()

    def train(self):
        self.mean_loss = 0.0
        for epoch in range(1, self.epochs + 1):
            t_1_b, t_b = self.load_batch(self.batch_size)

            def closure():
                global loss
                optimizer.zero_grad()
                pred = self.rnn(t_1_b)
                shaped_pred = pred.reshape(-1, 2)
                loss = self.criterion(abs(shaped_pred), abs(t_b))
                loss.backward()

                return loss

            optimizer = optim.Adam(self.rnn.parameters(), 1e-3)
            optimizer.step(closure)
            self.mean_loss += loss.item()

            if epoch % 100 == 0:
                print("Epoch: {} | Loss: {:.6f}".format(epoch, self.mean_loss))
                self.mean_loss = 0

    def savefig(self, filepath):
        self.fig.savefig(filepath)

    def _plot(self):
        t_1_b, t_b = self.load_batch(1)
        pred = self.rnn(t_1_b).detach().numpy().reshape(-1, 2)

        real = t_1_b.numpy().reshape(-1, 2)
        x, y = self.xy.T
        self.ax[0].plot(x, y, label="Real")
        self.ax[0].plot(real[:, 0], real[:, 1], label="Real batch")
        self.ax[0].plot(pred[:, 0], pred[:, 1], label="Pred")

        self.ax[1].scatter(real[:, 0], real[:, 1], label="Real")
        self.ax[1].scatter(pred[:, 0], pred[:, 1], label="Pred")

        for a in self.ax:
            a.legend()

    def plot(self, interactive=True):
        if interactive and (plt.get_backend() == "agg"):
            logging.ERROR("Not able to use interactive plotting in mpl `agg` mode.")
            # interactive = False
        elif interactive:
            while True:
                for a in self.ax:
                    a.clear()
                self._plot()
                plt.pause(1)
                plt.show(block=False)
        else:
            self._plot()
            return self.fig


import torch
import torch.nn as nn


def make_mlp(dim_list, activation="relu", batch_norm=True, dropout=0):
    layers = []
    for dim_in, dim_out in zip(dim_list[:-1], dim_list[1:]):
        layers.append(nn.Linear(dim_in, dim_out))
        if batch_norm:
            layers.append(nn.BatchNorm1d(dim_out))
        if activation == "relu":
            layers.append(nn.ReLU())
        elif activation == "leakyrelu":
            layers.append(nn.LeakyReLU())
        if dropout > 0:
            layers.append(nn.Dropout(p=dropout))
    return nn.Sequential(*layers)


def get_noise(shape, noise_type):
    if noise_type == "gaussian":
        return torch.randn(*shape).cuda()
    elif noise_type == "uniform":
        return torch.rand(*shape).sub_(0.5).mul_(2.0).cuda()
    raise ValueError('Unrecognized noise type "%s"' % noise_type)


class Encoder(nn.Module):
    """Encoder is part of both TrajectoryGenerator and
    TrajectoryDiscriminator"""

    def __init__(
        self, embedding_dim=64, h_dim=64, mlp_dim=1024, num_layers=1, dropout=0.0
    ):
        super(Encoder, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers

        self.encoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        self.spatial_embedding = nn.Linear(2, embedding_dim)

    def init_hidden(self, batch):
        return (
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
            torch.zeros(self.num_layers, batch, self.h_dim).cuda(),
        )

    def forward(self, obs_traj):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        Output:
        - final_h: Tensor of shape (self.num_layers, batch, self.h_dim)
        """
        # Encode observed Trajectory
        batch = obs_traj.size(1)
        obs_traj_embedding = self.spatial_embedding(obs_traj.view(-1, 2))
        obs_traj_embedding = obs_traj_embedding.view(-1, batch, self.embedding_dim)
        state_tuple = self.init_hidden(batch)
        output, state = self.encoder(obs_traj_embedding, state_tuple)
        final_h = state[0]
        return final_h


class Decoder(nn.Module):
    """Decoder is part of TrajectoryGenerator"""

    def __init__(
        self,
        seq_len,
        embedding_dim=64,
        h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        pooling_type="pool_net",
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super(Decoder, self).__init__()

        self.seq_len = seq_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.embedding_dim = embedding_dim
        self.pool_every_timestep = pool_every_timestep

        self.decoder = nn.LSTM(embedding_dim, h_dim, num_layers, dropout=dropout)

        if pool_every_timestep:
            if pooling_type == "pool_net":
                self.pool_net = PoolHiddenNet(
                    embedding_dim=self.embedding_dim,
                    h_dim=self.h_dim,
                    mlp_dim=mlp_dim,
                    bottleneck_dim=bottleneck_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                )
            elif pooling_type == "spool":
                self.pool_net = SocialPooling(
                    h_dim=self.h_dim,
                    activation=activation,
                    batch_norm=batch_norm,
                    dropout=dropout,
                    neighborhood_size=neighborhood_size,
                    grid_size=grid_size,
                )

            mlp_dims = [h_dim + bottleneck_dim, mlp_dim, h_dim]
            self.mlp = make_mlp(
                mlp_dims, activation=activation, batch_norm=batch_norm, dropout=dropout
            )

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.hidden2pos = nn.Linear(h_dim, 2)

    def forward(self, last_pos, last_pos_rel, state_tuple, seq_start_end):
        """
        Inputs:
        - last_pos: Tensor of shape (batch, 2)
        - last_pos_rel: Tensor of shape (batch, 2)
        - state_tuple: (hh, ch) each tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - pred_traj: tensor of shape (self.seq_len, batch, 2)
        """
        batch = last_pos.size(0)
        pred_traj_fake_rel = []
        decoder_input = self.spatial_embedding(last_pos_rel)
        decoder_input = decoder_input.view(1, batch, self.embedding_dim)

        for _ in range(self.seq_len):
            output, state_tuple = self.decoder(decoder_input, state_tuple)
            rel_pos = self.hidden2pos(output.view(-1, self.h_dim))
            curr_pos = rel_pos + last_pos

            if self.pool_every_timestep:
                decoder_h = state_tuple[0]
                pool_h = self.pool_net(decoder_h, seq_start_end, curr_pos)
                decoder_h = torch.cat([decoder_h.view(-1, self.h_dim), pool_h], dim=1)
                decoder_h = self.mlp(decoder_h)
                decoder_h = torch.unsqueeze(decoder_h, 0)
                state_tuple = (decoder_h, state_tuple[1])

            embedding_input = rel_pos

            decoder_input = self.spatial_embedding(embedding_input)
            decoder_input = decoder_input.view(1, batch, self.embedding_dim)
            pred_traj_fake_rel.append(rel_pos.view(batch, -1))
            last_pos = curr_pos

        pred_traj_fake_rel = torch.stack(pred_traj_fake_rel, dim=0)
        return pred_traj_fake_rel, state_tuple[0]


class PoolHiddenNet(nn.Module):
    """Pooling module as proposed in our paper"""

    def __init__(
        self,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
    ):
        super(PoolHiddenNet, self).__init__()

        self.mlp_dim = 1024
        self.h_dim = h_dim
        self.bottleneck_dim = bottleneck_dim
        self.embedding_dim = embedding_dim

        mlp_pre_dim = embedding_dim + h_dim
        mlp_pre_pool_dims = [mlp_pre_dim, 512, bottleneck_dim]

        self.spatial_embedding = nn.Linear(2, embedding_dim)
        self.mlp_pre_pool = make_mlp(
            mlp_pre_pool_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tensor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch
        - end_pos: Tensor of shape (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, bottleneck_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_end_pos = end_pos[start:end]
            # Repeat -> H1, H2, H1, H2
            curr_hidden_1 = curr_hidden.repeat(num_ped, 1)
            # Repeat position -> P1, P2, P1, P2
            curr_end_pos_1 = curr_end_pos.repeat(num_ped, 1)
            # Repeat position -> P1, P1, P2, P2
            curr_end_pos_2 = self.repeat(curr_end_pos, num_ped)
            curr_rel_pos = curr_end_pos_1 - curr_end_pos_2
            curr_rel_embedding = self.spatial_embedding(curr_rel_pos)
            mlp_h_input = torch.cat([curr_rel_embedding, curr_hidden_1], dim=1)
            curr_pool_h = self.mlp_pre_pool(mlp_h_input)
            curr_pool_h = curr_pool_h.view(num_ped, num_ped, -1).max(1)[0]
            pool_h.append(curr_pool_h)
        pool_h = torch.cat(pool_h, dim=0)
        return pool_h


class SocialPooling(nn.Module):
    """Current state of the art pooling mechanism:
    http://cvgl.stanford.edu/papers/CVPR16_Social_LSTM.pdf"""

    def __init__(
        self,
        h_dim=64,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        neighborhood_size=2.0,
        grid_size=8,
        pool_dim=None,
    ):
        super(SocialPooling, self).__init__()
        self.h_dim = h_dim
        self.grid_size = grid_size
        self.neighborhood_size = neighborhood_size
        if pool_dim:
            mlp_pool_dims = [grid_size * grid_size * h_dim, pool_dim]
        else:
            mlp_pool_dims = [grid_size * grid_size * h_dim, h_dim]

        self.mlp_pool = make_mlp(
            mlp_pool_dims, activation=activation, batch_norm=batch_norm, dropout=dropout
        )

    def get_bounds(self, ped_pos):
        top_left_x = ped_pos[:, 0] - self.neighborhood_size / 2
        top_left_y = ped_pos[:, 1] + self.neighborhood_size / 2
        bottom_right_x = ped_pos[:, 0] + self.neighborhood_size / 2
        bottom_right_y = ped_pos[:, 1] - self.neighborhood_size / 2
        top_left = torch.stack([top_left_x, top_left_y], dim=1)
        bottom_right = torch.stack([bottom_right_x, bottom_right_y], dim=1)
        return top_left, bottom_right

    def get_grid_locations(self, top_left, other_pos):
        cell_x = torch.floor(
            ((other_pos[:, 0] - top_left[:, 0]) / self.neighborhood_size)
            * self.grid_size
        )
        cell_y = torch.floor(
            ((top_left[:, 1] - other_pos[:, 1]) / self.neighborhood_size)
            * self.grid_size
        )
        grid_pos = cell_x + cell_y * self.grid_size
        return grid_pos

    def repeat(self, tensor, num_reps):
        """
        Inputs:
        -tensor: 2D tensor of any shape
        -num_reps: Number of times to repeat each row
        Outpus:
        -repeat_tensor: Repeat each row such that: R1, R1, R2, R2
        """
        col_len = tensor.size(1)
        tensor = tensor.unsqueeze(dim=1).repeat(1, num_reps, 1)
        tensor = tensor.view(-1, col_len)
        return tensor

    def forward(self, h_states, seq_start_end, end_pos):
        """
        Inputs:
        - h_states: Tesnsor of shape (num_layers, batch, h_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - end_pos: Absolute end position of obs_traj (batch, 2)
        Output:
        - pool_h: Tensor of shape (batch, h_dim)
        """
        pool_h = []
        for _, (start, end) in enumerate(seq_start_end):
            start = start.item()
            end = end.item()
            num_ped = end - start
            grid_size = self.grid_size * self.grid_size
            curr_hidden = h_states.view(-1, self.h_dim)[start:end]
            curr_hidden_repeat = curr_hidden.repeat(num_ped, 1)
            curr_end_pos = end_pos[start:end]
            curr_pool_h_size = (num_ped * grid_size) + 1
            curr_pool_h = curr_hidden.new_zeros((curr_pool_h_size, self.h_dim))
            # curr_end_pos = curr_end_pos.data
            top_left, bottom_right = self.get_bounds(curr_end_pos)

            # Repeat position -> P1, P2, P1, P2
            curr_end_pos = curr_end_pos.repeat(num_ped, 1)
            # Repeat bounds -> B1, B1, B2, B2
            top_left = self.repeat(top_left, num_ped)
            bottom_right = self.repeat(bottom_right, num_ped)

            grid_pos = self.get_grid_locations(top_left, curr_end_pos).type_as(
                seq_start_end
            )
            # Make all positions to exclude as non-zero
            # Find which peds to exclude
            x_bound = (curr_end_pos[:, 0] >= bottom_right[:, 0]) + (
                curr_end_pos[:, 0] <= top_left[:, 0]
            )
            y_bound = (curr_end_pos[:, 1] >= top_left[:, 1]) + (
                curr_end_pos[:, 1] <= bottom_right[:, 1]
            )

            within_bound = x_bound + y_bound
            within_bound[0 :: num_ped + 1] = 1  # Don't include the ped itself
            within_bound = within_bound.view(-1)

            # This is a tricky way to get scatter add to work. Helps me avoid a
            # for loop. Offset everything by 1. Use the initial 0 position to
            # dump all uncessary adds.
            grid_pos += 1
            total_grid_size = self.grid_size * self.grid_size
            offset = torch.arange(
                0, total_grid_size * num_ped, total_grid_size
            ).type_as(seq_start_end)

            offset = self.repeat(offset.view(-1, 1), num_ped).view(-1)
            grid_pos += offset
            grid_pos[within_bound != 0] = 0
            grid_pos = grid_pos.view(-1, 1).expand_as(curr_hidden_repeat)

            curr_pool_h = curr_pool_h.scatter_add(0, grid_pos, curr_hidden_repeat)
            curr_pool_h = curr_pool_h[1:]
            pool_h.append(curr_pool_h.view(num_ped, -1))

        pool_h = torch.cat(pool_h, dim=0)
        pool_h = self.mlp_pool(pool_h)
        return pool_h


class TrajectoryGenerator(nn.Module):
    """Modified from @agrimgupta92's https://github.com/agrimgupta92/sgan/blob/master/sgan/models.py."""

    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        encoder_h_dim=64,
        decoder_h_dim=128,
        mlp_dim=1024,
        num_layers=1,
        noise_dim=(0,),
        noise_type="gaussian",
        noise_mix_type="ped",
        pooling_type=None,
        pool_every_timestep=True,
        dropout=0.0,
        bottleneck_dim=1024,
        activation="relu",
        batch_norm=True,
        neighborhood_size=2.0,
        grid_size=8,
    ):
        super(TrajectoryGenerator, self).__init__()

        if pooling_type and pooling_type.lower() == "none":
            pooling_type = None

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.mlp_dim = mlp_dim
        self.encoder_h_dim = encoder_h_dim
        self.decoder_h_dim = decoder_h_dim
        self.embedding_dim = embedding_dim
        self.noise_dim = noise_dim
        self.num_layers = num_layers
        self.noise_type = noise_type
        self.noise_mix_type = noise_mix_type
        self.pooling_type = pooling_type
        self.noise_first_dim = 0
        self.pool_every_timestep = pool_every_timestep
        self.bottleneck_dim = 1024

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=encoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        self.decoder = Decoder(
            pred_len,
            embedding_dim=embedding_dim,
            h_dim=decoder_h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            pool_every_timestep=pool_every_timestep,
            dropout=dropout,
            bottleneck_dim=bottleneck_dim,
            activation=activation,
            batch_norm=batch_norm,
            pooling_type=pooling_type,
            grid_size=grid_size,
            neighborhood_size=neighborhood_size,
        )

        if pooling_type == "pool_net":
            self.pool_net = PoolHiddenNet(
                embedding_dim=self.embedding_dim,
                h_dim=encoder_h_dim,
                mlp_dim=mlp_dim,
                bottleneck_dim=bottleneck_dim,
                activation=activation,
                batch_norm=batch_norm,
            )
        elif pooling_type == "spool":
            self.pool_net = SocialPooling(
                h_dim=encoder_h_dim,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
                neighborhood_size=neighborhood_size,
                grid_size=grid_size,
            )

        if self.noise_dim[0] == 0:
            self.noise_dim = None
        else:
            self.noise_first_dim = noise_dim[0]

        # Decoder Hidden
        if pooling_type:
            input_dim = encoder_h_dim + bottleneck_dim
        else:
            input_dim = encoder_h_dim

        if self.mlp_decoder_needed():
            mlp_decoder_context_dims = [
                input_dim,
                mlp_dim,
                decoder_h_dim - self.noise_first_dim,
            ]

            self.mlp_decoder_context = make_mlp(
                mlp_decoder_context_dims,
                activation=activation,
                batch_norm=batch_norm,
                dropout=dropout,
            )

    def add_noise(self, _input, seq_start_end, user_noise=None):
        """
        Inputs:
        - _input: Tensor of shape (_, decoder_h_dim - noise_first_dim)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Outputs:
        - decoder_h: Tensor of shape (_, decoder_h_dim)
        """
        if not self.noise_dim:
            return _input

        if self.noise_mix_type == "global":
            noise_shape = (seq_start_end.size(0),) + self.noise_dim
        else:
            noise_shape = (_input.size(0),) + self.noise_dim

        if user_noise is not None:
            z_decoder = user_noise
        else:
            z_decoder = get_noise(noise_shape, self.noise_type)

        if self.noise_mix_type == "global":
            _list = []
            for idx, (start, end) in enumerate(seq_start_end):
                start = start.item()
                end = end.item()
                _vec = z_decoder[idx].view(1, -1)
                _to_cat = _vec.repeat(end - start, 1)
                _list.append(torch.cat([_input[start:end], _to_cat], dim=1))
            decoder_h = torch.cat(_list, dim=0)
            return decoder_h

        decoder_h = torch.cat([_input, z_decoder], dim=1)

        return decoder_h

    def mlp_decoder_needed(self):
        if (
            self.noise_dim
            or self.pooling_type
            or self.encoder_h_dim != self.decoder_h_dim
        ):
            return True
        else:
            return False

    def forward(self, obs_traj, obs_traj_rel, seq_start_end, user_noise=None):
        """
        Inputs:
        - obs_traj: Tensor of shape (obs_len, batch, 2)
        - obs_traj_rel: Tensor of shape (obs_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch.
        - user_noise: Generally used for inference when you want to see
        relation between different types of noise and outputs.
        Output:
        - pred_traj_rel: Tensor of shape (self.pred_len, batch, 2)
        """
        batch = obs_traj_rel.size(1)
        # Encode seq
        final_encoder_h = self.encoder(obs_traj_rel)
        # Pool States
        if self.pooling_type:
            end_pos = obs_traj[-1, :, :]
            pool_h = self.pool_net(final_encoder_h, seq_start_end, end_pos)
            # Construct input hidden states for decoder
            mlp_decoder_context_input = torch.cat(
                [final_encoder_h.view(-1, self.encoder_h_dim), pool_h], dim=1
            )
        else:
            mlp_decoder_context_input = final_encoder_h.view(-1, self.encoder_h_dim)

        # Add Noise
        if self.mlp_decoder_needed():
            noise_input = self.mlp_decoder_context(mlp_decoder_context_input)
        else:
            noise_input = mlp_decoder_context_input
        decoder_h = self.add_noise(noise_input, seq_start_end, user_noise=user_noise)
        decoder_h = torch.unsqueeze(decoder_h, 0)

        decoder_c = torch.zeros(self.num_layers, batch, self.decoder_h_dim).cuda()

        state_tuple = (decoder_h, decoder_c)
        last_pos = obs_traj[-1]
        last_pos_rel = obs_traj_rel[-1]
        # Predict Trajectory

        decoder_out = self.decoder(last_pos, last_pos_rel, state_tuple, seq_start_end)
        pred_traj_fake_rel, final_decoder_h = decoder_out

        return pred_traj_fake_rel


class TrajectoryDiscriminator(nn.Module):
    def __init__(
        self,
        obs_len,
        pred_len,
        embedding_dim=64,
        h_dim=64,
        mlp_dim=1024,
        num_layers=1,
        activation="relu",
        batch_norm=True,
        dropout=0.0,
        d_type="local",
    ):
        super(TrajectoryDiscriminator, self).__init__()

        self.obs_len = obs_len
        self.pred_len = pred_len
        self.seq_len = obs_len + pred_len
        self.mlp_dim = mlp_dim
        self.h_dim = h_dim
        self.d_type = d_type

        self.encoder = Encoder(
            embedding_dim=embedding_dim,
            h_dim=h_dim,
            mlp_dim=mlp_dim,
            num_layers=num_layers,
            dropout=dropout,
        )

        real_classifier_dims = [h_dim, mlp_dim, 1]
        self.real_classifier = make_mlp(
            real_classifier_dims,
            activation=activation,
            batch_norm=batch_norm,
            dropout=dropout,
        )
        if d_type == "global":
            mlp_pool_dims = [h_dim + embedding_dim, mlp_dim, h_dim]
            self.pool_net = PoolHiddenNet(
                embedding_dim=embedding_dim,
                h_dim=h_dim,
                mlp_dim=mlp_pool_dims,
                bottleneck_dim=h_dim,
                activation=activation,
                batch_norm=batch_norm,
            )

    def forward(self, traj, traj_rel, seq_start_end=None):
        """
        Inputs:
        - traj: Tensor of shape (obs_len + pred_len, batch, 2)
        - traj_rel: Tensor of shape (obs_len + pred_len, batch, 2)
        - seq_start_end: A list of tuples which delimit sequences within batch
        Output:
        - scores: Tensor of shape (batch,) with real/fake scores
        """
        final_h = self.encoder(traj_rel)
        # Note: In case of 'global' option we are using start_pos as opposed to
        # end_pos. The intution being that hidden state has the whole
        # trajectory and relative postion at the start when combined with
        # trajectory information should help in discriminative behavior.
        if self.d_type == "local":
            classifier_input = final_h.squeeze()
        else:
            classifier_input = self.pool_net(final_h.squeeze(), seq_start_end, traj[0])
        scores = self.real_classifier(classifier_input)
        return scores
