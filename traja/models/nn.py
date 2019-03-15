#! /usr/local/env python3
"""Pytorch visualization code modified from Chad Jensen's implementation
(https://discuss.pytorch.org/t/lstm-for-sequence-prediction/22021/3)."""
try:
    import torch
except ImportError:
    raise ImportError("pytorch is not installed. Install it via pytorch.org")
import traja
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import logging

nb_steps = 10

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


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
            interactive = False
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
