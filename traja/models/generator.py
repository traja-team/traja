"""Generate time series from model"""

import plotly.express as px
import torch
from traja.models.predictive_models.ae import MultiModelAE
from traja.models.generative_models.vae import MultiModelVAE
from traja.models.generative_models.vaegan import MultiModelVAEGAN
from traja.models.predictive_models.irl import MultiModelIRL
from traja.models.predictive_models.lstm import LSTM
from traja.models.utils import load
import matplotlib.pyplot as plt
import os

device = "cuda" if torch.cuda.is_available() else "cpu"


class Generate:
    def __init__(
        self,
        model_type: str = None,
        model_path: str = None,
        model_hyperparameters: dict = None,
        model: torch.nn.Module = None,
    ):
        """Generate a batch of future steps from a random latent state of Multi variate multi label models 

        Args:
            model_type (str, optional): Type of model ['vae','vaegan','custom']. Defaults to None.
            model_path (str, optional): Path to trained model (model.pt). Defaults to None.
            model_hyperparameters (dict, optional): [description]. Defaults to None.
            model (torch.nn.Module, optional): Custom model from user. Defaults to None
        """

        self.model_type = model_type
        self.model_path = model_path
        self.model_hyperparameters = model_hyperparameters

        # Batch size and time step size
        self.batch_size = self.model_hyperparameters["batch_size"]
        self.num_future = self.model_hyperparameters["num_future"]

        if self.model_type == "vae":
            self.model = MultiModelVAE(**self.model_hyperparameters)

        if self.model_type == "vaegan":
            self.model = MultiModelVAEGAN(**self.model_hyperparameters)

        if self.model_type == "custom":
            assert model is not None
            self.model = model(**self.model_hyperparameters)

    def generate_batch(self):

        # Load the model
        model = load(self.model, self.model_hyperparameters, self.model_path)

        if self.model_type == "vae":
            # Random noise
            z = (
                torch.empty(self.batch_size, self.model_hyperparameters.latent_size)
                .normal_(mean=0, std=0.1)
                .to(device)
            )
            # Generate trajectories from the noise
            out = model.decoder(z, self.num_future).cpu().detach().numpy()
            out = out.reshape(out.shape[0] * out.shape[1], out.shape[2])
            try:
                cat = model.classifier(z)
                print(
                    "IDs in this batch of synthetic data", torch.max(cat, 1).indices + 1
                )
            except:
                pass

            plt.figure(figsize=(12, 4))
            plt.plot(out[:, 0], label="Generated x: Longitude")
            plt.plot(out[:, 1], label="Generated y: Latitude")
            plt.legend()

            fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 5), sharey=True)
            fig.set_size_inches(20, 5)

            for i in range(2):
                for j in range(5):
                    ax[i, j].plot(
                        out[:, 0][
                            (i + j) * self.num_future : (i + j) * self.num_future
                            + self.num_future
                        ],
                        out[:, 1][
                            (i + j) * self.num_future : (i + j) * self.num_future
                            + self.num_future
                        ],
                        label="Animal ID {}".format(
                            (torch.max(cat, 1).indices + 1).detach()[i + j]
                        ),
                        color="g",
                    )
                    ax[i, j].legend()
            plt.show()

            return out

        elif self.model_type == "vaegan" or "custom":
            return NotImplementedError

    def generate_timeseries(num_steps):
        return NotImplementedError


class Predict:
    def __init__(
        self,
        model_type: str = None,
        model_path: str = None,
        model_hyperparameters: dict = None,
        model: torch.nn.Module = None,
    ):
        """Generate a batch of future steps from a random latent state of Multi variate multi label models 

        Args:
            model_type (str, optional): Type of model ['ae','irl','lstm','custom']. Defaults to None.
            model_path (str, optional): [description]. Defaults to None.
            model_hyperparameters (dict, optional): [description]. Defaults to None.
            model (torch.nn.Module, optional): Custom model from user. Defaults to None
        """

        self.model_type = model_type
        self.model_path = model_path
        self.model_hyperparameters = model_hyperparameters

        # Batch size and time step size
        self.batch_size = self.model_hyperparameters["batch_size"]
        self.num_future = self.model_hyperparameters["num_future"]

        if self.model_type == "ae":
            self.model = MultiModelAE(**self.model_hyperparameters)

        if self.model_type == "lstm":
            self.model = LSTM(**self.model_hyperparameters)

        if self.model_type == "irl":
            self.model = MultiModelIRL(**self.model_hyperparameters)

        if self.model_type == "custom":
            assert model is not None
            self.model = model(**self.model_hyperparameters)
