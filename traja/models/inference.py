"""Generate time series from model"""

import matplotlib.pyplot as plt
import numpy as np
import torch

from traja.models.generative_models.vae import MultiModelVAE
from traja.models.generative_models.vaegan import MultiModelVAEGAN
from traja.models.predictive_models.ae import MultiModelAE
from traja.models.predictive_models.lstm import LSTM

device = "cuda" if torch.cuda.is_available() else "cpu"


class Generator:
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

        if self.model_type == "vae":
            self.model = MultiModelVAE(**self.model_hyperparameters)

        if self.model_type == "vaegan":
            self.model = MultiModelVAEGAN(**self.model_hyperparameters)

        if self.model_type == "custom":
            assert model is not None
            self.model = model(**self.model_hyperparameters)

        (
            self.generated_category,
            self.generated_data,
        ) = (None, None)

    def generate(self, num_steps, classify=True, scaler=None, plot_data=True):

        self.model.to(device)
        if self.model_type == "vae":
            # Random noise
            z = (
                torch.empty(
                    self.model_hyperparameters["batch_size"],
                    self.model_hyperparameters["latent_size"],
                )
                .normal_(mean=0, std=0.1)
                .to(device)
            )
            # Generate trajectories from the noise
            self.generated_data = (
                self.model.decoder(z, num_steps).cpu().detach().numpy()
            )
            self.generated_data = self.generated_data.reshape(
                self.generated_data.shape[0] * self.generated_data.shape[1],
                self.generated_data.shape[2],
            )
            if classify:
                try:
                    self.generated_category = self.model.classifier(z)
                    print(
                        "IDs in this batch of synthetic data",
                        torch.max(self.generated_category, 1).indices.detach() + 1,
                    )
                except Exception as error:
                    print("Classifier not found: " + repr(error))

            # Scale original data and generated data

            # Rescaling predicted data
            self.generated_data = scaler.inverse_transform(self.generated_data)

            # TODO:Depreself.generated_categoryed;Slicing the data into batches
            self.generated_data = np.array(
                [
                    self.generated_data[i : i + num_steps]
                    for i in range(0, len(self.generated_data), num_steps)
                ]
            )

            # Reshape [batch_size*num_steps,input_dim]
            self.generated_data = self.generated_data.reshape(
                self.generated_data.shape[0] * self.generated_data.shape[1],
                self.generated_data.shape[2],
            )

            if plot_data:
                fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 5), sharey=True)
                fig.set_size_inches(20, 5)

                for i in range(2):
                    for j in range(5):
                        if classify:
                            try:
                                label = "Animal ID {}".format(
                                    (
                                        torch.max(self.generated_category, 1).indices
                                        + 1
                                    ).detach()[i + j]
                                )
                            except Exception as error:
                                print("Classifier not found:" + repr(error))
                        else:
                            label = ""
                        ax[i, j].plot(
                            self.generated_data[:, 0][
                                (i + j) * num_steps : (i + j) * num_steps + num_steps
                            ],
                            self.generated_data[:, 1][
                                (i + j) * num_steps : (i + j) * num_steps + num_steps
                            ],
                            label=label,
                            color="g",
                        )
                        ax[i, j].legend()
                plt.show()

            return self.generated_data

        elif self.model_type == "vaegan" or "custom":
            return NotImplementedError

class Predictor:
    def __init__(
        self,
        model_type: str = None,
        model_path: str = None,
        model_hyperparameters: dict = None,
        model: torch.nn.Module = None,
    ):
        """Generate a batch of future steps from a random latent state of Multi variate multi label models

        Args:
            model_type (str, optional): Type of model ['ae','lstm','custom']. Defaults to None.
            model_path (str, optional): [description]. Defaults to None.
            model_hyperparameters (dict, optional): [description]. Defaults to None.
            model (torch.nn.Module, optional): Custom model from user. Defaults to None
        """

        self.model_type = model_type
        self.model_path = model_path
        self.model_hyperparameters = model_hyperparameters

        if self.model_type == "ae":
            self.model = MultiModelAE(
                num_regressor_layers=2,
                regressor_hidden_size=32,
                num_regressor_parameters=3,
                **self.model_hyperparameters,
            )

        if self.model_type == "lstm":
            self.model = LSTM(**self.model_hyperparameters)

        if self.model_type == "custom":
            assert model is not None
            self.model = model(**self.model_hyperparameters)

        (
            self.predicted_category,
            self.target_data,
            self.target_data_,
            self.predicted_data,
            self.predicted_data_,
        ) = (None, None, None, None, None)

    def predict(self, data_loader, num_steps, scaler, classify=True):
        """[summary]

        Args:
            data_loader ([type]): [description]
            num_steps ([type]): [description]
            scaler (dict): Scalers of the target data. This scale the model predictions to the scale of the target (future steps).
                        : This scaler will be returned by the traja data preprocessing and loading helper function.
            classify (bool, optional): [description]. Defaults to True.

        Returns:
            [type]: [description]
        """

        self.model.to(device)
        if self.model_type == "ae":
            for data, target, self.generated_category in data_loader:
                data, target = data.to(device), target.to(device)
                self.predicted_data = self.model.encoder(data)
                self.predicted_data = self.model.latent(self.generated_data)
                self.predicted_data = self.model.decoder(self.generated_data)
                if classify:
                    self.generated_category = self.model.classifier(self.predicted_data)

                target = target.cpu().detach().numpy()
                target = target.reshape(
                    target.shape[0] * target.shape[1], target.shape[2]
                )
                self.predicted_data = self.predicted_data.cpu().detach().numpy()
                self.predicted_data = self.predicted_data.reshape(
                    self.predicted_data.shape[0] * self.predicted_data.shape[1],
                    self.predicted_data.shape[2],
                )

                # Rescaling predicted data
                for i in range(self.predicted_data.shape[1]):
                    s_s = scaler.inverse_transform(
                        self.predicted_data[:, i].reshape(-1, 1)
                    )
                    s_s = np.reshape(s_s, len(s_s))
                    self.predicted_data[:, i] = s_s

                predicted_data = np.array(
                    [
                        self.predicted_data[i : i + num_steps]
                        for i in range(0, len(self.predicted_data), num_steps)
                    ]
                )
                # Rescaling target data
                self.target_data = target.copy()
                for i in range(self.target_data.shape[1]):
                    s_s = scaler.inverse_transform(
                        self.target_data[:, i].reshape(-1, 1)
                    )
                    s_s = np.reshape(s_s, len(s_s))
                    self.target_data[:, i] = s_s
                self.target_data = np.array(
                    [
                        self.target_data[i : i + num_steps]
                        for i in range(0, len(self.target_data), num_steps)
                    ]
                )

                # Reshape [batch_size*num_steps,input_dim]
                predicted_data_ = predicted_data.reshape(
                    self.predicted_data.shape[0] * self.predicted_data.shape[1],
                    self.predicted_data.shape[2],
                )
                self.target_data_ = self.target_data.reshape(
                    self.target_data.shape[0] * self.target_data.shape[1],
                    self.target_data.shape[2],
                )

                fig, ax = plt.subplots(nrows=2, ncols=5, figsize=(16, 5), sharey=False)
                fig.set_size_inches(40, 20)
                for i in range(2):
                    for j in range(5):
                        ax[i, j].plot(
                            predicted_data_[:, 0][
                                (i + j) * num_steps : (i + j) * num_steps + num_steps
                            ],
                            predicted_data_[:, 1][
                                (i + j) * num_steps : (i + j) * num_steps + num_steps
                            ],
                            label=f"Predicted ID {self.generated_categoryegory[i + j]}",
                        )

                        ax[i, j].plot(
                            self.target_data_[:, 0][
                                (i + j) * num_steps : (i + j) * num_steps + num_steps
                            ],
                            self.target_data_[:, 1][
                                (i + j) * num_steps : (i + j) * num_steps + num_steps
                            ],
                            label=f"Target ID {self.generated_category[i + j]}",
                            color="g",
                        )
                        ax[i, j].legend()

                        plt.autoscale(True, axis="y", tight=False)
                plt.show()

                return predicted_data

        elif self.model_type == "vaegan" or "custom":
            return NotImplementedError
