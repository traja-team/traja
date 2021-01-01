import torch
import matplotlib.pyplot as plt
import numpy as np
import collections
from numpy import math
import os, json


class TimeDistributed(torch.nn.Module):
    """ Time distributed wrapper compatible with linear/dense pytorch layer modules"""

    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x):

        # Linear layer accept 2D input
        if len(x.size()) <= 2:
            return self.module(x)

        # Squash samples and timesteps into a single axis
        x_reshape = x.contiguous().view(
            -1, x.size(-1)
        )  # (samples * timesteps, input_size)
        out = self.module(x_reshape)

        # We have to reshape Y back to the target shape
        if self.batch_first:
            out = out.contiguous().view(
                x.size(0), -1, out.size(-1)
            )  # (samples, timesteps, output_size)
        else:
            out = out.view(
                -1, x.size(1), out.size(-1)
            )  # (timesteps, samples, output_size)

        return out


def save(model, hyperparameters, PATH=None):
    """Save the trained model(.pth) along with its hyperparameters as a json (hyper.json) at the user defined Path
    Parameters:
    -----------
        model (torch.nn.Module): Trained Model
        hyperparameters(dict): Hyperparameters of the model
        PATH (str): Directory path to save the trained model and its hyperparameters
    Returns: 
    ---------
        None
    """
    if not isinstance(hyperparameters, dict):
        raise Exception("Invalid argument, hyperparameters must be dict")
    # Save
    if PATH is None:
        PATH = os.getcwd() + "model.pt"
    torch.save(model.state_dict(), PATH)
    _dir, _ = os.path.split(PATH)
    if hyperparameters is not None:
        with open("./hypers.json", "w") as fp:
            json.dump(hyperparameters, fp, sort_keys=False)
    print("Model saved at {} ".format(_dir))


def load(model, PATH=None):
    """Load trained model from PATH using the model_hyperparameters saved in the
    Parameters:
    -----------
        model (torch.nn.Module): Type of the model ['ae','vae','vaegan','irl','lstm','custom']
        model_hyperparameters (dict): Dictionary of hyperparameters used to initiate model
        PATH (str): Directory path of the model
    Returns:
    ---------
        model(torch.nn.module): Model
    """
    # Hyperparameters
    if PATH is None:
        PATH = os.getcwd() + "/model.pt"
        print(f"Model loaded from {PATH}")
    else:
        raise Exception("Model not found at " f"{PATH}")

    # Get hyperparameters from the model path
    PATH, _ = os.path.split(PATH)
    try:
        with open(PATH + "/hypers.json") as f:
            hypers = json.load(f)
    except:
        raise Exception("Hyper parameters not found at " f"{PATH}")

    # Load
    model = model(**hypers)
    # Load state of the model
    model.load_state_dict(torch.load(PATH))

    return model


def read_hyperparameters(hyperparameter_json):
    """Read the json file and return the hyperparameters as dict

    Args:
        hyperparameter_json (json): Json file containing the hyperparameters of the trained model

    Returns:
        [dict]: Python dictionary of the hyperparameters 
    """
    with open(hyperparameter_json) as f_in:
        return json.load(f_in)

