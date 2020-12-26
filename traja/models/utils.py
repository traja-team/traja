import torch
import matplotlib.pyplot as plt
import numpy as np
import collections
from numpy import math


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


def save_model(model, PATH):
    """[summary]

    Args:
        model ([type]): [description]
        PATH ([type]): [description]
    """

    # PATH = "state_dict_model.pt"
    # Save
    torch.save(model.state_dict(), PATH)
    print("Model saved at {}".format(PATH))


def load_model(model, model_hyperparameters, PATH):
    """[summary]

    Args:
        model ([type]): [description]
        model_hyperparameters ([type]): [description]
        PATH ([type]): [description]

    Returns:
        [type]: [description]
    """
    # Load
    model = model(model_hyperparameters)
    model.load_state_dict(torch.load(PATH))

    return model
