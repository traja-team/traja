"""This module contains the variational autoencoders - GAN and its variants
1. classic VAE-GAN
2. *****

Loss functions:
1. MSE
2. Huber Loss"""

import torch


class MultiModelVAEGAN(torch.nn.Module):
    def __init__(self, *model_hyperparameters, **kwargs):
        super(MultiModelVAEGAN, self).__init__()

        for dictionary in model_hyperparameters:
            for key in dictionary:
                setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])

    def __new__(cls):
        pass

    def forward(self, *input: None, **kwargs: None):
        return NotImplementedError
