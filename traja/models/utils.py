import torch
import matplotlib.pyplot as plt
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
        x_reshape = x.contiguous().view(-1, x.size(-1))  # (samples * timesteps, input_size)
        out = self.module(x_reshape)

        # We have to reshape Y back to the target shape
        if self.batch_first:
            out = out.contiguous().view(x.size(0), -1, out.size(-1))  # (samples, timesteps, output_size)
        else:
            out = out.view(-1, x.size(1), out.size(-1))  # (timesteps, samples, output_size)

        return out
    
def get_optimizers(model_type, model, lr=0.0001):
    r"""Optimizers for each network in the model

    Args:
        model_type ([type]): [description]
        model ([type]): [description]
        lr (float, optional): [description]. Defaults to 0.0001.

    Returns:
        [type]: [description]
    """
    
    if model_type == 'ae' or 'vae':
        # Optimizers for each network in the model
        encoder_optimizer = torch.optim.Adam(model.encoder.parameters(), lr=lr)
        latent_optimizer = torch.optim.Adam(model.latent.parameters(), lr=lr)
        decoder_optimizer = torch.optim.Adam(model.decoder.parameters(), lr=lr)
        classifier_optimizer = torch.optim.Adam(model.classifier.parameters(), lr=lr)
        return [encoder_optimizer, latent_optimizer, decoder_optimizer, classifier_optimizer]
    
    elif model_type == 'vaegan':
        return NotImplementedError
    
    else: # LSTM
        return NotImplementedError

def get_lrschedulers(model_type, encoder_optimizer, decoder_optimizer, latent_optimizer, classifier_optimizer, factor=0.1, patience=10):
    
    r"""Learning rate scheduler for each network in the model
    NOTE: Scheduler metric should be test set loss

    Args:
        model_type ([type]): [description]
        encoder_optimizer ([type]): [description]
        decoder_optimizer ([type]): [description]
        latent_optimizer ([type]): [description]
        classifier_optimizer ([type]): [description]
        factor (float, optional): [description]. Defaults to 0.1.
        patience (int, optional): [description]. Defaults to 10.

    Returns:
        [type]: [description]
    """
    
    if model_type == 'ae' or 'vae':
        encoder_scheduler = ReduceLROnPlateau(encoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True)
        decoder_scheduler = ReduceLROnPlateau(decoder_optimizer, mode='max', factor=factor, patience=patience, verbose=True)
        latent_scheduler = ReduceLROnPlateau(latent_optimizer, mode='max', factor=factor, patience=patience, verbose=True)
        classifier_scheduler = ReduceLROnPlateau(classifier_optimizer, mode='max', factor=factor, patience=patience, verbose=True)
        return [encoder_scheduler, decoder_scheduler, latent_scheduler, classifier_scheduler]
    
    elif model_type == 'vaegan':
        return NotImplementedError
    
    else: # LSTM
        return NotImplementedError
    
    
    
def save_model(model, PATH):
    r"""[summary]

    Args:
        model ([type]): [description]
        PATH ([type]): [description]
    """
    
    # PATH = "state_dict_model.pt"
    # Save
    torch.save(model.state_dict(), PATH)
    print('Model saved at {}'.format(PATH))
    
def load_model(model,model_hyperparameters, PATH):
    r"""[summary]

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






