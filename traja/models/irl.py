""" Implementation of Inverse Reinforcement Learning algorithm for Time series"""
import torch

class MultiModelIRL(torch.nn.Module):
    def __init__(self,*model_hyperparameters, **kwargs):
        super(MultiModelIRL,self).__init__()
        
        for dictionary in model_hyperparameters:
            for key in dictionary:
                    setattr(self, key, dictionary[key])
        for key in kwargs:
            setattr(self, key, kwargs[key])
            
    def __new__(cls):
        pass
    
    def forward(self, *input:None, **kwargs: None):
        return NotImplementedError
        
        