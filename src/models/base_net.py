import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from abc import ABC, abstractmethod

from src.utils.config import default_config

# For debugging
from src.utils.device import print_params

class BaseNet(torch.nn.Module, ABC):
    """ Base class for maze networks, containing necessary methods common to all models"""

    def __init__(self, config=default_config):
        torch.nn.Module.__init__(self)

        self.eval() # Set the model to evaluation mode by default
    
    @abstractmethod
    def name(self):
        pass

    @abstractmethod
    def input_to_latent(self, inputs, grad=False):
        pass

    @abstractmethod
    def latent_forward(self, latents, inputs, iters, grad=False):
        pass
    
    @abstractmethod
    def latent_to_output(self, latents, grad=False):
        pass
    
    @abstractmethod
    def output_to_prediction(self, outputs, grad=False):
        pass

    def predict(self, inputs, iters, grad=False):
        """Compute predictions from the inputs"""
        latents = self.input_to_latent(inputs)
        latents = self.latent_forward(latents, inputs, iters)
        outputs = self.latent_to_output(latents)
        predictions = self.output_to_prediction(outputs, inputs)

        return predictions
