import numpy as np
import torch
from torch import autograd

from src.models.base_net import BaseNet
from src.models.pi_net_original import DEQNet
from src.utils.config import default_config

class PINet(BaseNet, DEQNet):
    """Original DeepThinking Network 2D model class, but with modifications for convenience:
    - Added methods inherited from BaseNet
    - Modified forward method to not return all_outputs
    """

    def __init__(self, *args, **kwargs):
        width = kwargs['width']
        in_channels = kwargs['in_channels']
        config = kwargs['config']
        # Pass different arguments to the super classes
        BaseNet.__init__(self)
        DEQNet.__init__(self, width, config, in_channels=in_channels)

    def name(self):
        return 'pi_net'
    
    def input_to_latent(self, inputs, grad=False):
        # Reset deq normalization after loading weights
        self.deq._reset(inputs)

        with torch.no_grad() if not grad else torch.enable_grad():
            latents = self.projection(inputs)
        return latents

    def latent_forward(self, latents, inputs, iters=1, grad=False):
        with torch.no_grad() if not grad else torch.enable_grad():

            if default_config['threshold'] == 'default':
                threshold = self.f_thres
            elif default_config['threshold'] == 'max_iter':
                threshold = max(iters)
            elif type(default_config['threshold']) == int:
                threshold = default_config['threshold']

            if type(iters) == int:
                self.layer_idx = [iters]
                latents = self.projection(inputs) if latents is None else latents
                func = lambda latents: self.deq(latents, inputs)
                result = self.f_solver(func, latents, threshold=threshold, stop_mode=self.stop_mode, layer_idx=self.layer_idx, name="forward")
                latents = result['result']
                return latents

            elif type(iters) == list:
                self.layer_idx = iters
                latents = self.projection(inputs) if latents is None else latents
                func = lambda latents: self.deq(latents, inputs)
                result = self.f_solver(func, latents, threshold=threshold, stop_mode=self.stop_mode, layer_idx=self.layer_idx, name="forward")
                latents_series = torch.stack(result['interm_vals'], dim=0)
                return latents_series

    def latent_to_output(self, latents, grad=False):
        with torch.no_grad() if not grad else torch.enable_grad():        
            if latents.dim() == 4:
                outputs = self.head(latents)
            elif latents.dim() == 5:
                outputs = torch.zeros((latents.size(0), latents.size(1), 2, latents.size(3), latents.size(4))).to(latents.device)
                for i in range(latents.size(1)):
                    outputs[:, i] = self.head(latents[:, i])
        return outputs

    def output_to_prediction(self, outputs, inputs, masked=True, grad=False):
        with torch.no_grad() if not grad else torch.enable_grad():
            if outputs.dim() == 4:
                unmasked_predictions = torch.argmax(outputs, dim=1)
                if masked:
                    mask, _ = torch.max(inputs, dim=1)
                    predictions = unmasked_predictions * mask
                else:
                    predictions = unmasked_predictions
                return predictions
            elif outputs.dim() == 5:
                unmasked_predictions = torch.argmax(outputs, dim=2)
                if masked:
                    mask, _ = torch.max(inputs, dim=1)
                    mask = mask.unsqueeze(0)
                    predictions = unmasked_predictions * mask
                else:
                    predictions = unmasked_predictions
                return predictions
    
    def predict(self, inputs, iters, grad=False):
        return super().predict(inputs, iters, grad=grad)