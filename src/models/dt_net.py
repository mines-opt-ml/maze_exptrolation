import torch

from src.models.base_net import BaseNet
from src.models.dt_net_original import DTNetOriginal

class DTNet(BaseNet, DTNetOriginal):
    """Original DeepThinking Network 2D model class, but with modifications for convenience:
    - Added methods inherited from BaseNet
    - Modified forward method to not return all_outputs
    """

    def __init__(self, *args, **kwargs):
        BaseNet.__init__(self, *args, **kwargs)
        DTNetOriginal.__init__(self, *args, **kwargs)

    def name(self):
        return 'dt_net'

    def forward(self, x, iters_to_do, interim_thought=None, **kwargs):
        """Forward pass of the network"""
        
        initial_thought = self.projection(x)

        if interim_thought is None:
            interim_thought = initial_thought

        #all_outputs = torch.zeros((x.size(0), iters_to_do, 2, x.size(2), x.size(3))).to(x.device)

        for i in range(iters_to_do):
            if self.recall:
                interim_thought = torch.cat([interim_thought, x], 1)
            interim_thought = self.recur_block(interim_thought)
            out = self.head(interim_thought)
            #all_outputs[:, i] = out

        #if self.training:
        return out, interim_thought

        #return all_outputs
    
    def input_to_latent(self, inputs, grad=False):
        with torch.no_grad() if not grad else torch.enable_grad():
            latents = self.projection(inputs)
        return latents

    def latent_forward(self, latents, inputs, iters=1, grad=False):
        """ 
        If iters is int, return latents after iter iterations.
        If iters is list, add iters dimension to latents and return latents after each iteration in iters.
        """ 
        with torch.no_grad() if not grad else torch.enable_grad():
            if type(iters) == int:
                for _ in range(iters):
                    latents = self.recur_block(torch.cat([latents, inputs], 1))
                return latents
            elif type(iters) == list:
                latents_series = torch.zeros((len(iters),) + tuple(latents.shape), dtype=latents.dtype, device=latents.device)
                for i in range(1, max(iters)+1):
                    latents = self.recur_block(torch.cat([latents, inputs], 1))
                    if i in iters:
                        latents_series[iters.index(i)] = latents
                return latents_series

    def latent_to_output(self, latents, grad=False):
        with torch.no_grad() if not grad else torch.enable_grad():
            if latents.dim() == 3 or latents.dim() == 4:
                outputs = self.head(latents)
            elif latents.dim() == 5:
                outputs = torch.zeros((latents.size(0), latents.size(1), 2, latents.size(3), latents.size(4))).to(latents.device)
                for i in range(latents.size(1)):
                    outputs[:, i] = self.head(latents[:, i])
        return outputs

    def output_to_prediction(self, outputs, inputs, grad=False, masked=True):
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