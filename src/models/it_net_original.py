"""
An alternate maze-solving model, with latent space equal to output space.

Based on code by Amandin Chyba.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.autograd as autograd
from torch.linalg import norm
import numpy as np

class BasicBlockV2(nn.Module):
    """Basic residual block class"""

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlockV2, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(nn.Conv2d(in_planes, planes, kernel_size=1, stride=stride, bias=False))

    def forward(self, u):
        out = self.conv1(u)
        out = F.relu(out)
        out = self.conv2(out) 
        out += self.shortcut(u)
        out = F.relu(out)
        return out

class ImplicitModule(nn.Module):
    def __init__(self, block, num_blocks, width, in_channels):
        super(ImplicitModule, self).__init__()
        self.in_planes = int(width)
        self.num_blocks = num_blocks 
        self.recur_inj = nn.Conv2d(2*in_channels, width, kernel_size=3, stride=1, padding=1, bias=False)
        self.recur_layer = self._make_layer(block, width, num_blocks, stride=1)
        conv2 = nn.Conv2d(width, 32, kernel_size=3, stride=1, padding=1, bias=False)
        conv3 = nn.Conv2d(32, 8, kernel_size=3, stride=1, padding=1, bias=False)
        conv4 = nn.Conv2d(8, in_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.head = nn.Sequential(conv2, nn.ReLU(), conv3, nn.ReLU(), conv4)

    def _make_layer(self, block, planes, num_blocks, stride=1):
        strides = [stride] + [1] * (num_blocks-1)
        layers = []
        for strd in strides:
            layers.append(block(self.in_planes, planes, strd))
            self.in_planes = planes
        return nn.ModuleList(layers)
    
    def forward(self, u, x):
        u = self.recur_inj(torch.cat([u, x], 1))
        for i in range(self.num_blocks):
            u = self.recur_layer[i](u)
        u = self.head(u)
        return u

class ITNetOriginal(nn.Module):
    """Modified ResNet model class"""

    def __init__(self, num_blocks=2, width=128, in_channels=3):
        super(ITNetOriginal, self).__init__()
        self.width = int(width)
        self.num_blocks = num_blocks
        self.in_channels = in_channels
        self.latent = ImplicitModule(BasicBlockV2, num_blocks, width, in_channels)
    
    def name(self):
        return 'it_net'
    
    def forward(self, x, u, iter, series=False):
        all_outputs = torch.zeros((x.size(0), iter+1, self.in_channels, 
                                   x.size(2), x.size(3))).to(x.device)
        all_outputs[:,0] = u

        with torch.no_grad():
            for i in range(iter-1):
                u = self.latent(u, x)
                all_outputs[:,i+1] = u
        
        u = u.detach().requires_grad_()
        u = self.latent(u, x)
        all_outputs[:,-1] = u
        
        if series:
            return all_outputs
        return u