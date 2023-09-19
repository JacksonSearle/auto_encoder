import torch
from torch import nn


class Encoder(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.layers = nn.ModuleList()
        for i in range(len(layers)-1):
            self.layers.append(nn.Linear(layers[i], layers[i+1]))
    def forward(self, x):
        for i, layer in enumerate(self.layers):
            if i == len(self.layers)-1:
                return layer(x)
            else:
                x = torch.relu(layer(x))