from encoder import Encoder
from decoder import Decoder
from torch import nn

class Model(nn.Module):
    def __init__(self, layers):
        super().__init__()
        self.encoder = Encoder(layers)
        self.decoder = Decoder(layers)
    def forward(self, x):
        encoded = self.encoder(x)
        reconstructed = self.decoder(encoded)
        return encoded, reconstructed