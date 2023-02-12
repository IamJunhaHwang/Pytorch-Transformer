import torch
import torch.nn as nn

class Transformer(nn.Module):
    """
    Transformer Body
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        eout = self.encoder(x)
        return eout

    def decode(self, c, z):
        # c: context, z: some sentence

        dout = self.decoder(c, z)
        return dout

    def forward(self, x, z):
        context = self.encode(x)
        target = self.decode(context, z)

        return target
