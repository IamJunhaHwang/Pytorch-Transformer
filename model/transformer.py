import torch.nn as nn


class Transformer(nn.Module):
    """
    Transformer Body - Encoder-Decoder Architecture
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, x):
        out = self.encoder(x)
        return out

    def decode(self, c, z):
        # c: context, z: some sentence
        out = self.decoder(c, z)
        return out

    def forward(self, x, z, mask):
        """
        :param x:
        :param z:
        :return:
        """
        context = self.encode(x, mask)
        target = self.decode(context, z)

        return target