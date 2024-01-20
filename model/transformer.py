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
        eout = self.encoder(x)
        return eout

    def decode(self, c, z):
        # c: context, z: some sentence

        dout = self.decoder(c, z)
        return dout

    def forward(self, x, z, mask):
        """
        :param x:
        :param z:
        :return:
        """
        context = self.encode(x, mask)
        target = self.decode(context, z)

        return target