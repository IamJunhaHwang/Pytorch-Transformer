import torch.nn as nn


class Transformer(nn.Module):
    """
    Transformer Body - Encoder-Decoder Architecture
    """

    def __init__(self, encoder, decoder):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def encode(self, src, src_mask):
        out = self.encoder(src, src_mask)
        return out

    def decode(self, c, z):
        # c: context, z: some sentence
        out = self.decoder(c, z)
        return out

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param x:
        :param z:
        :return:
        """
        context = self.encode(src, src_mask)
        target = self.decode(context, tgt_mask)

        return target
