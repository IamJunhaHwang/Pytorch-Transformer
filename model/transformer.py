import torch.nn as nn
import torch


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

    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        # c: context, z: some sentence
        out = self.decoder(tgt, encoder_out, src_mask, tgt_mask)
        return out

    def make_subsequent_mask(size):
        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

        return subsequent_mask == 0

    def forward(self, src, tgt, src_mask, tgt_mask):
        """
        :param x:
        :param z:
        :return:
        """
        context = self.encode(src, src_mask)
        target = self.decode(context, src_mask, tgt, tgt_mask)

        return target
