import torch.nn as nn
import torch


class Transformer(nn.Module):
    """
    Transformer Body - Encoder-Decoder Architecture
    """

    def __init__(self, encoder, decoder, src_embed, tgt_embed, generator):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.src_embed = src_embed
        self.tgt_embed = tgt_embed
        self.generator = generator

    def encode(self, src, src_mask):
        out = self.encoder(self.src_embed(src), src_mask)
        return out

    def decode(self, encoder_out, src_mask, tgt, tgt_mask):
        # c: context, z: some sentence
        out = self.decoder(self.tgt_embed(tgt), encoder_out, src_mask, tgt_mask)
        return out

    def make_subsequent_mask(self, size):
        "Mask out subsequent positions."

        attn_shape = (1, size, size)
        subsequent_mask = torch.triu(torch.ones(attn_shape), diagonal=1).type(torch.uint8)

        return subsequent_mask == 0

    def forward(self, src, tgt, src_mask, tgt_mask):
        context = self.encode(src, src_mask)
        target = self.decode(context, src_mask, tgt, tgt_mask)

        return target
