import torch.nn as nn
import copy
from utils.LayerNorm import LayerNorm


class Decoder(nn.Module):
    """
    The decoder is also composed of a stack of N identical layers.
    """

    def __init__(self, decoder_block, N):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(decoder_block) for _ in range(N)])  # N-Encoder Blocks
        self.norm = LayerNorm(decoder_block.size)

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        """
        :param x:
        :param encoder_out:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        for layer in self.layers:
            x = layer(x, encoder_out, src_mask, tgt_mask)

        return self.norm(x)