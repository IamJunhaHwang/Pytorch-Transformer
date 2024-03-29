import torch.nn as nn
import copy
from utils.LayerNorm import LayerNorm


class Encoder(nn.Module):
    """
    The encoder is composed of a stack of N identical layers.
    """

    def __init__(self, encoder_block, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(N)])  # N-Encoder Blocks
        self.norm = LayerNorm(encoder_block.size)

    def forward(self, x, mask):
        """
        :param x:
        :return:
        """
        for layer in self.layers:
            x = layer(x, mask)

        # why norm??
        return self.norm(x)