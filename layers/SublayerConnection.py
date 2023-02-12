import torch.nn as nn
from Transformer.layers.LayerNorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    Residual Connection + Layer Normalization

    Note that it was updated after writing the paper about below things

    1. output is changed to x + SubLayer(LayerNorm(x))
    2. Layer Normalization is added the final outputs of encoder/decoder at last
    """

    def __init__(self, size, d_rate):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(d_rate)

    def forward(self, x, sublayer):
        # According to the paper, apply dropout to the output of each sub-layer
        return x + self.dropout(sublayer(self.norm(x)))