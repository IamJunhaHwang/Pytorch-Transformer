import torch.nn as nn
from layers.LayerNorm import LayerNorm


class SublayerConnection(nn.Module):
    """
    Residual Connection + Layer Normalization

    Note that it was updated about below things after writing the paper.
    - Ref: https://github.com/harvardnlp/annotated-transformer/issues/92

    1. output is changed to x + SubLayer(LayerNorm(x))
    2. Layer Normalization is added the final outputs of encoder/decoder at last
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # According to the paper, apply dropout to the output of each sub-layer
        # x + dropout(Sublayer(LayerNorm(x)))
        return x + self.dropout(sublayer(self.norm(x)))