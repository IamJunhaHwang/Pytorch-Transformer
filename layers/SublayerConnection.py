import torch.nn as nn
from Transformer.layers.LayerNorm import LayerNorm

class SublayerConnection(nn.Module):
    """
    Residual Connection + Layer Normalization
    """

    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        return x + self.dropout(sublayer(self.norm(x)))