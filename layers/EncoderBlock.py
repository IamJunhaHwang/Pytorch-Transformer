import torch.nn as nn
from Transformer.layers.SublayerConnection import SublayerConnection

class EncoderBlock(nn.Module):

    def __init__(self, size, attention, ffnn, dropout):
        super(EncoderBlock, self).__init__()

        self.size = size
        self.attention = attention
        self.ffnn = ffnn
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.attention(x, x, x, mask))

        return self.sublayer[1](x, self.ffnn)