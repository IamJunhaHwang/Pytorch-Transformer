import torch.nn as nn
from Transformer.layers.SublayerConnection import SublayerConnection

class DecoderBlock(nn.Module):

    def __init__(self, size, attention, masked_attention, ffnn, dropout):
        super(DecoderBlock, self).__init__()

        self.size = size
        self.attention = attention
        self.masked_attention = masked_attention
        self.ffnn = ffnn
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, memory, src_mask, tgt_mask):
        """

        :param x:
        :param memory:
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        m = memory
        x = self.sublayer[0](x, lambda x: self.masked_attention(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.attention(x, m, m, src_mask))

        return self.sublayer[2](x, self.ffnn)