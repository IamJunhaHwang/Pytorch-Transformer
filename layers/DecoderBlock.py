import torch.nn as nn
from utils.SublayerConnection import SublayerConnection


class DecoderBlock(nn.Module):
    """
    In addition to the two sub-layers in each encoder layer, the decoder inserts a third sub-layer, which
    performs multi-head attention over the output of the encoder stack. Similar to the encoder, we employ
    residual connections around each of the sub-layers, followed by layer normalization.
    """

    def __init__(self, size, self_attention, cross_attention, feed_forward, dropout):
        super(DecoderBlock, self).__init__()

        self.size = size
        self.self_attention = self_attention
        self.cross_attention = cross_attention
        self.ffnn = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(3)])

    def forward(self, x, encoder_out, src_mask, tgt_mask):
        """
        :param x:
        :param encoder_out: Encoder output
        :param src_mask:
        :param tgt_mask:
        :return:
        """
        e = encoder_out
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.cross_attention(x, e, e, src_mask))

        return self.sublayer[2](x, self.ffnn)
