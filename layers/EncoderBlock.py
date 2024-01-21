import torch.nn as nn
from layers.SublayerConnection import SublayerConnection


class EncoderBlock(nn.Module):
    """
    Each layer has two sub-layers: 1) multi-head self-attention mechanism,
                                   2) position-wise fully connected feed-forward network.
    """

    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderBlock, self).__init__()

        self.size = size
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        :param x: input for residual connection
        :param mask: mask for padding
        :return:
        """

        # 첫 번째 sublayer, attention의 출력 값이 아닌 attention 함수 자체를 넘겨주기 위해 lambda 사용
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))

        # 두 번째 sublayer
        return self.sublayer[1](x, self.feed_forward)
