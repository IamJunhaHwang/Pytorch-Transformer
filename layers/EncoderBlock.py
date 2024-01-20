import torch.nn as nn
from layers.SublayerConnection import SublayerConnection

class EncoderBlock(nn.Module):

    def __init__(self, size, self_attention, feed_forward, dropout):
        super(EncoderBlock, self).__init__()

        self.size = size
        self.self_attention = self_attention
        self.feed_forward = feed_forward
        self.sublayer = nn.ModuleList([SublayerConnection(size, dropout) for _ in range(2)])

    def forward(self, x, mask):
        """
        :param x:
        :param mask:
        :return:
        """

        # 첫 번째 sublayer, attention의 결과가 아닌 각 변수를 가진 attention 함수 자체를 넘겨주기 위해 lambda 사용
        x = self.sublayer[0](x, lambda x: self.self_attention(x, x, x, mask))

        # 두 번째 sublayer
        return self.sublayer[1](x, self.feed_forward)
