import torch.nn as nn


class PositionWiseFeedForwardLayer(nn.Module):
    """
    FFN(x) = max(0, xW1 + b1)W2 + b2

    Each of the layers in our encoder and decoder contains a fully connected feed-forward network.
    This consists of two linear transformations with a ReLU activation in between.
    """
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionWiseFeedForwardLayer, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(self.w_1(x).relu()))