import torch
import torch.nn as nn
import math

class PositionalEncoding(nn.Module):
    """
    Implement the PE function.
    """

    def __init__(self, d_model, dropout, max_len=512):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p = dropout)

        # Compute the positional encodings once in log space.
        # Ref : https://ai.stackexchange.com/questions/41670/why-use-exponential-and-log-in-positional-encoding-of-transformer
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1)  # shape == [max_len x 1]
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model)  # shape == [ (d_model/2), ]
        )
        pe[:, 0::2] = torch.sin(position * div_term)  # shape == [max_len x (d_model/2) ]
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # shape == [1 x max_len, d_model]
        self.register_buffer("pe", pe)

    def forward(self, x):
        x = x + self.pe[:, : x.size(1)].requires_grad_(False)
        return self.dropout(x)
