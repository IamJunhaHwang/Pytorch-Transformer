import torch.nn as nn
import copy

class Encoder(nn.Module):
    """Encoder is consist with N-layers"""

    def __init__(self, encoder_block, N):
        super(Encoder, self).__init__()
        self.layers = nn.ModuleList([copy.deepcopy(encoder_block) for _ in range(N)])  # N-Encoder Blocks

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out)

        return out