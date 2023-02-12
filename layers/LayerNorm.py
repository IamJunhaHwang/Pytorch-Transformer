import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """
    Layer Normalization

    This method was introduced by Ba et al. in https://arxiv.org/abs/1607.06450
    Example: https://paperswithcode.com/method/layer-normalization
    """

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))  # scale
        self.b_2 = nn.Parameter(torch.zeros(features))  # bias
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2