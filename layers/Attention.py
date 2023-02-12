import torch
import math

def attention(query, key, value, mask=None, dropout=None):
    "Compute Scaled Dot Product Attention"

    d_k = query.size(-1) # Q = n x d_k
    scores = torch.matmul(query, key.transpose(-2,-1)) / math.sqrt(d_k)
    # scores = [n x d_k] ' [d_k x n](transposed one) / sqrt(d_k)

    if mask is not None:  # pad_masking
        scores = scores.masked_fill(mask == 0, -1e9)  # 0인 부분을 -1e9(-inf)로 채우기

    p_attn = scores.softmax(dim=-1)  # scores shape = [n x n]

    if dropout is not None:
        p_attn = dropout(p_attn)

    return torch.matmul(p_attn, value), p_attn