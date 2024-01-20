import torch
import math


def attention(query, key, value, mask=None, dropout=None):
    """Compute Scaled Dot Product Attention
    query, key, value == [n_batch, h, seq_len, d_k]
    batch_size : mini-batch size
    h : # of head
    seq_len : input sequence length
    d_k : dimension of Key(Q, V도 같음)
    """

    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # scores = [n_batch * h * seq_len * d_k] matmul [n_batch * h * d_k * seq_len](transposed) / sqrt(d_k)
    # scores == [n_batch * h * seq_len * seq_len]

    if mask is not None:  # pad_masking
        scores = scores.masked_fill(mask == 0, -1e9)  # 0인 부분을 -1e9(-inf)로 채우기
        # masked_fill : https://pytorch.org/docs/stable/generated/torch.Tensor.masked_fill_.html

    p_attn = scores.softmax(dim=-1)  # scores shape = [n_batch * h * seq_len * seq_len]

    if dropout is not None:
        p_attn = dropout(p_attn)

    # return = [n_batch * h * seq_len * seq_len] matmul [n_batch * h * seq_len * d_k]
    # return == [n_batch * h * seq_len * d_k]
    return torch.matmul(p_attn, value), p_attn
