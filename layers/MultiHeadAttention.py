import torch.nn as nn
from layers.Attention import attention


class MultiHeadAttention(nn.Module):
    """
    h : num of head
    d_model : Whole model size
    d_k : 병렬 처리를 위해 나눈 차원; d_embed
    linears : Q, K ,V, Linear layer(attention 이후의)  [각 가중치 행렬 역할]
    """

    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model % h == 0, "d_model % h != 0"

        self.d_k = d_model // h
        self.h = h
        self.linears = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(4)])  # d_model = d_k * h
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        """
        query, key, value : Input sequence embeddings [n_batch * seq_len * d_embed(d_model)]
        mask : [n_batch * seq_len * seq_len]
        """

        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatch = query.size(0)

        # 1) Do all the linear projections in batch from d_model => h x d_k
        # x(q,k,v) == [n_batch * seq_len * d_embed],  linear == [d_model * d_model]; d_model == d_embed
        # -> lin(x) == [n_batch * seq_len * d_model]
        # -> lin(x).view == [n_batch * seq_len(-1이라 shape 계산 해서 남는 것 자동으로) * h * d_k]
        # -> lin(x).view.transpose == [n_batch * h * seq_len * d_k], attention 계산 함수가 끝쪽 shape이 [seq_len, d_k]인 것을 기대하므로
        query, key, value = [
            lin(x).view(nbatch, -1, self.h, self.d_k).transpose(1, 2)
            for lin, x in zip(self.linears, (query, key, value))  # Linear가 4갠데 unpack error가 안나나?
        ]

        # 2) Apply attention on all the projected vectors in batch.
        x, self.attn = attention(
            query, key, value, mask=mask, dropout=self.dropout
        )  # x == [n_batch * h * seq_len * d_k]

        # input shape[n_batch * seq_len * d_model]과 같게 만들어야 함
        # x.transpose == [n_batch * seq_len * h * d_k]
        # x.transpose.view == [n_batch * seq_len * d_model]
        x = (
            x.transpose(1, 2)
            .contiguous()
            .view(nbatch, -1, self.h * self.d_k)
        )

        del query
        del key
        del value

        # 마지막 linear를 거침 -> [n_batch * seq_len * d_model]
        return self.linears[-1](x)
