import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from math import sqrt
import os


class TriangularCausalMask():
    """Regular Triangular Masking
    """
    def __init__(self, B, L, device="cpu"):
        """Constructor

        Args:
            B (int): Batch size
            L (int): sequence_length
            device (str, optional): _description_. Defaults to "cpu".
        """
        # B1LL because B is batch LL create a square matrix 1 helps for broadcasting used in masked_fill later on/
        mask_shape = [B, 1, L, L]
        with torch.no_grad():
            self._mask = torch.triu(torch.ones(mask_shape, dtype=torch.bool), diagonal=1).to(device)

    @property
    def mask(self):
        return self._mask


class AnomalyAttention(nn.Module):
    """Anamoly Attention class describe in the paper
    """
    def __init__(self, win_size, mask_flag=True, scale=None, attention_dropout=0.0, output_attention=False):
        """Constructor
        """
        super(AnomalyAttention, self).__init__()
        self.scale = scale
        self.mask_flag = mask_flag #mask flag
        self.output_attention = output_attention # output attention flag
        self.dropout = nn.Dropout(attention_dropout)
        window_size = win_size #token size
        self.distances = torch.zeros((window_size, window_size)).cuda() #prior distance calculator
        for i in range(window_size):
            for j in range(window_size):
                self.distances[i][j] = abs(i - j) # how far away i and j are

    def forward(self, queries, keys, values, sigma, attn_mask):
        """How this class deal with the input

        Args:
            queries (Tensor): Queries of Transformer
            keys (Tensor): Key of Transformer
            values (Tensor): Value of Transformer
            sigma (Tensor): Prior Association Sigma
            attn_mask (Tensor): Attention mask

        Returns:
            tuple: Reconstruction, series prior and sigma
        """
        B, L, H, E = queries.shape # Batch, Seq_length, Head, Embedding
        _, S, _, D = values.shape # Batch same as quries, sequence length of values, Head same as queries, Embedding of value as D
        scale = self.scale or 1. / sqrt(E) #E here is the embedding size

        scores = torch.einsum("blhe,bshe->bhls", queries, keys) #dot product between the embedding dimension of queries and keys Matmul
        if self.mask_flag: #If require an attention mask create a traingular causal mask
            if attn_mask is None:
                attn_mask = TriangularCausalMask(B, L, device=queries.device)
            scores.masked_fill_(attn_mask.mask, -np.inf) #fill masking position with negative inf for softmax since softmax of negative inf become 0.
        attn = scale * scores #apply the scale to the scores (query * value)

        sigma = sigma.transpose(1, 2)  # B L H ->  B H L #Batch Sequence length head to batch head seq_length
        window_size = attn.shape[-1]
        sigma = torch.sigmoid(sigma * 5) + 1e-5 #normalize sigma and adds numeric stability using 1e-5
        sigma = torch.pow(3, sigma) - 1 #make sigma larger make sigma start from 0
        sigma = sigma.unsqueeze(-1).repeat(1, 1, 1, window_size)  # B H L L #add new last dimension and repeat for window_size stamp for the last sequence length.
        prior = self.distances.unsqueeze(0).unsqueeze(0).repeat(sigma.shape[0], sigma.shape[1], 1, 1).cuda() #broadcast the distances
        prior = 1.0 / (math.sqrt(2 * math.pi) * sigma) * torch.exp(-prior ** 2 / 2 / (sigma ** 2)) #prior calculation from the paper

        series = self.dropout(torch.softmax(attn, dim=-1)) #Transformer final softmax
        V = torch.einsum("bhls,bshd->blhd", series, values) #KQ matmul V

        if self.output_attention:
            return (V.contiguous(), series, prior, sigma)
        else:
            return (V.contiguous(), None)


class AttentionLayer(nn.Module):
    """The full antteion Layer
    """
    def __init__(self, attention, d_model, n_heads, d_keys=None,
                 d_values=None):
        """Constructor

        Args:
            attention (nn layers): some attention layer we want to use (AnomalyAttention)
            d_model (int): dimension of the model
            n_heads (int): heads of transformer
            d_keys (int, optional): dimension of keys or else d_model//n_heads . Defaults to None.
            d_values (int, optional): dimension of values or else d_model//n_heads. Defaults to None.
        """
        super(AttentionLayer, self).__init__()

        d_keys = d_keys or (d_model // n_heads)
        d_values = d_values or (d_model // n_heads)
        self.norm = nn.LayerNorm(d_model)
        self.inner_attention = attention
        self.query_projection = nn.Linear(d_model,
                                          d_keys * n_heads)
        self.key_projection = nn.Linear(d_model,
                                        d_keys * n_heads)
        self.value_projection = nn.Linear(d_model,
                                          d_values * n_heads)
        self.sigma_projection = nn.Linear(d_model,
                                          n_heads)
        self.out_projection = nn.Linear(d_values * n_heads, d_model)

        self.n_heads = n_heads

    def forward(self, queries, keys, values, attn_mask):
        """Regular Transformer Forward

        Args:
            queries (Tensor): Queries of Transformer
            keys (Tensor): Key of Transformer
            values (Tensor): Value of Transformer
            attn_mask (Tensor): Attention mask

        Returns:
            tuple: Reconstruction, series prior and sigma
        """
        B, L, _ = queries.shape
        _, S, _ = keys.shape
        H = self.n_heads #how many head to split the embedding
        x = queries
        queries = self.query_projection(queries).view(B, L, H, -1) #Linear layer x*Wq
        keys = self.key_projection(keys).view(B, S, H, -1) #Linear layer x*Wk
        values = self.value_projection(values).view(B, S, H, -1) #Linear layer x*Wv
        sigma = self.sigma_projection(x).view(B, L, H) #Sigma is obtained from query passing through linear

        out, series, prior, sigma = self.inner_attention(
            queries,
            keys,
            values,
            sigma,
            attn_mask
        ) #Like regular attention after passing through the linear layer pass through the attention
        out = out.view(B, L, -1) #take out the head dimension to get back to embedding dimenion

        return self.out_projection(out), series, prior, sigma
